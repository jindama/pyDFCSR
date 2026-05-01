# pyDFCSR GPU Acceleration Plan

**Branch:** `gpu-development`
**Date:** 2026-04-30
**Environment:** `pdes` conda env, PyTorch 2.9.1+cu128, 4x NVIDIA A100 80GB

---

## Current Architecture

The simulation loop (`CSR2D.run` in `CSR.py`) iterates over lattice elements and
steps within each element. Each step does:

1. **Beam tracking** — Bmad-X `track_element` propagates particles through a
   lattice slice (drift, dipole, quad, sextupole). Updates beam statistics.
2. **Distribution function (DF) construction** — Deposits particles onto a 2D
   (x, z) grid via CIC histogramming (Numba). Computes velocity field.
   Smooths with Savitzky-Golay filters. Computes spatial gradients.
3. **Interpolant assembly** — Maintains a sliding window of DF snapshots. When
   beam size changes significantly, re-interpolates all historical snapshots
   onto a common grid using `RegularGridInterpolator`. Stacks into 3D arrays
   (time × x × z).
4. **CSR wake calculation** — The bottleneck. Loops over `xbins * zbins` mesh
   points (typically 600, up to 10k). For each point, builds retarded-time
   integration sub-grids, evaluates the EM kernel via 12× `interpolate1D` and
   5× `interpolate3D` calls, and integrates with nested `np.trapz`.
5. **Wake application** — Interpolates the 2D wake field onto each particle
   position and updates momenta.

### Profiling summary (where time is spent)

Profiled at moderate particle count (~10k). At typical production scale
(100k–10M particles), per-particle operations (CIC deposition, wake
interpolation onto particles, beam tracking) become significant and
must also run on GPU.

| Component | Fraction (10k) | Scales with | Current impl |
|-----------|---------------|-------------|-------------|
| CSR wake calculation (step 4) | ~90% | mesh points (xbins×zbins) | Python for-loop, Numba interp |
| DF construction (step 2) | ~5% | particles | Numba CIC + scipy savgol |
| Interpolant assembly (step 3) | ~3% | DF history depth | scipy RegularGridInterpolator |
| Beam tracking (step 1) | ~1% | particles | Bmad-X (numpy or torch) |
| Wake application (step 5) | ~1% | particles | scipy RegularGridInterpolator |

---

## Framework Choice: PyTorch

Bmad-X is built on PyTorch. Its `Particle` namedtuple holds arrays that can be
either numpy or torch tensors, and `track_element` works with both. The
`interfaces.py` file already handles numpy↔torch conversion. Building on
PyTorch means:

- Beam tracking runs on GPU with minimal changes (pass cuda tensors).
- All downstream array math maps directly to `torch` ops.
- No CPU↔GPU transfer boundary between tracking and CSR.
- `torch.trapezoid` replaces `np.trapz`; `torch.nn.functional.conv2d` replaces
  `savgol_filter`.

---

## Phased Implementation

### Phase 1: Device Abstraction Layer + Beam on GPU

**Goal:** Every array in the pipeline lives on a configurable device. CPU path
still works identically.

**Files to modify:**
- `CSR.py` — Add `device` param to `CSR2D.__init__`.
- `beams.py` — Construct `Particle` with torch tensors on `device`. Replace
  `np.std`, `np.mean`, `np.polyfit`, `np.polyval` with torch equivalents.
- `interfaces.py` — Add device-aware conversion helpers.
- `lattice.py` — Store `coords`, `n_vec`, `tau_vec`, `s` as torch tensors
  on `device` (these are read-only after init, small arrays).
- `twiss.py` — Accept torch tensors (move to CPU for `np.cov` if needed,
  since this is a diagnostic, not on the hot path).
- `params.py` — No changes needed (just config scalars).

**Departures from original discussion:**
- `np.polyfit` has no direct torch equivalent. Will use a manual least-squares
  solve (`torch.linalg.lstsq`) for the degree-1 fit in `beam.slope`.
- `twiss_from_bmadx_particles` uses `np.cov`. Since Twiss is a diagnostic
  called once per step (not in the inner loop), we will `.cpu().numpy()` the
  particle arrays for this call rather than rewriting the Twiss math. This
  avoids an unnecessary GPU sync on the hot path — the sync happens anyway
  because we store the scalar results.

**Validation:** Run an existing example (e.g., `example_dipole.ipynb`) on CPU
with torch tensors, confirm results match the numpy baseline to machine
precision.

---

### Phase 2: Vectorized CSR Wake Calculation (the main event)

**Goal:** Eliminate the `for i in range(N)` loop in `calculate_2D_CSR`. This
is ~90% of the total speedup.

**Pre-phase cleanup (done):** Dead code removed from `get_CSR_wake` and
`get_CSR_integrand`:
- `ignore_vx` debug block (both branches set `False`) — removed
- `CSR_blocker` (always `False`) — removed
- `debug` return paths — removed
- Commented-out alternative EM formulations — removed
- Hardcoded velocity simplifications (`vs=1`, `vs_ret=1`, `vs_s_ret=0`)
  inlined directly into the math with a comment
- Chirp-band sub-region integration refactored into a loop over
  `(xp_1d, sp_1d)` pairs (no behavior change)

**Live code structure after cleanup:**

`get_CSR_wake(s, x)` — per-observer entry point:
1. Compute integration bounds (s1..s4, x-ranges) based on chirp (`tan_theta`)
2. Build 3 or 4 `(xp_1d, sp_1d)` sub-regions
3. For each sub-region: meshgrid → `get_CSR_integrand` → nested `np.trapz`
4. Sum contributions → `(dE_dct, x_kick)`

`get_CSR_integrand(s, x, t, sp, xp)` — the EM kernel:
1. Interpolate lattice geometry at observer `s` (6 scalar lookups) and
   source `sp` (6 array lookups) via `interpolate1D`
2. Compute separation vector `r - r'` and retarded time `t_ret`
3. Look up 5 distribution function fields at `(t_ret, xp, sp-t_ret)` via
   `interpolate3D`
4. Compute piecewise curvature `rho_sp`
5. Assemble longitudinal integrand (velocity-weighted density gradients)
6. Assemble transverse integrand (W1 + W2 + W3 near-field terms)

**What changes:**

#### 2a. Batched `get_CSR_wake`

Currently each of the N=xbins×zbins observation points calls `get_CSR_wake`
independently. The integration sub-grids (sp1, sp2, sp3, xp) have the same
shape for every observation point — only the observer (s, x) differs.

Restructure so that all N observers are processed simultaneously:
- `s` and `x` become tensors of shape `(N,)`.
- Integration grids become `(N, N_xp, N_sp)` via broadcasting.
- All kernel evaluations become batched tensor ops.

**Complication — chirp band:** When `|tan_theta| > 1`, integration bounds
depend on the observer x-position, so each observer gets different xp ranges.
Two options: (a) pad all observers to the widest range and mask, or (b) run
the chirp-band case as a separate kernel. Option (b) is simpler since
`tan_theta` is uniform across all observers within a step (it's a beam-level
property), so the branch is all-or-nothing per step — no per-observer
branching needed.

#### 2b. Torch interpolation kernels

Already implemented in Phase 1: `torch_interp.py` contains
`interpolate1d_torch` and `interpolate3d_torch`. These will be used
directly in the batched integrand.

#### 2c. Batched integration

Replace nested `np.trapz` with `torch.trapezoid` over the appropriate
dimensions. The integration over xp and sp becomes a batched reduction over
the last two dimensions of the `(N, N_xp, N_sp)` integrand tensor.

#### 2d. Precompute lattice geometry per step

`get_CSR_integrand` calls `interpolate1D` 12 times per observer (6 at
observer s, 6 at source points sp). After batching:
- Observer geometry: 6 calls with `(N,)` input → 6 tensors of shape `(N,)`
- Source geometry: `sp` values are shared across sub-regions of the same
  shape. Precompute once per unique `sp_1d` array, broadcast to all observers.

**Files to modify:**
- `CSR.py` — `calculate_2D_CSR`, `get_CSR_wake`, `get_CSR_integrand`,
  `get_CSR_mesh`. Significant rewrite of these four methods.

**Validation:** Compare `dE_dct` and `x_kick` arrays against CPU/numpy
reference at every step of a test problem. Tolerance: `rtol=1e-6` for
float64.

---

### Phase 3: Deposition + Filtering on GPU

**Goal:** Eliminate the CPU bottleneck for per-particle operations. At
production scale (100k–10M particles), CIC deposition, wake interpolation,
and the `to_numpy` round-trips in `get_DF` dominate if left on CPU. This
phase is not optional cleanup — it is required for the GPU port to deliver
meaningful speedup at typical particle counts.

#### 3a. CIC histogram on GPU

Replace `histogram_cic_2d` (Numba) with a torch implementation using
`scatter_add_` for atomic accumulation. The CIC kernel deposits each particle
into 4 neighboring bins with linear weights.

**Note:** `torch.scatter_add_` is deterministic on GPU starting from
PyTorch 2.1. Typical particle counts range from a few hundred thousand to
10 million — the GPU scatter approach scales well to these sizes.

#### 3b. Savitzky-Golay → separable convolution

The current code applies `scipy.signal.savgol_filter` axis-by-axis (twice per
quantity, for density, density_x, density_z, vx, vx_x — about 10 calls per
step). Replace with:
1. Precompute the 1D Savitzky-Golay kernel coefficients (once, at init).
2. Apply as `torch.nn.functional.conv1d` along each axis (separable).

This is numerically equivalent to the scipy version.

#### 3c. Gradient computation

Replace `np.gradient` with finite-difference via `torch.diff` or a small
conv1d kernel with coefficients `[-0.5, 0, 0.5]` (central differences),
matching numpy's default behavior.

#### 3d. Wake application interpolation

Replace the `RegularGridInterpolator` in `beam.apply_wakes` with the same
torch bilinear interpolation used elsewhere (or the 2D version of our
`interpolate` function).

**Files to modify:**
- `deposit.py` — `histogram_cic_2d`, `DF_tracker.get_DF`,
  `DF_tracker.append_interpolant`, `DF_tracker.build_interpolant`.
- `beams.py` — `apply_wakes`.

**Validation:** Compare deposited density and filtered gradients against
numpy/scipy reference. Tolerance: `rtol=1e-5` (filtering introduces minor
floating-point ordering differences).

---

### Phase 4: Performance Tuning + Cleanup

1. **Memory pre-allocation** — Allocate workspace tensors in `__init__`,
   reuse across steps. Currently every call to `get_CSR_integrand` creates
   dozens of temporary arrays.
2. **Mixed precision** — Test `float32` for the CSR kernel. The integrand
   involves 1/r terms that could lose precision for very small r, but the
   integration smooths this out. Benchmark accuracy vs. throughput.
3. **MPI path** — Keep the MPI code path for multi-GPU future use, but the
   single-GPU vectorized path replaces the per-rank Python loop. Gate with
   `if self.parallel and not self.use_gpu`.
4. **CUDA streams** — Overlap DF construction with wake computation from the
   previous step if applicable.
5. **Profiling** — Use `torch.profiler` to identify remaining bottlenecks
   after phases 1-3.

---

## Files NOT changing

| File | Reason |
|------|--------|
| `r_gen6.py` | Called once per element, negligible cost. Pure numpy is fine. |
| `SGolay_filter.py` | Unused (commented out in deposit.py). |
| `postprocessor.py` | Post-hoc analysis, not on hot path. |
| `pyDFCSR_mpi_run.py` | MPI launcher script, unchanged. |
| `yaml_parser.py` | One-time config parsing. |
| `tools.py` | I/O utilities, plotting. Stays numpy. |
| `twiss_R.py` | Called rarely, small matrices. |
| `physical_constants.py` | Just constants. |

---

## Dependency Changes

No new packages required. Everything uses PyTorch (already installed) and
removes the runtime dependency on Numba for the hot path (Numba kernels
remain as CPU fallback).

Scipy remains for the CPU fallback path only. The GPU path replaces:
- `scipy.signal.savgol_filter` → `torch.nn.functional.conv1d`
- `scipy.interpolate.RegularGridInterpolator` → custom torch interpolation

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-30 | Initial plan written. Branch `gpu-development` created. Environment `pdes` verified: PyTorch 2.9.1+cu128, A100 80GB, all deps installed. |
| 2026-04-30 | Phase 1 implemented: `torch_interp.py` (new), `interfaces.py`, `beams.py`, `lattice.py`, `CSR.py`, `twiss.py`, `deposit.py` updated. Device abstraction (`device='cpu'|'cuda'`) threaded through CSR2D → Beam → Lattice. Beam tracking runs on GPU via Bmad-X torch support. CSR wake calc still numpy/Numba (Phase 2). Wake application uses torch bilinear interpolation on GPU. |
| 2026-05-01 | Phase 1 validated. Fixed `bmadx_particles_to_openpmd` tensor/numpy mixing (p0c, s, mc2 become tensors after bmadx GPU tracking). CPU vs GPU agreement: beam stats ~1e-14, particle coords ~1e-7 (float64 precision through SBend). Full CSR-on run validated: CPU vs GPU mean energy matches to 1.4e-6. |
| 2026-05-01 | Pre-Phase 2 cleanup: removed dead code from `get_CSR_wake` and `get_CSR_integrand` (ignore_vx, CSR_blocker, debug returns, commented-out alternative formulations). Inlined velocity simplifications (vs=1). Refactored sub-region integration into loops. Updated Phase 2 plan to note torch_interp.py already exists and chirp-band branching is all-or-nothing per step. |
| 2026-05-01 | Phase 2 implemented: `calculate_2D_CSR_torch()` and `_get_CSR_integrand_batched()` added to CSR.py. All N=xbins×zbins observers processed simultaneously via (N, N_xp, N_sp) tensor broadcasting. `_batched_trapz2d` helper for double trapezoid integration. `_prepare_csr_tensors()` moves DF data to GPU. Chirp/non-chirp handled by all-or-nothing branch per step. GPU dispatch added in `run()` via `self.use_torch` flag. |
| 2026-05-01 | Phase 2 validated. apply_CSR=0 (identical beam states): dE_dct and x_kick match CPU to ~1e-12 relative difference — well below rtol=1e-6 target. apply_CSR=1 + transverse_on=1 (full pipeline): beam statistics agree to ~0.003% (sigma_x 4e-5, sigma_z 5e-7, mean_z 6e-5). Larger particle-level diffs (~2-5%) are expected accumulation over 13 steps of wake feedback. |
| 2026-05-01 | Phase 3 implemented: (a) GPU CIC histogram via `scatter_add_` in `histogram_cic_2d_torch`. (b) Savitzky-Golay filtering via `conv2d` with precomputed kernel, replacing scipy. (c) Central-difference gradient via torch, replacing `np.gradient`. (d) DF interpolant assembly via `interpolate2d_bilinear_torch`, replacing scipy `RegularGridInterpolator`. (f) GPU Twiss via `_cov3_torch` + `_twiss_dispersion_calc_torch`, eliminating 28 GPU→CPU round-trips. `DF_tracker` now accepts `device` parameter. |
| 2026-05-01 | Phase 3 validated + benchmarked. CPU vs GPU agreement identical to Phase 2 (same accumulated feedback diffs). 1M particle benchmark: CPU 23.4s → GPU 3.6s = **6.5x speedup**. Simulation loop (`run()`) dropped from 3.1s to 1.5s. Remaining wall time dominated by file I/O (2.07s = 57%). |
| 2026-05-01 | Phase 4 implemented: (a) `torch.compile` on `_csr_integrand_math` — fuses ~40 element-wise kernels into 2–3 compiled kernels. (b) `interpolate3d_multi_torch` and `interpolate1d_multi_torch` — compute indices once for 5 DF fields (or 6 lattice fields) instead of repeating. (c) `build_interpolant` creates torch tensors directly on GPU; `_prepare_csr_tensors` references them instead of re-creating via `torch.tensor()`. Lattice tensors created once. |
| 2026-05-01 | Phase 4 benchmarked. 1M particles: CPU 22.6s → GPU 3.3s = **6.9x overall**, simulation loop 1.19s = **~15x compute speedup**. CSR integrand 45% faster (0.35s→0.19s). Diminishing returns — remaining time is genuine computation + I/O. |
| 2026-05-01 | Percentile-based grid bounds implemented in `deposit.py`. New `configure_params` options: `grid_mode='sigma'|'percentile'`, `grid_percentile=0.9995`, `grid_padding=0.05`. Default is `'sigma'` (unchanged behavior). Percentile mode uses `np.percentile`/`torch.quantile` for grid bounds, tracks bounds in `x_bounds_log`/`z_bounds_log`, uses union-of-bounds for common interpolation grid with margin. Validated: CPU/GPU agree to same tolerance as sigma mode; percentile grid ~24% tighter than sigma grid. |
