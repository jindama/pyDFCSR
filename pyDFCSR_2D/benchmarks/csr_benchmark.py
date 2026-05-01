"""
CSR GPU benchmark: per-step timing breakdown across particle counts.
Uses the full CSR2D.run() with I/O disabled, extracts per-step timings
by instrumenting the inner loop categories.
"""
import sys, time, json
sys.path.insert(0, '/home/lew/obedc/DFCSR/pyDFCSR')
import numpy as np
import torch
from pyDFCSR_2D.CSR import CSR2D
from pyDFCSR_2D.interfaces import to_numpy

CONFIG = '/tmp/tmph7koqv_v/config_full.yaml'
PARTICLE_COUNTS = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]


def make_beam(csr, n, device):
    np.random.seed(42)
    if device != 'cpu':
        make = lambda arr: torch.tensor(arr, device=device, dtype=torch.float64)
    else:
        make = lambda arr: arr
    csr.beam.particle = csr.beam.particle._replace(
        x=make(np.random.normal(0, 5e-5, n)),
        px=make(np.random.normal(0, 1e-6, n)),
        y=make(np.random.normal(0, 5e-5, n)),
        py=make(np.random.normal(0, 1e-6, n)),
        z=make(np.random.normal(0, 5e-5, n)),
        pz=make(np.random.normal(0, 1e-4, n)),
    )


def sync(device):
    if device != 'cpu':
        torch.cuda.synchronize()


def run_benchmark(n_particles, device):
    """Run the full simulation with I/O disabled, timing each category per step.
    Returns per-step averages over the bend steps only (where CSR is active)."""
    csr = CSR2D(CONFIG, device=device)
    csr.CSR_params.apply_CSR = 1
    csr.CSR_params.transverse_on = 1
    csr.CSR_params.compute_CSR = 1
    csr.CSR_params.write_beam = None
    csr.CSR_params.write_wakes = False
    make_beam(csr, n_particles, device)

    # Monkey-patch the run loop to collect per-step timings
    step_timings = []

    orig_get_DF = csr.DF_tracker.get_DF
    orig_append_DF = csr.DF_tracker.append_DF
    orig_append_interp = csr.DF_tracker.append_interpolant
    orig_build_interp = csr.DF_tracker.build_interpolant
    orig_get_mesh = csr.get_CSR_mesh
    orig_calc_csr = csr.calculate_2D_CSR_torch if csr.use_torch else csr.calculate_2D_CSR
    orig_apply = csr.beam.apply_wakes

    timing_acc = {}

    def timed_section(name, fn, *args, **kwargs):
        sync(device)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        sync(device)
        timing_acc[name] = timing_acc.get(name, 0) + (time.perf_counter() - t0)
        return result

    # Override methods to add timing
    def patched_get_DF(*a, **kw):
        return timed_section('compute_df', orig_get_DF, *a, **kw)

    def patched_append_DF(*a, **kw):
        return timed_section('compute_df', orig_append_DF, *a, **kw)

    def patched_append_interp(*a, **kw):
        return timed_section('compute_df', orig_append_interp, *a, **kw)

    def patched_build_interp(*a, **kw):
        return timed_section('compute_df', orig_build_interp, *a, **kw)

    def patched_get_mesh(*a, **kw):
        return timed_section('compute_csr', orig_get_mesh, *a, **kw)

    # We need to patch calculate_2D_CSR or calculate_2D_CSR_torch
    def patched_calc_csr(*a, **kw):
        return timed_section('compute_csr', orig_calc_csr, *a, **kw)

    def patched_apply(*a, **kw):
        return timed_section('apply', orig_apply, *a, **kw)

    csr.DF_tracker.get_DF = patched_get_DF
    csr.DF_tracker.append_DF = patched_append_DF
    csr.DF_tracker.append_interpolant = patched_append_interp
    csr.DF_tracker.build_interpolant = patched_build_interp
    csr.get_CSR_mesh = patched_get_mesh
    if csr.use_torch:
        csr.calculate_2D_CSR_torch = patched_calc_csr
    else:
        csr.calculate_2D_CSR = patched_calc_csr
    csr.beam.apply_wakes = patched_apply

    # Hook into update_statistics to capture per-step boundaries
    orig_update_stats = csr.update_statistics
    def patched_update_stats(step):
        # Save accumulated timings for this step, then reset
        if timing_acc:
            step_timings.append(dict(timing_acc))
        timing_acc.clear()
        return orig_update_stats(step)
    csr.update_statistics = patched_update_stats

    # Suppress print output
    import io as _io
    import contextlib
    with contextlib.redirect_stdout(_io.StringIO()):
        csr.run()

    # Only keep steps where CSR was computed (have 'compute_csr' key)
    csr_steps = [s for s in step_timings if 'compute_csr' in s]

    # Discard first 2 CSR steps (warmup / re-interpolation)
    if len(csr_steps) > 3:
        csr_steps = csr_steps[2:]

    result = {}
    for cat in ['compute_df', 'compute_csr', 'apply']:
        vals = [s.get(cat, 0) for s in csr_steps]
        result[cat] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    totals = [sum(s.get(c, 0) for c in ['compute_df', 'compute_csr', 'apply']) for s in csr_steps]
    result['total'] = {'mean': float(np.mean(totals)), 'std': float(np.std(totals))}
    result['n_steps'] = len(csr_steps)
    return result


if __name__ == '__main__':
    results = {}

    # GPU warmup (torch.compile)
    print("GPU warmup...")
    _ = run_benchmark(100_000, 'cuda')
    print("Warmup done.\n")

    for n in PARTICLE_COUNTS:
        print(f"=== N = {n:,} ===")
        for device in ['cpu', 'cuda']:
            label = f"{device}_{n}"
            print(f"  {device}...", end=' ', flush=True)
            t0 = time.perf_counter()
            results[label] = run_benchmark(n, device)
            elapsed = time.perf_counter() - t0
            r = results[label]
            print(f"done ({elapsed:.1f}s, {r['total']['mean']*1e3:.2f} ms/step, "
                  f"df={r['compute_df']['mean']*1e3:.2f} csr={r['compute_csr']['mean']*1e3:.2f} "
                  f"apply={r['apply']['mean']*1e3:.2f}, n_steps={r['n_steps']})")

    with open('/tmp/csr_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/csr_benchmark_results.json")

    print("\n" + "="*80)
    print(f"{'N':>12s}  {'CPU (ms)':>10s}  {'GPU (ms)':>10s}  {'Speedup':>8s}  "
          f"{'CPU df':>8s}  {'CPU csr':>8s}  {'CPU app':>8s}  "
          f"{'GPU df':>8s}  {'GPU csr':>8s}  {'GPU app':>8s}")
    print("-"*80)
    for n in PARTICLE_COUNTS:
        c = results[f'cpu_{n}']
        g = results[f'cuda_{n}']
        ct = c['total']['mean']*1e3
        gt = g['total']['mean']*1e3
        print(f"{n:>12,d}  {ct:>10.2f}  {gt:>10.2f}  {ct/gt:>7.1f}x  "
              f"{c['compute_df']['mean']*1e3:>8.2f}  {c['compute_csr']['mean']*1e3:>8.2f}  {c['apply']['mean']*1e3:>8.2f}  "
              f"{g['compute_df']['mean']*1e3:>8.2f}  {g['compute_csr']['mean']*1e3:>8.2f}  {g['apply']['mean']*1e3:>8.2f}")
