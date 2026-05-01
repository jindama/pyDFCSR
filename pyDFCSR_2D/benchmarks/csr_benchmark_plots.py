"""
Generate three benchmark charts from csr_benchmark_results.json:
  1. Log-log total compute time/step vs N (CPU vs GPU)
  2. Stacked bar: time/step by category (compute DF, compute CSR, apply)
  3. Stacked bar: CSR sub-breakdown (placeholder until sub-kernel instrumentation)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'csr_benchmark_results.json')) as f:
    data = json.load(f)

PARTICLE_COUNTS = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
CATEGORIES = ['compute_df', 'compute_csr', 'apply']
CAT_LABELS = ['Compute DF', 'Compute CSR', 'Apply wake']
CAT_COLORS = ['#4C72B0', '#DD8452', '#55A868']

N = np.array(PARTICLE_COUNTS)
N_labels = ['100k', '500k', '1M', '5M', '10M']


def get_vals(device):
    totals, dfs, csrs, apps = [], [], [], []
    for n in PARTICLE_COUNTS:
        d = data[f'{device}_{n}']
        totals.append(d['total']['mean'] * 1e3)
        dfs.append(d['compute_df']['mean'] * 1e3)
        csrs.append(d['compute_csr']['mean'] * 1e3)
        apps.append(d['apply']['mean'] * 1e3)
    return (np.array(totals), np.array(dfs),
            np.array(csrs), np.array(apps))


cpu_total, cpu_df, cpu_csr, cpu_app = get_vals('cpu')
gpu_total, gpu_df, gpu_csr, gpu_app = get_vals('cuda')

# ============================================================
# Chart 1: Log-log total time/step
# ============================================================
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.loglog(N, cpu_total, 'o-', color='#C44E52', linewidth=2, markersize=8, label='CPU (NumPy/SciPy)')
ax1.loglog(N, gpu_total, 's-', color='#4C72B0', linewidth=2, markersize=8, label='GPU (PyTorch, A100)')

for i, n in enumerate(N):
    speedup = cpu_total[i] / gpu_total[i]
    ax1.annotate(f'{speedup:.0f}x',
                 xy=(n, gpu_total[i]), xytext=(0, -18),
                 textcoords='offset points', ha='center', fontsize=9,
                 color='#4C72B0', fontweight='bold')

ax1.set_xlabel('Number of particles', fontsize=12)
ax1.set_ylabel('Time per step (ms)', fontsize=12)
ax1.set_title('CSR Computation Time per Step', fontsize=14)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, which='both', alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
ax1.set_xlim(7e4, 1.5e7)
fig1.tight_layout()
outpath = os.path.join(SCRIPT_DIR, 'csr_bench_loglog.png')
fig1.savefig(outpath, dpi=150)
print(f'Saved {outpath}')

# ============================================================
# Chart 2: Stacked bars — time/step by category, CPU vs GPU
# ============================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

x = np.arange(len(PARTICLE_COUNTS))
bar_width = 0.6

for ax, device_label, vals_df, vals_csr, vals_app, title in [
    (ax2a, 'CPU', cpu_df, cpu_csr, cpu_app, 'CPU (NumPy/SciPy)'),
    (ax2b, 'GPU', gpu_df, gpu_csr, gpu_app, 'GPU (PyTorch, A100)'),
]:
    bottom = np.zeros(len(PARTICLE_COUNTS))
    for vals, label, color in zip(
        [vals_df, vals_csr, vals_app], CAT_LABELS, CAT_COLORS
    ):
        bars = ax.bar(x, vals, bar_width, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(N_labels, fontsize=11)
    ax.set_xlabel('Number of particles', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

ax2a.set_ylabel('Time per step (ms)', fontsize=12)

# Use log scale so GPU bars are visible
ax2a.set_yscale('log')
ax2b.set_yscale('log')
ax2a.set_ylim(1, 15000)
ax2b.set_ylim(1, 15000)

fig2.suptitle('Per-Step Timing Breakdown', fontsize=14, y=1.02)
fig2.tight_layout()
outpath = os.path.join(SCRIPT_DIR, 'csr_bench_breakdown.png')
fig2.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')

# ============================================================
# Chart 2b: Same but linear scale, side-by-side CPU/GPU per N
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 5.5))

bar_width = 0.35
x = np.arange(len(PARTICLE_COUNTS))

# CPU bars
bottom_cpu = np.zeros(len(PARTICLE_COUNTS))
for vals, label, color in zip(
    [cpu_df, cpu_csr, cpu_app], CAT_LABELS, CAT_COLORS
):
    ax3.bar(x - bar_width/2, vals, bar_width, bottom=bottom_cpu,
            label=f'{label}' if color == CAT_COLORS[0] else label,
            color=color, edgecolor='white', linewidth=0.5)
    bottom_cpu += vals

# GPU bars
bottom_gpu = np.zeros(len(PARTICLE_COUNTS))
for vals, label, color in zip(
    [gpu_df, gpu_csr, gpu_app], CAT_LABELS, CAT_COLORS
):
    ax3.bar(x + bar_width/2, vals, bar_width, bottom=bottom_gpu,
            color=color, edgecolor='white', linewidth=0.5,
            hatch='///', alpha=0.85)
    bottom_gpu += vals

# Annotate speedup
for i in range(len(PARTICLE_COUNTS)):
    speedup = cpu_total[i] / gpu_total[i]
    ax3.annotate(f'{speedup:.0f}x',
                 xy=(x[i] + bar_width/2, gpu_total[i]),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold', color='#4C72B0')

ax3.set_xticks(x)
ax3.set_xticklabels(N_labels, fontsize=11)
ax3.set_xlabel('Number of particles', fontsize=12)
ax3.set_ylabel('Time per step (ms)', fontsize=12)
ax3.set_title('Per-Step Timing Breakdown: CPU (solid) vs GPU (hatched)', fontsize=13)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CAT_COLORS[0], label='Compute DF'),
    Patch(facecolor=CAT_COLORS[1], label='Compute CSR'),
    Patch(facecolor=CAT_COLORS[2], label='Apply wake'),
    Patch(facecolor='gray', label='CPU (solid)'),
    Patch(facecolor='gray', hatch='///', alpha=0.85, label='GPU (hatched)'),
]
ax3.legend(handles=legend_elements, fontsize=10, loc='upper left')
ax3.set_yscale('log')
ax3.set_ylim(1, 15000)
ax3.grid(True, axis='y', alpha=0.3)

fig3.tight_layout()
outpath = os.path.join(SCRIPT_DIR, 'csr_bench_sidebyside.png')
fig3.savefig(outpath, dpi=150)
print(f'Saved {outpath}')

plt.close('all')
print('\nDone.')
