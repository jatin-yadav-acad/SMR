#!/usr/bin/env python
"""
fig5_silicon_pareto.py
======================
Generates Figure 5: Physical Silicon & CMOS Energy Dual Pareto Frontiers.
Panel A: Measured Step Energy on RTX 5070 Ti Silicon vs Terminal Class-IL Accuracy (Split CIFAR-100).
         Highlighting SMR/D-SMR in upper-left quadrant (2085.74 mJ, 52.40% AA).
Panel B: Task Horizon T in {1..50} vs Cumulative 28nm CMOS Energy Dissipation (Joules).
         Showing SMR linear scaling O(T) (357.5 J) vs Replay quadratic scaling O(T^2) (4836.5 J, 13.53x gap).

Outputs:
  - manuscript/figures/fig5_silicon_pareto.pdf
  - manuscript/figures/fig5_silicon_pareto.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('manuscript/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.titlesize': 11
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 1.95), dpi=300)

# =====================================================================
# Panel A: Physical Silicon Energy vs Class-IL Accuracy (RTX 5070 Ti)
# =====================================================================
# Methods and points: (Energy mJ, Class-IL AA %, label, color, marker)
pts = [
    (2085.74, 52.40, 'D-SMR (Plastic)', '#2F855A', '*', 180),
    (2085.74, 44.60, 'D-SMR (Frozen)', '#38A169', 'o', 80),
    (2085.74, 38.80, 'SMR-OSM (Static)', '#2B6CB0', '^', 80),
    (2085.74, 18.45, 'Static SMR', '#68D391', 's', 70),
    (4183.26, 55.10, 'DER++ (Replay)', '#E53E3E', 'D', 70),
    (4310.50, 54.80, 'FOSTER (Replay)', '#C53030', '^', 70),
    (2760.23, 2.10, 'EWC (Fisher)', '#D69E2E', 'v', 70),
    (3343.32, 4.80, 'PackNet', '#DD6B20', 'P', 70),
    (2024.69, 3.42, 'Finetune', '#718096', 'X', 70),
]

for energy, acc, name, col, mark, size in pts:
    ax1.scatter(energy, acc, color=col, marker=mark, s=size, edgecolors='#1A202C', lw=0.6, zorder=4)
    # Staggered offsets to prevent any collision
    if name == 'D-SMR (Plastic)':
        ax1.annotate(name, (energy, acc), xytext=(energy + 70, acc + 3.2),
                     fontsize=6.8, fontweight='bold', color=col, va='bottom')
    elif name == 'D-SMR (Frozen)':
        ax1.annotate(name, (energy, acc), xytext=(energy + 70, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')
    elif name == 'SMR-OSM (Static)':
        ax1.annotate(name, (energy, acc), xytext=(energy + 70, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')
    elif name == 'Static SMR':
        ax1.annotate(name, (energy, acc), xytext=(energy + 70, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')
    elif name == 'DER++ (Replay)':
        ax1.annotate(name, (energy, acc), xytext=(energy - 35, acc - 5.5),
                     fontsize=6.8, fontweight='bold', color=col, ha='right', va='top')
    elif name == 'FOSTER (Replay)':
        ax1.annotate(name, (energy, acc), xytext=(energy + 35, acc + 1.5),
                     fontsize=6.8, fontweight='bold', color=col, ha='left', va='center')
    elif name == 'Finetune':
        ax1.annotate(name, (energy, acc), xytext=(energy + 60, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')
    elif name == 'EWC (Fisher)':
        ax1.annotate(name, (energy, acc), xytext=(energy + 60, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')
    elif name == 'PackNet':
        ax1.annotate(name, (energy, acc), xytext=(energy + 60, acc),
                     fontsize=6.8, fontweight='bold', color=col, va='center')

# Pareto frontier contour curve
pareto_x = [2085.74, 4183.26]
pareto_y = [52.40, 55.10]
ax1.plot(pareto_x, pareto_y, ls='--', color='#2F855A', alpha=0.7, lw=1.2)

ax1.axvspan(1850, 2220, color='#F0FFF4', alpha=0.45, zorder=0)
ax1.text(1870, 28, 'Buffer-Free SMR\n($2.01\\times$ Energy)', fontsize=6.6,
         color='#22543D', fontweight='bold', ha='left', va='center',
         bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#38A169", lw=0.8, alpha=0.92))

ax1.set_xlabel('Measured Step Energy (mJ on RTX silicon)', fontsize=8.5)
ax1.set_ylabel('Split CIFAR-100 Class-IL (%)', fontsize=8.5)
ax1.set_title('(a) Physical Silicon Energy vs. Accuracy', fontsize=9.5, fontweight='bold')
ax1.set_xlim(1850, 4850)
ax1.set_ylim(-2, 64)
ax1.grid(True, ls='--', alpha=0.4)

# =====================================================================
# Panel B: 28nm CMOS Energy Dissipation vs Task Horizon T in {1..50}
# =====================================================================
T = np.arange(1, 51)

# Analytical CMOS modeling:
# SMR: linear scaling 357.5 J at T=50 -> E(T) = 7.15 * T
# Replay: quadratic scaling 4836.5 J at T=50 -> E(T) = 1.9346 * T^2
e_smr = 7.15 * T
e_replay = 1.9346 * (T ** 2)

ax2.plot(T, e_smr, color='#2F855A', lw=2.0, label=r'SMR $\mathcal{O}(T)$ ($357.5\,\mathrm{J}$ total)')
ax2.plot(T, e_replay, color='#E53E3E', lw=2.0, ls='--', label=r'Replay $\mathcal{O}(T^2)$ ($4836.5\,\mathrm{J}$ total)')

# Fill area showing energy gap
ax2.fill_between(T, e_smr, e_replay, color='#FEB2B2', alpha=0.35, label=r'$13.53\times$ Energy Advantage')

ax2.text(35, 1300, r'$\mathbf{13.53\times}$ Energy Advantage' + '\n' + r'($4479.0\,\mathrm{J}$ off-chip DRAM saved)',
         ha='center', va='center', fontsize=6.8, fontweight='bold', color='#742A2A',
         bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec="#E53E3E", lw=0.9, alpha=0.92))

ax2.set_xlabel(r'Task Horizon $T$ ($1 \dots 50$ Tasks)', fontsize=8.5)
ax2.set_ylabel('Cumulative CMOS Energy (Joules)', fontsize=8.5)
ax2.set_title('(b) 28nm Edge CMOS Energy Horizon', fontsize=9.5, fontweight='bold')
ax2.set_xlim(1, 50)
ax2.set_ylim(0, 5200)
ax2.grid(True, ls='--', alpha=0.4)
ax2.legend(loc='upper left', fontsize=6.5, frameon=True, framealpha=0.92)

plt.tight_layout()
fig.savefig('manuscript/figures/fig5_silicon_pareto.pdf', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig5_silicon_pareto.png', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig5_hardware_pareto.pdf', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig5_hardware_pareto.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Generated manuscript/figures/fig5_silicon_pareto and fig5_hardware_pareto (.pdf and .png)")
