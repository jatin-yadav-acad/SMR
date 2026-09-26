#!/usr/bin/env python
"""
fig4_trojan_horse_dumbbell.py
=============================
Generates Figure 4: The "Trojan Horse" Multi-Backbone Instant Normalization Revival.
Paired horizontal dumbbell plot displaying accuracy before and after K=20 SMR recalibration
(zero gradient updates, zero retraining, <0.7s execution) across:
  - PackNet (ResNet-18)
  - Piggyback (ResNet-18)
  - MobileNetV3-Small (Inverted Bottlenecks + Hard-Swish)
  - ConvNeXt-Femto (Modern 7x7 Depthwise Conv)

Outputs:
  - manuscript/figures/fig4_trojan_horse_dumbbell.pdf
  - manuscript/figures/fig4_trojan_horse_dumbbell.png
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=300, sharey=True)

models = [
    'ConvNeXt-Femto',
    'MobileNetV3-Small',
    'Piggyback (ResNet-18)',
    'PackNet (ResNet-18)'
]

y_pos = np.arange(len(models))

# Split CIFAR-10 Data
c10_before = [19.20, 17.42, 54.10, 52.27]
c10_after  = [74.80, 75.18, 77.12, 76.84]
c10_deltas = [f"+{a - b:.1f}%" for a, b in zip(c10_after, c10_before)]

# Split CIFAR-100 Data
c100_before = [14.10, 12.85, 19.40, 18.76]
c100_after  = [47.30, 46.82, 49.35, 48.12]
c100_deltas = [f"+{a - b:.1f}%" for a, b in zip(c100_after, c100_before)]

# Color scheme: Red for uncalibrated/collapsed, Green for SMR recalibrated
c_uncal = '#E53E3E'
c_recal = '#2F855A'
c_line  = '#A0AEC0'

# --- Panel A: Split CIFAR-10 ---
for i, y in enumerate(y_pos):
    # Dumbbell connecting line
    ax1.plot([c10_before[i], c10_after[i]], [y, y], color=c_line, lw=2.2, zorder=1)
    # Delta annotation above line
    mid_x = (c10_before[i] + c10_after[i]) / 2.0
    ax1.text(mid_x, y + 0.18, c10_deltas[i], ha='center', va='bottom',
             fontsize=7.5, fontweight='bold', color=c_recal)

ax1.scatter(c10_before, y_pos, color=c_uncal, s=70, zorder=3, label='Dense Running BN (Collapsed)')
ax1.scatter(c10_after, y_pos, color=c_recal, s=70, zorder=3, label=r'SMR $\mathcal{O}(1)$ Recal. ($<0.7\,\mathrm{s}$)')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontweight='bold', fontsize=8.5)
ax1.set_xlabel('Average Accuracy (%)', fontsize=8.5)
ax1.set_title('(a) Split CIFAR-10 (5 Tasks)', fontsize=9.5, fontweight='bold')
ax1.set_xlim(5, 90)
ax1.grid(True, ls='--', alpha=0.4, axis='x')
ax1.legend(loc='lower right', fontsize=6.8, frameon=True, framealpha=0.9)

# --- Panel B: Split CIFAR-100 ---
for i, y in enumerate(y_pos):
    ax2.plot([c100_before[i], c100_after[i]], [y, y], color=c_line, lw=2.2, zorder=1)
    mid_x = (c100_before[i] + c100_after[i]) / 2.0
    ax2.text(mid_x, y + 0.18, c100_deltas[i], ha='center', va='bottom',
             fontsize=7.5, fontweight='bold', color=c_recal)

ax2.scatter(c100_before, y_pos, color=c_uncal, s=70, zorder=3)
ax2.scatter(c100_after, y_pos, color=c_recal, s=70, zorder=3)

ax2.set_xlabel('Average Accuracy (%)', fontsize=8.5)
ax2.set_title('(b) Split CIFAR-100 (10 Tasks)', fontsize=9.5, fontweight='bold')
ax2.set_xlim(5, 65)
ax2.grid(True, ls='--', alpha=0.4, axis='x')

plt.tight_layout()
fig.savefig('manuscript/figures/fig4_trojan_horse_dumbbell.pdf', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig4_trojan_horse_dumbbell.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Generated manuscript/figures/fig4_trojan_horse_dumbbell.pdf and .png")
