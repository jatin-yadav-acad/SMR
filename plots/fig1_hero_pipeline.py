#!/usr/bin/env python
"""
fig1_hero_pipeline.py
=====================
Generates Figure 1: The Hero System Overview Schematic for Sparse Mechanistic Routing (SMR).
Spans 3 chronological phases:
  Phase 1: Sensory-Decision Split (Dense shared visual primitives conv1..layer3)
  Phase 2: O(1) Forward BatchNorm Recalibration (<0.7s, K=20 micro-batches, zero backprop)
  Phase 3: Autonomous Conformal Test-Time Routing (O(1) Fast-Gate, |C(x)| <= 2, |M|=0)

Outputs:
  - manuscript/figures/fig1_hero_pipeline.pdf
  - manuscript/figures/fig1_hero_pipeline.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs('manuscript/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(10.0, 2.7), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis('off')

# Background container panels for 3 phases
# Phase 1: Blue tint
p1 = FancyBboxPatch((1, 2), 29, 31, boxstyle="round,pad=0.5,rounding_size=1.2",
                    facecolor='#F0F4F8', edgecolor='#B0C4DE', lw=1.5)
ax.add_patch(p1)

# Phase 2: Coral/Red tint
p2 = FancyBboxPatch((32, 2), 34, 31, boxstyle="round,pad=0.5,rounding_size=1.2",
                    facecolor='#FFF5F5', edgecolor='#FEB2B2', lw=1.5)
ax.add_patch(p2)

# Phase 3: Green tint
p3 = FancyBboxPatch((68, 2), 31, 31, boxstyle="round,pad=0.5,rounding_size=1.2",
                    facecolor='#F0FFF4', edgecolor='#9AE6B4', lw=1.5)
ax.add_patch(p3)

# Phase Headers
ax.text(15.5, 30.5, "Phase 1: Sensory-Decision Split", fontsize=10, fontweight='bold',
        color='#1A365D', ha='center', va='center')
ax.text(49.0, 30.5, r"Phase 2: Dual-Mode Signal Restoration", fontsize=10, fontweight='bold',
        color='#742A2A', ha='center', va='center')
ax.text(83.5, 30.5, "Phase 3: Autonomous Class-IL Routing", fontsize=10, fontweight='bold',
        color='#22543D', ha='center', va='center')

# Subtitle labels
ax.text(15.5, 28.5, r"Shared Visual Primitives ($\kappa^2 \leq 7.2\times 10^{-4}$)", fontsize=7.5,
        color='#4A5568', ha='center', va='center')
ax.text(49.0, 28.5, r"Conv/BN: $\mathcal{O}(1)$ Recalibration  |  Transformer/LN: Rank Slicing", fontsize=7.2,
        color='#4A5568', ha='center', va='center')
ax.text(83.5, 28.5, r"$\mathcal{O}(1)$ Fast-Gate + Conformal Triage", fontsize=7.5,
        color='#4A5568', ha='center', va='center')

# --- Phase 1 Details ---
# Input Image Box
b_in = FancyBboxPatch((2.5, 14), 4.5, 9, boxstyle="square,pad=0",
                      facecolor='#E2E8F0', edgecolor='#718096', lw=1.2)
ax.add_patch(b_in)
ax.text(4.75, 18.5, "Input\n$x$", fontsize=8, ha='center', va='center', fontweight='bold')

# Arrow
ax.annotate('', xy=(8.5, 18.5), xytext=(7.0, 18.5),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#4A5568'))

# Sensory Backbone
b_sens = FancyBboxPatch((8.5, 8), 12.5, 18, boxstyle="round,pad=0.2,rounding_size=0.6",
                        facecolor='#2B6CB0', edgecolor='#1A365D', lw=1.2)
ax.add_patch(b_sens)
ax.text(14.75, 21.5, "Sensory Backbone", fontsize=8.5, color='white', ha='center', fontweight='bold')
ax.text(14.75, 17.5, "conv1 to layer3", fontsize=7.5, color='#EBF8FF', ha='center')
ax.text(14.75, 14.0, "Shared Dense Primitives", fontsize=7, color='#EBF8FF', ha='center')
ax.text(14.75, 10.5, r"$\|\Delta h_{\mathrm{sens}}\|_2 \approx 0$ (Invariant)", fontsize=6.8, color='#BEE3F8', ha='center')

# Arrow to feature vector
ax.annotate('', xy=(23.5, 18.5), xytext=(21.0, 18.5),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#4A5568'))

# Feature vector
b_feat = FancyBboxPatch((23.5, 12), 4.8, 13, boxstyle="round,pad=0.2,rounding_size=0.4",
                        facecolor='#CBD5E0', edgecolor='#4A5568', lw=1.2)
ax.add_patch(b_feat)
ax.text(25.9, 19.5, r"$h_{\mathrm{sens}}(x)$", fontsize=8, ha='center', va='center', fontweight='bold')
ax.text(25.9, 14.5, r"$\mathbb{R}^{256}$", fontsize=7.5, ha='center', va='center', color='#2D3748')

# --- Connecting Arrow P1 -> P2 ---
ax.annotate('', xy=(33.5, 18.5), xytext=(28.5, 18.5),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='#3182CE'))

# --- Phase 2 Details ---
# Sub-circuits box (Branch 1: Conv/BatchNorm)
b_sub = FancyBboxPatch((33.5, 14.2), 14.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                       facecolor='#FFF', edgecolor='#E53E3E', lw=1.2)
ax.add_patch(b_sub)
ax.text(40.75, 23.2, "Regime 1: Conv / BN", fontsize=6.8, fontweight='bold', color='#9B2C2C', ha='center')
ax.text(40.75, 20.4, r"Pruned: $\rho_{\mathrm{in}}=0.08$ (8% width)", fontsize=6.0, color='#742A2A', ha='center')
ax.text(40.75, 17.8, r"Affine shift $\Delta\mu < 0$ (clamped)", fontsize=6.0, color='#E53E3E', ha='center', fontweight='bold')
ax.text(40.75, 15.4, r"$\to \mathcal{O}(1)$ Recal ($K{=}20$, $<0.7$s)", fontsize=6.0, color='#22543D', ha='center', fontweight='bold')

# Branch 2: Transformer / LayerNorm / RMSNorm
b_recal = FancyBboxPatch((50.0, 14.2), 14.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                         facecolor='#FEFCBF', edgecolor='#D69E2E', lw=1.2)
ax.add_patch(b_recal)
ax.text(57.25, 23.2, "Regime 2: ViTs / LLMs", fontsize=6.8, fontweight='bold', color='#744210', ha='center')
ax.text(57.25, 20.4, "Token-Dynamic LN/RMSNorm", fontsize=6.0, color='#744210', ha='center')
ax.text(57.25, 17.8, r"SMR-LoRA Rank Slicing ($r_t$)", fontsize=6.0, color='#B7791F', ha='center', fontweight='bold')
ax.text(57.25, 15.4, r"$\Theta_{\mathrm{dec}} \equiv 0$ | Zero Interference", fontsize=5.8, color='#744210', ha='center', fontweight='bold')

# Arrow between Branch 1 and Branch 2
ax.annotate('', xy=(50.0, 19.0), xytext=(48.0, 19.0),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color='#D69E2E'))

# Restored status badge below P2
b_badge = FancyBboxPatch((33.5, 3.0), 31.0, 10.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                         facecolor='#C6F6D5', edgecolor='#38A169', lw=1.2)
ax.add_patch(b_badge)
ax.text(49.0, 9.2, "Dual-Regime Continual Stability Guaranteed!", fontsize=7.0, fontweight='bold', color='#22543D', ha='center')
ax.text(49.0, 5.8, r"CNNs: Signal Revived ($18.8\% \to 48.1\%$)  |  LLMs: Exact Invariance ($\mathrm{FM} = 0.00\%$)", fontsize=5.6, color='#276749', ha='center')

# --- Connecting Arrow P2 -> P3 ---
ax.annotate('', xy=(69.5, 18.5), xytext=(64.5, 18.5),
            arrowprops=dict(arrowstyle="->", lw=1.8, color='#38A169'))

# --- Phase 3 Details ---
# Fast-Gate
b_fg = FancyBboxPatch((70.0, 14.2), 11.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                      facecolor='#276749', edgecolor='#1C4532', lw=1.2)
ax.add_patch(b_fg)
ax.text(75.75, 23.2, r"$\mathcal{O}(1)$ Fast-Gate", fontsize=7.2, color='white', ha='center', fontweight='bold')
ax.text(75.75, 20.2, r"Query Prototypes $\mu_{c,t}$", fontsize=6.2, color='#E2E8F0', ha='center')
ax.text(75.75, 17.5, r"Constant $10.10\,\mathrm{ms}$", fontsize=6.2, color='#9AE6B4', ha='center', fontweight='bold')
ax.text(75.75, 15.4, r"($9.2\times$ faster)", fontsize=5.8, color='#E2E8F0', ha='center')

# Conformal Triage Box
b_conf = FancyBboxPatch((83.5, 14.2), 14.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                        facecolor='#FFF', edgecolor='#319795', lw=1.2)
ax.add_patch(b_conf)
ax.text(90.75, 23.2, "Conformal Triage", fontsize=7.2, color='#234E52', ha='center', fontweight='bold')
ax.text(90.75, 20.2, r"$\geq 95.0\%$ Coverage (Full)", fontsize=6.0, color='#285E61', ha='center')
ax.text(90.75, 17.5, r"Truncated $|\mathcal{C}| \leq 2$: $81.4\% - 89.2\%$", fontsize=5.6, color='#234E52', ha='center', fontweight='bold')
ax.text(90.75, 15.4, r"Edge Pareto knee budget", fontsize=5.6, color='#319795', ha='center')

# Arrow between Fast-Gate and Conformal
ax.annotate('', xy=(83.5, 19.5), xytext=(81.5, 19.5),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#2D3748'))

# Final Decision Heads / Output below P3
b_out = FancyBboxPatch((70.0, 2.5), 28.0, 9.6, boxstyle="round,pad=0.2,rounding_size=0.4",
                       facecolor='#EBF8FF', edgecolor='#3182CE', lw=1.2)
ax.add_patch(b_out)
ax.text(84.0, 9.6, r"Autonomous Class-IL Prediction ($|\mathcal{M}|=0$)", fontsize=7.0, fontweight='bold', color='#1A365D', ha='center')
ax.text(84.0, 6.9, r"Evaluate target heads $t \in \mathcal{C}(x)$ via standardized fusion", fontsize=5.8, color='#2B6CB0', ha='center')
ax.text(84.0, 4.3, "Zero Replay Buffers  |  Zero DRAM Traffic  |  Zero Privacy Risk", fontsize=5.4, color='#4A5568', ha='center')

# Arrow from Conformal to Final Output
ax.annotate('', xy=(90.75, 12.1), xytext=(90.75, 14.2),
            arrowprops=dict(arrowstyle="->", lw=1.2, color='#3182CE'))

plt.tight_layout()
fig.savefig('manuscript/figures/fig1_hero_pipeline.pdf', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig1_hero_pipeline.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Generated manuscript/figures/fig1_hero_pipeline.pdf and .png")
