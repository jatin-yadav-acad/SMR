#!/usr/bin/env python
"""
Generate publication-quality vector PDF/PNG Figure 5:
Empirical Depth Analysis of Sparse Mechanistic Routing (SMR).
Panels:
(a) Recalibration Sample Efficiency (K Micro-Batches vs. Accuracy Recovery & Clamping)
(b) Layer-Wise Normalization Paradox & Sensory-Decision Partitioning Proof
(c) Capacity-Sparsity Frontier (Retention ratio rho vs. Final AA and T_max)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 7
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "results_final", "depth_experiments.json")
OUT_DIR = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_depth_figure():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.0), dpi=300)

    # --- PANEL A: K-SWEEP ---
    k_data = data["recalibration_k_sweep"]
    ks = [d["K_micro_batches"] for d in k_data]
    accs = [d["accuracy"] for d in k_data]
    clamps = [d["negative_clamped_pct"] for d in k_data]

    ax1_twin = ax1.twinx()
    l1 = ax1.plot(ks, accs, color="#10b981", marker="o", lw=2.2, label="Test Accuracy (%)")
    l2 = ax1_twin.plot(ks, clamps, color="#ef4444", marker="s", lw=1.8, linestyle="--", label="Clamped Signals (%)")

    ax1.axvline(20, color="#6366f1", linestyle=":", lw=1.8, label="Optimal K=20")
    ax1.annotate("Optimal K=20\n(97.70% Plateau)", xy=(20, 97.7), xytext=(24, 95.2),
                 arrowprops=dict(facecolor='#6366f1', edgecolor='#4338ca', shrink=0.08, width=1.2, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#eef2ff", ec="#6366f1", lw=1.0),
                 fontsize=8.5, fontweight='bold', color='#3730a3')

    ax1.set_xlabel("Calibration Micro-Batches ($K$)", fontweight='bold')
    ax1.set_ylabel("Task 0 Accuracy (%)", fontweight='bold', color="#065f46")
    ax1_twin.set_ylabel("Pre-ReLU Clamped Signals (%)", fontweight='bold', color="#991b1b")
    ax1.set_title("(a) Recalibration Sample Efficiency", fontweight='bold')
    ax1.set_ylim(93, 99)
    ax1_twin.set_ylim(6, 15)
    ax1.grid(True)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)

    # --- PANEL B: LAYER-WISE PROFILE ---
    lw_data = data["layerwise_clamping_profile"]
    layers = ["layer1\n(Conv 64)", "layer2\n(Conv 128)", "layer3\n(Conv 256)", "layer4\n(Conv 512)"]
    uncal_accs = [d["uncal_acc"] for d in lw_data]
    recal_accs = [d["recal_acc"] for d in lw_data]

    x = np.arange(len(layers))
    w = 0.32
    ax2.bar(x - w/2, uncal_accs, w, label="Uncalibrated Pruned", color="#f87171", edgecolor="#dc2626")
    ax2.bar(x + w/2, recal_accs, w, label="Recalibrated (SMR)", color="#34d399", edgecolor="#059669")
    ax2.axhline(50.0, color="#64748b", linestyle=":", lw=1.5, label="Chance Level (50%)")

    ax2.annotate("Sensory Breakdown\n(Early Layers Collapse Primitives)",
                 xy=(1, 50), xytext=(0.1, 28),
                 arrowprops=dict(facecolor='#ef4444', edgecolor='#b91c1c', shrink=0.08, width=1.2, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#ef4444", lw=1.0),
                 fontsize=8, fontweight='bold', color='#991b1b')

    ax2.annotate("Decision Routing\n(97.7% Restored)",
                 xy=(3, 97.7), xytext=(2.1, 104),
                 arrowprops=dict(facecolor='#10b981', edgecolor='#047857', shrink=0.08, width=1.2, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ecfdf5", ec="#10b981", lw=1.0),
                 fontsize=8, fontweight='bold', color='#065f46')

    ax2.set_xlabel("Convolutional Stage", fontweight='bold')
    ax2.set_ylabel("Task Accuracy (%)", fontweight='bold')
    ax2.set_title("(b) Sensory-Decision Layer Profile", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_ylim(20, 114)
    ax2.grid(axis='y')
    ax2.legend(loc="lower right", framealpha=0.9)

    # --- PANEL C: CAPACITY-SPARSITY FRONTIER ---
    cs_data = data["capacity_sparsity_sweep"]
    rhos = [d["rho"] for d in cs_data]
    final_aas = [d["final_AA"] for d in cs_data]
    t_maxs = [d["T_max_theoretical"] for d in cs_data]

    ax3_twin = ax3.twinx()
    l3 = ax3.plot([r*100 for r in rhos], final_aas, color="#3b82f6", marker="^", lw=2.2, label="Final AA (%)")
    l4 = ax3_twin.step([r*100 for r in rhos], t_maxs, color="#f59e0b", lw=2.0, where="mid", linestyle="-.", label="Orthogonal Capacity ($T_{max}$)")

    ax3.set_xlabel("Channel Retention Ratio $\\rho$ (%)", fontweight='bold')
    ax3.set_ylabel("Split CIFAR-10 Final AA (%)", fontweight='bold', color="#1e40af")
    ax3_twin.set_ylabel("Max Orthogonal Tasks ($T_{max}$)", fontweight='bold', color="#92400e")
    ax3.set_title("(c) Capacity-Sparsity Frontier", fontweight='bold')
    ax3.set_ylim(90, 96)
    ax3_twin.set_ylim(0, 24)
    ax3.grid(True)

    lines3 = l3 + l4
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc="center right", framealpha=0.9)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig5_depth_analysis.pdf")
    png_path = os.path.join(OUT_DIR, "fig5_depth_analysis.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Generated {pdf_path} and {png_path} successfully!")


if __name__ == '__main__':
    generate_depth_figure()
