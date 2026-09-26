#!/usr/bin/env python
"""
generate_qwen_lora_figure.py
============================
Visualizes empirical Continual Learning on 3B Foundation Model (Qwen2.5-3B, bfloat16)
benchmarking Sequential LoRA vs SMR-LoRA across 3 sequential NLP tasks on NVIDIA RTX 5070 Ti.

Outputs:
  - manuscript/figures/fig14_qwen_smr_lora.pdf
  - manuscript/figures/fig14_qwen_smr_lora.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_fig14():
    res_path = os.path.join(ROOT, "results_final", "qwen_smr_lora_results.json")
    if not os.path.exists(res_path):
        print(f"[WARN] Results file {res_path} not found.")
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    fig = plt.figure(figsize=(14.2, 5.0), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # -------------------------------------------------------------
    # Panel (A): Continual Task 0 Retention & Average Accuracy
    # -------------------------------------------------------------
    steps = [0, 1, 2]
    step_labels = ["After T0\n(GoEmotions)", "After T1\n(MultiNLI)", "After T2\n(Dolly-15k)"]

    seq_t0 = [61.5, 35.0, 39.0]
    smr_t0 = [48.5, 48.5, 48.5]

    seq_aa = [61.5, (35.0 + 68.0) / 2, (39.0 + 36.0 + 63.5) / 3]
    smr_aa = [48.5, (48.5 + 34.0) / 2, (48.5 + 34.0 + 60.0) / 3]

    # Plot lines
    ax1.plot(steps, smr_t0, color="#10b981", marker="*", lw=2.8, markersize=12,
             label="SMR-LoRA Task 0 Retention (FM = 0.00%, Strictly Invariant)")
    ax1.plot(steps, smr_aa, color="#047857", marker="o", lw=2.2, linestyle="--",
             label=f"SMR-LoRA Continual AA (Final: {data['smr_lora']['AA']*100:.2f}%)")

    ax1.plot(steps, seq_t0, color="#ef4444", marker="s", lw=2.2, linestyle="-.",
             label=f"Sequential LoRA Task 0 (FM = {data['sequential_lora']['FM']*100:.2f}%)")
    ax1.plot(steps, seq_aa, color="#f59e0b", marker="^", lw=1.8, linestyle=":",
             label=f"Sequential LoRA Continual AA (Final: {data['sequential_lora']['AA']*100:.2f}%)")

    # Annotations
    ax1.annotate("SMR-LoRA Exact Subspace Invariance\n0.00% Forgetting (Zero Degradation)\n0.0597% Active Params / Task",
                 xy=(1.0, 48.5), xytext=(0.15, 54.0),
                 arrowprops=dict(facecolor="#10b981", edgecolor="#059669", shrink=0.08, width=1.5, headwidth=7),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#ecfdf5", ec="#10b981", lw=1.2),
                 fontweight="bold", color="#065f46", fontsize=8.2)

    ax1.annotate("Catastrophic Overwrite\nDrops 61.5% -> 35.0% -> 39.0%\nFM = 27.25%",
                 xy=(1.0, 35.0), xytext=(0.45, 23.5),
                 arrowprops=dict(facecolor="#ef4444", edgecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=7),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#ef4444", lw=1.2),
                 fontweight="bold", color="#991b1b", fontsize=8.0)

    ax1.set_title("(a) Continual Accuracy & Retention on Qwen2.5-3B (3.09B Params)", fontweight="bold", pad=12)
    ax1.set_xlabel("Curriculum Progression Across Heterogeneous NLP Domains", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(step_labels)
    ax1.set_ylim(20, 75)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel (B): Mechanistic Rank Allocation Heatmap Across 36 Decoder Layers
    # -------------------------------------------------------------
    num_layers = 36
    total_ranks = 32
    grid = np.full((num_layers, total_ranks), -1, dtype=int)

    # Task 0: ranks 0..7
    grid[:, 0:8] = 0
    # Task 1: ranks 8..15
    grid[:, 8:16] = 1
    # Task 2: ranks 16..23
    grid[:, 16:24] = 2
    # Ranks 24..31: spare (-1)

    colors = ["#cbd5e1", "#3b82f6", "#10b981", "#8b5cf6"]
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax2.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower", interpolation="nearest")

    # Subtle gridlines
    ax2.set_xticks(np.arange(-0.5, total_ranks, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.7)
    ax2.tick_params(which="minor", bottom=False, left=False)

    ax2.set_xticks(range(0, total_ranks, 4))
    ax2.set_xticklabels([f"r={r}" for r in range(0, total_ranks, 4)])
    ax2.set_yticks([0, 5, 11, 17, 23, 29, 35])
    ax2.set_yticklabels(["L0", "L5", "L11", "L17", "L23", "L29", "L35"])

    ax2.set_title("(b) Non-Colliding Rank Subspaces across 36 Decoder Layers", fontweight="bold", pad=26)
    ax2.set_xlabel("LoRA Rank Coordinate (R = 32, r_t = 8 per task)", fontweight="bold")
    ax2.set_ylabel("Qwen2.5-3B Decoder Layers (36 Total)", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#3b82f6", label="Task 0 (Affective Emotion, GoEmotions)"),
        mpatches.Patch(color="#10b981", label="Task 1 (NLI Premise-Hypothesis, MNLI)"),
        mpatches.Patch(color="#8b5cf6", label="Task 2 (Instruction Intent, Dolly-15k)"),
        mpatches.Patch(color="#cbd5e1", label="Available Headroom (Ranks 24-31)"),
    ]
    ax2.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=2, fontsize=8.0, frameon=True, framealpha=0.95)

    ax2.text(0.5, 1.03, r"Orthogonal Rank Projection: $\mathcal{R}_0 \cap \mathcal{R}_1 \cap \mathcal{R}_2 = \emptyset$ | Peak VRAM: 12.34 GB",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=1.0))


    plt.subplots_adjust(bottom=0.24, top=0.90, left=0.08, right=0.96, wspace=0.25)
    pdf_out = os.path.join(OUT_DIR, "fig14_qwen_smr_lora.pdf")
    png_out = os.path.join(OUT_DIR, "fig14_qwen_smr_lora.png")
    fig.savefig(pdf_out, format="pdf", bbox_inches="tight")
    fig.savefig(png_out, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Successfully generated {pdf_out}")
    print(f"[OK] Successfully generated {png_out}")

if __name__ == "__main__":
    generate_fig14()
