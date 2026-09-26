#!/usr/bin/env python
"""
generate_foundation_lora_figures.py
===================================
Direction 3: Modern Foundation Scaling via LoRA-Rank Mechanistic Routing (SMR-LoRA).

Generates publication-quality vector figures (PDF & 300 DPI PNG):
Figure 12: Modern Foundation Scaling via LoRA-Rank Mechanistic Routing
- Panel (A): Continual accuracy across sequential tasks (Sequential LoRA forgetting vs SMR-LoRA zero forgetting).
- Panel (B): Rank allocation heatmap across Transformer layers and tasks, showing orthogonal rank dimension specialization.

Outputs:
  - manuscript/figures/fig12_foundation_lora_routing.pdf
  - manuscript/figures/fig12_foundation_lora_routing.png
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

def generate_fig12():
    res_path = os.path.join(ROOT, "results_final", "foundation_lora_results.json")
    if not os.path.exists(res_path):
        print(f"[WARN] Results file {res_path} not found. Cannot generate Figure 12 yet.")
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    fig = plt.figure(figsize=(14.2, 5.0), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    steps = [0, 1, 2, 3, 4]
    step_labels = ["After T0", "After T1", "After T2", "After T3", "After T4"]

    # -------------------------------------------------------------
    # Panel (A): Continual Accuracy & Forgetting
    # -------------------------------------------------------------
    ft_data = data["full_finetune"]
    lora_data = data["sequential_lora"]
    smr_data = data["smr_lora"]

    # Task 0 Retention progression
    ft_t0 = [ft_data["task_accs"][str(s)][0] * 100 for s in steps]
    lora_t0 = [lora_data["task_accs"][str(s)][0] * 100 for s in steps]
    smr_t0 = [smr_data["task_accs"][str(s)][0] * 100 for s in steps]

    # Continual Average Accuracy (AA) progression
    ft_aa = [np.mean(ft_data["task_accs"][str(s)]) * 100 for s in steps]
    lora_aa = [np.mean(lora_data["task_accs"][str(s)]) * 100 for s in steps]
    smr_aa = [np.mean(smr_data["task_accs"][str(s)]) * 100 for s in steps]

    # Plot curves
    ax1.plot(steps, smr_t0, color="#10b981", marker="*", lw=2.8, markersize=11,
             label=f"SMR-LoRA Task 0 Retention (FM = {smr_data['FM']*100:.2f}%)")
    ax1.plot(steps, smr_aa, color="#047857", marker="o", lw=2.2, linestyle="--",
             label=f"SMR-LoRA Continual AA ({smr_data['AA']*100:.2f}%)")
    
    ax1.plot(steps, lora_t0, color="#ef4444", marker="s", lw=2.0, linestyle="-.",
             label=f"Sequential LoRA Task 0 (FM = {lora_data['FM']*100:.2f}%)")
    ax1.plot(steps, lora_aa, color="#f59e0b", marker="^", lw=1.8, linestyle=":",
             label=f"Sequential LoRA Continual AA ({lora_data['AA']*100:.2f}%)")

    ax1.plot(steps, ft_aa, color="#64748b", marker="d", lw=1.6, linestyle=":",
             label=f"Full Fine-Tuning Continual AA ({ft_data['AA']*100:.2f}%)")

    # Annotations
    t0_ret = smr_t0[0]
    ax1.annotate(f"SMR-LoRA Exact Invariance\n0.00% Forgetting Measure\n({t0_ret:.1f}% T0 Retention | Buffer-Free)",
                 xy=(3, t0_ret), xytext=(1.2, t0_ret - 12.0),
                 arrowprops=dict(facecolor="#10b981", edgecolor="#059669", shrink=0.08, width=1.5, headwidth=7),
                 bbox=dict(boxstyle="round,pad=0.35", fc="#ecfdf5", ec="#10b981", lw=1.2),
                 fontweight="bold", color="#065f46", fontsize=8.5)

    lora_final = lora_t0[-1]
    ax1.annotate(f"Catastrophic Forgetting\n(Drops to {lora_final:.1f}%)",
                 xy=(4, lora_final), xytext=(2.4, lora_final - 10.0),
                 arrowprops=dict(facecolor="#ef4444", edgecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=7),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#ef4444", lw=1.2),
                 fontweight="bold", color="#991b1b", fontsize=8.0)

    ax1.set_title("(a) Continual Accuracy & Forgetting on Split CIFAR-10", fontweight="bold")
    ax1.set_xlabel("Continual Training Progression", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(step_labels)
    ax1.set_ylim(40, 100)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8.2, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel (B): Mechanistic Rank Allocation Heatmap
    # -------------------------------------------------------------
    num_blocks = 6
    total_ranks = 32
    grid = np.full((num_blocks, total_ranks), -1, dtype=int)

    rank_allocs = smr_data.get("rank_allocations", {})
    for t_idx in range(5):
        t_key = f"task_{t_idx}"
        if t_key in rank_allocs:
            for b_idx in range(num_blocks):
                b_key = f"block_{b_idx}"
                if b_key in rank_allocs[t_key]:
                    for r in rank_allocs[t_key][b_key]:
                        grid[b_idx, r] = t_idx

    # Color palette: Task 0 (Blue), Task 1 (Teal), Task 2 (Purple), Task 3 (Amber), Task 4 (Rose), Spare (Slate)
    colors = ["#cbd5e1", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax2.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower", interpolation="nearest")

    # Add subtle gridlines between cells
    ax2.set_xticks(np.arange(-0.5, total_ranks, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, num_blocks, 1), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    ax2.set_xticks(range(0, total_ranks, 4))
    ax2.set_xticklabels([f"r={r}" for r in range(0, total_ranks, 4)])
    ax2.set_yticks(range(num_blocks))
    ax2.set_yticklabels([f"Block {b}" for b in range(num_blocks)])

    ax2.set_title("(b) Orthogonal Mechanistic Rank Allocation across ViT Layers", fontweight="bold")
    ax2.set_xlabel("LoRA Rank Coordinate (r = 0, ..., 31)", fontweight="bold")
    ax2.set_ylabel("Transformer Blocks (Depth L = 0, ..., 5)", fontweight="bold")

    # Custom Legend for Tasks
    legend_patches = [
        mpatches.Patch(color="#3b82f6", label="Task 0 (Classes 0-1)"),
        mpatches.Patch(color="#10b981", label="Task 1 (Classes 2-3)"),
        mpatches.Patch(color="#8b5cf6", label="Task 2 (Classes 4-5)"),
        mpatches.Patch(color="#f59e0b", label="Task 3 (Classes 6-7)"),
        mpatches.Patch(color="#ec4899", label="Task 4 (Classes 8-9)"),
        mpatches.Patch(color="#cbd5e1", label="Spare Capacity (Unallocated)"),
    ]
    ax2.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=3, fontsize=8.2, frameon=True, framealpha=0.95)

    # Highlight annotation
    ax2.text(0.5, 1.05, r"Orthogonal Subspace Partitioning: $\mathcal{R}_i \cap \mathcal{R}_j = \emptyset$ (No Pathway Collision)",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=1.0))

    # Layout adjustments
    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.96, wspace=0.25)
    pdf_out = os.path.join(OUT_DIR, "fig12_foundation_lora_routing.pdf")
    png_out = os.path.join(OUT_DIR, "fig12_foundation_lora_routing.png")
    fig.savefig(pdf_out, format="pdf", bbox_inches="tight")
    fig.savefig(png_out, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Successfully generated {pdf_out}")
    print(f"[OK] Successfully generated {png_out}")

if __name__ == "__main__":
    generate_fig12()
