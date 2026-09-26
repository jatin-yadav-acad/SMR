"""
generate_representation_figures.py
==================================
Generates publication-grade vector figures (PDF) for:
Figure 8: Internal Representation Geometry and Spectral Criticality across Continual Horizons.

Panels:
(a) Cross-Layer CKA Similarity Matrix across [conv1, layer1, layer2, layer3, layer4].
(b) Covariance Eigenspectra and Power-Law Decay (alpha-exponent fit).
(c) Effective Rank (erank) Dimensionality across sequential continual learning steps (T0 -> T4).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Professional Serif Styling matching ICLR manuscript
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.autolayout': False
})

def generate_figure(results_path="results_final/representation_geometry_results.json",
                    output_pdf="manuscript/figures/fig8_representation_geometry.pdf"):
    if not os.path.exists(results_path):
        print(f"Error: Results file '{results_path}' not found.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # Average across all seeds
    num_seeds = len(data)
    first_seed = data[0]
    
    # 1. Cross-Layer CKA Matrix for SMR (averaged over seeds)
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    cka_matrices = []
    for s_data in data:
        step_metrics = s_data['smr']['step_metrics']
        # Take final step
        final_cka = np.array(step_metrics[-1]['cross_cka'])
        cka_matrices.append(final_cka)
    mean_cka = np.mean(cka_matrices, axis=0)
    
    # 2. Eigenspectra for layer4 at step 0 vs step 4
    top_evals_smr_t0 = np.mean([s['smr']['step_metrics'][0]['spectra']['layer4']['top_evals'] for s in data], axis=0)
    top_evals_smr_t4 = np.mean([s['smr']['step_metrics'][-1]['spectra']['layer4']['top_evals'] for s in data], axis=0)
    top_evals_uncal_t4 = np.mean([s['uncalibrated_smr']['step_metrics'][-1]['spectra']['layer4']['top_evals'] for s in data], axis=0)
    top_evals_ft_t4 = np.mean([s['finetune']['step_metrics'][-1]['spectra']['layer4']['top_evals'] for s in data], axis=0)
    
    # Normalize spectra for comparison
    top_evals_smr_t0 = top_evals_smr_t0 / (np.sum(top_evals_smr_t0) + 1e-10)
    top_evals_smr_t4 = top_evals_smr_t4 / (np.sum(top_evals_smr_t4) + 1e-10)
    top_evals_uncal_t4 = top_evals_uncal_t4 / (np.sum(top_evals_uncal_t4) + 1e-10)
    top_evals_ft_t4 = top_evals_ft_t4 / (np.sum(top_evals_ft_t4) + 1e-10)
    
    # 3. Effective Rank trajectory across steps (T0 -> T4)
    steps = [0, 1, 2, 3, 4]
    erank_smr = [np.mean([s['smr']['step_metrics'][t]['spectra']['layer4']['erank'] for s in data]) for t in steps]
    erank_uncal = [np.mean([s['uncalibrated_smr']['step_metrics'][t]['spectra']['layer4']['erank'] for s in data]) for t in steps]
    erank_ft = [np.mean([s['finetune']['step_metrics'][t]['spectra']['layer4']['erank'] for s in data]) for t in steps]
    erank_ewc = [np.mean([s['ewc']['step_metrics'][t]['spectra']['layer4']['erank'] for s in data]) for t in steps]
    
    # Mean alpha values
    alpha_smr = np.mean([s['smr']['step_metrics'][-1]['spectra']['layer4']['alpha'] for s in data])
    alpha_uncal = np.mean([s['uncalibrated_smr']['step_metrics'][-1]['spectra']['layer4']['alpha'] for s in data])
    alpha_ft = np.mean([s['finetune']['step_metrics'][-1]['spectra']['layer4']['alpha'] for s in data])
    
    # Create Figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), gridspec_kw={'width_ratios': [1.05, 1.15, 1.15]})
    
    # -------------------------------------------------------------
    # Panel (a): Cross-layer CKA Heatmap
    # -------------------------------------------------------------
    ax_a = axes[0]
    layer_labels = [r'conv1', r'layer1', r'layer2', r'layer3', r'layer4']
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(mean_cka, annot=True, fmt=".2f", cmap=cmap, cbar=True,
                xticklabels=layer_labels, yticklabels=layer_labels, ax=ax_a,
                vmin=0.2, vmax=1.0, annot_kws={"size": 8.5})
    ax_a.set_title(r"$\mathbf{(a)}$ Cross-Layer CKA Alignment (SMR)", pad=10)
    ax_a.set_xlabel(r"Target Stage")
    ax_a.set_ylabel(r"Source Stage")
    
    # -------------------------------------------------------------
    # Panel (b): Covariance Eigenspectra & Power-Law Decay
    # -------------------------------------------------------------
    ax_b = axes[1]
    ranks = np.arange(1, len(top_evals_smr_t4) + 1)
    
    # Critical reference line alpha = 1.0
    crit_ref = 0.4 * (ranks ** -1.0)
    ax_b.loglog(ranks, crit_ref, 'k--', lw=1.8, label=r'Criticality ($\alpha=1.0$)')
    
    ax_b.loglog(ranks, top_evals_smr_t4, 'o-', color='#1b7837', lw=2.0, ms=4,
                label=rf'SMR (Ours, $\alpha={alpha_smr:.2f}$)')
    ax_b.loglog(ranks, top_evals_uncal_t4, 's--', color='#d95f02', lw=1.8, ms=3.5,
                label=rf'Uncalibrated SMR ($\alpha={alpha_uncal:.2f}$)')
    ax_b.loglog(ranks, top_evals_ft_t4, '^:', color='#7570b3', lw=1.8, ms=3.5,
                label=rf'Finetuning ($\alpha={alpha_ft:.2f}$)')
    
    ax_b.set_title(r"$\mathbf{(b)}$ Covariance Eigenspectra $\lambda_i \propto i^{-\alpha}$", pad=10)
    ax_b.set_xlabel(r"Eigenvalue Rank $i$")
    ax_b.set_ylabel(r"Normalized Variance $\lambda_i / \sum_j \lambda_j$")
    ax_b.grid(True, which="both", ls=":", alpha=0.5)
    ax_b.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel (c): Effective Rank (erank) across Continual Steps
    # -------------------------------------------------------------
    ax_c = axes[2]
    task_labels = [r'$T_0$', r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$']
    
    ax_c.plot(steps, erank_smr, 'o-', color='#1b7837', lw=2.2, ms=6, label=r'SMR (Calibrated)')
    ax_c.plot(steps, erank_uncal, 's--', color='#d95f02', lw=1.8, ms=5, label=r'Uncalibrated SMR')
    ax_c.plot(steps, erank_ewc, 'd-.', color='#e7298a', lw=1.8, ms=5, label=r'EWC (Kirkpatrick)')
    ax_c.plot(steps, erank_ft, '^:', color='#7570b3', lw=1.8, ms=5, label=r'Finetuning (SGD)')
    
    ax_c.set_xticks(steps)
    ax_c.set_xticklabels(task_labels)
    ax_c.set_title(r"$\mathbf{(c)}$ Effective Dimensionality $\mathrm{erank}(\mathbf{\Sigma})$", pad=10)
    ax_c.set_xlabel(r"Sequential Continual Task Step")
    ax_c.set_ylabel(r"Effective Rank $\exp(H(p))$")
    ax_c.grid(True, ls=":", alpha=0.5)
    ax_c.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    png_path = output_pdf.replace('.pdf', '.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully generated publication figures:")
    print(f"  PDF: {output_pdf}")
    print(f"  PNG: {png_path}")

if __name__ == "__main__":
    generate_figure()
