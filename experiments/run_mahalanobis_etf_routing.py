#!/usr/bin/env python
"""
run_mahalanobis_etf_routing.py
==============================
Autonomous ETF Mahalanobis Routing for Task-Free Class-IL (Direction 1).

The Problem:
Modular networks suffer in Class-IL when using uncalibrated Helmholtz free energy
because foreign tasks produce uncalibrated logit magnitudes, leading to severe misrouting.

The Solution:
Exploits Equiangular Tight Frame (ETF) geometry in representation space (layer4 pooled features).
Under terminal training, within-class covariance collapses (Sigma_W -> 0), so foreign-task
inputs land far off-manifold in the orthogonal subspace.

Action:
1. Extract representations h(x) (layer4 pooled features, D=512) from ResNet-18 on Split CIFAR-10
   and Split CIFAR-100.
2. For each task t, compute class prototypes mu_{c, t} and regularized covariance:
   Sigma_t = Sigma_{W, t} + lambda * I
3. Evaluate task-agnostic routing using Mahalanobis distance:
   score(x, t) = min_{c in C_t} (h(x) - mu_{c, t})^T Sigma_t^{-1} (h(x) - mu_{c, t})
   Assign x to task t* = argmin_t score(x, t), then predict class c* within task t*.
4. Benchmark AUROC of task identification and resulting Class-IL accuracy vs Helmholtz free-energy routing.
5. Log results to results_final/mahalanobis_etf_results.json.
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import ResNet18
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    recalibrate_bn, apply_inference_routing, snapshot_full_state,
    snapshot_bn_state, snapshot_fc_state, merge_masks,
    fine_tune_decision_subcircuit, evaluate_accuracy
)
from experiments.run_final_experiments import get_cifar10_tasks
from experiments.run_cifar100_scaled_smr import get_cifar100_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_active_classes(model, active_classes):
    model.active_classes = active_classes
    if hasattr(model, 'head') and hasattr(model.head, 'current_active_classes'):
        model.head.current_active_classes = active_classes


def train_and_extract_task_subcircuits(tasks, num_classes, epochs_per_task=15, lr=0.03, prune_percentile=80, ft_epochs=4, ckpt_path=None):
    """
    Trains SMR sequentially across tasks and returns isolated subcircuit states.
    If ckpt_path exists, loads and returns cached canonical subcircuits.
    """
    model = ResNet18(num_classes=num_classes, cifar_style=True).to(DEVICE)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  [CACHE] Loading canonical subcircuits from {ckpt_path}...")
        cached_data = torch.load(ckpt_path, map_location=DEVICE)
        return model, cached_data["subcircuits"]

    task_subcircuits = []
    cum_masks = {}
    saved_weights = {}

    for t_idx, task_info in enumerate(tasks):
        set_active_classes(model, task_info['cum'])
        print(f"  [Train Task {t_idx}] Classes: {task_info['classes']} (Epochs: {epochs_per_task})...")
        train_task(model, task_info['train'], DEVICE, epochs=epochs_per_task, lr=lr,
                   protection_masks=cum_masks if t_idx > 0 else None,
                   saved_weights=saved_weights if t_idx > 0 else None,
                   use_amp=True)

        importance = compute_neuron_importance(model, task_info['train'], DEVICE, num_batches=20)
        t_mask = compute_smr_mask(importance, percentile=prune_percentile, prior_protection=cum_masks if t_idx > 0 else None)
        
        print(f"    Specializing isolated decision sub-circuit ({ft_epochs} epochs)...")
        fine_tune_decision_subcircuit(model, task_info['train'], t_mask, task_info['classes'], DEVICE, epochs=ft_epochs, lr=0.01)

        current_state = snapshot_full_state(model)
        base_bn = snapshot_bn_state(model)
        base_fc = snapshot_fc_state(model)

        apply_inference_routing(model, current_state, t_mask, base_bn, base_fc, DEVICE, calibration_loader=task_info['train'])
        cal_state = snapshot_full_state(model)
        model.load_state_dict(current_state) # restore for next task

        task_subcircuits.append({
            "task_id": t_idx,
            "classes": task_info["classes"],
            "state_dict": cal_state,
            "mask": t_mask
        })

        if t_idx == 0:
            cum_masks = t_mask
        else:
            cum_masks = merge_masks(cum_masks, t_mask)
        saved_weights = snapshot_full_state(model)

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"subcircuits": task_subcircuits}, ckpt_path)
        print(f"  [CACHE] Saved canonical subcircuits to {ckpt_path}")

    return model, task_subcircuits


def compute_task_prototypes_and_covariance(model, task_subcircuit, train_loader, device, reg_lambda=1e-3, max_samples=2000):
    """
    Extracts penultimate features h(x) on task t's training data,
    computes class prototypes mu_{c, t} and regularized covariance Sigma_t = Sigma_{W, t} + lambda * I.
    """
    model.load_state_dict(task_subcircuit["state_dict"])
    model.eval()

    classes = task_subcircuit["classes"]
    feats_by_class = {c: [] for c in classes}
    total_samples = 0

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            h = model.features(x).cpu() # [B, D]
            for i in range(x.size(0)):
                label = y[i].item()
                if label in feats_by_class:
                    feats_by_class[label].append(h[i])
                    total_samples += 1
            if total_samples >= max_samples:
                break

    D = 512
    prototypes = {}
    diffs = []

    for c in classes:
        if len(feats_by_class[c]) > 0:
            class_feats = torch.stack(feats_by_class[c], dim=0) # [N_c, D]
            mu_c = torch.mean(class_feats, dim=0) # [D]
            prototypes[c] = mu_c
            diffs.append(class_feats - mu_c.unsqueeze(0))
        else:
            prototypes[c] = torch.zeros(D)

    # Within-class covariance Sigma_{W, t}
    if len(diffs) > 0:
        all_diffs = torch.cat(diffs, dim=0) # [N, D]
        N = all_diffs.size(0)
        Sigma_W = (all_diffs.t() @ all_diffs) / max(1, N) # [D, D]
    else:
        Sigma_W = torch.zeros(D, D)

    # Regularized covariance Sigma_t = Sigma_{W, t} + lambda * I
    Sigma_reg = Sigma_W + reg_lambda * torch.eye(D)
    
    # Compute inverse covariance Sigma_t^{-1}
    try:
        Sigma_inv = torch.linalg.inv(Sigma_reg)
    except Exception:
        Sigma_inv = torch.linalg.pinv(Sigma_reg)

    return prototypes, Sigma_inv, Sigma_W


def evaluate_routing_benchmark(model, task_subcircuits, task_prototypes, task_cov_invs, tasks, device):
    """
    Evaluates:
    1. AUROC of task identification (Mahalanobis vs Free Energy)
    2. Task routing accuracy (Mahalanobis vs Free Energy)
    3. Class-IL top-1 classification accuracy (Mahalanobis vs Free Energy)
    4. Confusion matrices
    5. Mahalanobis distance separation statistics (in-task vs foreign-task)
    """
    num_tasks = len(tasks)
    
    # Storage for all test samples
    all_maha_scores = []   # list of [num_tasks] scores for each sample
    all_energy_scores = [] # list of [num_tasks] scores for each sample
    all_true_tasks = []    # list of true task indices
    all_true_labels = []   # list of true class labels
    
    # For class prediction
    maha_pred_classes = []
    energy_pred_classes = []
    
    # In-task vs Foreign-task Mahalanobis distances
    in_task_distances = []
    foreign_task_distances = []

    print("\n--- Running Evaluation over All Test Tasks ---")
    for true_t, task_info in enumerate(tasks):
        test_loader = task_info['test']
        for x, y in test_loader:
            x = x.to(device)
            B = x.size(0)
            
            # Evaluate across all candidate task subcircuits
            batch_maha_scores = []
            batch_energy_scores = []
            batch_maha_best_classes = []
            batch_energy_best_classes = []

            for cand_t, candidate in enumerate(task_subcircuits):
                model.load_state_dict(candidate["state_dict"])
                model.eval()
                with torch.no_grad():
                    h = model.features(x)   # [B, D]
                    logits = model(x)       # [B, num_classes]
                    cand_classes = candidate["classes"]
                    cand_logits = logits[:, cand_classes] # [B, |C_cand|]

                    # 1. Helmholtz Free Energy: E(x, cand_t) = - logsumexp(logits / T)
                    energy = - torch.logsumexp(cand_logits, dim=1) # [B]
                    batch_energy_scores.append(energy.cpu().numpy())
                    
                    # Best class under Free Energy (highest logit)
                    best_cls_energy_idx = torch.argmax(cand_logits, dim=1).cpu().numpy()
                    best_cls_energy = [cand_classes[idx] for idx in best_cls_energy_idx]
                    batch_energy_best_classes.append(best_cls_energy)

                    # 2. Mahalanobis Distance: min_{c in C_t} (h(x) - mu_{c, t})^T Sigma_t^{-1} (h(x) - mu_{c, t})
                    cand_protos = task_prototypes[cand_t]
                    Sigma_inv = task_cov_invs[cand_t].to(device)
                    
                    class_dists = []
                    for c in cand_classes:
                        mu_c = cand_protos[c].to(device)
                        diff = h - mu_c.unsqueeze(0) # [B, D]
                        # dist = (diff @ Sigma_inv * diff).sum(dim=1)
                        dist_c = torch.sum((diff @ Sigma_inv) * diff, dim=1) # [B]
                        class_dists.append(dist_c)
                        
                    class_dists_stacked = torch.stack(class_dists, dim=1) # [B, |C_cand|]
                    min_dist, best_c_idx = torch.min(class_dists_stacked, dim=1)
                    
                    batch_maha_scores.append(min_dist.cpu().numpy())
                    best_cls_maha = [cand_classes[idx] for idx in best_c_idx.cpu().numpy()]
                    batch_maha_best_classes.append(best_cls_maha)

            # Stack for this batch: [B, num_tasks]
            batch_maha_scores = np.stack(batch_maha_scores, axis=1)
            batch_energy_scores = np.stack(batch_energy_scores, axis=1)

            for b in range(B):
                m_scores = batch_maha_scores[b]
                e_scores = batch_energy_scores[b]
                
                all_maha_scores.append(m_scores)
                all_energy_scores.append(e_scores)
                all_true_tasks.append(true_t)
                all_true_labels.append(y[b].item())
                
                # Best task assignment
                p_task_maha = int(np.argmin(m_scores))
                p_task_energy = int(np.argmin(e_scores))
                
                maha_pred_classes.append(batch_maha_best_classes[p_task_maha][b])
                energy_pred_classes.append(batch_energy_best_classes[p_task_energy][b])
                
                # Record distance separation
                in_task_distances.append(float(m_scores[true_t]))
                for other_t in range(num_tasks):
                    if other_t != true_t:
                        foreign_task_distances.append(float(m_scores[other_t]))

    all_maha_scores = np.array(all_maha_scores)     # [N_total, num_tasks]
    all_energy_scores = np.array(all_energy_scores) # [N_total, num_tasks]
    all_true_tasks = np.array(all_true_tasks)       # [N_total]
    all_true_labels = np.array(all_true_labels)     # [N_total]
    maha_pred_classes = np.array(maha_pred_classes)
    energy_pred_classes = np.array(energy_pred_classes)

    total_samples = len(all_true_labels)

    # 1. AUROC of Task Identification (per task and average)
    maha_aurocs = []
    energy_aurocs = []

    for t in range(num_tasks):
        y_binary = (all_true_tasks == t).astype(int)
        # For Mahalanobis: lower distance = higher score -> use negative distance
        score_m = - all_maha_scores[:, t]
        # For Helmholtz Free Energy: lower energy = higher score -> use negative energy
        score_e = - all_energy_scores[:, t]
        
        try:
            auc_m = float(roc_auc_score(y_binary, score_m))
        except Exception:
            auc_m = 0.5
        try:
            auc_e = float(roc_auc_score(y_binary, score_e))
        except Exception:
            auc_e = 0.5

        maha_aurocs.append(auc_m)
        energy_aurocs.append(auc_e)

    mean_maha_auroc = float(np.mean(maha_aurocs))
    mean_energy_auroc = float(np.mean(energy_aurocs))

    # 2. Task Routing Accuracy & Confusion Matrix
    pred_tasks_maha = np.argmin(all_maha_scores, axis=1)
    pred_tasks_energy = np.argmin(all_energy_scores, axis=1)

    cm_maha = np.zeros((num_tasks, num_tasks), dtype=int)
    cm_energy = np.zeros((num_tasks, num_tasks), dtype=int)

    for i in range(total_samples):
        cm_maha[all_true_tasks[i], pred_tasks_maha[i]] += 1
        cm_energy[all_true_tasks[i], pred_tasks_energy[i]] += 1

    task_routing_acc_maha = float(np.diag(cm_maha).sum() / total_samples * 100.0)
    task_routing_acc_energy = float(np.diag(cm_energy).sum() / total_samples * 100.0)

    # 3. Class-IL Top-1 Classification Accuracy
    class_il_acc_maha = float(np.mean(maha_pred_classes == all_true_labels) * 100.0)
    class_il_acc_energy = float(np.mean(energy_pred_classes == all_true_labels) * 100.0)

    # 4. Distance Separation Statistics
    mean_in_dist = float(np.mean(in_task_distances))
    std_in_dist = float(np.std(in_task_distances))
    mean_foreign_dist = float(np.mean(foreign_task_distances))
    std_foreign_dist = float(np.std(foreign_task_distances))
    separation_ratio = float(mean_foreign_dist / max(1e-5, mean_in_dist))

    print(f"\n==================================================================")
    print(f"BENCHMARK RESULTS (N_test = {total_samples} samples, {num_tasks} tasks)")
    print(f"==================================================================")
    print(f"Metric                           | Free Energy | ETF Mahalanobis | Gain / Delta")
    print(f"------------------------------------------------------------------")
    print(f"Task Identification AUROC        | {mean_energy_auroc*100:6.2f}%    | {mean_maha_auroc*100:6.2f}%         | +{(mean_maha_auroc-mean_energy_auroc)*100:+.2f}%")
    print(f"Task Routing Accuracy            | {task_routing_acc_energy:6.2f}%    | {task_routing_acc_maha:6.2f}%         | +{task_routing_acc_maha-task_routing_acc_energy:+.2f}%")
    print(f"Class-IL Top-1 Accuracy          | {class_il_acc_energy:6.2f}%    | {class_il_acc_maha:6.2f}%         | +{class_il_acc_maha-class_il_acc_energy:+.2f}%")
    print(f"------------------------------------------------------------------")
    print(f"Mahalanobis In-Task Distance     : {mean_in_dist:.2f} +/- {std_in_dist:.2f}")
    print(f"Mahalanobis Foreign-Task Distance: {mean_foreign_dist:.2f} +/- {std_foreign_dist:.2f}")
    print(f"Off-Manifold Separation Ratio    : {separation_ratio:.2f}x")
    print(f"==================================================================")

    return {
        "num_tasks": num_tasks,
        "total_samples": total_samples,
        "task_id_auroc": {
            "free_energy_per_task": energy_aurocs,
            "mahalanobis_per_task": maha_aurocs,
            "free_energy_mean": mean_energy_auroc,
            "mahalanobis_mean": mean_maha_auroc,
            "auroc_gain": mean_maha_auroc - mean_energy_auroc
        },
        "task_routing_acc": {
            "free_energy": task_routing_acc_energy,
            "mahalanobis": task_routing_acc_maha,
            "routing_acc_gain": task_routing_acc_maha - task_routing_acc_energy
        },
        "class_il_acc": {
            "free_energy": class_il_acc_energy,
            "mahalanobis": class_il_acc_maha,
            "class_il_acc_gain": class_il_acc_maha - class_il_acc_energy
        },
        "confusion_matrices": {
            "free_energy": cm_energy.tolist(),
            "mahalanobis": cm_maha.tolist()
        },
        "distance_separation": {
            "in_task_mean": mean_in_dist,
            "in_task_std": std_in_dist,
            "foreign_task_mean": mean_foreign_dist,
            "foreign_task_std": std_foreign_dist,
            "separation_ratio": separation_ratio,
            "sample_in_dists": in_task_distances[:500],
            "sample_foreign_dists": foreign_task_distances[:500]
        }
    }


def run_etf_mahalanobis_experiment():
    print("=" * 80)
    print("RUNNING AUTONOMOUS ETF MAHALANOBIS ROUTING BENCHMARK (DIRECTION 1)")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    set_seed(42)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "split_cifar10": {},
        "split_cifar100": {}
    }

    # =============================================================
    # 1. SPLIT CIFAR-10 (5 Tasks, 10 Classes)
    # =============================================================
    print("\n" + "#" * 80)
    print("BENCHMARK 1: SPLIT CIFAR-10 (5 TASKS, 10 CLASSES)")
    print("#" * 80)
    c10_tasks = get_cifar10_tasks(data_dir, batch_size=128, num_tasks=5)

    print("Step 1: Training SMR and Extracting Isolated Subcircuits...")
    c10_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'canonical_cifar10_smr.pt')
    model_c10, c10_subcircuits = train_and_extract_task_subcircuits(
        c10_tasks, num_classes=10, epochs_per_task=15, lr=0.03, prune_percentile=80, ft_epochs=4, ckpt_path=c10_ckpt
    )

    print("Step 2: Computing Task Class Prototypes and Regularized Covariance Inverses...")
    c10_prototypes = {}
    c10_cov_invs = {}
    for t_idx, sc in enumerate(c10_subcircuits):
        protos, cov_inv, sigma_w = compute_task_prototypes_and_covariance(
            model_c10, sc, c10_tasks[t_idx]['train'], DEVICE, reg_lambda=1e-3
        )
        c10_prototypes[t_idx] = protos
        c10_cov_invs[t_idx] = cov_inv
        tr_cov = float(torch.trace(sigma_w).item())
        print(f"  Task {t_idx} Prototypes: {list(protos.keys())} | Tr(Sigma_W): {tr_cov:.4f}")

    print("Step 3: Evaluating Task-Agnostic Routing and Class-IL Accuracy...")
    c10_results = evaluate_routing_benchmark(
        model_c10, c10_subcircuits, c10_prototypes, c10_cov_invs, c10_tasks, DEVICE
    )
    results["split_cifar10"] = c10_results

    # =============================================================
    # 2. SPLIT CIFAR-100 (10 Tasks, 100 Classes)
    # =============================================================
    print("\n" + "#" * 80)
    print("BENCHMARK 2: SPLIT CIFAR-100 (10 TASKS, 100 CLASSES)")
    print("#" * 80)
    c100_tasks = get_cifar100_tasks(batch_size=128)

    print("Step 1: Training SMR and Extracting Isolated Subcircuits...")
    c100_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'canonical_cifar100_smr.pt')
    model_c100, c100_subcircuits = train_and_extract_task_subcircuits(
        c100_tasks, num_classes=100, epochs_per_task=12, lr=0.03, prune_percentile=92, ft_epochs=4, ckpt_path=c100_ckpt
    )

    print("Step 2: Computing Task Class Prototypes and Regularized Covariance Inverses...")
    c100_prototypes = {}
    c100_cov_invs = {}
    for t_idx, sc in enumerate(c100_subcircuits):
        protos, cov_inv, sigma_w = compute_task_prototypes_and_covariance(
            model_c100, sc, c100_tasks[t_idx]['train'], DEVICE, reg_lambda=1e-3
        )
        c100_prototypes[t_idx] = protos
        c100_cov_invs[t_idx] = cov_inv
        tr_cov = float(torch.trace(sigma_w).item())
        print(f"  Task {t_idx} Prototypes: {len(protos)} classes | Tr(Sigma_W): {tr_cov:.4f}")

    print("Step 3: Evaluating Task-Agnostic Routing and Class-IL Accuracy...")
    c100_results = evaluate_routing_benchmark(
        model_c100, c100_subcircuits, c100_prototypes, c100_cov_invs, c100_tasks, DEVICE
    )
    results["split_cifar100"] = c100_results

    # Save to results_final/mahalanobis_etf_results.json
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_final')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'mahalanobis_etf_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Mahalanobis ETF results successfully saved to: {out_file}")

    return results


if __name__ == '__main__':
    run_etf_mahalanobis_experiment()
