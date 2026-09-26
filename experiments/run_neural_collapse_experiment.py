"""
run_neural_collapse_experiment.py
=================================
Empirically investigates the Neural Collapse (NC1, NC2, NC3) geometric dynamics
across sequential continual learning on Split CIFAR-10.

Measures:
- NC1 (Variability Collapse): Tr(Sigma_W * Sigma_B^+) / C >= 0
- NC2 (Equiangular Simplex ETF): Pairwise cosine similarity -> -1/(C-1) = -1.0, and norm uniformity
- NC3 (Self-Duality): Alignment between centered class means and linear classifier weights

Evaluates Task 0 representations at each sequential task step t in {0, 1, 2, 3, 4}
across:
1. Finetune (Sequential SGD)
2. EWC (Elastic Weight Consolidation)
3. Uncalibrated SMR (Stale BatchNorm)
4. SMR (Full framework with O(1) Recalibration and Parameter Invariance)
"""

import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import ResNet18
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    recalibrate_bn, apply_inference_routing, snapshot_full_state,
    snapshot_bn_state, snapshot_fc_state, merge_masks,
    fine_tune_decision_subcircuit, EWCRegularizer, evaluate_task_accuracy
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cifar10_tasks(batch_size=128):
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    root = os.path.join("data", "cifar10")
    train_ds = datasets.CIFAR10(root, True, download=True, transform=tr)
    test_ds = datasets.CIFAR10(root, False, download=True, transform=te)
    
    num_tasks = 5
    classes_per_task = 2
    tasks = []
    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        tri = [i for i, l in enumerate(train_ds.targets) if l in cls]
        tei = [i for i, l in enumerate(test_ds.targets) if l in cls]
        tasks.append({
            "id": t,
            "classes": cls,
            "cum": (t + 1) * classes_per_task,
            "train": DataLoader(Subset(train_ds, tri), batch_size=batch_size, shuffle=True, pin_memory=True),
            "test": DataLoader(Subset(test_ds, tei), batch_size=batch_size, shuffle=False, pin_memory=True)
        })
    return tasks

def extract_task0_features(model, task0_test_loader, device):
    """Extract penultimate layer features and labels for Task 0."""
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for x, y in task0_test_loader:
            x = x.to(device)
            feat = model.features(x)
            all_features.append(feat.cpu())
            all_labels.append(y.cpu())
    H = torch.cat(all_features, dim=0).numpy() # [N, D]
    Y = torch.cat(all_labels, dim=0).numpy()    # [N]
    return H, Y

def compute_neural_collapse_metrics(H, Y, classifier_weight, classes=[0, 1]):
    """
    Computes NC1, NC2, NC3 for the given features and classifier weights.
    Uses exact rank-1 pseudoinverse for C=2 to avoid numerical eigenvalue noise.
    """
    C = len(classes)
    D = H.shape[1]
    
    # Class means
    class_means = []
    for c in classes:
        mask = (Y == c)
        if np.sum(mask) > 0:
            class_means.append(np.mean(H[mask], axis=0))
        else:
            class_means.append(np.zeros(D))
    class_means = np.stack(class_means, axis=0) # [C, D]
    global_mean = np.mean(class_means, axis=0, keepdims=True) # [1, D]
    
    # Centered class means
    M_centered = class_means - global_mean # [C, D]
    
    # Within-class covariance Sigma_W
    Sigma_W = np.zeros((D, D))
    for i, c in enumerate(classes):
        mask = (Y == c)
        if np.sum(mask) > 0:
            diff = H[mask] - class_means[i:i+1] # [N_c, D]
            Sigma_W += np.dot(diff.T, diff) / len(H[mask])
    Sigma_W /= C
    
    # Between-class covariance Sigma_B
    # For C=2: v = (mu_0 - mu_1)/2, Sigma_B = v @ v.T
    v = M_centered[0] # [D]
    norm_v_sq = np.dot(v, v)
    
    if norm_v_sq > 1e-12:
        # Exact rank-1 pseudo-inverse: Sigma_B^+ = (v @ v.T) / ||v||^4
        # Tr(Sigma_W @ Sigma_B^+) = (v.T @ Sigma_W @ v) / ||v||^4
        v_SigmaW_v = float(np.dot(v, np.dot(Sigma_W, v)))
        nc1 = float(v_SigmaW_v / (C * (norm_v_sq ** 2)))
    else:
        nc1 = float('nan')
        
    tr_Sigma_W = float(np.trace(Sigma_W))
    tr_Sigma_B = float(norm_v_sq)
    
    # NC2: Equiangular Simplex ETF
    norms = np.linalg.norm(M_centered, axis=1, keepdims=True) + 1e-8
    M_normalized = M_centered / norms # [C, D]
    cosine_matrix = np.dot(M_normalized, M_normalized.T) # [C, C]
    
    if C == 2:
        nc2_cos = float(cosine_matrix[0, 1]) # Ideal is -1.000
    else:
        off_diag = cosine_matrix[~np.eye(C, dtype=bool)]
        nc2_cos = float(np.mean(off_diag))
        
    norm_ratio = float(np.std(norms) / (np.mean(norms) + 1e-8))
    
    # NC3: Self-Duality
    W_task = classifier_weight[classes] # [C, D]
    W_centered = W_task - np.mean(W_task, axis=0, keepdims=True)
    W_norms = np.linalg.norm(W_centered, axis=1, keepdims=True) + 1e-8
    W_normalized = W_centered / W_norms
    
    cos_align = []
    for c in range(C):
        cos_align.append(np.dot(M_normalized[c], W_normalized[c]))
    nc3_align = float(np.mean(cos_align)) # Ideal is 1.000
    
    return {
        "NC1_Tr_SigmaW_SigmaBinv": nc1,
        "tr_Sigma_W": tr_Sigma_W,
        "tr_Sigma_B": tr_Sigma_B,
        "NC2_pairwise_cosine": nc2_cos,
        "NC2_norm_uniformity": norm_ratio,
        "NC3_weight_mean_alignment": nc3_align
    }

def run_nc_experiment():
    print("=" * 80)
    print("RUNNING NEURAL COLLAPSE (NC1, NC2, NC3) CONTINUAL LEARNING PROBE")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    os.makedirs("results_final", exist_ok=True)
    out_file = os.path.join("results_final", "neural_collapse_results.json")
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    else:
        results = {}
        
    set_seed(42)
    tasks = get_cifar10_tasks(batch_size=128)
    task0_classes = tasks[0]["classes"]
    
    # -------------------------------------------------------------
    # METHOD 1: FINETUNE (Sequential SGD)
    # -------------------------------------------------------------
    if "finetune" not in results:
        print("\n[PROBE] Method: Finetune")
        set_seed(42)
        model_ft = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
        ft_nc_history = []
        
        for t in range(len(tasks)):
            task = tasks[t]
            model_ft.head.current_active_classes = task["cum"]
            train_task(model_ft, task["train"], DEVICE, epochs=10, lr=0.03)
            
            # Probe Task 0 geometry
            H0, Y0 = extract_task0_features(model_ft, tasks[0]["test"], DEVICE)
            W = model_ft.head.classifier.weight.detach().cpu().numpy()
            metrics = compute_neural_collapse_metrics(H0, Y0, W, classes=task0_classes)
            
            model_ft.head.current_active_classes = tasks[0]["cum"]
            acc0 = evaluate_task_accuracy(model_ft, tasks[0]["test"], DEVICE, task0_classes)
            metrics["task0_acc"] = float(acc0)
            metrics["step_t"] = t
            ft_nc_history.append(metrics)
            print(f"  Step T={t} | Task 0 Acc: {acc0:.4f} | NC1: {metrics['NC1_Tr_SigmaW_SigmaBinv']:.4f} | "
                  f"NC2 Cos: {metrics['NC2_pairwise_cosine']:.4f} | NC3 Align: {metrics['NC3_weight_mean_alignment']:.4f}")
            
        results["finetune"] = ft_nc_history
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  [Incremental Save] Finetune completed and saved.")
    else:
        print("\n[PROBE] Finetune already completed. Reusing cached results.")
        
    # -------------------------------------------------------------
    # METHOD 2: EWC
    # -------------------------------------------------------------
    if "ewc" not in results:
        print("\n[PROBE] Method: EWC")
        set_seed(42)
        model_ewc = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
        ewc_reg = EWCRegularizer(model_ewc, lambda_ewc=5000.0)
        ewc_nc_history = []
        
        for t in range(len(tasks)):
            task = tasks[t]
            model_ewc.head.current_active_classes = task["cum"]
            model_ewc.train()
            opt = torch.optim.SGD(model_ewc.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
            for ep in range(10):
                for x, y in task["train"]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    out = model_ewc(x)
                    loss = F.cross_entropy(out, y) + ewc_reg.penalty(model_ewc)
                    loss.backward()
                    opt.step()
            ewc_reg.compute_fisher(model_ewc, task["train"], DEVICE)
            
            # Probe Task 0 geometry
            H0, Y0 = extract_task0_features(model_ewc, tasks[0]["test"], DEVICE)
            W = model_ewc.head.classifier.weight.detach().cpu().numpy()
            metrics = compute_neural_collapse_metrics(H0, Y0, W, classes=task0_classes)
            
            model_ewc.head.current_active_classes = tasks[0]["cum"]
            acc0 = evaluate_task_accuracy(model_ewc, tasks[0]["test"], DEVICE, task0_classes)
            metrics["task0_acc"] = float(acc0)
            metrics["step_t"] = t
            ewc_nc_history.append(metrics)
            print(f"  Step T={t} | Task 0 Acc: {acc0:.4f} | NC1: {metrics['NC1_Tr_SigmaW_SigmaBinv']:.4f} | "
                  f"NC2 Cos: {metrics['NC2_pairwise_cosine']:.4f} | NC3 Align: {metrics['NC3_weight_mean_alignment']:.4f}")
            
        results["ewc"] = ewc_nc_history
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  [Incremental Save] EWC completed and saved.")
    else:
        print("\n[PROBE] EWC already completed. Reusing cached results.")
        
    # -------------------------------------------------------------
    # METHOD 3: UNCALIBRATED MODULAR (Stale BN - Normalization Paradox)
    # -------------------------------------------------------------
    if "uncalibrated_modular" not in results:
        print("\n[PROBE] Method: Uncalibrated Modular (Stale BN)")
        set_seed(42)
        model_uncal = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
        model_uncal.head.current_active_classes = tasks[0]["cum"]
        train_task(model_uncal, tasks[0]["train"], DEVICE, epochs=15, lr=0.03)
        
        imp0 = compute_neuron_importance(model_uncal, tasks[0]["train"], DEVICE, num_batches=20)
        mask0 = compute_smr_mask(imp0, percentile=85)
        
        task0_state = snapshot_full_state(model_uncal)
        uncal_nc_history = []
        
        apply_inference_routing(model_uncal, task0_state, mask0, snapshot_bn_state(model_uncal),
                                snapshot_fc_state(model_uncal), DEVICE, calibration_loader=None)
        H0, Y0 = extract_task0_features(model_uncal, tasks[0]["test"], DEVICE)
        W = model_uncal.head.classifier.weight.detach().cpu().numpy()
        metrics_uncal = compute_neural_collapse_metrics(H0, Y0, W, classes=task0_classes)
        acc0_uncal = evaluate_task_accuracy(model_uncal, tasks[0]["test"], DEVICE, task0_classes)
        metrics_uncal["task0_acc"] = float(acc0_uncal)
        metrics_uncal["step_t"] = 0
        uncal_nc_history.append(metrics_uncal)
        print(f"  Uncalibrated (Task 0) | Acc: {acc0_uncal:.4f} | NC1: {metrics_uncal['NC1_Tr_SigmaW_SigmaBinv']:.4f} | "
              f"NC2 Cos: {metrics_uncal['NC2_pairwise_cosine']:.4f} | Tr(Sigma_W): {metrics_uncal['tr_Sigma_W']:.4f}")
        results["uncalibrated_modular"] = uncal_nc_history
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  [Incremental Save] Uncalibrated Modular completed and saved.")
    else:
        print("\n[PROBE] Uncalibrated Modular already completed. Reusing cached results.")
        
    # -------------------------------------------------------------
    # METHOD 4: SMR (Ours: Mechanistic Routing + O(1) Recalibration + Exact Invariance)
    # -------------------------------------------------------------
    if "smr" not in results:
        print("\n[PROBE] Method: SMR (Ours)")
        set_seed(42)
        model_smr = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
        smr_nc_history = []
        
        task_masks = {}
        task_bn_states = {}
        task_fc_states = {}
        global_protection = {}
        saved_protected_weights = {}
        
        for t in range(len(tasks)):
            task = tasks[t]
            model_smr.head.current_active_classes = task["cum"]
            
            train_task(model_smr, task["train"], DEVICE, epochs=15, lr=0.03,
                       protection_masks=global_protection,
                       saved_weights=saved_protected_weights, use_amp=True)
            
            imp = compute_neuron_importance(model_smr, task["train"], DEVICE, num_batches=20)
            t_mask = compute_smr_mask(imp, percentile=85, prior_protection=global_protection)
            
            fine_tune_decision_subcircuit(model_smr, task["train"], t_mask, task["classes"], DEVICE, epochs=4, lr=0.01)
            
            curr_state = snapshot_full_state(model_smr)
            apply_inference_routing(model_smr, curr_state, t_mask, snapshot_bn_state(model_smr),
                                    snapshot_fc_state(model_smr), DEVICE, calibration_loader=task["train"])
            task_masks[t] = deepcopy(t_mask)
            task_bn_states[t] = snapshot_bn_state(model_smr)
            task_fc_states[t] = snapshot_fc_state(model_smr)
            task_isolated_state = snapshot_full_state(model_smr)
            if t == 0:
                task0_isolated_state = deepcopy(task_isolated_state)
            model_smr.load_state_dict(curr_state)
            
            saved_protected_weights = snapshot_full_state(model_smr)
            global_protection = merge_masks(global_protection, t_mask)
            
            # Probe Task 0 geometry via inference routing to Task 0's isolated subcircuit
            full_trained_state = snapshot_full_state(model_smr)
            model_smr.load_state_dict(task0_isolated_state)
            model_smr.head.current_active_classes = tasks[0]["cum"]
            
            H0, Y0 = extract_task0_features(model_smr, tasks[0]["test"], DEVICE)
            W = model_smr.head.classifier.weight.detach().cpu().numpy()
            metrics = compute_neural_collapse_metrics(H0, Y0, W, classes=task0_classes)
            acc0 = evaluate_task_accuracy(model_smr, tasks[0]["test"], DEVICE, task0_classes)
            metrics["task0_acc"] = float(acc0)
            metrics["step_t"] = t
            smr_nc_history.append(metrics)
            print(f"  Step T={t} | Task 0 Acc: {acc0:.4f} | NC1: {metrics['NC1_Tr_SigmaW_SigmaBinv']:.4f} | "
                  f"NC2 Cos: {metrics['NC2_pairwise_cosine']:.4f} | NC3 Align: {metrics['NC3_weight_mean_alignment']:.4f}")
            
            # Restore model_smr so next task trains on unpruned state
            model_smr.load_state_dict(full_trained_state)
            
        results["smr"] = smr_nc_history
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  [Incremental Save] SMR completed and saved.")
    else:
        print("\n[PROBE] SMR already completed. Reusing cached results.")
        
    print(f"\n[ALL COMPLETED] Neural Collapse experiment results saved to {out_file}!")
    return results

if __name__ == "__main__":
    run_nc_experiment()
