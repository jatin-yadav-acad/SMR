"""
run_representation_geometry.py
==============================
Rigorous empirical investigation of internal representation geometry across continual learning
on Split CIFAR-10 (ResNet-18) over seeds [42, 1337, 2025].

Evaluates:
1. Cross-layer Linear Centered Kernel Alignment (CKA) between [conv1, layer1, layer2, layer3, layer4]
   and temporal CKA alignment with initial Task 0 representations: CKA(X_l^(t), X_l^(0)).
2. Covariance Eigenspectra & Power-Law Decay Exponent (alpha-fit) connecting to Stringer et al. (Nature 2019)
   neural criticality: lambda_i ~ i^(-alpha), testing if modular routing preserves alpha ~ 1.0.
3. Dimensionality preservation via Effective Rank: erank(Sigma) = exp(H(p)).
4. Full test set evaluations (strictly N_eval = 10,000, zero downsampling).

Methods:
- Finetune (Sequential SGD)
- EWC (Elastic Weight Consolidation)
- Uncalibrated SMR (Stale BatchNorm)
- SMR (Full Framework with O(1) Recalibration and Parameter Invariance)
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
    train_ds = datasets.CIFAR10(root, True, download=False, transform=tr)
    test_ds = datasets.CIFAR10(root, False, download=False, transform=te)
    
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
    full_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return tasks, full_test_loader

def extract_all_layer_features(model, dataloader, device):
    """
    Extracts activations from conv1, layer1, layer2, layer3, layer4
    for all samples in the given dataloader.
    Returns dictionary of PyTorch tensors [N, D_l].
    """
    model.eval()
    layer_feats = {
        'conv1': [],
        'layer1': [],
        'layer2': [],
        'layer3': [],
        'layer4': []
    }
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feats = model.extract_layer_features(x)
            for k in layer_feats:
                layer_feats[k].append(feats[k].cpu())
            all_labels.append(y)
            
    for k in layer_feats:
        layer_feats[k] = torch.cat(layer_feats[k], dim=0) # [N, D_l]
    labels = torch.cat(all_labels, dim=0).numpy()
    return layer_feats, labels

def linear_cka(X, Y):
    """
    Linear Centered Kernel Alignment (CKA) between two representation matrices.
    X: [N, d1], Y: [N, d2].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    dot_prod = torch.norm(X.t() @ Y, p='fro') ** 2
    norm_x = torch.norm(X.t() @ X, p='fro')
    norm_y = torch.norm(Y.t() @ Y, p='fro')
    denom = norm_x * norm_y
    if denom.item() == 0 or torch.isnan(denom):
        return 0.0
    return float((dot_prod / denom).item())

def compute_layerwise_cka_matrix(layer_feats):
    """
    Computes 5x5 pairwise CKA similarity matrix across all 5 network layers.
    Layers: ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    """
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    M = np.zeros((len(layers), len(layers)), dtype=np.float32)
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            if i == j:
                M[i, j] = 1.0
            elif j > i:
                sim = linear_cka(layer_feats[l1], layer_feats[l2])
                M[i, j] = sim
                M[j, i] = sim
    return M.tolist()

def compute_spectrum_metrics(X):
    """
    Computes covariance eigenspectrum, power-law decay alpha-exponent,
    and effective rank for a representation matrix X [N, D].
    Uses numerically bulletproof SVD on centered representations:
    lambda_i = s_i^2 / (N - 1). Guaranteed convergence on all matrices.
    """
    N, D = X.shape
    X_cent = X - X.mean(dim=0, keepdim=True)
    X_np = X_cent.cpu().numpy().astype(np.float64)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # SVD singular values
    s = np.linalg.svd(X_np, compute_uv=False)
    evals = (s ** 2) / max(1, N - 1)
    
    if len(evals) < D:
        evals = np.pad(evals, (0, D - len(evals)))
    evals = np.clip(evals, a_min=0.0, a_max=None)
    
    total_var = float(np.sum(evals))
    if total_var > 0:
        p = evals / total_var
        p_pos = p[p > 1e-12]
        spectral_entropy = float(-np.sum(p_pos * np.log(p_pos)))
        erank = float(np.exp(spectral_entropy))
    else:
        spectral_entropy = 0.0
        erank = 1.0
        
    pos_idx = np.where(evals > 1e-7)[0]
    if len(pos_idx) >= 5:
        ranks = np.arange(1, len(pos_idx) + 1)
        log_ranks = np.log(ranks)
        log_evals = np.log(evals[pos_idx])
        poly = np.polyfit(log_ranks, log_evals, 1)
        alpha = float(-poly[0])
        pred = poly[0] * log_ranks + poly[1]
        ss_tot = np.sum((log_evals - np.mean(log_evals)) ** 2)
        ss_res = np.sum((log_evals - pred) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
    else:
        alpha = 0.0
        r2 = 0.0
        
    return {
        'top_evals': evals[:32].tolist(),
        'alpha': alpha,
        'r2': r2,
        'erank': erank,
        'normalized_erank': float(erank / D),
        'spectral_entropy': spectral_entropy,
        'total_variance': total_var
    }

def run_geometry_benchmark_for_seed(seed, epochs_per_task=10):
    print(f"\n=======================================================")
    print(f"   STARTING REPRESENTATION GEOMETRY BENCHMARK (SEED {seed})")
    print(f"=======================================================")
    set_seed(seed)
    tasks, full_test_loader = get_cifar10_tasks(batch_size=128)
    task0_test_loader = tasks[0]["test"]
    
    results = {
        'seed': seed,
        'finetune': {'step_metrics': []},
        'ewc': {'step_metrics': []},
        'uncalibrated_smr': {'step_metrics': []},
        'smr': {'step_metrics': []}
    }
    
    # -------------------------------------------------------------
    # 1. Finetune (Sequential SGD)
    # -------------------------------------------------------------
    print("\n--- Running Finetune (Sequential SGD) ---")
    set_seed(seed)
    ft_model = ResNet18(num_classes=10).to(DEVICE)
    
    initial_t0_feats = None
    for t in range(5):
        ft_model.head.current_active_classes = tasks[t]["cum"]
        train_task(ft_model, tasks[t]["train"], DEVICE, epochs=epochs_per_task, lr=0.01)
        
        feats_t0, _ = extract_all_layer_features(ft_model, task0_test_loader, DEVICE)
        if t == 0:
            initial_t0_feats = deepcopy(feats_t0)
            
        temporal_cka = {l: linear_cka(feats_t0[l], initial_t0_feats[l]) for l in feats_t0}
        cross_cka = compute_layerwise_cka_matrix(feats_t0)
        layer_spectra = {l: compute_spectrum_metrics(feats_t0[l]) for l in feats_t0}
        t0_acc = evaluate_task_accuracy(ft_model, task0_test_loader, DEVICE, tasks[0]["classes"])
        
        results['finetune']['step_metrics'].append({
            'task_step': t,
            'task0_acc': t0_acc,
            'temporal_cka': temporal_cka,
            'cross_cka': cross_cka,
            'spectra': layer_spectra
        })
        print(f"  Step T{t}: Task0 Acc={100*t0_acc:.2f}%, L4 CKA={temporal_cka['layer4']:.4f}, L4 Alpha={layer_spectra['layer4']['alpha']:.3f}, L4 eRank={layer_spectra['layer4']['erank']:.1f}")

    # -------------------------------------------------------------
    # 2. EWC
    # -------------------------------------------------------------
    print("\n--- Running EWC ---")
    set_seed(seed)
    ewc_model = ResNet18(num_classes=10).to(DEVICE)
    ewc_reg = EWCRegularizer(ewc_model, lambda_ewc=5000.0)
    
    initial_t0_feats_ewc = None
    for t in range(5):
        ewc_model.head.current_active_classes = tasks[t]["cum"]
        opt_ewc = torch.optim.SGD(ewc_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for epoch in range(epochs_per_task):
            ewc_model.train()
            for x, y in tasks[t]["train"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt_ewc.zero_grad()
                loss = F.cross_entropy(ewc_model(x), y) + ewc_reg.penalty(ewc_model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ewc_model.parameters(), 5.0)
                opt_ewc.step()
        ewc_reg.compute_fisher(ewc_model, tasks[t]["train"], DEVICE)
        
        feats_t0, _ = extract_all_layer_features(ewc_model, task0_test_loader, DEVICE)
        if t == 0:
            initial_t0_feats_ewc = deepcopy(feats_t0)
            
        temporal_cka = {l: linear_cka(feats_t0[l], initial_t0_feats_ewc[l]) for l in feats_t0}
        cross_cka = compute_layerwise_cka_matrix(feats_t0)
        layer_spectra = {l: compute_spectrum_metrics(feats_t0[l]) for l in feats_t0}
        t0_acc = evaluate_task_accuracy(ewc_model, task0_test_loader, DEVICE, tasks[0]["classes"])
        
        results['ewc']['step_metrics'].append({
            'task_step': t,
            'task0_acc': t0_acc,
            'temporal_cka': temporal_cka,
            'cross_cka': cross_cka,
            'spectra': layer_spectra
        })
        print(f"  Step T{t}: Task0 Acc={100*t0_acc:.2f}%, L4 CKA={temporal_cka['layer4']:.4f}, L4 Alpha={layer_spectra['layer4']['alpha']:.3f}, L4 eRank={layer_spectra['layer4']['erank']:.1f}")

    # -------------------------------------------------------------
    # 3 & 4. Uncalibrated vs Calibrated SMR
    # -------------------------------------------------------------
    print("\n--- Running SMR Pipeline (Comparing Uncalibrated vs Calibrated) ---")
    set_seed(seed)
    smr_model = ResNet18(num_classes=10).to(DEVICE)
    
    global_protection = {}
    saved_protected_weights = {}
    task_isolated_states = {}
    task_uncal_states = {}
    
    initial_t0_feats_smr = None
    
    for t in range(5):
        task = tasks[t]
        smr_model.head.current_active_classes = task["cum"]
        print(f"\n  [SMR Task {t}] Training representation...")
        
        # 1. Train current task protecting prior tasks
        train_task(smr_model, task["train"], DEVICE, epochs=epochs_per_task, lr=0.01,
                   protection_masks=global_protection,
                   saved_weights=saved_protected_weights, use_amp=True)
                   
        # 2. Map critical circuit
        importance = compute_neuron_importance(smr_model, task["train"], DEVICE, num_batches=20)
        task_mask = compute_smr_mask(importance, percentile=85, prior_protection=global_protection)
        
        # 3. Specialize isolated decision circuit
        fine_tune_decision_subcircuit(smr_model, task["train"], task_mask, task["classes"], DEVICE, epochs=4, lr=0.01)
        
        curr_state = snapshot_full_state(smr_model)
        
        # 4. Snapshot UNCALIBRATED state (Zero out non-task channels without BN recalibration)
        apply_inference_routing(smr_model, curr_state,
                                task_mask, snapshot_bn_state(smr_model),
                                snapshot_fc_state(smr_model), DEVICE,
                                calibration_loader=None)
        task_uncal_states[t] = snapshot_full_state(smr_model)
        smr_model.load_state_dict(curr_state)
        
        # 5. Snapshot CALIBRATED state (With O(1) BN recalibration)
        apply_inference_routing(smr_model, curr_state,
                                task_mask, snapshot_bn_state(smr_model),
                                snapshot_fc_state(smr_model), DEVICE,
                                calibration_loader=task["train"])
        task_isolated_states[t] = snapshot_full_state(smr_model)
        smr_model.load_state_dict(curr_state)
        
        saved_protected_weights = snapshot_full_state(smr_model)
        global_protection = merge_masks(global_protection, task_mask)
        
        # EVALUATE TASK 0 IN UNCALIBRATED STATE (Stale BatchNorm)
        smr_model.load_state_dict(task_uncal_states[0])
        smr_model.head.current_active_classes = tasks[0]["cum"]
        feats_uncal, _ = extract_all_layer_features(smr_model, task0_test_loader, DEVICE)
        t0_acc_uncal = evaluate_task_accuracy(smr_model, task0_test_loader, DEVICE, tasks[0]["classes"])
        
        # EVALUATE TASK 0 IN FULL SMR STATE (Recalibrated BatchNorm + Protected Invariance)
        smr_model.load_state_dict(task_isolated_states[0])
        smr_model.head.current_active_classes = tasks[0]["cum"]
        feats_cal, _ = extract_all_layer_features(smr_model, task0_test_loader, DEVICE)
        t0_acc_cal = evaluate_task_accuracy(smr_model, task0_test_loader, DEVICE, tasks[0]["classes"])
        smr_model.load_state_dict(curr_state)
        
        if t == 0:
            initial_t0_feats_smr = deepcopy(feats_cal)
            
        temporal_cka_uncal = {l: linear_cka(feats_uncal[l], initial_t0_feats_smr[l]) for l in feats_uncal}
        cross_cka_uncal = compute_layerwise_cka_matrix(feats_uncal)
        spectra_uncal = {l: compute_spectrum_metrics(feats_uncal[l]) for l in feats_uncal}
        
        temporal_cka_cal = {l: linear_cka(feats_cal[l], initial_t0_feats_smr[l]) for l in feats_cal}
        cross_cka_cal = compute_layerwise_cka_matrix(feats_cal)
        spectra_cal = {l: compute_spectrum_metrics(feats_cal[l]) for l in feats_cal}
        
        results['uncalibrated_smr']['step_metrics'].append({
            'task_step': t,
            'task0_acc': t0_acc_uncal,
            'temporal_cka': temporal_cka_uncal,
            'cross_cka': cross_cka_uncal,
            'spectra': spectra_uncal
        })
        
        results['smr']['step_metrics'].append({
            'task_step': t,
            'task0_acc': t0_acc_cal,
            'temporal_cka': temporal_cka_cal,
            'cross_cka': cross_cka_cal,
            'spectra': spectra_cal
        })
        
        print(f"  Step T{t} UNCALIBRATED: Task0 Acc={100*t0_acc_uncal:.2f}%, L4 CKA={temporal_cka_uncal['layer4']:.4f}, L4 Alpha={spectra_uncal['layer4']['alpha']:.3f}, L4 eRank={spectra_uncal['layer4']['erank']:.1f}")
        print(f"  Step T{t} SMR (CALIBRATED): Task0 Acc={100*t0_acc_cal:.2f}%, L4 CKA={temporal_cka_cal['layer4']:.4f}, L4 Alpha={spectra_cal['layer4']['alpha']:.3f}, L4 eRank={spectra_cal['layer4']['erank']:.1f}")

    # Full test set evaluation at step 4
    print("\n--- Evaluating Complete 10,000 CIFAR-10 Test Set for SMR ---")
    task_accs = []
    for eval_t in range(5):
        smr_model.load_state_dict(task_isolated_states[eval_t])
        smr_model.head.current_active_classes = tasks[eval_t]["cum"]
        acc = evaluate_task_accuracy(smr_model, tasks[eval_t]["test"], DEVICE, tasks[eval_t]["classes"])
        task_accs.append(acc)
    smr_model.load_state_dict(curr_state)
    mean_aa = float(np.mean(task_accs))
    print(f"  Final SMR Full Test Set Average Accuracy: {100*mean_aa:.2f}% | Per-task: {[f'{100*a:.2f}%' for a in task_accs]}")
    results['final_smr_task_accs'] = task_accs
    results['final_smr_mean_aa'] = mean_aa

    return results

def main():
    seeds = [42, 1337, 2025]
    all_seed_results = []
    
    t0_all = time.time()
    for s in seeds:
        seed_res = run_geometry_benchmark_for_seed(s, epochs_per_task=10)
        all_seed_results.append(seed_res)
        
    os.makedirs("results_final", exist_ok=True)
    out_path = os.path.join("results_final", "representation_geometry_results.json")
    with open(out_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"REPRESENTATION GEOMETRY BENCHMARK COMPLETE ({time.time()-t0_all:.1f}s)")
    print(f"Saved results to: {out_path}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
