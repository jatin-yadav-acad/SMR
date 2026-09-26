"""
run_cifar10_c_robustness.py
===========================
Evaluates Out-of-Distribution (OOD) Corruption Robustness of continually learned
models on the full CIFAR-10-C benchmark (19 corruption types, Hendrycks & Dietterich 2019).

Compares:
1. Finetune
2. EWC
3. Replay (DER++)
4. SMR (Ours)

Measures:
- Mean Corruption Accuracy (mCA) across all 19 corruptions
- Category breakdowns: Noise, Blur, Weather, Digital
- Task retention under environmental distribution shift
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import ResNet18
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    recalibrate_bn, apply_inference_routing, snapshot_full_state,
    snapshot_bn_state, snapshot_fc_state, merge_masks,
    fine_tune_decision_subcircuit, EWCRegularizer, ReplayBuffer,
    evaluate_task_accuracy
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "brightness", "spatter",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"
]

CATEGORIES = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "Blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "Weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"]
}

def get_clean_cifar10_tasks(batch_size=128):
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    root = os.path.join("data", "cifar10")
    train_ds = datasets.CIFAR10(root, True, download=True, transform=tr)
    
    num_tasks = 5
    classes_per_task = 2
    tasks = []
    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        tri = [i for i, l in enumerate(train_ds.targets) if l in cls]
        tasks.append({
            "id": t,
            "classes": cls,
            "cum": (t + 1) * classes_per_task,
            "train": DataLoader(Subset(train_ds, tri), batch_size=batch_size, shuffle=True, pin_memory=True),
        })
    return tasks

def load_cifar10_c_task_loaders(cifar_c_dir, corruption_name, severity=3, batch_size=128):
    """
    Loads corrupted test images for a specific corruption and severity level (1-5),
    split into 5 task evaluation loaders.
    """
    data_path = os.path.join(cifar_c_dir, f"{corruption_name}.npy")
    label_path = os.path.join(cifar_c_dir, "labels.npy")
    
    data = np.load(data_path)     # [50000, 32, 32, 3]
    labels = np.load(label_path) # [50000]
    
    # Slice severity level (each level is 10000 samples)
    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    sub_data = data[start_idx:end_idx]
    sub_labels = labels[start_idx:end_idx]
    
    # Normalize: uint8 [0, 255] -> float [0, 1] -> standardized
    # shape [N, 3, 32, 32]
    tensor_data = torch.from_numpy(sub_data).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    tensor_data = (tensor_data - mean) / std
    tensor_labels = torch.from_numpy(sub_labels).long()
    
    # Create task loaders
    classes_per_task = 2
    task_loaders = {}
    for t in range(5):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        cls_set = set(cls)
        mask = torch.tensor([l.item() in cls_set for l in tensor_labels])
        t_x = tensor_data[mask]
        t_y = tensor_labels[mask]
        ds = TensorDataset(t_x, t_y)
        task_loaders[t] = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return task_loaders

def evaluate_model_on_cifar10_c(model, task_loaders, device, eval_fn):
    """Evaluates average accuracy across all 5 tasks for a given corruption."""
    accs = []
    for t in range(5):
        acc = eval_fn(model, t, task_loaders[t], device)
        accs.append(acc)
    return float(np.mean(accs)), accs

def run_cifar10_c_experiment():
    print("=" * 80)
    print("RUNNING CIFAR-10-C CORRUPTION ROBUSTNESS CONTINUAL LEARNING BENCHMARK")
    print(f"Device: {DEVICE} | Corruptions: {len(CORRUPTIONS)} | Severity: 3")
    print("=" * 80)
    
    cifar_c_dir = "CIFAR-10-C"
    clean_tasks = get_clean_cifar10_tasks(batch_size=128)
    
    results = {}
    
    # =========================================================================
    # 1. TRAIN & EVALUATE SMR
    # =========================================================================
    print("\n--- Training SMR Model ---")
    set_seed(42)
    model_smr = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
    task_masks = {}
    task_bn_states = {}
    task_fc_states = {}
    task_states = {}
    global_protection = {}
    saved_protected_weights = {}
    
    for t in range(5):
        task = clean_tasks[t]
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
        task_states[t] = snapshot_full_state(model_smr)
        model_smr.load_state_dict(curr_state)
        
        saved_protected_weights = snapshot_full_state(model_smr)
        global_protection = merge_masks(global_protection, t_mask)
        print(f"  [SMR] Task {t} completed.")
        
    def eval_smr_task(m, t_id, loader, dev):
        m.load_state_dict(task_states[t_id])
        m.head.current_active_classes = (t_id + 1) * 2
        cls = list(range(t_id * 2, (t_id + 1) * 2))
        return evaluate_task_accuracy(m, loader, dev, cls)
        
    print("\n--- Evaluating SMR on CIFAR-10-C ---")
    smr_c_accs = {}
    for c_name in CORRUPTIONS:
        loaders = load_cifar10_c_task_loaders(cifar_c_dir, c_name, severity=3)
        mean_acc, per_task = evaluate_model_on_cifar10_c(model_smr, loaders, DEVICE, eval_smr_task)
        smr_c_accs[c_name] = mean_acc
        print(f"  {c_name:<20s} : {mean_acc:.4f} (Tasks: {[f'{a:.2f}' for a in per_task]})")
    results["smr"] = smr_c_accs
    
    # =========================================================================
    # 2. TRAIN & EVALUATE REPLAY (DER++)
    # =========================================================================
    print("\n--- Training Replay (DER++) Model ---")
    set_seed(42)
    model_replay = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
    replay_buf = ReplayBuffer(capacity_per_class=50) # 500 total
    
    for t in range(5):
        task = clean_tasks[t]
        model_replay.head.current_active_classes = task["cum"]
        model_replay.train()
        opt = torch.optim.SGD(model_replay.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        for ep in range(15):
            for x, y in task["train"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = model_replay(x)
                loss = F.cross_entropy(out, y)
                rx, ry = replay_buf.sample(x.size(0))
                if rx is not None:
                    rx, ry = rx.to(DEVICE), ry.to(DEVICE)
                    rout = model_replay(rx)
                    loss += F.cross_entropy(rout, ry)
                loss.backward()
                opt.step()
        replay_buf.add_task(task["train"], DEVICE)
        print(f"  [Replay] Task {t} completed.")
        
    def eval_standard_task(m, t_id, loader, dev):
        m.head.current_active_classes = (t_id + 1) * 2
        cls = list(range(t_id * 2, (t_id + 1) * 2))
        return evaluate_task_accuracy(m, loader, dev, cls)
        
    print("\n--- Evaluating Replay on CIFAR-10-C ---")
    replay_c_accs = {}
    for c_name in CORRUPTIONS:
        loaders = load_cifar10_c_task_loaders(cifar_c_dir, c_name, severity=3)
        mean_acc, per_task = evaluate_model_on_cifar10_c(model_replay, loaders, DEVICE, eval_standard_task)
        replay_c_accs[c_name] = mean_acc
        print(f"  {c_name:<20s} : {mean_acc:.4f} (Tasks: {[f'{a:.2f}' for a in per_task]})")
    results["replay"] = replay_c_accs

    # =========================================================================
    # 3. TRAIN & EVALUATE EWC
    # =========================================================================
    print("\n--- Training EWC Model ---")
    set_seed(42)
    model_ewc = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
    ewc_reg = EWCRegularizer(model_ewc, lambda_ewc=5000.0)
    for t in range(5):
        task = clean_tasks[t]
        model_ewc.head.current_active_classes = task["cum"]
        model_ewc.train()
        opt = torch.optim.SGD(model_ewc.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        for ep in range(15):
            for x, y in task["train"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = model_ewc(x)
                loss = F.cross_entropy(out, y) + ewc_reg.penalty(model_ewc)
                loss.backward()
                opt.step()
        ewc_reg.compute_fisher(model_ewc, task["train"], DEVICE)
        print(f"  [EWC] Task {t} completed.")
        
    print("\n--- Evaluating EWC on CIFAR-10-C ---")
    ewc_c_accs = {}
    for c_name in CORRUPTIONS:
        loaders = load_cifar10_c_task_loaders(cifar_c_dir, c_name, severity=3)
        mean_acc, per_task = evaluate_model_on_cifar10_c(model_ewc, loaders, DEVICE, eval_standard_task)
        ewc_c_accs[c_name] = mean_acc
        print(f"  {c_name:<20s} : {mean_acc:.4f} (Tasks: {[f'{a:.2f}' for a in per_task]})")
    results["ewc"] = ewc_c_accs
    
    # =========================================================================
    # 4. TRAIN & EVALUATE FINETUNE
    # =========================================================================
    print("\n--- Training Finetune Model ---")
    set_seed(42)
    model_ft = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
    for t in range(5):
        task = clean_tasks[t]
        model_ft.head.current_active_classes = task["cum"]
        train_task(model_ft, task["train"], DEVICE, epochs=15, lr=0.03)
        print(f"  [Finetune] Task {t} completed.")
        
    print("\n--- Evaluating Finetune on CIFAR-10-C ---")
    ft_c_accs = {}
    for c_name in CORRUPTIONS:
        loaders = load_cifar10_c_task_loaders(cifar_c_dir, c_name, severity=3)
        mean_acc, per_task = evaluate_model_on_cifar10_c(model_ft, loaders, DEVICE, eval_standard_task)
        ft_c_accs[c_name] = mean_acc
        print(f"  {c_name:<20s} : {mean_acc:.4f} (Tasks: {[f'{a:.2f}' for a in per_task]})")
    results["finetune"] = ft_c_accs

    # Compute category summaries
    summary = {}
    for method, c_dict in results.items():
        summary[method] = {
            "overall_mCA": float(np.mean(list(c_dict.values()))),
            "by_category": {}
        }
        for cat_name, cat_corrs in CATEGORIES.items():
            cat_vals = [c_dict[c] for c in cat_corrs if c in c_dict]
            summary[method]["by_category"][cat_name] = float(np.mean(cat_vals))
            
    results["summary"] = summary
    
    out_file = os.path.join("results_final", "cifar10_c_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] CIFAR-10-C results saved to {out_file}!")
    
    print("\n" + "=" * 80)
    print("CIFAR-10-C BENCHMARK SUMMARY (Mean Corruption Accuracy, Severity 3):")
    print(f"{'Method':<15s} | {'Overall mCA':<12s} | {'Noise':<8s} | {'Blur':<8s} | {'Weather':<8s} | {'Digital':<8s}")
    print("-" * 70)
    for m in ["finetune", "ewc", "replay", "smr"]:
        s = summary[m]
        print(f"{m:<15s} | {s['overall_mCA']*100:.2f}%       | {s['by_category']['Noise']*100:.1f}%   | "
              f"{s['by_category']['Blur']*100:.1f}%   | {s['by_category']['Weather']*100:.1f}%   | {s['by_category']['Digital']*100:.1f}%")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_cifar10_c_experiment()
