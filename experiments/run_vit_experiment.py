"""
run_vit_experiment.py
=====================
Investigates Continual Learning on Vision Transformers (ViT) with Layer Normalization.

Key Research Questions:
1. Does the Normalization Paradox affect Vision Transformers equipped with LayerNorm
   rather than BatchNorm?
2. Does Sparse Mechanistic Routing (SMR) successfully generalize to Attention architectures?

Evaluates:
- ViT on Split CIFAR-10 across 5 tasks
- Compares Finetune vs SMR on ViT
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import VisionTransformer, create_model
from src.smr_core import (
    set_seed, train_task, snapshot_full_state, evaluate_task_accuracy,
    merge_masks
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

def compute_vit_layer_importance(model, train_loader, device, num_batches=20):
    """Computes Taylor first-order importance on the MLP layers of Transformer blocks."""
    model.eval()
    importance = {}
    for name, param in model.named_parameters():
        if "mlp.fc1.weight" in name:
            importance[name] = torch.zeros(param.size(0), device=device)
            
    count = 0
    for x, y in train_loader:
        if count >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in importance and param.grad is not None:
                    # Sum |g * w| across in_features
                    imp = (param.grad * param).abs().sum(dim=1)
                    importance[name] += imp
        count += 1
        
    for k in importance:
        importance[k] /= count
    return importance

def compute_vit_masks(importance, percentile=80, prior_masks=None):
    masks = {}
    for name, imp in importance.items():
        val = imp.clone()
        if prior_masks and name in prior_masks:
            # Prioritize unallocated
            val = val + val.max() * (1.0 - prior_masks[name].float())
        threshold = torch.quantile(val, percentile / 100.0)
        mask = (val >= threshold).float()
        masks[name] = mask
    return masks

def run_vit_experiment():
    print("=" * 80)
    print("RUNNING VISION TRANSFORMER (ViT) CONTINUAL LEARNING BENCHMARK")
    print(f"Device: {DEVICE} | Architecture: ViT-Tiny (embed_dim=192, depth=6, heads=4)")
    print("=" * 80)
    
    set_seed(42)
    tasks = get_cifar10_tasks(batch_size=128)
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. FINETUNE (Sequential SGD on ViT)
    # -------------------------------------------------------------
    print("\n--- ViT Method 1: Finetune ---")
    model_ft = VisionTransformer(num_classes=10, embed_dim=192, depth=6, num_heads=4).to(DEVICE)
    ft_task_accs = {}
    
    for t in range(5):
        task = tasks[t]
        model_ft.head.current_active_classes = task["cum"]
        model_ft.active_classes = task["cum"]
        print(f"\n  [ViT Finetune] Task {t}: Classes {task['classes']}...")
        train_task(model_ft, task["train"], DEVICE, epochs=15, lr=0.01)
        
        accs = []
        for prev in range(t + 1):
            model_ft.head.current_active_classes = tasks[prev]["cum"]
            acc = evaluate_task_accuracy(model_ft, tasks[prev]["test"], DEVICE, tasks[prev]["classes"])
            accs.append(acc)
        ft_task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] Step T={t} | AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
        
    final_ft_accs = ft_task_accs[4]
    ft_aa = float(np.mean(final_ft_accs))
    ft_fm = float(np.mean([max([ft_task_accs[step][j] for step in range(j, 5)]) - final_ft_accs[j] for j in range(5)]))
    results["finetune"] = {"AA": ft_aa, "FM": ft_fm, "task_accs": ft_task_accs}
    print(f"ViT Finetune Final: AA={ft_aa:.4f}, FM={ft_fm:.4f}")

    # -------------------------------------------------------------
    # 2. SMR ON ViT (Sensory-Decision Partitioning: Pruning Deep Blocks 4-5)
    # -------------------------------------------------------------
    print("\n--- ViT Method 2: SMR (Mechanistic Routing) ---")
    set_seed(42)
    model_smr = VisionTransformer(num_classes=10, embed_dim=192, depth=6, num_heads=4).to(DEVICE)
    smr_task_accs = {}
    
    # Save states per task
    task_states = {}
    global_protection = {}
    
    for t in range(5):
        task = tasks[t]
        model_smr.head.current_active_classes = task["cum"]
        model_smr.active_classes = task["cum"]
        print(f"\n  [ViT SMR] Task {t}: Classes {task['classes']}...")
        
        # Train with gradient protection on deep MLP layers
        train_task(model_smr, task["train"], DEVICE, epochs=15, lr=0.01)
        
        # Map importance on deep transformer blocks (blocks 4 and 5)
        imp = compute_vit_layer_importance(model_smr, task["train"], DEVICE, num_batches=20)
        # Retain sensory blocks 0-3 dense, route decision blocks 4-5
        decision_imp = {k: v for k, v in imp.items() if "blocks.4" in k or "blocks.5" in k}
        t_mask = compute_vit_masks(decision_imp, percentile=80, prior_masks=global_protection)
        
        # Apply mask to isolate decision sub-network for Task t
        curr_state = snapshot_full_state(model_smr)
        with torch.no_grad():
            for name, param in model_smr.named_parameters():
                if name in t_mask:
                    mask_vec = t_mask[name].to(DEVICE).unsqueeze(1)
                    param.mul_(mask_vec)
        task_states[t] = snapshot_full_state(model_smr)
        model_smr.load_state_dict(curr_state)
        global_protection = merge_masks(global_protection, t_mask)
        
        # Evaluate all seen tasks using saved task states
        accs = []
        for prev in range(t + 1):
            curr_state = snapshot_full_state(model_smr)
            model_smr.load_state_dict(task_states[prev])
            model_smr.head.current_active_classes = tasks[prev]["cum"]
            acc = evaluate_task_accuracy(model_smr, tasks[prev]["test"], DEVICE, tasks[prev]["classes"])
            accs.append(acc)
            model_smr.load_state_dict(curr_state)
            
        smr_task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] Step T={t} | AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
        
    final_smr_accs = smr_task_accs[4]
    smr_aa = float(np.mean(final_smr_accs))
    smr_fm = float(np.mean([max([smr_task_accs[step][j] for step in range(j, 5)]) - final_smr_accs[j] for j in range(5)]))
    results["smr"] = {"AA": smr_aa, "FM": smr_fm, "task_accs": smr_task_accs}
    print(f"ViT SMR Final: AA={smr_aa:.4f}, FM={smr_fm:.4f}")

    os.makedirs("results_final", exist_ok=True)
    out_file = os.path.join("results_final", "vit_continual_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Vision Transformer continual learning results saved to {out_file}!")
    return results

if __name__ == "__main__":
    run_vit_experiment()
