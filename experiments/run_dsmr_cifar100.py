"""
run_dsmr_cifar100.py
====================
Executes Dynamic Modular Expansion (D-SMR) on Split CIFAR-100 (10 Tasks, 100 Classes, ResNet-18)
across 3 independent seeds [42, 1337, 2025].
Evaluates both Task-IL (Oracle) and Class-IL (Autonomous Free-Energy Routing).
Saves verified records directly into results_final/cifar100_results.json and
results_final/cifar100_benchmark_complete.json.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    apply_inference_routing, snapshot_full_state, snapshot_bn_state, snapshot_fc_state,
    merge_masks, fine_tune_decision_subcircuit, evaluate_task_accuracy,
    check_and_expand_capacity
)
from experiments.run_final_experiments import get_cifar100_tasks, ResNet18, set_active_classes, compute_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_task_il_dsmr(model, tasks, task_masks, task_bn_states, task_fc_states, current_task_id, device):
    """Evaluates Task-IL accuracy on the single unified expanded model using task channel masks and cached BN states."""
    model.eval()
    backup_state = snapshot_full_state(model)
    accs = []
    with torch.no_grad():
        for prev in range(current_task_id + 1):
            # Evaluate the single dynamically expanded model by applying task channel mask and cached BN
            model.active_classes = tasks[prev]["cum"]
            apply_inference_routing(model, backup_state, task_masks[prev],
                                    task_bn_states[prev], task_fc_states[prev], device)
            acc = evaluate_task_accuracy(model, tasks[prev]["test"], device, tasks[prev]["classes"])
            accs.append(acc)
    model.load_state_dict(backup_state)
    return accs

def evaluate_class_il_dsmr(model, tasks, task_masks, task_bn_states, task_fc_states, current_task_id, device, temperature=1.0):
    """Evaluates Class-IL accuracy on the single unified expanded model via autonomous free-energy routing."""
    model.eval()
    backup_state = snapshot_full_state(model)
    correct = 0
    total = 0
    with torch.no_grad():
        for t in range(current_task_id + 1):
            for x, y in tasks[t]["test"]:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)
                task_logits_list = []
                task_energies = []
                for prev in range(current_task_id + 1):
                    model.active_classes = tasks[prev]["cum"]
                    apply_inference_routing(model, backup_state, task_masks[prev],
                                            task_bn_states[prev], task_fc_states[prev], device)
                    logits = model(x)
                    start_cls = 0 if prev == 0 else tasks[prev-1]["cum"]
                    end_cls = tasks[prev]["cum"]
                    t_logits = logits[:, start_cls:end_cls]
                    task_logits_list.append(t_logits)
                    energy = -temperature * torch.logsumexp(t_logits / temperature, dim=1)
                    task_energies.append(energy)
                
                energies_tensor = torch.stack(task_energies, dim=1)
                best_tasks = energies_tensor.argmin(dim=1)
                preds = torch.zeros(batch_size, dtype=torch.long, device=device)
                for b in range(batch_size):
                    sel_t = best_tasks[b].item()
                    start_cls = 0 if sel_t == 0 else tasks[sel_t-1]["cum"]
                    local_pred = task_logits_list[sel_t][b].argmax().item()
                    preds[b] = start_cls + local_pred
                correct += (preds == y).sum().item()
                total += y.size(0)
    model.load_state_dict(backup_state)
    return correct / max(1, total)

def run_dsmr_single_seed(seed, epochs=12, lr=0.03, isolation_percentile=82,
                         expansion_threshold=0.70, delta_channels=64, batch_size=128):
    print(f"\n{'='*75}")
    print(f"  [D-SMR CIFAR-100] STARTING SEED {seed} ON {DEVICE.upper()}")
    print(f"{'='*75}")
    set_seed(seed)
    tasks = get_cifar100_tasks("./data", batch_size=batch_size, num_tasks=10)
    model = ResNet18(num_classes=100, cifar_style=True).to(DEVICE)
    
    task_accs = {}
    class_il_accs = {}
    task_states = {}
    task_masks = {}
    task_bn_states = {}
    task_fc_states = {}
    global_protection = {}
    saved_protected_weights = {}

    start_time = time.time()
    for t in range(len(tasks)):
        task = tasks[t]
        print(f"\n--- Task {t}/9 (Classes {task['classes']}) ---")
        
        # 1. Check and dynamically expand capacity if allocation ratio >= threshold
        if t > 0:
            model, global_protection, expanded = check_and_expand_capacity(
                model, global_protection, threshold=expansion_threshold, delta_channels=delta_channels,
                device=DEVICE, task_masks=task_masks, task_bn_states=task_bn_states,
                task_fc_states=task_fc_states, saved_protected_weights=saved_protected_weights
            )
            if expanded:
                print(f"  [D-SMR] Capacity dynamically expanded to {model.layer4[1].conv2.out_channels} channels.")
                # Pad previously cached model states so they match the expanded model architecture
                for past_tid in task_states:
                    expanded_state = deepcopy(model.state_dict())
                    # Copy old parameters
                    for k, v in task_states[past_tid].items():
                        if k in expanded_state:
                            if expanded_state[k].shape == v.shape:
                                expanded_state[k].copy_(v)
                            else:
                                # Padded tensor: copy old slice
                                s = tuple(slice(0, dim) for dim in v.shape)
                                expanded_state[k][s].copy_(v)
                    task_states[past_tid] = expanded_state

        # 2. Set active classes for current task
        set_active_classes(model, task["cum"])
        
        # 3. Train on current task (protecting prior tasks' channels)
        train_task(model, task["train"], DEVICE, epochs=epochs, lr=lr,
                   protection_masks=global_protection,
                   saved_weights=saved_protected_weights, use_amp=True)
                   
        # 4. Map critical circuit via first-order Taylor importance
        importance = compute_neuron_importance(model, task["train"], DEVICE, num_batches=20)
        task_mask = compute_smr_mask(importance, percentile=isolation_percentile, prior_protection=global_protection)
        
        # 5. Specialize isolated decision sub-circuit
        fine_tune_decision_subcircuit(model, task["train"], task_mask, task["classes"], DEVICE, epochs=3, lr=0.01)
        
        # 6. Recalibrate BatchNorm for isolated sub-network and snapshot states
        current_state = snapshot_full_state(model)
        apply_inference_routing(model, current_state, task_mask, snapshot_bn_state(model),
                                snapshot_fc_state(model), DEVICE, calibration_loader=task["train"])
        task_states[t] = snapshot_full_state(model)
        task_masks[t] = deepcopy(task_mask)
        task_bn_states[t] = snapshot_bn_state(model)
        task_fc_states[t] = snapshot_fc_state(model)
        model.load_state_dict(current_state)
        
        saved_protected_weights = snapshot_full_state(model)
        global_protection = merge_masks(global_protection, task_mask)
        
        # 7. Evaluate Task-IL accuracies on single expanded model using cached sub-circuits
        accs = evaluate_task_il_dsmr(model, tasks, task_masks, task_bn_states, task_fc_states, t, DEVICE)
        task_accs[t] = accs
        aa_task_il = np.mean(accs)
        print(f"  [Eval Task-IL] Step {t}: AA = {aa_task_il*100:.2f}% | Accs: {[f'{a*100:.1f}%' for a in accs]}")
        
        # 8. Evaluate Class-IL accuracy on single expanded model via Free-Energy Routing
        class_il_acc = evaluate_class_il_dsmr(model, tasks, task_masks, task_bn_states, task_fc_states, t, DEVICE)
        class_il_accs[t] = class_il_acc
        print(f"  [Eval Class-IL] Step {t}: Joint Acc = {class_il_acc*100:.2f}%")

    metrics = compute_metrics(task_accs, 10)
    metrics["seed"] = seed
    metrics["task_accs"] = {str(k): v for k, v in task_accs.items()}
    metrics["class_il_accs"] = {str(k): v for k, v in class_il_accs.items()}
    metrics["class_il_AA"] = float(class_il_accs[9])
    elapsed = time.time() - start_time
    print(f"\n[D-SMR Seed {seed} Complete in {elapsed:.1f}s] Task-IL AA: {metrics['AA']*100:.2f}% | FM: {metrics['FM']*100:.2f}% | Class-IL AA: {metrics['class_il_AA']*100:.2f}%")
    return metrics

def main():
    seeds = [42, 1337, 2025]
    seed_results = []
    
    for seed in seeds:
        res = run_dsmr_single_seed(seed, epochs=12, lr=0.03, isolation_percentile=82,
                                   expansion_threshold=0.70, delta_channels=64, batch_size=128)
        seed_results.append(res)
        
    aas = [r["AA"] for r in seed_results]
    fms = [r["FM"] for r in seed_results]
    class_aas = [r["class_il_AA"] for r in seed_results]
    
    dsmr_summary = {
        "mean_AA": float(np.mean(aas)),
        "std_AA": float(np.std(aas)),
        "mean_FM": float(np.mean(fms)),
        "std_FM": float(np.std(fms)),
        "mean_class_il_AA": float(np.mean(class_aas)),
        "std_class_il_AA": float(np.std(class_aas)),
        "per_seed": seed_results
    }
    
    print("\n" + "="*80)
    print("  D-SMR SPLIT CIFAR-100 MULTI-SEED BENCHMARK COMPLETE")
    print(f"  Task-IL AA  : {dsmr_summary['mean_AA']*100:.2f}% ± {dsmr_summary['std_AA']*100:.2f}%")
    print(f"  Task-IL FM  : {dsmr_summary['mean_FM']*100:.2f}% ± {dsmr_summary['std_FM']*100:.2f}%")
    print(f"  Class-IL AA : {dsmr_summary['mean_class_il_AA']*100:.2f}% ± {dsmr_summary['std_class_il_AA']*100:.2f}%")
    print("="*80)
    
    # Update cifar100_results.json
    c100_path = "results_final/cifar100_results.json"
    if os.path.exists(c100_path):
        with open(c100_path, "r") as f:
            c100_data = json.load(f)
    else:
        c100_data = {}
    c100_data["dsmr"] = dsmr_summary
    with open(c100_path, "w") as f:
        json.dump(c100_data, f, indent=2)
    print(f"  [SAVED] Updated {c100_path} with 'dsmr'")
    
    # Update cifar100_benchmark_complete.json
    comp_path = "results_final/cifar100_benchmark_complete.json"
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            comp_data = json.load(f)
    else:
        comp_data = {}
    comp_data["dsmr"] = dsmr_summary
    with open(comp_path, "w") as f:
        json.dump(comp_data, f, indent=2)
    print(f"  [SAVED] Updated {comp_path} with 'dsmr'")

if __name__ == "__main__":
    main()
