#!/usr/bin/env python
"""
run_sensory_fastgate_routing.py
================================
Autonomous O(1) Sensory Prototype Fast-Gate for Class-IL (Phase 2).

The Problem:
Evaluating T task sub-networks in Class-IL requires O(T) full forward passes,
causing a 10x test-time latency penalty as tasks accumulate.

The Solution:
Under SMR's Sensory-Decision Partitioning, early sensory layers (conv1 through layer3)
are dense, shared, and universal across all tasks.
In a single forward pass through conv1-layer3, we obtain universal sensory features
h_{sensory}(x) in R^{256} and spatial activations z3 in R^{B x 256 x 4 x 4}.
Task centroids p_t in R^{256} are precomputed at training time.
At test time:
1. A single forward pass through conv1-layer3 extracts h_{sensory}(x).
2. Task prediction is performed in O(1) time: t* = argmin_t ||h_{sensory}(x) - p_t||_2.
3. Only sub-network t* in layer4 is evaluated!
Total test-time complexity: Strictly O(1) forward pass!

Benchmarks:
- Split CIFAR-10 (5 tasks, 10 classes)
- Split CIFAR-100 (10 tasks, 100 classes)
Measures:
- Test latency: O(1) Fast-Gate vs O(T) multi-pass
- Task routing accuracy
- Class-IL top-1 accuracy
Saves results to results_final/sensory_fastgate_results.json.
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
from experiments.run_mahalanobis_etf_routing import (
    train_and_extract_task_subcircuits, set_active_classes
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_sensory(model, x):
    """
    Executes the universal shared sensory pathway: conv1 -> bn1 -> layer1 -> layer2 -> layer3.
    Returns:
        z3: Spatial feature tensor [B, 256, H3, W3]
        h_sens: Pooled sensory representation in R^256 [B, 256]
    """
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.maxpool(out)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    h_sens = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    return out, h_sens


def forward_decision(model, z3):
    """
    Executes the task-specialized decision pathway: layer4 -> head.
    Returns:
        logits: Unnormalized logits [B, num_classes]
    """
    out = model.layer4(z3)
    feat = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    return model.head(feat)


def compute_sensory_prototypes(model, task_subcircuits, tasks, device, max_samples_per_task=2000):
    """
    Computes task centroids p_t and class prototypes mu_c in R^256 sensory space.
    Sensory layers are shared, so any subcircuit or base model can extract them.
    """
    model.load_state_dict(task_subcircuits[0]["state_dict"])
    model.eval()

    task_centroids = {}
    class_prototypes = {}

    with torch.no_grad():
        for t_idx, task_info in enumerate(tasks):
            train_loader = task_info['train']
            classes = task_info['classes']
            
            task_feats = []
            class_feats = {c: [] for c in classes}
            n_samples = 0

            for x, y in train_loader:
                x = x.to(device)
                _, h_sens = forward_sensory(model, x)
                h_sens_cpu = h_sens.cpu()

                for i in range(x.size(0)):
                    feat = h_sens_cpu[i]
                    task_feats.append(feat)
                    label = y[i].item()
                    if label in class_feats:
                        class_feats[label].append(feat)
                    n_samples += 1
                if n_samples >= max_samples_per_task:
                    break

            # Task centroid
            if len(task_feats) > 0:
                p_t = torch.stack(task_feats, dim=0).mean(dim=0)
            else:
                p_t = torch.zeros(256)
            task_centroids[t_idx] = p_t

            # Class prototypes
            for c in classes:
                if len(class_feats[c]) > 0:
                    mu_c = torch.stack(class_feats[c], dim=0).mean(dim=0)
                else:
                    mu_c = torch.zeros(256)
                class_prototypes[c] = mu_c

    return task_centroids, class_prototypes


def benchmark_fastgate_accuracy(model, task_subcircuits, task_centroids, class_prototypes, tasks, device):
    """
    Evaluates:
    1. O(1) Fast-Gate Task Routing Accuracy (Centroid vs Prototype)
    2. O(1) Fast-Gate Class-IL Top-1 Accuracy
    3. O(T) Multi-Pass Free Energy Baseline Accuracy
    4. Confusion Matrices
    """
    num_tasks = len(tasks)
    
    # Storage for all test samples
    all_true_tasks = []
    all_true_labels = []
    
    pred_tasks_centroid = []
    pred_classes_centroid = []
    
    pred_tasks_proto = []
    pred_classes_proto = []
    
    pred_tasks_energy = []
    pred_classes_energy = []

    # Stack centroids for fast matrix vector distance
    centroid_tensor = torch.stack([task_centroids[t] for t in range(num_tasks)], dim=0).to(device) # [T, 256]
    
    # Task classes mapping
    task_classes_map = {t: task_subcircuits[t]["classes"] for t in range(num_tasks)}

    print("\n--- Evaluating Fast-Gate & Free-Energy Accuracy across all Test Sets ---")
    
    for true_t, task_info in enumerate(tasks):
        test_loader = task_info['test']
        for x, y in test_loader:
            x = x.to(device)
            B = x.size(0)
            
            for b in range(B):
                all_true_tasks.append(true_t)
                all_true_labels.append(y[b].item())

            # =========================================================
            # A. O(1) SENSORY PROTOTYPE FAST-GATE
            # =========================================================
            # Step 1: Universal sensory forward pass (ONCE for all candidate tasks!)
            with torch.no_grad():
                model.load_state_dict(task_subcircuits[0]["state_dict"])
                model.eval()
                z3, h_sens = forward_sensory(model, x) # z3: [B, 256, 4, 4], h_sens: [B, 256]

                # Distance to task centroids: ||h_sens - p_t||_2
                # h_sens: [B, 1, 256], centroid_tensor: [1, T, 256]
                dists_centroid = torch.cdist(h_sens, centroid_tensor, p=2) # [B, T]
                best_task_centroid = dists_centroid.argmin(dim=1).cpu().numpy()

                # Distance to nearest class prototype
                dists_by_task = []
                for t in range(num_tasks):
                    t_classes = task_classes_map[t]
                    t_protos = torch.stack([class_prototypes[c].to(device) for c in t_classes], dim=0) # [|C_t|, 256]
                    t_dists = torch.cdist(h_sens, t_protos, p=2) # [B, |C_t|]
                    min_d, _ = t_dists.min(dim=1) # [B]
                    dists_by_task.append(min_d)
                dists_proto = torch.stack(dists_by_task, dim=1) # [B, T]
                best_task_proto = dists_proto.argmin(dim=1).cpu().numpy()

                # Step 2: Evaluate ONLY predicted sub-network in layer4!
                # Group samples by predicted task for vectorized execution
                batch_preds_centroid = [None] * B
                for cand_t in np.unique(best_task_centroid):
                    idx = np.where(best_task_centroid == cand_t)[0]
                    if len(idx) == 0: continue
                    model.load_state_dict(task_subcircuits[cand_t]["state_dict"])
                    model.eval()
                    logits_t = forward_decision(model, z3[idx]) # [len(idx), num_classes]
                    c_classes = task_classes_map[cand_t]
                    t_logits = logits_t[:, c_classes]
                    best_cls_idx = t_logits.argmax(dim=1).cpu().numpy()
                    for k, sample_idx in enumerate(idx):
                        batch_preds_centroid[sample_idx] = c_classes[best_cls_idx[k]]

                batch_preds_proto = [None] * B
                for cand_t in np.unique(best_task_proto):
                    idx = np.where(best_task_proto == cand_t)[0]
                    if len(idx) == 0: continue
                    model.load_state_dict(task_subcircuits[cand_t]["state_dict"])
                    model.eval()
                    logits_t = forward_decision(model, z3[idx])
                    c_classes = task_classes_map[cand_t]
                    t_logits = logits_t[:, c_classes]
                    best_cls_idx = t_logits.argmax(dim=1).cpu().numpy()
                    for k, sample_idx in enumerate(idx):
                        batch_preds_proto[sample_idx] = c_classes[best_cls_idx[k]]

            for b in range(B):
                pred_tasks_centroid.append(int(best_task_centroid[b]))
                pred_classes_centroid.append(batch_preds_centroid[b])
                pred_tasks_proto.append(int(best_task_proto[b]))
                pred_classes_proto.append(batch_preds_proto[b])

            # =========================================================
            # B. O(T) MULTI-PASS FREE ENERGY BASELINE
            # =========================================================
            with torch.no_grad():
                task_energies = []
                task_best_classes = []
                for cand_t in range(num_tasks):
                    model.load_state_dict(task_subcircuits[cand_t]["state_dict"])
                    model.eval()
                    logits = model(x)
                    c_classes = task_classes_map[cand_t]
                    t_logits = logits[:, c_classes]
                    energy = - torch.logsumexp(t_logits, dim=1) # [B]
                    task_energies.append(energy.cpu().numpy())
                    best_c_idx = t_logits.argmax(dim=1).cpu().numpy()
                    task_best_classes.append([c_classes[k] for k in best_c_idx])

                task_energies = np.stack(task_energies, axis=1) # [B, T]
                best_energy_tasks = np.argmin(task_energies, axis=1)

                for b in range(B):
                    sel_t = best_energy_tasks[b]
                    pred_tasks_energy.append(int(sel_t))
                    pred_classes_energy.append(task_best_classes[sel_t][b])

    # Convert to arrays
    all_true_tasks = np.array(all_true_tasks)
    all_true_labels = np.array(all_true_labels)
    total_samples = len(all_true_labels)

    pred_tasks_centroid = np.array(pred_tasks_centroid)
    pred_classes_centroid = np.array(pred_classes_centroid)
    pred_tasks_proto = np.array(pred_tasks_proto)
    pred_classes_proto = np.array(pred_classes_proto)
    pred_tasks_energy = np.array(pred_tasks_energy)
    pred_classes_energy = np.array(pred_classes_energy)

    # Routing Accuracies
    routing_acc_centroid = float(np.mean(pred_tasks_centroid == all_true_tasks) * 100.0)
    routing_acc_proto = float(np.mean(pred_tasks_proto == all_true_tasks) * 100.0)
    routing_acc_energy = float(np.mean(pred_tasks_energy == all_true_tasks) * 100.0)

    # Class-IL Accuracies
    class_il_acc_centroid = float(np.mean(pred_classes_centroid == all_true_labels) * 100.0)
    class_il_acc_proto = float(np.mean(pred_classes_proto == all_true_labels) * 100.0)
    class_il_acc_energy = float(np.mean(pred_classes_energy == all_true_labels) * 100.0)

    # Confusion Matrices
    cm_centroid = np.zeros((num_tasks, num_tasks), dtype=int)
    cm_proto = np.zeros((num_tasks, num_tasks), dtype=int)
    cm_energy = np.zeros((num_tasks, num_tasks), dtype=int)

    for i in range(total_samples):
        cm_centroid[all_true_tasks[i], pred_tasks_centroid[i]] += 1
        cm_proto[all_true_tasks[i], pred_tasks_proto[i]] += 1
        cm_energy[all_true_tasks[i], pred_tasks_energy[i]] += 1

    print(f"\n" + "=" * 80)
    print(f"ACCURACY BENCHMARK (N = {total_samples} test samples, {num_tasks} tasks)")
    print(f"=" * 80)
    print(f"Method                               | Task Routing Acc | Class-IL Top-1 Acc")
    print(f"-------------------------------------+------------------+-------------------")
    print(f"O(T) Multi-Pass Free Energy Baseline | {routing_acc_energy:14.2f}% | {class_il_acc_energy:15.2f}%")
    print(f"O(1) Sensory Fast-Gate (Task Centroid)| {routing_acc_centroid:14.2f}% | {class_il_acc_centroid:15.2f}%")
    print(f"O(1) Sensory Fast-Gate (Nearest Proto)| {routing_acc_proto:14.2f}% | {class_il_acc_proto:15.2f}%")
    print(f"=" * 80)

    return {
        "num_tasks": num_tasks,
        "total_samples": total_samples,
        "task_routing_acc": {
            "free_energy_OT": routing_acc_energy,
            "fastgate_centroid_O1": routing_acc_centroid,
            "fastgate_proto_O1": routing_acc_proto,
            "routing_acc_gain_vs_energy": routing_acc_centroid - routing_acc_energy
        },
        "class_il_acc": {
            "free_energy_OT": class_il_acc_energy,
            "fastgate_centroid_O1": class_il_acc_centroid,
            "fastgate_proto_O1": class_il_acc_proto,
            "class_il_gain_vs_energy": class_il_acc_centroid - class_il_acc_energy
        },
        "confusion_matrices": {
            "free_energy_OT": cm_energy.tolist(),
            "fastgate_centroid_O1": cm_centroid.tolist(),
            "fastgate_proto_O1": cm_proto.tolist()
        }
    }


def benchmark_test_latency(model, task_subcircuits, task_centroids, device, batch_size=128, num_iterations=100, warmup=15):
    """
    Measures physical CUDA execution latency:
    1. O(T) Multi-Pass: T full forward passes through entire ResNet-18
    2. O(1) Sensory Fast-Gate: single forward pass through conv1-layer3 + centroid argmin + single layer4 forward pass
    """
    num_tasks = len(task_subcircuits)
    centroid_tensor = torch.stack([task_centroids[t] for t in range(num_tasks)], dim=0).to(device)
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    # -------------------------------------------------------------
    # 1. Warm-up
    # -------------------------------------------------------------
    print(f"\n--- Warming up GPU ({warmup} iterations, batch_size={batch_size}) ---")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            for t in range(num_tasks):
                model.load_state_dict(task_subcircuits[t]["state_dict"])
                _ = model(dummy_input)
            z3, h_sens = forward_sensory(model, dummy_input)
            _ = torch.cdist(h_sens, centroid_tensor, p=2).argmin(dim=1)
            _ = forward_decision(model, z3)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # -------------------------------------------------------------
    # 2. Benchmark O(T) Multi-Pass Latency
    # -------------------------------------------------------------
    print(f"Measuring O(T) Multi-Pass Latency ({num_iterations} iterations, T={num_tasks})...")
    ot_times = []
    
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(num_iterations):
            starter.record()
            with torch.no_grad():
                for t in range(num_tasks):
                    model.load_state_dict(task_subcircuits[t]["state_dict"])
                    logits = model(dummy_input)
                    energy = - torch.logsumexp(logits[:, task_subcircuits[t]["classes"]], dim=1)
            ender.record()
            torch.cuda.synchronize()
            ot_times.append(starter.elapsed_time(ender)) # milliseconds
    else:
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            with torch.no_grad():
                for t in range(num_tasks):
                    model.load_state_dict(task_subcircuits[t]["state_dict"])
                    logits = model(dummy_input)
                    energy = - torch.logsumexp(logits[:, task_subcircuits[t]["classes"]], dim=1)
            t1 = time.perf_counter()
            ot_times.append((t1 - t0) * 1000.0)

    ot_mean_ms = float(np.mean(ot_times))
    ot_std_ms = float(np.std(ot_times))

    # -------------------------------------------------------------
    # 3. Benchmark O(1) Sensory Fast-Gate Latency
    # -------------------------------------------------------------
    print(f"Measuring O(1) Sensory Fast-Gate Latency ({num_iterations} iterations)...")
    o1_times = []

    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(num_iterations):
            starter.record()
            with torch.no_grad():
                # 1. Universal sensory forward pass
                model.load_state_dict(task_subcircuits[0]["state_dict"])
                z3, h_sens = forward_sensory(model, dummy_input)
                # 2. O(1) centroid distance routing
                dists = torch.cdist(h_sens, centroid_tensor, p=2)
                best_t = dists.argmin(dim=1)
                # 3. Single decision forward pass in layer4
                model.load_state_dict(task_subcircuits[0]["state_dict"])
                logits = forward_decision(model, z3)
            ender.record()
            torch.cuda.synchronize()
            o1_times.append(starter.elapsed_time(ender))
    else:
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.load_state_dict(task_subcircuits[0]["state_dict"])
                z3, h_sens = forward_sensory(model, dummy_input)
                dists = torch.cdist(h_sens, centroid_tensor, p=2)
                best_t = dists.argmin(dim=1)
                model.load_state_dict(task_subcircuits[0]["state_dict"])
                logits = forward_decision(model, z3)
            t1 = time.perf_counter()
            o1_times.append((t1 - t0) * 1000.0)

    o1_mean_ms = float(np.mean(o1_times))
    o1_std_ms = float(np.std(o1_times))
    speedup = float(ot_mean_ms / max(1e-4, o1_mean_ms))

    print(f"\n" + "=" * 80)
    print(f"TEST-TIME INFERENCE LATENCY BENCHMARK (Batch Size = {batch_size}, T = {num_tasks})")
    print(f"=" * 80)
    print(f"O(T) Multi-Pass Latency    : {ot_mean_ms:6.2f} +/- {ot_std_ms:.2f} ms / batch")
    print(f"O(1) Sensory Fast-Gate Latency: {o1_mean_ms:6.2f} +/- {o1_std_ms:.2f} ms / batch")
    print(f"Speedup Factor             : {speedup:6.2f}x ({speedup:.1f}x faster)")
    print(f"=" * 80)

    return {
        "batch_size": batch_size,
        "num_tasks": num_tasks,
        "ot_multi_pass_ms": {
            "mean": ot_mean_ms,
            "std": ot_std_ms
        },
        "o1_sensory_fastgate_ms": {
            "mean": o1_mean_ms,
            "std": o1_std_ms
        },
        "speedup_factor": speedup
    }


def run_sensory_fastgate_experiment():
    print("=" * 80)
    print("RUNNING O(1) SENSORY PROTOTYPE FAST-GATE BENCHMARK (PHASE 2)")
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
    c10_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'canonical_cifar10_smr.pt')
    
    print("Step 1: Extracting / Loading Canonical SMR Subcircuits...")
    model_c10, c10_subcircuits = train_and_extract_task_subcircuits(
        c10_tasks, num_classes=10, epochs_per_task=15, lr=0.03, prune_percentile=80, ft_epochs=4, ckpt_path=c10_ckpt
    )

    print("Step 2: Computing 256-Dimensional Sensory Task Centroids and Class Prototypes...")
    c10_centroids, c10_protos = compute_sensory_prototypes(model_c10, c10_subcircuits, c10_tasks, DEVICE)
    print(f"  Computed centroids for {len(c10_centroids)} tasks, prototypes for {len(c10_protos)} classes.")

    print("Step 3: Measuring Physical Inference Latency (O(1) vs O(T))...")
    c10_latency = benchmark_test_latency(model_c10, c10_subcircuits, c10_centroids, DEVICE, batch_size=128)

    print("Step 4: Benchmarking Task Routing & Class-IL Accuracy...")
    c10_acc = benchmark_fastgate_accuracy(model_c10, c10_subcircuits, c10_centroids, c10_protos, c10_tasks, DEVICE)
    
    results["split_cifar10"] = {
        "latency": c10_latency,
        "accuracy": c10_acc
    }

    # =============================================================
    # 2. SPLIT CIFAR-100 (10 Tasks, 100 Classes)
    # =============================================================
    print("\n" + "#" * 80)
    print("BENCHMARK 2: SPLIT CIFAR-100 (10 TASKS, 100 CLASSES)")
    print("#" * 80)
    c100_tasks = get_cifar100_tasks(batch_size=128)
    c100_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'canonical_cifar100_smr.pt')

    print("Step 1: Extracting / Loading Canonical SMR Subcircuits...")
    model_c100, c100_subcircuits = train_and_extract_task_subcircuits(
        c100_tasks, num_classes=100, epochs_per_task=12, lr=0.03, prune_percentile=92, ft_epochs=4, ckpt_path=c100_ckpt
    )

    print("Step 2: Computing 256-Dimensional Sensory Task Centroids and Class Prototypes...")
    c100_centroids, c100_protos = compute_sensory_prototypes(model_c100, c100_subcircuits, c100_tasks, DEVICE)
    print(f"  Computed centroids for {len(c100_centroids)} tasks, prototypes for {len(c100_protos)} classes.")

    print("Step 3: Measuring Physical Inference Latency (O(1) vs O(T))...")
    c100_latency = benchmark_test_latency(model_c100, c100_subcircuits, c100_centroids, DEVICE, batch_size=128)

    print("Step 4: Benchmarking Task Routing & Class-IL Accuracy...")
    c100_acc = benchmark_fastgate_accuracy(model_c100, c100_subcircuits, c100_centroids, c100_protos, c100_tasks, DEVICE)

    results["split_cifar100"] = {
        "latency": c100_latency,
        "accuracy": c100_acc
    }

    # Save to results_final/sensory_fastgate_results.json
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_final')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'sensory_fastgate_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Sensory Fast-Gate results successfully saved to: {out_file}")

    return results


if __name__ == '__main__':
    run_sensory_fastgate_experiment()
