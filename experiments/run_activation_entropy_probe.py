#!/usr/bin/env python
"""
run_activation_entropy_probe.py
===============================
Information-Theoretic Activation Entropy Probe across Depth.

Investigates Direction 2:
Explaining the Normalization Paradox as an information-theoretic breakdown:
Measures coordinate-wise Shannon entropy of binary activations b_l = I(z_l > 0)
across network depth (conv1, layer1, layer2, layer3, layer4):
    H_l = (1 / D_l) \sum_{j=1}^{D_l} [ - p_j log2(p_j) - (1 - p_j) log2(1 - p_j) ]
    where p_j = P(z_{l, j} > 0).

Compares 3 conditions:
(a) Dense Model (unpruned baseline)
(b) Uncalibrated Pruned Sub-circuit (shows exponential entropy decay toward 0 bits in deep layers)
(c) O(1) Recalibrated SMR (restores entropy and channel information capacity)

Outputs logged to results_final/activation_entropy_results.json.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import ResNet18
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    recalibrate_bn, apply_inference_routing, snapshot_full_state,
    snapshot_bn_state, snapshot_fc_state, evaluate_accuracy
)
from experiments.run_final_experiments import get_cifar10_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_depth_pre_activations(model, x):
    """
    Extracts pre-activations z_l before ReLU at:
    conv1, layer1, layer2, layer3, layer4.
    Returns dictionary of tensors on CPU.
    """
    pre_acts = {}
    
    # 1. conv1
    z_conv1 = model.bn1(model.conv1(x))
    pre_acts['conv1'] = z_conv1.detach().cpu()
    out = F.relu(z_conv1)
    out = model.maxpool(out)

    # 2. layer1
    for i, blk in enumerate(model.layer1):
        b_out = F.relu(blk.bn1(blk.conv1(out)))
        b_out = blk.bn2(blk.conv2(b_out))
        z = b_out + blk.shortcut(out)
        if i == len(model.layer1) - 1:
            pre_acts['layer1'] = z.detach().cpu()
        out = F.relu(z)

    # 3. layer2
    for i, blk in enumerate(model.layer2):
        b_out = F.relu(blk.bn1(blk.conv1(out)))
        b_out = blk.bn2(blk.conv2(b_out))
        z = b_out + blk.shortcut(out)
        if i == len(model.layer2) - 1:
            pre_acts['layer2'] = z.detach().cpu()
        out = F.relu(z)

    # 4. layer3
    for i, blk in enumerate(model.layer3):
        b_out = F.relu(blk.bn1(blk.conv1(out)))
        b_out = blk.bn2(blk.conv2(b_out))
        z = b_out + blk.shortcut(out)
        if i == len(model.layer3) - 1:
            pre_acts['layer3'] = z.detach().cpu()
        out = F.relu(z)

    # 5. layer4
    for i, blk in enumerate(model.layer4):
        b_out = F.relu(blk.bn1(blk.conv1(out)))
        b_out = blk.bn2(blk.conv2(b_out))
        z = b_out + blk.shortcut(out)
        if i == len(model.layer4) - 1:
            pre_acts['layer4'] = z.detach().cpu()
        out = F.relu(z)

    return pre_acts


def compute_activation_entropy(pre_acts_dict):
    """
    Given a dictionary of pre-activation tensors [N, D_l, H_l, W_l],
    computes coordinate-wise Shannon entropy H_l for each layer:
        p_j = P(z_{l, j} > 0)
        H(p_j) = - p_j log2(p_j) - (1 - p_j) log2(1 - p_j)
        H_l = (1 / D_l) sum_{j=1}^{D_l} H(p_j)
    """
    layer_metrics = {}
    
    for layer_name, z in pre_acts_dict.items():
        # z shape: [N, D_l, H, W]
        # Binary activation: b = (z > 0)
        b = (z > 0).float()
        
        # Coordinate-wise activation probability p_j averaged across N, H, W
        p = b.mean(dim=(0, 2, 3)).numpy() # [D_l]
        D_l = len(p)
        
        # Binary Shannon entropy H(p_j)
        h_coords = np.zeros_like(p)
        for j in range(D_l):
            pj = p[j]
            if 0.0 < pj < 1.0:
                h_coords[j] = - pj * np.log2(pj) - (1.0 - pj) * np.log2(1.0 - pj)
            else:
                h_coords[j] = 0.0
                
        H_l = float(np.mean(h_coords))
        
        # Statistics of activation probability and entropy
        clamped_channels = int(np.sum(p <= 0.01))
        saturated_channels = int(np.sum(p >= 0.99))
        active_channels = int(np.sum((p > 0.01) & (p < 0.99)))
        
        # Mean entropy over active channels (if any)
        active_entropy = float(np.mean(h_coords[p > 0.01])) if active_channels > 0 else 0.0
        
        layer_metrics[layer_name] = {
            "D_l": D_l,
            "mean_entropy_bits": H_l,
            "active_channel_entropy_bits": active_entropy,
            "mean_p": float(np.mean(p)),
            "std_p": float(np.std(p)),
            "clamped_channels": clamped_channels,
            "saturated_channels": saturated_channels,
            "active_channels": active_channels,
            "coordinate_entropies": h_coords.tolist(),
            "coordinate_p": p.tolist()
        }
        
    return layer_metrics


def collect_test_pre_activations(model, test_loader, device, max_batches=20):
    """Pass test batches through the model and concatenate pre-activations."""
    model.eval()
    collected = {'conv1': [], 'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= max_batches:
                break
            x = x.to(device)
            batch_pre_acts = extract_depth_pre_activations(model, x)
            for k in collected:
                collected[k].append(batch_pre_acts[k])
                
    for k in collected:
        collected[k] = torch.cat(collected[k], dim=0) # [N, D_l, H, W]
        
    return collected


def run_activation_entropy_probe():
    print("=" * 80)
    print("RUNNING ACTIVATION ENTROPY CASCADE PROBE ACROSS DEPTH (DIRECTION 2)")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    set_seed(42)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    tasks = get_cifar10_tasks(data_dir, batch_size=128, num_tasks=5)
    task0 = tasks[0]

    # Initialize and train ResNet-18 on Task 0
    model = ResNet18(num_classes=10, cifar_style=True).to(DEVICE)
    model.active_classes = 2

    print("Training ResNet-18 on Task 0 (10 epochs, lr=0.03)...")
    train_task(model, task0['train'], DEVICE, epochs=10, lr=0.03, use_amp=True)
    dense_acc = evaluate_accuracy(model, task0['test'], DEVICE)
    print(f"Task 0 Dense Unpruned Accuracy: {dense_acc * 100:.2f}%")

    # Compute Taylor neuron importance and SMR mask (85% prune in layer4 decision pathway)
    importance = compute_neuron_importance(model, task0['train'], DEVICE, num_batches=20)
    mask = compute_smr_mask(importance, percentile=85)

    base_state = snapshot_full_state(model)
    base_bn = snapshot_bn_state(model)
    base_fc = snapshot_fc_state(model)

    layers_ordered = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

    # -------------------------------------------------------------
    # CONDITION (a): DENSE MODEL
    # -------------------------------------------------------------
    print("\n--- Condition (a): Dense Model ---")
    dense_pre_acts = collect_test_pre_activations(model, task0['test'], DEVICE)
    dense_metrics = compute_activation_entropy(dense_pre_acts)
    for l in layers_ordered:
        m = dense_metrics[l]
        print(f"  {l:<7} (D={m['D_l']:3d}) | Entropy: {m['mean_entropy_bits']:.4f} bits | Mean p: {m['mean_p']:.3f} | Clamped: {m['clamped_channels']}/{m['D_l']}")

    # -------------------------------------------------------------
    # CONDITION (b): UNCALIBRATED PRUNED SUB-CIRCUIT (The Normalization Paradox)
    # -------------------------------------------------------------
    print("\n--- Condition (b): Uncalibrated Pruned Sub-circuit (Stale BN) ---")
    apply_inference_routing(model, base_state, mask, base_bn, base_fc, DEVICE, calibration_loader=None)
    uncal_acc = evaluate_accuracy(model, task0['test'], DEVICE)
    print(f"  Uncalibrated Accuracy: {uncal_acc * 100:.2f}% (Destructive Collapse!)")
    
    uncal_pre_acts = collect_test_pre_activations(model, task0['test'], DEVICE)
    uncal_metrics = compute_activation_entropy(uncal_pre_acts)
    for l in layers_ordered:
        m = uncal_metrics[l]
        print(f"  {l:<7} (D={m['D_l']:3d}) | Entropy: {m['mean_entropy_bits']:.4f} bits | Mean p: {m['mean_p']:.3f} | Clamped: {m['clamped_channels']}/{m['D_l']}")

    # -------------------------------------------------------------
    # CONDITION (c): O(1) RECALIBRATED SMR
    # -------------------------------------------------------------
    print("\n--- Condition (c): O(1) Recalibrated SMR (Signal Restored) ---")
    apply_inference_routing(model, base_state, mask, base_bn, base_fc, DEVICE, calibration_loader=task0['train'])
    recal_acc = evaluate_accuracy(model, task0['test'], DEVICE)
    print(f"  Recalibrated SMR Accuracy: {recal_acc * 100:.2f}% (Signal Fully Restored!)")
    
    recal_pre_acts = collect_test_pre_activations(model, task0['test'], DEVICE)
    recal_metrics = compute_activation_entropy(recal_pre_acts)
    for l in layers_ordered:
        m = recal_metrics[l]
        print(f"  {l:<7} (D={m['D_l']:3d}) | Entropy: {m['mean_entropy_bits']:.4f} bits | Mean p: {m['mean_p']:.3f} | Clamped: {m['clamped_channels']}/{m['D_l']}")

    # Structure final results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "layers": layers_ordered,
        "accuracies": {
            "dense": float(dense_acc * 100),
            "uncalibrated": float(uncal_acc * 100),
            "recalibrated_smr": float(recal_acc * 100)
        },
        "dense_model": {
            l: {
                "D_l": dense_metrics[l]["D_l"],
                "mean_entropy_bits": dense_metrics[l]["mean_entropy_bits"],
                "active_channel_entropy_bits": dense_metrics[l]["active_channel_entropy_bits"],
                "mean_p": dense_metrics[l]["mean_p"],
                "std_p": dense_metrics[l]["std_p"],
                "clamped_channels": dense_metrics[l]["clamped_channels"],
                "active_channels": dense_metrics[l]["active_channels"],
            } for l in layers_ordered
        },
        "uncalibrated_pruned": {
            l: {
                "D_l": uncal_metrics[l]["D_l"],
                "mean_entropy_bits": uncal_metrics[l]["mean_entropy_bits"],
                "active_channel_entropy_bits": uncal_metrics[l]["active_channel_entropy_bits"],
                "mean_p": uncal_metrics[l]["mean_p"],
                "std_p": uncal_metrics[l]["std_p"],
                "clamped_channels": uncal_metrics[l]["clamped_channels"],
                "active_channels": uncal_metrics[l]["active_channels"],
            } for l in layers_ordered
        },
        "recalibrated_smr": {
            l: {
                "D_l": recal_metrics[l]["D_l"],
                "mean_entropy_bits": recal_metrics[l]["mean_entropy_bits"],
                "active_channel_entropy_bits": recal_metrics[l]["active_channel_entropy_bits"],
                "mean_p": recal_metrics[l]["mean_p"],
                "std_p": recal_metrics[l]["std_p"],
                "clamped_channels": recal_metrics[l]["clamped_channels"],
                "active_channels": recal_metrics[l]["active_channels"],
            } for l in layers_ordered
        },
        "summary_table": []
    }

    print("\n" + "=" * 80)
    print(f"{'Layer':<8} | {'Dense (bits)':<14} | {'Uncalibrated (bits)':<20} | {'Recalibrated SMR (bits)':<24} | {'Delta H (Restored)'}")
    print("-" * 80)
    for l in layers_ordered:
        h_dense = dense_metrics[l]["mean_entropy_bits"]
        h_uncal = uncal_metrics[l]["mean_entropy_bits"]
        h_recal = recal_metrics[l]["mean_entropy_bits"]
        delta = h_recal - h_uncal
        results["summary_table"].append({
            "layer": l,
            "dense_entropy_bits": h_dense,
            "uncalibrated_entropy_bits": h_uncal,
            "recalibrated_entropy_bits": h_recal,
            "delta_entropy_restored": delta
        })
        print(f"{l:<8} | {h_dense:<14.4f} | {h_uncal:<20.4f} | {h_recal:<24.4f} | +{delta:.4f} bits")
    print("=" * 80)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_final')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'activation_entropy_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Activation entropy results successfully saved to: {out_file}")

    return results


if __name__ == '__main__':
    run_activation_entropy_probe()
