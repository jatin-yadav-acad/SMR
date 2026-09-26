#!/usr/bin/env python
"""
Extract authentic empirical activation distributions from ResNet-18 (layer4.1.bn2)
across:
1. Dense baseline
2. Uncalibrated sparse sub-circuit (The Normalization Paradox)
3. SMR Recalibrated sparse sub-circuit (Signal Restored)
"""
import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from experiments.run_final_experiments import ResNet18, get_cifar10_tasks, set_seed
from src.smr_core import (
    train_task, compute_neuron_importance, compute_smr_mask,
    apply_inference_routing, snapshot_full_state,
    snapshot_bn_state, snapshot_fc_state, evaluate_accuracy
)


def capture_activations(model_instance, loader, device):
    activations = []
    def hook_fn(module, input, output):
        # output of bn2 before ReLU
        activations.append(output.detach().cpu().numpy())
    
    # Hook bn2 in layer4.1
    hook = model_instance.layer4[1].bn2.register_forward_hook(hook_fn)
    model_instance.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model_instance(x)
            if len(activations) >= 5:  # 5 batches = 640 samples
                break
    hook.remove()
    all_acts = np.concatenate(activations, axis=0).flatten()
    return all_acts


def extract_activations(data_dir='./data', output_path='results_final/empirical_activations.npz'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Empirical Activation Extraction] Using device: {device}')

    set_seed(42)
    tasks = get_cifar10_tasks(data_dir, batch_size=128, num_tasks=5)
    task0 = tasks[0]

    model = ResNet18(num_classes=10, cifar_style=True).to(device)
    model.active_classes = task0['cum']

    print('Training ResNet-18 on Task 0 (10 epochs)...')
    train_task(model, task0['train'], device, epochs=10, lr=0.03, use_amp=True)

    test_acc_dense = evaluate_accuracy(model, task0['test'], device)
    print(f'Task 0 Dense Accuracy: {test_acc_dense*100:.2f}%')

    # Compute Taylor importance and SMR mask (85% prune = 15% active in layer4)
    importance = compute_neuron_importance(model, task0['train'], device, num_batches=20)
    mask = compute_smr_mask(importance, percentile=85)

    # Save snapshot
    base_state = snapshot_full_state(model)
    base_bn = snapshot_bn_state(model)
    base_fc = snapshot_fc_state(model)

    # Condition 1: Dense unpruned model
    acts_dense = capture_activations(model, task0['test'], device)

    # Condition 2: Uncalibrated sparse model (The Paradox)
    apply_inference_routing(model, base_state, mask, base_bn, base_fc, device, calibration_loader=None)
    test_acc_uncal = evaluate_accuracy(model, task0['test'], device)
    print(f'Task 0 Uncalibrated Sparse Accuracy (The Paradox): {test_acc_uncal*100:.2f}%')
    acts_stale = capture_activations(model, task0['test'], device)

    # Condition 3: SMR Recalibrated sparse model
    apply_inference_routing(model, base_state, mask, base_bn, base_fc, device, calibration_loader=task0['train'])
    test_acc_recal = evaluate_accuracy(model, task0['test'], device)
    print(f'Task 0 Recalibrated Sparse Accuracy: {test_acc_recal*100:.2f}%')
    acts_recal = capture_activations(model, task0['test'], device)

    # Verification of percentages below 0
    neg_dense = float(np.mean(acts_dense < 0) * 100)
    neg_stale = float(np.mean(acts_stale < 0) * 100)
    neg_recal = float(np.mean(acts_recal < 0) * 100)

    print(f'Negative Signals Clamped by ReLU:')
    print(f'  Dense:        {neg_dense:.1f}%')
    print(f'  Stale Sparse: {neg_stale:.1f}% (Severe Clamping!)')
    print(f'  Recalibrated: {neg_recal:.1f}% (Restored Activation Range)')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        dense=acts_dense,
        stale=acts_stale,
        recal=acts_recal,
        stats={
            'neg_dense': neg_dense,
            'neg_stale': neg_stale,
            'neg_recal': neg_recal,
            'acc_dense': float(test_acc_dense),
            'acc_stale': float(test_acc_uncal),
            'acc_recal': float(test_acc_recal),
        }
    )
    print(f'Saved empirical activations to {output_path}')


if __name__ == '__main__':
    extract_activations()
