"""
Unit Tests for Split-SVHN Continual Learning Pipeline
=====================================================
Verifies dataset loading, partitioning into 5 tasks, forward pass,
and SMR circuit masking invariance on SVHN digits.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_svhn_experiment import get_svhn_tasks, ResNet18
from src.smr_core import (
    compute_neuron_importance,
    compute_smr_mask,
    merge_masks
)

def test_svhn_task_partitioning():
    """Verify that SVHN is cleanly partitioned into 5 tasks with 2 classes each."""
    data_root = "./data"
    tasks = get_svhn_tasks(data_root, batch_size=32, num_tasks=5)
    
    assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"
    
    for t, task in enumerate(tasks):
        assert task["id"] == t
        assert len(task["classes"]) == 2
        assert task["classes"] == [2 * t, 2 * t + 1]
        assert task["cum"] == 2 * (t + 1)
        
        # Test batch shape
        x, y = next(iter(task["train"]))
        assert x.dim() == 4
        assert x.shape[1] == 3
        assert x.shape[2] == 32 and x.shape[3] == 32
        assert all(label.item() in task["classes"] for label in y)

def test_svhn_model_forward():
    """Verify that ResNet18 processes SVHN tensors and applies class masking."""
    model = ResNet18(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    
    # Active classes = 2
    model.active_classes = 2
    out = model(x)
    assert out.shape == (4, 10)
    assert not torch.isneginf(out[:, :2]).any()
    assert torch.isneginf(out[:, 2:]).all()

def test_svhn_smr_isolation():
    """Verify Taylor importance and circuit mask generation on SVHN data."""
    tasks = get_svhn_tasks("./data", batch_size=16, num_tasks=5)
    task0 = tasks[0]
    
    model = ResNet18(num_classes=10)
    model.active_classes = 2
    device = torch.device("cpu")
    
    importance = compute_neuron_importance(model, task0["train"], device, num_batches=2)
    assert len(importance) > 0
    
    mask = compute_smr_mask(importance, percentile=80)
    assert len(mask) > 0
    assert all(k in importance for k in mask)
    
    # Verify non-empty active channels
    total_active = sum(m.sum().item() for m in mask.values())
    assert total_active > 0
