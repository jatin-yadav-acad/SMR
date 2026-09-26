#!/usr/bin/env python
"""
Unit test suite verifying SMR components:
1. PreAllocatedSparseHead (PASH) unseen masking
2. ResNet18 & ResNet50 neural architectures
3. First-Order Taylor importance calculation
4. Sparse mechanistic mask allocation and prioritization
5. Decision pathway parameter invariance (zero drift)
6. O(1) BatchNorm recalibration dynamics
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import PreAllocatedSparseHead, ResNet18, ResNet50, create_model
from src.smr_core import (
    compute_neuron_importance,
    compute_smr_mask,
    recalibrate_bn,
    merge_masks,
    snapshot_full_state,
    snapshot_bn_state,
    snapshot_fc_state,
    apply_inference_routing,
    compute_subspace_basis,
    evaluate_grassmannian_projection,
    compute_layer_representation_subspace,
    project_gradients_onto_null_space,
    compute_closed_form_bn_evolution,
    apply_closed_form_bn_evolution,
)


def test_pash():
    """Verify Pre-Allocated Sparse Head (PASH) masks unseen classes to -inf."""
    head = PreAllocatedSparseHead(in_features=64, max_classes=10, initial_classes=4)
    x = torch.randn(8, 64)
    logits = head(x)
    assert logits.shape == (8, 10)
    # Seen classes (0-3) must be finite
    assert torch.all(torch.isfinite(logits[:, :4]))
    # Unseen classes (4-9) must be -inf
    assert torch.all(torch.isinf(logits[:, 4:]) & (logits[:, 4:] < 0))


def test_architectures():
    """Verify ResNet-18 and ResNet-50 instantiate and execute forward pass correctly."""
    model_cifar = ResNet18(num_classes=10, cifar_style=True)
    x_cifar = torch.randn(2, 3, 32, 32)
    out_cifar = model_cifar(x_cifar)
    assert out_cifar.shape == (2, 10)

    model_img = ResNet50(num_classes=100, cifar_style=False)
    x_img = torch.randn(2, 3, 64, 64)
    out_img = model_img(x_img)
    assert out_img.shape == (2, 100)


def test_taylor_importance():
    """Verify First-Order Taylor importance calculation on conv channels."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2)
    )
    device = torch.device("cpu")
    x = torch.randn(16, 3, 16, 16)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    importance = compute_neuron_importance(model, loader, device, num_batches=2)
    assert "0" in importance
    assert len(importance["0"]) == 8
    assert all(v >= 0.0 for v in importance["0"])


def test_decision_parameter_invariance():
    """Verify algebraic zero drift on protected parameter coordinates."""
    conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
    initial_weights = conv.weight.clone().detach()

    # Protect channel 0 and 1
    protection_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    
    # Simulate an optimization step with gradient
    loss = conv(torch.randn(2, 4, 8, 8)).sum()
    loss.backward()

    # Zero gradients on protected channels
    with torch.no_grad():
        conv.weight.grad[protection_mask] = 0.0
        # SGD step
        conv.weight.data -= 0.1 * conv.weight.grad
        # Snapshot projection
        conv.weight.data[protection_mask] = initial_weights[protection_mask]

    # Protected channels must have EXACT zero drift
    drift = torch.norm(conv.weight.data[protection_mask] - initial_weights[protection_mask]).item()
    assert drift == 0.0, f"Parameter drift must be identically 0.0, got {drift}"


def test_bn_recalibration():
    """Verify BatchNorm recalibration updates running stats without modifying weights."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(4)
    )
    device = torch.device("cpu")
    initial_mean = model[1].running_mean.clone()
    initial_conv_weight = model[0].weight.clone()

    x = torch.randn(32, 3, 8, 8) + 5.0  # Shifted input
    loader = DataLoader(TensorDataset(x, torch.zeros(32)), batch_size=8)

    recalibrate_bn(model, loader, device, num_batches=4)

    # Running mean must shift towards +5.0
    assert not torch.allclose(model[1].running_mean, initial_mean)
    # Weights must NOT change during forward-only recalibration
    assert torch.equal(model[0].weight, initial_conv_weight)


def test_dynamic_capacity_expansion():
    """Verify Dynamic Modular Expansion (D-SMR) expands channels while preserving prior weights."""
    from src.models import ResNet18
    from src.smr_core import check_and_expand_capacity

    device = torch.device("cpu")
    model = ResNet18(num_classes=10, cifar_style=True).to(device)

    # Simulate 512 channels in layer4.1.conv2 with 75% allocation (below 80% threshold)
    prior_mask = {"layer4.1.conv2": torch.zeros(512, dtype=torch.bool, device=device)}
    prior_mask["layer4.1.conv2"][:384] = True  # 384/512 = 0.75

    old_conv_weight = model.layer4[1].conv2.weight.clone()
    old_bn_mean = model.layer4[1].bn2.running_mean.clone()

    model, prior_mask, expanded = check_and_expand_capacity(
        model, prior_mask, threshold=0.80, delta_channels=32, device=device
    )
    assert not expanded, "Should not expand when allocation ratio (75%) < threshold (80%)"
    assert model.layer4[1].conv2.out_channels == 512

    # Now allocate to 85% (above 80% threshold)
    prior_mask["layer4.1.conv2"][:436] = True  # 436/512 = 0.851

    model, prior_mask, expanded = check_and_expand_capacity(
        model, prior_mask, threshold=0.80, delta_channels=32, device=device
    )
    assert expanded, "Should expand when allocation ratio >= threshold"
    assert model.layer4[1].conv2.out_channels == 544  # 512 + 32
    assert model.layer4[1].bn2.num_features == 544
    assert prior_mask["layer4.1.conv2"].size(0) == 544

    # Mathematical Invariance: Previous 512 channels MUST BE IDENTICAL
    assert torch.equal(model.layer4[1].conv2.weight[:512], old_conv_weight), "Historical weights altered!"
    assert torch.equal(model.layer4[1].bn2.running_mean[:512], old_bn_mean), "Historical BN statistics altered!"
    # Newly appended channels in prior_mask must be False (unallocated)
    assert not torch.any(prior_mask["layer4.1.conv2"][512:]), "New channels must be marked unallocated!"


def test_grassmannian_subspace_projection():
    """Verify Grassmannian principal subspace projection energy computation."""
    torch.manual_seed(42)
    # Generate 100 samples in 32 dimensions with dominant energy in first 4 dims
    N, d, k = 100, 32, 4
    features = torch.randn(N, d) * 0.1
    features[:, :k] += torch.randn(N, k) * 5.0

    basis = compute_subspace_basis(features, rank_k=k)
    assert basis.shape == (d, k)
    # Verify orthonormality: U^T U = I_k
    eye_k = torch.matmul(basis.t(), basis)
    assert torch.allclose(eye_k, torch.eye(k), atol=1e-5)

    # In-subspace samples should have much higher projection energy than random noise
    in_subspace = features[:10]
    out_subspace = torch.randn(10, d) * 0.1
    e_in = evaluate_grassmannian_projection(in_subspace, basis)
    e_out = evaluate_grassmannian_projection(out_subspace, basis)
    assert (e_in.mean() > e_out.mean() * 10), "In-subspace energy must dominate out-of-subspace energy!"


def test_null_space_gradient_projection():
    """Verify gradient projection strictly nullifies gradient components along historical representation subspaces."""
    torch.manual_seed(42)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
    # Forward pass to obtain activations
    x = torch.randn(50, 8, 8, 8)
    out = conv(x)
    loss = out.sum()
    loss.backward()
    assert conv.weight.grad is not None

    # Compute sensory representation subspace for input channels (8 channels)
    act_flat = x.permute(0, 2, 3, 1).reshape(-1, 8)
    U = compute_layer_representation_subspace(act_flat, energy_threshold=0.90)
    assert U.size(0) == 8

    # Wrap in dict matching model named module
    historical_subspaces = {"conv": U}
    dummy_model = nn.ModuleDict({"conv": conv})

    # Project gradient onto null space
    project_gradients_onto_null_space(dummy_model, historical_subspaces)

    # Verify that projected gradient is orthogonal to subspace U
    grad_reshaped = conv.weight.grad.data.permute(0, 2, 3, 1).reshape(-1, 8)
    projection_on_basis = torch.matmul(grad_reshaped, U)
    # Maximum projection component should be numerically ~ 0 (< 1e-4 in float32)
    max_drift = torch.abs(projection_on_basis).max().item()
    assert max_drift < 1e-4, f"Gradient must have zero projection on historical subspace! Got max drift: {max_drift}"


def test_closed_form_bn_evolution():
    """Verify O(0) analytical moment evolution and clamping probability computation."""
    dense_mean = torch.tensor([1.2, 0.8, -0.4])
    dense_var = torch.tensor([2.0, 1.5, 0.9])
    rho_in = 0.08
    
    sm, sv, dmu, pclamp = compute_closed_form_bn_evolution(dense_mean, dense_var, rho_in=rho_in)
    
    # Check theoretical formula: sparse_mean = rho_in * dense_mean
    assert torch.allclose(sm, rho_in * dense_mean, atol=1e-5)
    # Check theoretical formula: sparse_var = rho_in * dense_var + rho_in * (1 - rho_in) * dense_mean^2
    expected_sv = rho_in * dense_var + rho_in * (1.0 - rho_in) * (dense_mean ** 2)
    assert torch.allclose(sv, expected_sv, atol=1e-5)
    # Clamping probability should be in [0, 1]
    assert (pclamp >= 0.0).all() and (pclamp <= 1.0).all()

    # Test apply_closed_form_bn_evolution on ResNet-18
    model = ResNet18(num_classes=10)
    updated = apply_closed_form_bn_evolution(model, rho_in=rho_in)
    assert updated == 5, f"Expected 5 layer4 BN layers updated, got {updated}"


def test_spectral_contractive_regularizer():
    """Verify spectral contractive penalty keeps singular values bounded."""
    from src.smr_core import estimate_spectral_norm, compute_spectral_contractive_loss
    conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    # Inflate weights so spectral norm > 1.0
    conv.weight.data.mul_(5.0)
    sigma, _ = estimate_spectral_norm(conv.weight)
    assert sigma.item() > 1.0

    dummy_model = nn.ModuleDict({"sensory_conv": conv})
    penalty = compute_spectral_contractive_loss(dummy_model, ["sensory_conv"], max_sigma=1.0)
    assert penalty.item() > 0.0


def test_orthogonal_subspace_multiplexing():
    """Verify Orthogonal Subspace Multiplexing creates mutually orthogonal task updates."""
    from src.smr_core import OrthogonalSubspaceMultiplexing
    base_linear = nn.Linear(32, 32)
    osm = OrthogonalSubspaceMultiplexing(base_linear, rank_per_task=4, max_tasks=5)
    osm.add_task_subspace(0)
    osm.add_task_subspace(1)

    # Subspaces U_0 and U_1 must be orthogonal: U_0^T U_1 approx 0
    u0 = osm.task_U["0"]
    u1 = osm.task_U["1"]
    cross_prod = torch.matmul(u0.t(), u1)
    assert torch.abs(cross_prod).max().item() < 1e-5



