"""
src: Sparse Mechanistic Routing (SMR) Continual Learning Package
"""

from src.models import (
    PreAllocatedSparseHead,
    ResNet18,
    ResNet50,
    create_model,
)

from src.smr_core import (
    set_seed,
    evaluate_accuracy,
    compute_neuron_importance,
    compute_smr_mask,
    merge_masks,
    snapshot_full_state,
    snapshot_bn_state,
    snapshot_fc_state,
    apply_inference_routing,
    train_task,
    EWCRegularizer,
    ReplayBuffer,
    evaluate_class_il_smr,
    evaluate_task_il_smr,
    recalibrate_bn,
    PackNetManager,
    fine_tune_decision_subcircuit,
    check_and_expand_capacity,
)

__all__ = [
    "PreAllocatedSparseHead",
    "ResNet18",
    "ResNet50",
    "create_model",
    "set_seed",
    "evaluate_accuracy",
    "compute_neuron_importance",
    "compute_smr_mask",
    "merge_masks",
    "snapshot_full_state",
    "snapshot_bn_state",
    "snapshot_fc_state",
    "apply_inference_routing",
    "train_task",
    "EWCRegularizer",
    "ReplayBuffer",
    "evaluate_class_il_smr",
    "evaluate_task_il_smr",
    "recalibrate_bn",
    "PackNetManager",
    "fine_tune_decision_subcircuit",
    "check_and_expand_capacity",
]

