"""
Unified Continual Learning Experiment Runner (ICLR Benchmark)
============================================================
Runs SMR + baselines across continual learning datasets with multiple seeds.
Fixes the Task 0 regression by correctly separating training from circuit isolation.

Usage: python run_final_experiments.py [--dataset cifar10] [--seeds 42 1337 2025]
"""
import os, json, time, argparse, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.smr_core import (
    set_seed, train_task, compute_neuron_importance, compute_smr_mask,
    recalibrate_bn, apply_inference_routing, snapshot_full_state,
    evaluate_accuracy, evaluate_class_il_smr, evaluate_task_il_smr,
    EWCRegularizer, ReplayBuffer, snapshot_bn_state, snapshot_fc_state, merge_masks,
    fine_tune_decision_subcircuit, evaluate_task_accuracy, PackNetManager,
    check_and_expand_capacity
)

def set_active_classes(model, active_classes):
    model.active_classes = active_classes
    if hasattr(model, 'head') and hasattr(model.head, 'current_active_classes'):
        model.head.current_active_classes = active_classes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Model =====
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride=stride, bias=False), nn.BatchNorm2d(out))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, cifar_style=True):
        super().__init__()
        self.cifar_style = cifar_style
        self.inp = 64
        if cifar_style:
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._layer(64, 2, 1)
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._layer(64, 2, 1)
            
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)
        self.active_classes = 0
        self.num_classes = num_classes

    def _layer(self, out, n, stride):
        layers = [BasicBlock(self.inp, out, stride)]
        self.inp = out
        for _ in range(n - 1):
            layers.append(BasicBlock(self.inp, out))
        return nn.Sequential(*layers)

    def features(self, x):
        if self.cifar_style:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)
            out = self.layer1(out)
        out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)

    def forward(self, x):
        logits = self.fc(self.features(x))
        if self.active_classes < self.num_classes:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, :self.active_classes] = 0.0
            logits = logits + mask
        return logits


# ===== Dataset Loaders =====
def get_cifar10_tasks(data_root, batch_size=64, num_tasks=5, img_size=32):
    tr_ops = []
    if img_size != 32: tr_ops.append(transforms.Resize((img_size, img_size)))
    tr_ops.extend([transforms.RandomCrop(img_size, padding=4 if img_size==32 else 16), transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    tr = transforms.Compose(tr_ops)
    
    te_ops = []
    if img_size != 32: te_ops.append(transforms.Resize((img_size, img_size)))
    te_ops.extend([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    te = transforms.Compose(te_ops)
    
    root = os.path.join(data_root, "cifar10")
    try:
        train_ds = datasets.CIFAR10(root, True, download=False, transform=tr)
        test_ds = datasets.CIFAR10(root, False, download=False, transform=te)
    except Exception:
        train_ds = datasets.CIFAR10(root, True, download=True, transform=tr)
        test_ds = datasets.CIFAR10(root, False, download=True, transform=te)
    return _split_dataset(train_ds, test_ds, num_tasks, 10, batch_size)

def get_cifar100_tasks(data_root, batch_size=64, num_tasks=10, img_size=32):
    tr_ops = []
    if img_size != 32: tr_ops.append(transforms.Resize((img_size, img_size)))
    tr_ops.extend([transforms.RandomCrop(img_size, padding=4 if img_size==32 else 16), transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    tr = transforms.Compose(tr_ops)

    te_ops = []
    if img_size != 32: te_ops.append(transforms.Resize((img_size, img_size)))
    te_ops.extend([transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    te = transforms.Compose(te_ops)

    root = os.path.join(data_root, "cifar100")
    try:
        train_ds = datasets.CIFAR100(root, True, download=False, transform=tr)
        test_ds = datasets.CIFAR100(root, False, download=False, transform=te)
    except Exception:
        train_ds = datasets.CIFAR100(root, True, download=True, transform=tr)
        test_ds = datasets.CIFAR100(root, False, download=True, transform=te)
    return _split_dataset(train_ds, test_ds, num_tasks, 100, batch_size)

def get_svhn_tasks(data_root, batch_size=64, num_tasks=5, img_size=32):
    tr_ops = []
    if img_size != 32: tr_ops.append(transforms.Resize((img_size, img_size)))
    tr_ops.extend([
        transforms.RandomCrop(img_size, padding=4 if img_size==32 else 16),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    tr = transforms.Compose(tr_ops)

    te_ops = []
    if img_size != 32: te_ops.append(transforms.Resize((img_size, img_size)))
    te_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    te = transforms.Compose(te_ops)

    root = os.path.join(data_root, "SVHN")
    train_ds = datasets.SVHN(root, split="train", download=False, transform=tr)
    test_ds = datasets.SVHN(root, split="test", download=False, transform=te)
    train_ds.targets = [int(l) for l in train_ds.labels]
    test_ds.targets = [int(l) for l in test_ds.labels]
    return _split_dataset(train_ds, test_ds, num_tasks, 10, batch_size)

def get_tinyimagenet_tasks(data_root, batch_size=64, num_tasks=10, img_size=32):
    """TinyImageNet-200: 200 classes, 64x64 images."""
    tr = transforms.Compose([transforms.Resize(img_size), transforms.RandomCrop(img_size, padding=4 if img_size==32 else 16),
         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821))])
    te = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
         transforms.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821))])
    root = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.exists(os.path.join(root, "train")):
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tr)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=te)
        return _split_dataset_imagefolder(train_ds, test_ds, num_tasks, 200, batch_size)
    else:
        print(f"[WARN] TinyImageNet not found at {root}. Skipping.")
        return None

def get_imagenet100_tasks(data_root, batch_size=64, num_tasks=10, img_size=224):
    """ImageNet-100: 100 classes subset of ImageNet (224x224)."""
    tr = transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    te = transforms.Compose([transforms.Resize(int(img_size*256/224)), transforms.CenterCrop(img_size),
         transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    root = os.path.join(data_root, "imagenet100")
    if os.path.exists(os.path.join(root, "train")):
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tr)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=te)
        return _split_dataset_imagefolder(train_ds, test_ds, num_tasks, 100, batch_size)
    else:
        print(f"[WARN] ImageNet-100 not found at {root}. Skipping.")
        return None

def _split_dataset(train_ds, test_ds, num_tasks, total_classes, batch_size):
    tasks = []
    classes_per_task = total_classes // num_tasks
    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        cum = (t + 1) * classes_per_task
        tri = [i for i, l in enumerate(train_ds.targets) if l in cls]
        tei = [i for i, l in enumerate(test_ds.targets) if l in cls]
        tasks.append({"id": t, "classes": cls, "cum": cum,
            "train": DataLoader(Subset(train_ds, tri), batch_size, shuffle=True,
                                num_workers=0, pin_memory=True),
            "test": DataLoader(Subset(test_ds, tei), batch_size,
                               num_workers=0, pin_memory=True)})
    return tasks

def _split_dataset_imagefolder(train_ds, test_ds, num_tasks, total_classes, batch_size):
    tasks = []
    classes_per_task = total_classes // num_tasks
    train_targets = [s[1] for s in train_ds.samples]
    test_targets = [s[1] for s in test_ds.samples]
    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        cum = (t + 1) * classes_per_task
        cls_set = set(cls)
        tri = [i for i, l in enumerate(train_targets) if l in cls_set]
        tei = [i for i, l in enumerate(test_targets) if l in cls_set]
        tasks.append({"id": t, "classes": cls, "cum": cum,
            "train": DataLoader(Subset(train_ds, tri), batch_size, shuffle=True,
                                num_workers=0, pin_memory=True),
            "test": DataLoader(Subset(test_ds, tei), batch_size,
                               num_workers=0, pin_memory=True)})
    return tasks

# ===== Method Runners =====

def run_finetune(model, tasks, device, epochs=10, lr=0.01, eval_mode="task_il"):
    """Naive sequential finetuning (lower bound)."""
    task_accs = {}
    for t in range(len(tasks)):
        task = tasks[t]
        set_active_classes(model, task["cum"])
        print(f"\n  [Finetune] Task {t}: Classes {task['classes'][:3]}...")
        train_task(model, task["train"], device, epochs=epochs, lr=lr)
        # Evaluate all seen tasks
        accs = []
        for prev in range(t + 1):
            set_active_classes(model, tasks[prev]["cum"])
            if eval_mode in ("task_il", "both"):
                acc = evaluate_task_accuracy(model, tasks[prev]["test"], device, tasks[prev]["classes"])
            else:
                acc = evaluate_accuracy(model, tasks[prev]["test"], device)
            accs.append(acc)
        task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
    return task_accs

def run_ewc(model, tasks, device, epochs=10, lr=0.01, lambda_ewc=5000.0, use_amp=True, eval_mode="task_il"):
    """Elastic Weight Consolidation baseline."""
    ewc = EWCRegularizer(model, lambda_ewc=lambda_ewc)
    task_accs = {}
    if 'scaler' not in locals():
        try:
            scaler = GradScaler(device_type=device.type, enabled=use_amp and device.type == 'cuda')
        except TypeError:
            scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    for t in range(len(tasks)):
        task = tasks[t]
        set_active_classes(model, task["cum"])
        print(f"\n  [EWC] Task {t}: Classes {task['classes'][:3]}...")
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        for ep in range(epochs):
            total_loss = correct = total = 0
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
                    out = model(x)
                    loss = F.cross_entropy(out, y) + ewc.penalty(model)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                with torch.no_grad():
                    total_loss += loss.item() * x.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            if epochs > 1 and (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"    Epoch {ep+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")
        ewc.compute_fisher(model, task["train"], device)
        accs = []
        for prev in range(t + 1):
            set_active_classes(model, tasks[prev]["cum"])
            if eval_mode in ("task_il", "both"):
                acc = evaluate_task_accuracy(model, tasks[prev]["test"], device, tasks[prev]["classes"])
            else:
                acc = evaluate_accuracy(model, tasks[prev]["test"], device)
            accs.append(acc)
        task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
    return task_accs

def run_packnet(model, tasks, device, epochs=10, lr=0.01, prune_ratio=0.5, retrain_epochs=3, eval_mode="task_il"):
    """PackNet modular baseline (Mallya & Lazebnik, CVPR 2018)."""
    packnet = PackNetManager(model)
    task_accs = {}
    for t in range(len(tasks)):
        task = tasks[t]
        set_active_classes(model, task["cum"])
        print(f"\n  [PackNet] Task {t}: Classes {task['classes'][:3]}...")
        packnet.train_task(task["train"], task["classes"], t, device,
                           epochs=epochs, lr=lr, prune_ratio=prune_ratio, retrain_epochs=retrain_epochs)
        accs = []
        for prev in range(t + 1):
            packnet.apply_task_mask(prev, device)
            set_active_classes(model, tasks[prev]["cum"])
            acc = evaluate_task_accuracy(model, tasks[prev]["test"], device, tasks[prev]["classes"])
            accs.append(acc)
        task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
    return task_accs

def run_replay(model, tasks, device, epochs=10, lr=0.01, buffer_per_class=50, use_amp=True, eval_mode="task_il"):
    """Experience Replay baseline."""
    buffer = ReplayBuffer(capacity_per_class=buffer_per_class)
    task_accs = {}
    if 'scaler' not in locals():
        try:
            scaler = GradScaler(device_type=device.type, enabled=use_amp and device.type == 'cuda')
        except TypeError:
            scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    for t in range(len(tasks)):
        task = tasks[t]
        set_active_classes(model, task["cum"])
        print(f"\n  [Replay] Task {t}: Classes {task['classes'][:3]}...")
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        for ep in range(epochs):
            total_loss = correct = total = 0
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                    rx, ry = buffer.sample(x.size(0))
                    if rx is not None:
                        rx, ry = rx.to(device), ry.to(device)
                        rout = model(rx)
                        loss += F.cross_entropy(rout, ry)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                with torch.no_grad():
                    total_loss += loss.item() * x.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            if epochs > 1 and (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"    Epoch {ep+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")
        buffer.add_task(task["train"], device)
        accs = []
        for prev in range(t + 1):
            set_active_classes(model, tasks[prev]["cum"])
            if eval_mode in ("task_il", "both"):
                acc = evaluate_task_accuracy(model, tasks[prev]["test"], device, tasks[prev]["classes"])
            else:
                acc = evaluate_accuracy(model, tasks[prev]["test"], device)
            accs.append(acc)
        task_accs[t] = accs
        aa = np.mean(accs)
        print(f"  [Eval Task-IL] AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in accs]}")
    return task_accs

def run_smr(model, tasks, device, epochs=10, lr=0.01, isolation_percentile=85, static_bn=True, eval_mode="both"):
    """
    Sparse Mechanistic Routing — FIXED VERSION.
    
    Key difference from old code: We NEVER freeze circuits during their own task's 
    training. The flow is:
      1. Train Task T (protecting only prior tasks' circuits)
      2. AFTER training, map Task T's critical circuit
      3. Save Task T's full state (weights + BN + FC) for future routing
      4. Merge Task T's mask into the global protection mask
    """
    task_accs = {}
    # History for inference routing
    task_masks = {}      # task_id -> mask dict
    task_bn_states = {}  # task_id -> bn state dict
    task_fc_states = {}  # task_id -> fc state dict
    
    # Global protection: union of all prior task masks
    global_protection = {}
    # Saved weights for enforcement
    saved_protected_weights = {}
    
    for t in range(len(tasks)):
        task = tasks[t]
        set_active_classes(model, task["cum"])
        print(f"\n  [SMR] Task {t}: Classes {task['classes'][:3]}...")
        
        # === TRAIN with protection of prior tasks only ===
        train_task(model, task["train"], device, epochs=epochs, lr=lr,
                   protection_masks=global_protection,
                   saved_weights=saved_protected_weights, use_amp=True)
        
        # === MAP this task's critical circuit AFTER training ===
        print(f"    Mapping critical circuit for Task {t}...")
        # Use training data for Taylor importance
        importance = compute_neuron_importance(model, task["train"], device, num_batches=20)
        task_mask = compute_smr_mask(importance, percentile=isolation_percentile, prior_protection=global_protection)
        
        # === SPECIALIZE isolated decision sub-circuit via fine-tuning ===
        print(f"    Specializing isolated decision sub-circuit (4 epochs)...")
        fine_tune_decision_subcircuit(model, task["train"], task_mask, task["classes"], device, epochs=4, lr=0.01)
        
        # === ONE-TIME recalibrate BN parameters for the isolated sub-network before saving ===
        if static_bn:
            print(f"    Recalibrating BN parameters at training time for Static Cached BN...")
            current_state = snapshot_full_state(model)
            apply_inference_routing(model, current_state,
                                    task_mask, snapshot_bn_state(model),
                                    snapshot_fc_state(model), device,
                                    calibration_loader=task["train"])
            task_masks[t] = deepcopy(task_mask)
            task_bn_states[t] = snapshot_bn_state(model)
            task_fc_states[t] = snapshot_fc_state(model)
            model.load_state_dict(current_state) # Restore model
        else:
            task_masks[t] = deepcopy(task_mask)
            task_bn_states[t] = snapshot_bn_state(model)
            task_fc_states[t] = snapshot_fc_state(model)
            
        # === Update global protected weights with the fine-tuned, calibrated state ===
        saved_protected_weights = snapshot_full_state(model)
        
        # === MERGE into global protection for next task ===
        global_protection = merge_masks(global_protection, task_mask)
        
        total_protected = sum(m.sum().item() for m in global_protection.values())
        total_ch = sum(m.numel() for m in global_protection.values())
        print(f"    Global protection: {int(total_protected)}/{total_ch} "
              f"({100*total_protected/max(total_ch,1):.1f}%)")
        
        # === EVALUATE via inference routing ===
        model.eval()
        pre_acc = evaluate_accuracy(model, task["test"], device)
        print(f"    [SMR] Pre-routing Acc (Task {t}): {pre_acc:.4f}")
        
        # Task-IL Accuracies
        if eval_mode in ("task_il", "both"):
            task_accs[t] = evaluate_task_il_smr(model, tasks, task_masks, task_bn_states, task_fc_states, t, device, static_bn=static_bn)
            aa = np.mean(task_accs[t])
            print(f"  [Eval Task-IL] AA={aa:.4f} | Accs: {[f'{a:.3f}' for a in task_accs[t]]}")
            
        # Class-IL Accuracy
        if eval_mode in ("class_il", "both"):
            class_il_acc = evaluate_class_il_smr(model, tasks, task_masks, task_bn_states, task_fc_states, t, device, static_bn=static_bn)
            print(f"  [Eval Class-IL] Joint Accuracy: {class_il_acc:.4f}")
            
    return task_accs

def run_joint(model, tasks, device, epochs=15, lr=0.01):
    """Joint training upper bound: train on ALL tasks simultaneously."""
    from torch.utils.data import ConcatDataset
    all_train = ConcatDataset([t["train"].dataset for t in tasks])
    set_active_classes(model, tasks[-1]["cum"])
    import os
    num_workers = 0 if os.name == 'nt' else 4
    joint_loader = DataLoader(all_train, batch_size=tasks[0]["train"].batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    print(f"\n  [Joint] Training on all {len(tasks)} tasks simultaneously...")
    train_task(model, joint_loader, device, epochs=epochs, lr=lr, use_amp=True)
    accs = []
    for t in range(len(tasks)):
        set_active_classes(model, tasks[t]["cum"])
        acc = evaluate_accuracy(model, tasks[t]["test"], device)
        accs.append(acc)
    print(f"  [Joint Eval] Accs: {[f'{a:.3f}' for a in accs]}")
    return {len(tasks) - 1: accs}

# ===== Metrics =====
def compute_metrics(task_accs, num_tasks):
    """Compute Average Accuracy (AA) and Forgetting Measure (FM)."""
    last = num_tasks - 1
    # Handle methods like JOINT that might only provide the final evaluation
    if last not in task_accs:
        # Try to find the largest available key
        available_keys = sorted(task_accs.keys())
        if not available_keys:
            return {"AA": 0.0, "FM": 0.0}
        last = available_keys[-1]
        
    final_accs = task_accs[last]
    aa = np.mean(final_accs)
    
    # FM = mean over tasks of (max acc ever seen - final acc)
    fm_vals = []
    for j in range(len(final_accs)):
        # Calculate max accuracy seen for task j up to the current last task
        relevant_accs = [task_accs[t][j] for t in range(j, last + 1) if t in task_accs and j < len(task_accs[t])]
        if relevant_accs:
            max_acc = max(relevant_accs)
            fm_vals.append(max_acc - final_accs[j])
            
    fm = np.mean(fm_vals) if fm_vals else 0.0
    return {"AA": float(aa), "FM": float(fm)}

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100","svhn","tinyimagenet","imagenet100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025])
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "vit"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--isolation_percentile", type=int, default=80)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--results_dir", default="./results_final")
    parser.add_argument("--methods", nargs="+",
                        default=["finetune", "ewc", "packnet", "replay", "smr", "joint"])
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--static_bn", action="store_true", default=True)
    parser.add_argument("--eval_mode", default="both", choices=["task_il", "class_il", "both"])
    parser.add_argument("--force", action="store_true", help="Force re-running even if results exist.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | Dataset: {args.dataset} | Seeds: {args.seeds} | AMP: {args.use_amp}")

    # Dataset config
    ds_config = {
        "cifar10": {"loader": get_cifar10_tasks, "num_classes": 10, "num_tasks": 5},
        "cifar100": {"loader": get_cifar100_tasks, "num_classes": 100, "num_tasks": 10},
        "svhn": {"loader": get_svhn_tasks, "num_classes": 10, "num_tasks": 5},
        "tinyimagenet": {"loader": get_tinyimagenet_tasks, "num_classes": 200, "num_tasks": 10},
        "imagenet100": {"loader": get_imagenet100_tasks, "num_classes": 100, "num_tasks": 10},
    }
    cfg = ds_config[args.dataset]

    method_runners = {
        "finetune": lambda m, t, d: run_finetune(m, t, d, epochs=args.epochs, lr=args.lr),
        "ewc": lambda m, t, d: run_ewc(m, t, d, epochs=args.epochs, lr=args.lr, use_amp=args.use_amp),
        "packnet": lambda m, t, d: run_packnet(m, t, d, epochs=args.epochs, lr=args.lr),
        "replay": lambda m, t, d: run_replay(m, t, d, epochs=args.epochs, lr=args.lr, use_amp=args.use_amp),
        "smr": lambda m, t, d: run_smr(m, t, d, epochs=args.epochs, lr=args.lr, isolation_percentile=args.isolation_percentile, static_bn=args.static_bn, eval_mode=args.eval_mode),
        "joint": lambda m, t, d: run_joint(m, t, d, epochs=args.epochs + 5, lr=args.lr),
    }

    file_prefix = f"{args.arch}_{args.dataset}" if args.arch != "resnet18" else args.dataset
    out_path = os.path.join(args.results_dir, f"{file_prefix}_results.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                all_results = json.load(f)
            print(f"[INFO] Loaded existing results from {out_path}. Methods already completed: {list(all_results.keys())}")
        except Exception as e:
            print(f"[WARN] Failed to load existing results from {out_path}: {e}")
            all_results = {}
    else:
        all_results = {}

    for method in args.methods:
        if method in all_results and not args.force:
            print(f"\nMethod {method.upper()} already computed and present in {out_path}. Skipping (use --force to re-run).")
            continue

        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()} | DATASET: {args.dataset.upper()}")
        print(f"{'='*70}")

        seed_results = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)
            try:
                img_size = 224 if args.arch == "vit" else 32
                if args.dataset == "imagenet100":
                    img_size = 224
                tasks = cfg["loader"](args.data_root, batch_size=args.batch_size, num_tasks=cfg["num_tasks"], img_size=img_size)
            except Exception as e:
                print(f"\n[ERROR] Failed to load dataset {args.dataset}: {e}")
                print("If this is a network error, please check your internet connection or manually download the dataset.")
                break
            
            if tasks is None:
                print(f"Dataset {args.dataset} not available. Skipping.")
                break
            cifar_style = (args.dataset not in ["imagenet100", "tinyimagenet"])
            if args.arch == "resnet18":
                model = ResNet18(num_classes=cfg["num_classes"], cifar_style=cifar_style).to(device)
            else:
                from src.models import create_model
                model = create_model(args.arch, initial_classes=cfg["num_classes"] // cfg["num_tasks"], max_classes=cfg["num_classes"], cifar_style=cifar_style).to(device)
            task_accs = method_runners[method](model, tasks, device)
            metrics = compute_metrics(task_accs, cfg["num_tasks"])
            metrics["task_accs"] = {str(k): v for k, v in task_accs.items()}
            metrics["seed"] = seed
            seed_results.append(metrics)
            print(f"  Seed {seed}: AA={metrics['AA']:.4f}, FM={metrics['FM']:.4f}")

        if seed_results:
            aas = [r["AA"] for r in seed_results]
            fms = [r["FM"] for r in seed_results]
            all_results[method] = {
                "mean_AA": float(np.mean(aas)),
                "std_AA": float(np.std(aas)),
                "mean_FM": float(np.mean(fms)),
                "std_FM": float(np.std(fms)),
                "per_seed": seed_results,
            }
            # Incremental Save
            out_path = os.path.join(args.results_dir, f"{file_prefix}_results.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [INCREMENTAL SAVE] {out_path} updated with {method}")

    # Save
    out_path = os.path.join(args.results_dir, f"{file_prefix}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {args.dataset.upper()}")
    print(f"{'Method':<15} | {'AA (mean±std)':<20} | {'FM (mean±std)':<20}")
    print("-" * 60)
    for method, r in all_results.items():
        if isinstance(r, dict) and "mean_AA" in r:
            print(f"{method:<15} | {r['mean_AA']:.4f}±{r['std_AA']:.4f}        "
                  f"| {r['mean_FM']:.4f}±{r['std_FM']:.4f}")

if __name__ == "__main__":
    main()
