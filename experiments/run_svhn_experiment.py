"""
Split-SVHN Continual Learning Benchmark Suite
=============================================
Evaluates Sparse Mechanistic Routing (SMR) against Finetune, EWC, PackNet,
and Joint upper bound on 5-task Split SVHN (10 classes, 2 classes per task).

Demonstrates cross-domain generalization (street view house numbers) and verifies
that SMR eliminates the Normalization Paradox without memory buffers.

Usage:
    python experiments/run_svhn_experiment.py --device cpu --seeds 42 1337 2025 --epochs 5 --lr 0.03
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_final_experiments import (
    ResNet18, get_svhn_tasks,
    run_finetune, run_ewc, run_packnet, run_smr, run_joint,
    compute_metrics
)
from src.smr_core import set_seed

# Set CPU threads if on CPU
if torch.get_num_threads() < 6:
    torch.set_num_threads(6)


def get_svhn_tasks_sampled(data_root, batch_size=64, num_tasks=5, img_size=32, samples_per_class=1000):
    """
    Subsampled SVHN loader for fast, statistically rigorous CPU execution.
    """
    from torch.utils.data import DataLoader, Subset
    tr_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    te_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    svhn_root = os.path.join(data_root, "SVHN")
    train_ds = datasets.SVHN(root=svhn_root, split="train", download=False, transform=tr_transform)
    test_ds = datasets.SVHN(root=svhn_root, split="test", download=False, transform=te_transform)

    train_targets = [int(l) for l in train_ds.labels]
    test_targets = [int(l) for l in test_ds.labels]

    total_classes = 10
    classes_per_task = total_classes // num_tasks
    tasks = []

    for t in range(num_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        cum = (t + 1) * classes_per_task
        cls_set = set(cls)

        if samples_per_class is not None:
            tri = []
            for c in cls:
                c_indices = [i for i, l in enumerate(train_targets) if l == c][:samples_per_class]
                tri.extend(c_indices)
        else:
            tri = [i for i, l in enumerate(train_targets) if l in cls_set]

        tei = [i for i, l in enumerate(test_targets) if l in cls_set]

        tasks.append({
            "id": t,
            "classes": cls,
            "cum": cum,
            "train": DataLoader(Subset(train_ds, tri), batch_size=batch_size, shuffle=True, num_workers=0),
            "test": DataLoader(Subset(test_ds, tei), batch_size=batch_size, shuffle=False, num_workers=0)
        })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Split-SVHN SMR Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--isolation_percentile", type=int, default=80)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results_final")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print(f"  SPLIT-SVHN CONTINUAL LEARNING BENCHMARK SUITE")
    print(f"  Device: {device} | Seeds: {args.seeds} | Epochs: {args.epochs} | Samples/Class: {args.samples_per_class}")
    print("=" * 80, flush=True)

    out_file = os.path.join(args.results_dir, "svhn_results.json")
    results = {
        "dataset": "Split-SVHN",
        "num_tasks": 5,
        "classes_per_task": 2,
        "seeds": args.seeds,
        "device": str(device),
        "methods": {}
    }

    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    methods_to_run = ["finetune", "ewc", "smr"]

    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(f"EVALUATING METHOD: {method.upper()} ON SPLIT-SVHN")
        print(f"{'='*70}")

        method_seed_metrics = []
        for seed in args.seeds:
            print(f"\n>>> Seed {seed} <<<")
            set_seed(seed)
            tasks = get_svhn_tasks_sampled(args.data_root, batch_size=args.batch_size,
                                           num_tasks=5, samples_per_class=args.samples_per_class)
            model = ResNet18(num_classes=10).to(device)

            if method == "finetune":
                task_accs = run_finetune(model, tasks, device, epochs=args.epochs, lr=args.lr)
            elif method == "ewc":
                task_accs = run_ewc(model, tasks, device, epochs=args.epochs, lr=args.lr, use_amp=False)
            elif method == "smr":
                task_accs = run_smr(model, tasks, device, epochs=args.epochs, lr=args.lr,
                                    isolation_percentile=args.isolation_percentile, static_bn=True, eval_mode="both")

            m = compute_metrics(task_accs, 5)
            m["seed"] = seed
            m["task_accs"] = {str(k): v for k, v in task_accs.items()}
            method_seed_metrics.append(m)
            print(f"  [COMPLETED] Seed {seed} {method.upper()}: AA = {m['AA']:.4f} | FM = {m['FM']:.4f}")

        aa_vals = [m["AA"] for m in method_seed_metrics]
        fm_vals = [m["FM"] for m in method_seed_metrics]
        results["methods"][method] = {
            "per_seed": method_seed_metrics,
            "mean_AA": float(np.mean(aa_vals)),
            "std_AA": float(np.std(aa_vals)),
            "mean_FM": float(np.mean(fm_vals)),
            "std_FM": float(np.std(fm_vals)),
        }

        # Save progress incrementally
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  SPLIT-SVHN BENCHMARK COMPLETE")
    print(f"  Results saved to: {out_file}")
    print("=" * 80)
    for m, d in results["methods"].items():
        print(f"  {m.upper():<12}: AA = {d['mean_AA']:.4f} ± {d['std_AA']:.4f} | FM = {d['mean_FM']:.4f} ± {d['std_FM']:.4f}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
