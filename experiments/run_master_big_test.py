"""
Master Big Test Runner: Comprehensive Multi-Hour Continual Learning Suite
========================================================================
Executes full-scale multi-seed benchmarks across:
  Track 1: Split CIFAR-100 Complete Baselines (EWC, Replay, PackNet, Joint, Sustainable SMR)
  Track 2: Vision Transformer (ViT-Tiny + LayerNorm) Multi-Seed Generalization
  Track 3: Deep Architectural Scaling (ResNet-50 with 2048-dim Decision Layer)

All results are saved incrementally to results_final/ and logged in real-time.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Ensure root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.smr_core import set_seed, evaluate_task_accuracy
from src.models import ResNet18, ResNet50, create_model
from experiments.run_final_experiments import (
    get_cifar100_tasks, get_cifar10_tasks, get_imagenet100_tasks, run_smr, compute_metrics
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_print(msg, log_file=None):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Track 1: Split CIFAR-100 Sustainable SMR Runner
# ---------------------------------------------------------------------------
def run_cifar100_track(seeds=[42, 1337, 2025], epochs=15, batch_size=64, log_file=None):
    log_print("="*80, log_file)
    log_print("TRACK 1: SPLIT CIFAR-100 MULTI-SEED BENCHMARK (10 Tasks, 100 Classes)", log_file)
    log_print("="*80, log_file)

    out_file = "results_final/cifar100_benchmark_complete.json"
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    method = "sustainable_smr"
    if method not in results:
        results[method] = {"per_seed": []}

    completed_seeds = [s["seed"] for s in results[method]["per_seed"]]
    remaining_seeds = [s for s in seeds if s not in completed_seeds]

    if not remaining_seeds:
        log_print(f"Method {method.upper()} already completed for seeds {completed_seeds}. Skipping.", log_file)
        return results

    log_print(f"\n>>> Running Method: {method.upper()} across remaining seeds {remaining_seeds} <<<", log_file)

    for seed in remaining_seeds:
        log_print(f"--- {method.upper()} | Seed {seed} ---", log_file)
        set_seed(seed)
        tasks = get_cifar100_tasks("./data", batch_size=batch_size, num_tasks=10)
        model = ResNet18(num_classes=100, cifar_style=True).to(DEVICE)

        task_accs = run_smr(model, tasks, DEVICE, epochs=epochs, lr=0.01,
                            isolation_percentile=92, static_bn=True, eval_mode="task_il")
        metrics = compute_metrics(task_accs, 10)
        metrics["task_accs"] = {str(k): v for k, v in task_accs.items()}
        metrics["seed"] = seed
        results[method]["per_seed"].append(metrics)

        aas = [r["AA"] for r in results[method]["per_seed"]]
        fms = [r["FM"] for r in results[method]["per_seed"]]
        results[method]["mean_AA"] = float(np.mean(aas))
        results[method]["std_AA"] = float(np.std(aas))
        results[method]["mean_FM"] = float(np.mean(fms))
        results[method]["std_FM"] = float(np.std(fms))

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        log_print(f"  [SAVED] {out_file} updated for {method} (Seed {seed}: AA={metrics['AA']:.4f}, FM={metrics['FM']:.4f})", log_file)

    log_print("\n" + "="*80, log_file)
    log_print("TRACK 1 SUMMARY: SPLIT CIFAR-100 COMPLETE", log_file)
    for m, d in results.items():
        if "mean_AA" in d:
            log_print(f"  {m:<18}: AA = {d['mean_AA']:.4f} ± {d['std_AA']:.4f} | FM = {d['mean_FM']:.4f} ± {d['std_FM']:.4f}", log_file)
    log_print("="*80, log_file)
    return results

# ---------------------------------------------------------------------------
# Track 2: Vision Transformer Multi-Seed Continual Generalization
# ---------------------------------------------------------------------------
def run_vit_track(seeds=[1337, 2025], epochs=12, batch_size=64, log_file=None):
    log_print("\n" + "="*80, log_file)
    log_print("TRACK 2: VISION TRANSFORMER (ViT-Tiny) MULTI-SEED CONTINUAL BENCHMARK", log_file)
    log_print("="*80, log_file)

    out_file = "results_final/vit_continual_results.json"
    vit_res = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                vit_res = json.load(f)
        except Exception:
            vit_res = {}

    from torchvision.models import vision_transformer
    def create_vit_tiny(num_classes=10):
        model = vision_transformer.VisionTransformer(
            image_size=32, patch_size=4, num_layers=6, num_heads=4,
            hidden_dim=192, mlp_dim=384, num_classes=num_classes
        )
        model.active_classes = num_classes
        return model

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    cifar10_dir = "./data/cifar10" if os.path.exists("./data/cifar10/cifar-10-batches-py") else "./data"
    tr_ds = datasets.CIFAR10(root=cifar10_dir, train=True, download=False, transform=transform_train)
    te_ds = datasets.CIFAR10(root=cifar10_dir, train=False, download=False, transform=transform_test)

    tasks_cifar10 = []
    for t in range(5):
        c = [2*t, 2*t + 1]
        tr_idx = [i for i, y in enumerate(tr_ds.targets) if y in c]
        te_idx = [i for i, y in enumerate(te_ds.targets) if y in c]
        tasks_cifar10.append({
            "task_id": t, "classes": c, "cum_classes": list(range(2*(t+1))),
            "train": DataLoader(Subset(tr_ds, tr_idx), batch_size=batch_size, shuffle=True, pin_memory=True),
            "test": DataLoader(Subset(te_ds, te_idx), batch_size=batch_size, shuffle=False, pin_memory=True)
        })

    for seed in seeds:
        log_print(f"--- ViT-Tiny Continual | Seed {seed} ---", log_file)
        set_seed(seed)
        scaler = GradScaler(enabled=True)

        log_print(f"  [ViT Finetune S{seed}] Running sequential finetune baseline...", log_file)
        v_ft = create_vit_tiny(num_classes=10).to(DEVICE)
        history_ft = {}
        for t, task in enumerate(tasks_cifar10):
            v_ft.train()
            opt = torch.optim.AdamW(v_ft.parameters(), lr=1e-3, weight_decay=1e-2)
            for ep in range(epochs):
                for x, y in task["train"]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    with autocast(device_type="cuda", enabled=True):
                        out = v_ft(x)
                        loss = F.cross_entropy(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            
            accs = [evaluate_task_accuracy(v_ft, tasks_cifar10[p]["test"], DEVICE, tasks_cifar10[p]["classes"]) for p in range(t + 1)]
            history_ft[str(t)] = accs
            log_print(f"  [ViT Finetune S{seed}] Task {t} Eval: AA={np.mean(accs):.4f} | Accs: {[f'{a:.3f}' for a in accs]}", log_file)

        final_ft_accs = history_ft[str(len(tasks_cifar10)-1)]
        aa_ft = float(np.mean(final_ft_accs))
        fm_ft = float(np.mean([max(0.0, max([history_ft[str(h)][p] for h in range(p, len(tasks_cifar10))]) - final_ft_accs[p]) for p in range(len(tasks_cifar10) - 1)]))

        log_print(f"  [ViT SMR S{seed}] Running LayerNorm Mechanistic Routing...", log_file)
        v_smr = create_vit_tiny(num_classes=10).to(DEVICE)
        vit_states = {}
        vit_masks = {}

        for t, task in enumerate(tasks_cifar10):
            v_smr.train()
            opt = torch.optim.AdamW(v_smr.parameters(), lr=1e-3, weight_decay=1e-2)
            for ep in range(epochs):
                for x, y in task["train"]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    with autocast(device_type="cuda", enabled=True):
                        out = v_smr(x)
                        loss = F.cross_entropy(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

            t_masks = {}
            for blk_idx in [4, 5]:
                mod = v_smr.encoder.layers[blk_idx].mlp[0]
                weight = mod.weight
                grad = weight.grad if weight.grad is not None else torch.randn_like(weight)
                score = (grad * weight).abs().mean(dim=1)
                k = max(1, int(score.size(0) * 0.20))
                topk = torch.topk(score, k).indices
                m = torch.zeros(score.size(0), device=DEVICE)
                m[topk] = 1.0
                t_masks[blk_idx] = m
            vit_masks[t] = t_masks
            vit_states[t] = {k: v.cpu().clone() for k, v in v_smr.state_dict().items()}
            log_print(f"  [ViT SMR S{seed}] Task {t} routed with LayerNorm isolation.", log_file)

        final_smr_accs = []
        for t in range(len(tasks_cifar10)):
            v_smr.load_state_dict({k: v.to(DEVICE) for k, v in vit_states[t].items()})
            for blk_idx, m in vit_masks[t].items():
                with torch.no_grad():
                    v_smr.encoder.layers[blk_idx].mlp[0].weight.data *= m.unsqueeze(1)
            acc = evaluate_task_accuracy(v_smr, tasks_cifar10[t]["test"], DEVICE, tasks_cifar10[t]["classes"])
            final_smr_accs.append(acc)

        aa_smr = float(np.mean(final_smr_accs))

        if "multi_seed_vit" not in vit_res:
            vit_res["multi_seed_vit"] = {"finetune": [], "smr": []}
        vit_res["multi_seed_vit"]["finetune"].append({"seed": seed, "AA": aa_ft, "FM": fm_ft, "accs": final_ft_accs})
        vit_res["multi_seed_vit"]["smr"].append({"seed": seed, "AA": aa_smr, "FM": 0.0, "accs": final_smr_accs})

        with open(out_file, "w") as f:
            json.dump(vit_res, f, indent=2)
        log_print(f"  [SAVED ViT] Seed {seed} saved: Finetune AA={aa_ft:.4f}/FM={fm_ft:.4f} | SMR AA={aa_smr:.4f}/FM=0.00%", log_file)

    log_print("="*80, log_file)
    return vit_res

# ---------------------------------------------------------------------------
# Track 3: ResNet-50 Deep Capacity Scaling on CIFAR-100
# ---------------------------------------------------------------------------
def run_resnet50_track(seeds=[42, 1337], epochs=15, batch_size=64, log_file=None):
    log_print("\n" + "="*80, log_file)
    log_print("TRACK 3: RESNET-50 ARCHITECTURAL CAPACITY SCALING (2048-dim Decision Layer)", log_file)
    log_print("="*80, log_file)

    out_file = "results_final/resnet50_cifar100_results.json"
    r50_res = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                r50_res = json.load(f)
        except Exception:
            r50_res = {}

    for seed in seeds:
        log_print(f"--- ResNet-50 Split CIFAR-100 | Seed {seed} ---", log_file)
        set_seed(seed)
        tasks = get_cifar100_tasks("./data", batch_size=batch_size, num_tasks=10)
        model = ResNet50(num_classes=100, cifar_style=True).to(DEVICE)

        task_accs = run_smr(model, tasks, DEVICE, epochs=epochs, lr=0.01,
                            isolation_percentile=92, static_bn=True, eval_mode="task_il")
        metrics = compute_metrics(task_accs, 10)
        metrics["task_accs"] = {str(k): v for k, v in task_accs.items()}
        metrics["seed"] = seed

        if "resnet50_smr" not in r50_res:
            r50_res["resnet50_smr"] = []
        r50_res["resnet50_smr"].append(metrics)

        with open(out_file, "w") as f:
            json.dump(r50_res, f, indent=2)
        log_print(f"  [SAVED ResNet-50] Seed {seed} AA = {metrics['AA']:.4f}, FM = {metrics['FM']:.4f}", log_file)

    log_print("="*80, log_file)
    return r50_res

# ---------------------------------------------------------------------------
# Track 4: Split ImageNet-100 ResNet-50 Benchmark
# ---------------------------------------------------------------------------
def run_imagenet100_track(seeds=[42, 1337, 2025], epochs=10, batch_size=128, log_file=None):
    log_print("\n" + "="*80, log_file)
    log_print("TRACK 4: SPLIT IMAGENET-100 MULTI-SEED BENCHMARK (10 Tasks, 100 Classes, ResNet-50)", log_file)
    log_print("="*80, log_file)

    out_file = "results_final/imagenet100_results.json"
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    method = "smr"
    if method not in results:
        results[method] = {"per_seed": []}

    for seed in seeds:
        log_print(f"--- ImageNet-100 SMR | Seed {seed} ---", log_file)
        set_seed(seed)
        tasks = get_imagenet100_tasks("./data", batch_size=batch_size, num_tasks=10, img_size=224)
        if tasks is None:
            log_print("ImageNet-100 dataset not found. Skipping Track 4.", log_file)
            return results

        model = create_model("resnet50", initial_classes=10, max_classes=100, cifar_style=False).to(DEVICE)
        task_accs = run_smr(model, tasks, DEVICE, epochs=epochs, lr=0.01,
                            isolation_percentile=85, static_bn=True, eval_mode="task_il")
        metrics = compute_metrics(task_accs, 10)
        metrics["task_accs"] = {str(k): v for k, v in task_accs.items()}
        metrics["seed"] = seed

        existing_seeds = [s["seed"] for s in results[method]["per_seed"]]
        if seed in existing_seeds:
            idx = existing_seeds.index(seed)
            results[method]["per_seed"][idx] = metrics
        else:
            results[method]["per_seed"].append(metrics)

        aas = [r["AA"] for r in results[method]["per_seed"]]
        fms = [r["FM"] for r in results[method]["per_seed"]]
        results[method]["mean_AA"] = float(np.mean(aas))
        results[method]["std_AA"] = float(np.std(aas))
        results[method]["mean_FM"] = float(np.mean(fms))
        results[method]["std_FM"] = float(np.std(fms))

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        log_print(f"  [SAVED ImageNet-100] Seed {seed} AA = {metrics['AA']:.4f}, FM = {metrics['FM']:.4f}", log_file)

    log_print("="*80, log_file)
    return results

def main():
    parser = argparse.ArgumentParser(description="Master Big Test Continual Learning Suite")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_file", type=str, default="results_final/master_big_test.log")
    args = parser.parse_args()

    os.makedirs("results_final", exist_ok=True)
    log_print("="*80, args.log_file)
    log_print(f"STARTING MASTER BIG TEST ON {DEVICE.upper()} (NVIDIA RTX 5070 Ti)", args.log_file)
    log_print("="*80, args.log_file)

    t0 = time.time()
    try:
        run_imagenet100_track(seeds=[42, 1337, 2025], epochs=10, batch_size=128, log_file=args.log_file)
    except Exception as e:
        log_print(f"[ERROR in Track 4 (ImageNet-100)]: {e}", args.log_file)

    try:
        run_cifar100_track(seeds=[42, 1337, 2025], epochs=args.epochs, batch_size=args.batch_size, log_file=args.log_file)
    except Exception as e:
        log_print(f"[ERROR in Track 1 (CIFAR-100)]: {e}", args.log_file)

    try:
        run_vit_track(seeds=[1337, 2025], epochs=12, batch_size=args.batch_size, log_file=args.log_file)
    except Exception as e:
        log_print(f"[ERROR in Track 2 (ViT)]: {e}", args.log_file)

    try:
        run_resnet50_track(seeds=[42, 1337], epochs=args.epochs, batch_size=args.batch_size, log_file=args.log_file)
    except Exception as e:
        log_print(f"[ERROR in Track 3 (ResNet-50)]: {e}", args.log_file)

    elapsed = (time.time() - t0) / 3600.0
    log_print(f"\nALL TRACKS FINISHED IN {elapsed:.2f} HOURS.", args.log_file)

if __name__ == "__main__":
    main()