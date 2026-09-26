#!/usr/bin/env python
"""
run_qwen14b_smr_lora.py
========================
Physical Silicon Benchmark: Continual Learning on 14B Foundation LLM (Qwen2.5-14B, 4-Bit/8-Bit BNB Quantized)
Hardware: NVIDIA GeForce RTX 5070 Ti (17.1 GB Physical GDDR7 VRAM, 64 GB Host RAM)

Curriculum across 3 heterogeneous linguistic domains:
  - Task 0: Affective Emotion Classification (GoEmotions)
  - Task 1: Natural Language Inference (MultiNLI)
  - Task 2: Instruction Intent Classification (Databricks Dolly-15k)

Compares:
  1. Standard Sequential LoRA (R=32, Unprotected Shared Subspace)
  2. SMR-LoRA (Ours: Non-Colliding Subspaces, r_t=8, Buffer-Free |M|=0)

Output:
  - results_final/qwen14b_smr_lora_results.json
"""

import os
os.environ["BNB_CUDA_VERSION"] = "130"

import sys
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================
# SMR-LoRA Architecture for 14B Decoder Layers
# =====================================================================

class SMRLoRALinear(nn.Module):
    """
    SMR-LoRA Adapter for 14B Quantized Linear Projections:
      h = base_linear(x) + (x @ A^T @ B^T) * (scaling * mask)
    """
    def __init__(self, base_layer: nn.Module, in_features: int, out_features: int, r: int = 32, lora_alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA projection matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.bfloat16, device=DEVICE))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.bfloat16, device=DEVICE))
        
        # Binary rank mask for coordinate isolation
        self.register_buffer("rank_mask", torch.ones(r, dtype=torch.bfloat16, device=DEVICE))
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.base_layer(x)
        # Apply rank-masked low-rank residual
        effective_B = self.lora_B * self.rank_mask.unsqueeze(0)
        lora_out = F.linear(F.linear(x.to(torch.bfloat16), self.lora_A), effective_B) * self.scaling
        return orig_out + lora_out.to(orig_out.dtype)


def apply_smr_lora_to_qwen14b(model, r=32, lora_alpha=32.0):
    """
    Instruments all 48 decoder layers of Qwen2.5-14B (q_proj and v_proj).
    Total: 96 adapted projections across 48 layers.
    """
    adapted_count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        # q_proj
        orig_q = layer.self_attn.q_proj
        smr_q = SMRLoRALinear(orig_q, orig_q.in_features, orig_q.out_features, r=r, lora_alpha=lora_alpha)
        layer.self_attn.q_proj = smr_q
        adapted_count += 1

        # v_proj
        orig_v = layer.self_attn.v_proj
        smr_v = SMRLoRALinear(orig_v, orig_v.in_features, orig_v.out_features, r=r, lora_alpha=lora_alpha)
        layer.self_attn.v_proj = smr_v
        adapted_count += 1

    # Freeze base model parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    return adapted_count


def set_active_ranks_for_task(model, active_ranks: list):
    for layer in model.model.layers:
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]:
            if isinstance(proj, SMRLoRALinear):
                proj.rank_mask.zero_()
                for r in active_ranks:
                    proj.rank_mask[r] = 1.0


def mask_frozen_rank_gradients(model, frozen_ranks: list):
    if not frozen_ranks:
        return
    for layer in model.model.layers:
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]:
            if isinstance(proj, SMRLoRALinear):
                if proj.lora_A.grad is not None:
                    proj.lora_A.grad[frozen_ranks, :] = 0.0
                if proj.lora_B.grad is not None:
                    proj.lora_B.grad[:, frozen_ranks] = 0.0


# =====================================================================
# Dataset Classes & Curriculum
# =====================================================================

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def prepare_nlp_curriculum(tokenizer, max_train=500, max_test=200):
    print("  Loading Task 0: GoEmotions...")
    ds_go = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    target_emotions = {0: 0, 1: 1, 2: 2, 3: 3}
    t0_texts, t0_labels = [], []
    for item in ds_go:
        labels = item["labels"]
        if len(labels) == 1 and labels[0] in target_emotions:
            t0_texts.append(item["text"])
            t0_labels.append(target_emotions[labels[0]])
        if len(t0_texts) >= max_train + max_test:
            break
    t0_train_texts = t0_texts[:max_train]
    t0_train_labels = t0_labels[:max_train]
    t0_test_texts = t0_texts[max_train:max_train + max_test]
    t0_test_labels = t0_labels[max_train:max_train + max_test]
    print(f"    Task 0 loaded: {len(t0_train_texts)} train, {len(t0_test_texts)} test")

    print("  Loading Task 1: MultiNLI...")
    ds_mnli = load_dataset("multi_nli", split="train")
    t1_texts, t1_labels = [], []
    for item in ds_mnli:
        if item["label"] in [0, 1, 2]:
            t1_texts.append(f"Premise: {item['premise']} Hypothesis: {item['hypothesis']}")
            t1_labels.append(item["label"])
        if len(t1_texts) >= max_train + max_test:
            break
    t1_train_texts = t1_texts[:max_train]
    t1_train_labels = t1_labels[:max_train]
    t1_test_texts = t1_texts[max_train:max_train + max_test]
    t1_test_labels = t1_labels[max_train:max_train + max_test]
    print(f"    Task 1 loaded: {len(t1_train_texts)} train, {len(t1_test_texts)} test")

    print("  Loading Task 2: Databricks Dolly-15k...")
    ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    target_cats = {"open_qa": 0, "closed_qa": 1, "classification": 2, "brainstorming": 3}
    t2_texts, t2_labels = [], []
    for item in ds_dolly:
        cat = item["category"]
        if cat in target_cats:
            text = f"Instruction: {item['instruction']} Context: {item.get('context', '')}"
            t2_texts.append(text)
            t2_labels.append(target_cats[cat])
        if len(t2_texts) >= max_train + max_test:
            break
    t2_train_texts = t2_texts[:max_train]
    t2_train_labels = t2_labels[:max_train]
    t2_test_texts = t2_texts[max_train:max_train + max_test]
    t2_test_labels = t2_labels[max_train:max_train + max_test]
    print(f"    Task 2 loaded: {len(t2_train_texts)} train, {len(t2_test_texts)} test")

    return [
        {
            "name": "GoEmotions (Affective Emotion)",
            "num_classes": 4,
            "train_dataset": TextClassificationDataset(t0_train_texts, t0_train_labels, tokenizer, max_length=64),
            "test_dataset": TextClassificationDataset(t0_test_texts, t0_test_labels, tokenizer, max_length=64),
        },
        {
            "name": "MultiNLI (NLI Reasoning)",
            "num_classes": 3,
            "train_dataset": TextClassificationDataset(t1_train_texts, t1_train_labels, tokenizer, max_length=64),
            "test_dataset": TextClassificationDataset(t1_test_texts, t1_test_labels, tokenizer, max_length=64),
        },
        {
            "name": "Dolly-15k (Instruction Intent)",
            "num_classes": 4,
            "train_dataset": TextClassificationDataset(t2_train_texts, t2_train_labels, tokenizer, max_length=64),
            "test_dataset": TextClassificationDataset(t2_test_texts, t2_test_labels, tokenizer, max_length=64),
        }
    ]


@torch.no_grad()
def evaluate_model_on_task(model, task_head, test_loader, active_ranks=None):
    model.eval()
    task_head.eval()
    if active_ranks is not None:
        set_active_ranks_for_task(model, active_ranks)
        
    correct = 0
    total = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        seq_lengths = attention_mask.sum(dim=1) - 1
        hidden = outputs.hidden_states[-1]
        batch_indices = torch.arange(input_ids.size(0), device=DEVICE)
        cls_repr = hidden[batch_indices, seq_lengths, :]
        
        logits = task_head(cls_repr)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return correct / total if total > 0 else 0.0


def run_qwen14b_continual_experiment():
    set_seed(42)
    print("=" * 80)
    print("  QWEN2.5-14B (14.77B PARAMETERS) SMR-LoRA CONTINUAL LEARNING BENCHMARK")
    print("  Hardware: NVIDIA GeForce RTX 5070 Ti (17.1 GB Physical GDDR7 VRAM)")
    print("=" * 80)

    # Use local cache snapshot if present, otherwise download from HF hub
    hf_cache_snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen2.5-14B-bnb-4bit/snapshots/7fb3a0220f2a2bb6a6565ca46b97bb3badb17ace")
    model_id = hf_cache_snapshot if os.path.exists(hf_cache_snapshot) else "unsloth/Qwen2.5-14B-bnb-4bit"
    local_only = os.path.exists(hf_cache_snapshot)
    print(f"Loading Qwen2.5-14B tokenizer from: {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = prepare_nlp_curriculum(tokenizer, max_train=500, max_test=200)

    hidden_size = 5120
    lora_r = 32
    r_per_task = 8
    total_base_params = 14770000000

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # -------------------------------------------------------------
    # BENCHMARK 1: Standard Sequential LoRA (R=32)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> BENCHMARK 1: STANDARD SEQUENTIAL LoRA (R=32, Unprotected Subspace Overwrite)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time_seq = time.time()

    print("Loading base Qwen2.5-14B in 4-bit NF4 precision...")
    model_seq = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    model_seq.gradient_checkpointing_enable()

    adapted_projs = apply_smr_lora_to_qwen14b(model_seq, r=lora_r)
    print(f"Attached SMRLoRALinear to {adapted_projs} projections across 48 decoder layers.")

    set_active_ranks_for_task(model_seq, list(range(lora_r)))
    seq_heads = [nn.Linear(hidden_size, t["num_classes"], dtype=torch.bfloat16, device=DEVICE) for t in tasks]
    acc_matrix_seq = np.zeros((len(tasks), len(tasks)))

    trainable_params_seq = sum(p.numel() for p in model_seq.parameters() if p.requires_grad)
    pct_seq = (trainable_params_seq / total_base_params) * 100
    print(f"Sequential LoRA Trainable Parameters: {trainable_params_seq:,} ({pct_seq:.4f}% of 14.8B model)")

    for stage, task in enumerate(tasks):
        print(f"\n--- [Sequential LoRA] Training Stage {stage}: {task['name']} ---")
        train_loader = DataLoader(task["train_dataset"], batch_size=2, shuffle=True)
        test_loaders = [DataLoader(t["test_dataset"], batch_size=4, shuffle=False) for t in tasks]
        
        head = seq_heads[stage]
        lora_params = [p for n, p in model_seq.named_parameters() if "lora_" in n and p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params + list(head.parameters()), lr=5e-4, weight_decay=0.01)

        model_seq.train()
        head.train()
        
        epochs = 1
        accum_steps = 2
        for epoch in range(epochs):
            total_loss = 0.0
            step_count = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model_seq(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                seq_lengths = attention_mask.sum(dim=1) - 1
                hidden = outputs.hidden_states[-1]
                batch_indices = torch.arange(input_ids.size(0), device=DEVICE)
                cls_repr = hidden[batch_indices, seq_lengths, :]
                
                logits = head(cls_repr)
                loss = F.cross_entropy(logits, labels) / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps
                step_count += 1
                
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            print(f"  Stage {stage} Epoch {epoch+1}/{epochs} Loss: {total_loss / step_count:.4f}")

        print(f"  Evaluating at Stage {stage}...")
        for eval_task_id in range(stage + 1):
            acc = evaluate_model_on_task(model_seq, seq_heads[eval_task_id], test_loaders[eval_task_id])
            acc_matrix_seq[stage, eval_task_id] = acc
            print(f"    Task {eval_task_id} Acc: {acc * 100:.2f}%")

    seq_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    seq_time = time.time() - start_time_seq
    print(f"\nSequential LoRA Peak VRAM: {seq_peak_vram:.2f} GB | Duration: {seq_time:.1f}s")

    del model_seq
    del seq_heads
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # BENCHMARK 2: SMR-LoRA (Ours: r_t=8 Dedicated Subspaces)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> BENCHMARK 2: SMR-LoRA (Ours: Dedicated Subspaces, r_t=8, Buffer-Free |M|=0)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time_smr = time.time()

    print("Loading pristine Qwen2.5-14B base model for SMR-LoRA...")
    model_smr = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    model_smr.gradient_checkpointing_enable()

    adapted_projs = apply_smr_lora_to_qwen14b(model_smr, r=lora_r)
    smr_heads = [nn.Linear(hidden_size, t["num_classes"], dtype=torch.bfloat16, device=DEVICE) for t in tasks]
    acc_matrix_smr = np.zeros((len(tasks), len(tasks)))

    allocated_ranks = {
        0: list(range(0, 8)),
        1: list(range(8, 16)),
        2: list(range(16, 24)),
    }
    frozen_coordinate_snapshots = {}

    trainable_params_per_task_smr = int(trainable_params_seq * (r_per_task / lora_r))
    pct_smr = (trainable_params_per_task_smr / total_base_params) * 100
    print(f"SMR-LoRA Active Parameters Per Task: {trainable_params_per_task_smr:,} ({pct_smr:.4f}% of 14.8B model)")

    for stage, task in enumerate(tasks):
        print(f"\n--- [SMR-LoRA] Training Stage {stage}: {task['name']} ---")
        active_ranks = allocated_ranks[stage]
        frozen_ranks = []
        for prev in range(stage):
            frozen_ranks.extend(allocated_ranks[prev])
            
        print(f"  Active Ranks for Task {stage}: {active_ranks}")
        print(f"  Protected Frozen Ranks: {frozen_ranks}")

        set_active_ranks_for_task(model_smr, active_ranks)

        train_loader = DataLoader(task["train_dataset"], batch_size=2, shuffle=True)
        test_loaders = [DataLoader(t["test_dataset"], batch_size=4, shuffle=False) for t in tasks]

        head = smr_heads[stage]
        lora_params = [p for n, p in model_smr.named_parameters() if "lora_" in n and p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params + list(head.parameters()), lr=5e-4, weight_decay=0.01)

        model_smr.train()
        head.train()

        epochs = 1
        accum_steps = 2
        for epoch in range(epochs):
            total_loss = 0.0
            step_count = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model_smr(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                seq_lengths = attention_mask.sum(dim=1) - 1
                hidden = outputs.hidden_states[-1]
                batch_indices = torch.arange(input_ids.size(0), device=DEVICE)
                cls_repr = hidden[batch_indices, seq_lengths, :]
                
                logits = head(cls_repr)
                loss = F.cross_entropy(logits, labels) / accum_steps
                loss.backward()

                mask_frozen_rank_gradients(model_smr, frozen_ranks)
                total_loss += loss.item() * accum_steps
                step_count += 1

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    for p_name, (layer_idx, is_q, is_A, snap) in frozen_coordinate_snapshots.items():
                        layer = model_smr.model.layers[layer_idx]
                        proj = layer.self_attn.q_proj if is_q else layer.self_attn.v_proj
                        with torch.no_grad():
                            if is_A:
                                proj.lora_A[snap["ranks"], :].copy_(snap["weights"])
                            else:
                                proj.lora_B[:, snap["ranks"]].copy_(snap["weights"])
                    optimizer.zero_grad()

            print(f"  Stage {stage} Epoch {epoch+1}/{epochs} Loss: {total_loss / step_count:.4f}")

        for l_idx, layer in enumerate(model_smr.model.layers):
            for is_q, proj in [(True, layer.self_attn.q_proj), (False, layer.self_attn.v_proj)]:
                snap_id_A = f"L{l_idx}_{'q' if is_q else 'v'}_A_T{stage}"
                snap_id_B = f"L{l_idx}_{'q' if is_q else 'v'}_B_T{stage}"
                frozen_coordinate_snapshots[snap_id_A] = (
                    l_idx, is_q, True,
                    {"ranks": active_ranks, "weights": proj.lora_A[active_ranks, :].detach().clone()}
                )
                frozen_coordinate_snapshots[snap_id_B] = (
                    l_idx, is_q, False,
                    {"ranks": active_ranks, "weights": proj.lora_B[:, active_ranks].detach().clone()}
                )

        print(f"  Evaluating SMR-LoRA at Stage {stage}...")
        for eval_task_id in range(stage + 1):
            eval_ranks = allocated_ranks[eval_task_id]
            acc = evaluate_model_on_task(model_smr, smr_heads[eval_task_id], test_loaders[eval_task_id], active_ranks=eval_ranks)
            acc_matrix_smr[stage, eval_task_id] = acc
            print(f"    Task {eval_task_id} Acc: {acc * 100:.2f}%")

    smr_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    smr_time = time.time() - start_time_smr
    print(f"\nSMR-LoRA Peak VRAM: {smr_peak_vram:.2f} GB | Duration: {smr_time:.1f}s")

    final_stage = len(tasks) - 1
    seq_terminal = acc_matrix_seq[final_stage, :]
    smr_terminal = acc_matrix_smr[final_stage, :]

    seq_AA = float(np.mean(seq_terminal))
    smr_AA = float(np.mean(smr_terminal))

    seq_FM_vals = []
    smr_FM_vals = []
    for j in range(len(tasks) - 1):
        seq_peak = np.max(acc_matrix_seq[j:len(tasks) - 1, j])
        seq_FM_vals.append(max(0.0, float(seq_peak - acc_matrix_seq[final_stage, j])))
        smr_peak = np.max(acc_matrix_smr[j:len(tasks) - 1, j])
        smr_FM_vals.append(max(0.0, float(smr_peak - acc_matrix_smr[final_stage, j])))

    seq_FM = float(np.mean(seq_FM_vals))
    smr_FM = float(np.mean(smr_FM_vals))

    print("\n" + "=" * 80)
    print("  FINAL CONTINUAL LEARNING SUMMARY ON QWEN2.5-14B (14.77B PARAMETERS)")
    print("=" * 80)
    print(f"{'Method':<20} | {'AA':<10} | {'FM':<10} | {'T0 Ret.':<10} | {'T1 Ret.':<10} | {'T2 Acc.':<10} | {'Peak VRAM':<10}")
    print("-" * 95)
    print(f"{'Sequential LoRA':<20} | {seq_AA*100:6.2f}%    | {seq_FM*100:6.2f}%    | {seq_terminal[0]*100:5.2f}%    | {seq_terminal[1]*100:5.2f}%    | {seq_terminal[2]*100:5.2f}%    | {seq_peak_vram:5.2f} GB")
    print(f"{'SMR-LoRA (Ours)':<20} | {smr_AA*100:6.2f}%    | {smr_FM*100:6.2f}%    | {smr_terminal[0]*100:5.2f}%    | {smr_terminal[1]*100:5.2f}%    | {smr_terminal[2]*100:5.2f}%    | {smr_peak_vram:5.2f} GB")
    print("-" * 95)

    results = {
        "model": "Qwen2.5-14B",
        "total_parameters": total_base_params,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_vram_gb": max(seq_peak_vram, smr_peak_vram),
        "sequential_lora": {
            "AA": seq_AA,
            "FM": seq_FM,
            "task_accs": acc_matrix_seq.tolist(),
            "terminal_accs": seq_terminal.tolist(),
            "tuned_params_per_task": trainable_params_seq,
            "pct_tuned": pct_seq,
            "execution_time_s": seq_time,
        },
        "smr_lora": {
            "AA": smr_AA,
            "FM": smr_FM,
            "task_accs": acc_matrix_smr.tolist(),
            "terminal_accs": smr_terminal.tolist(),
            "tuned_params_per_task": trainable_params_per_task_smr,
            "pct_tuned": pct_smr,
            "execution_time_s": smr_time,
        },
        "continual_delta": {
            "AA_gain": smr_AA - seq_AA,
            "FM_reduction": seq_FM - smr_FM,
        }
    }

    out_path = "results_final/qwen14b_smr_lora_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results successfully written to {out_path}")

if __name__ == "__main__":
    run_qwen14b_continual_experiment()
