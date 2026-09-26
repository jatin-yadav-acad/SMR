"""
run_qwen7b_smr_lora.py
======================
Continual Learning on 7.6B Foundation Model: Qwen2.5-7B (7.62B Parameters, 8B Class).
Benchmarks Sparse Mechanistic Routing (SMR-LoRA) on Large Language Models across
a sequential 3-task continual NLP curriculum on physical NVIDIA RTX 5070 Ti silicon:

Tasks:
  Task 0: Affective Emotion Classification (GoEmotions, 4 classes)
  Task 1: Natural Language Inference (MultiNLI, 3 classes)
  Task 2: Task Intent Classification (Databricks Dolly-15k, 4 classes)

Paradigms Compared:
  1. Standard Sequential LoRA (R=32, Unprotected Subspace Overwrite, Severe Catastrophic Forgetting)
  2. SMR-LoRA (Ours: Non-Colliding Rank Subspace Allocation, Exact 0.00% Forgetting, |M|=0)

Logs authentic metrics and GPU telemetry to results_final/qwen7b_smr_lora_results.json.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# SMR-LoRA Linear Adapter Module for Qwen2 Architecture
# =====================================================================

class SMRLoRALinear(nn.Module):
    """
    LoRA Linear Adapter supporting selective rank masking, dual-defense
    gradient projection, and coordinate-level parameter snapshotting.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 32, lora_alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        self.d_in = base_linear.in_features
        self.d_out = base_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r
        
        # Base weights frozen bitwise
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
            
        # LoRA weights in bfloat16 to match Qwen2.5 base precision
        self.lora_A = nn.Parameter(torch.randn(r, self.d_in, dtype=torch.bfloat16, device=DEVICE) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.d_out, r, dtype=torch.bfloat16, device=DEVICE))
        
        # Active rank mask: shape (r,), 1.0 for active ranks, 0.0 for masked/unallocated
        self.register_buffer("rank_mask", torch.ones(r, dtype=torch.bfloat16, device=DEVICE))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Apply active rank mask on A before projecting through B
        active_A = self.lora_A * self.rank_mask.unsqueeze(1)
        active_B = self.lora_B * self.rank_mask.unsqueeze(0)
        lora_out = (x @ active_A.t()) @ active_B.t() * self.scaling
        return base_out + lora_out


def apply_smr_lora_to_qwen7b(model, r: int = 32, lora_alpha: float = 32.0):
    """
    Attaches SMRLoRALinear modules to self-attention Query and Value projection layers
    across all 28 transformer decoder blocks.
    """
    adapted_count = 0
    for layer in model.model.layers:
        layer.self_attn.q_proj = SMRLoRALinear(layer.self_attn.q_proj, r=r, lora_alpha=lora_alpha)
        layer.self_attn.v_proj = SMRLoRALinear(layer.self_attn.v_proj, r=r, lora_alpha=lora_alpha)
        adapted_count += 2
    return adapted_count


def set_active_ranks_for_task(model, active_ranks: list):
    """
    Configures rank_mask buffer across all LoRA modules to isolate active task sub-manifold.
    """
    for layer in model.model.layers:
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]:
            if isinstance(proj, SMRLoRALinear):
                proj.rank_mask.zero_()
                for r in active_ranks:
                    proj.rank_mask[r] = 1.0


def mask_frozen_rank_gradients(model, frozen_ranks: list):
    """
    Dual-Defense Gradient Masking: zeroes out backpropagated gradients for historical
    task rank coordinates, guaranteeing exact bitwise coordinate invariance.
    """
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
# NLP Curriculum Datasets
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
    """
    Curriculum of 3 tasks:
      Task 0: GoEmotions (4 classes: admiration=0, amusement=1, anger=2, annoyance=3)
      Task 1: MultiNLI (3 classes: entailment=0, neutral=1, contradiction=2)
      Task 2: Dolly-15k (4 classes: open_qa=0, closed_qa=1, classification=2, brainstorming=3)
    """
    print("--- Preparing Continual NLP Curriculum ---")
    
    # Task 0: GoEmotions
    print("  Loading Task 0: GoEmotions...")
    ds_emo = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    t0_target_classes = [0, 1, 2, 3] # admiration, amusement, anger, annoyance
    t0_texts, t0_labels = [], []
    for item in ds_emo:
        labels = item["labels"]
        for c in t0_target_classes:
            if c in labels:
                t0_texts.append(item["text"])
                t0_labels.append(c)
                break
        if len(t0_texts) >= max_train + max_test:
            break
    
    t0_train_texts = t0_texts[:max_train]
    t0_train_labels = t0_labels[:max_train]
    t0_test_texts = t0_texts[max_train:max_train + max_test]
    t0_test_labels = t0_labels[max_train:max_train + max_test]
    print(f"    Task 0 loaded: {len(t0_train_texts)} train, {len(t0_test_texts)} test")

    # Task 1: MultiNLI
    print("  Loading Task 1: MultiNLI...")
    ds_nli = load_dataset("multi_nli", split="train")
    t1_texts, t1_labels = [], []
    for item in ds_nli:
        lbl = item["label"]
        if lbl in [0, 1, 2]:
            text = f"Premise: {item['premise']} Hypothesis: {item['hypothesis']}"
            t1_texts.append(text)
            t1_labels.append(lbl)
        if len(t1_texts) >= max_train + max_test:
            break
            
    t1_train_texts = t1_texts[:max_train]
    t1_train_labels = t1_labels[:max_train]
    t1_test_texts = t1_texts[max_train:max_train + max_test]
    t1_test_labels = t1_labels[max_train:max_train + max_test]
    print(f"    Task 1 loaded: {len(t1_train_texts)} train, {len(t1_test_texts)} test")

    # Task 2: Databricks Dolly 15k
    print("  Loading Task 2: Databricks Dolly-15k...")
    ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    target_cats = {
        "open_qa": 0,
        "closed_qa": 1,
        "classification": 2,
        "brainstorming": 3
    }
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

    tasks = [
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
    return tasks


# =====================================================================
# Evaluation Routine
# =====================================================================

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
        # Sequence representation from final layer last non-pad token
        seq_lengths = attention_mask.sum(dim=1) - 1
        hidden = outputs.hidden_states[-1]
        batch_indices = torch.arange(input_ids.size(0), device=DEVICE)
        cls_repr = hidden[batch_indices, seq_lengths, :] # [B, hidden_size]
        
        logits = task_head(cls_repr)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return correct / total if total > 0 else 0.0


# =====================================================================
# Main Benchmark Execution
# =====================================================================

def run_qwen7b_continual_experiment():
    set_seed(42)
    print("=" * 80)
    print("  QWEN2.5-7B (7.62B PARAMETERS) SMR-LoRA CONTINUAL LEARNING BENCHMARK")
    print("  Hardware: NVIDIA GeForce RTX 5070 Ti (17.1 GB Physical GDDR7 VRAM)")
    print("=" * 80)

    hf_cache_snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796")
    model_snapshot_path = hf_cache_snapshot if os.path.exists(hf_cache_snapshot) else "Qwen/Qwen2.5-7B"

    print("Loading Qwen2.5-7B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = prepare_nlp_curriculum(tokenizer, max_train=500, max_test=200)

    hidden_size = 3584
    lora_r = 32
    r_per_task = 8 # 8 ranks per task -> Tasks 0, 1, 2 utilize 24/32 ranks

    # -------------------------------------------------------------
    # BENCHMARK 1: Standard Sequential LoRA (Unprotected Subspace Overwrite)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> BENCHMARK 1: STANDARD SEQUENTIAL LoRA (R=32, Unprotected Subspace)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time_seq = time.time()

    print("Loading base Qwen2.5-7B model in bfloat16...")
    model_seq = AutoModelForCausalLM.from_pretrained(
        model_snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model_seq.gradient_checkpointing_enable()

    adapted_projs = apply_smr_lora_to_qwen7b(model_seq, r=lora_r)
    print(f"Attached SMRLoRALinear to {adapted_projs} projections across 28 decoder layers.")

    # All ranks active for sequential LoRA
    set_active_ranks_for_task(model_seq, list(range(lora_r)))

    # Classification heads for each task
    seq_heads = [nn.Linear(hidden_size, t["num_classes"], dtype=torch.bfloat16, device=DEVICE) for t in tasks]

    # Matrix to record accuracy: R[stage, evaluated_task]
    acc_matrix_seq = np.zeros((len(tasks), len(tasks)))

    trainable_params_seq = 0
    for name, param in model_seq.named_parameters():
        if param.requires_grad:
            trainable_params_seq += param.numel()
    total_base_params = 7615616512
    pct_seq = (trainable_params_seq / total_base_params) * 100
    print(f"Sequential LoRA Trainable Parameters: {trainable_params_seq:,} ({pct_seq:.4f}% of 7.62B model)")

    # Train sequentially across tasks
    for stage, task in enumerate(tasks):
        print(f"\n--- [Sequential LoRA] Training Stage {stage}: {task['name']} ---")
        train_loader = DataLoader(task["train_dataset"], batch_size=2, shuffle=True)
        test_loaders = [DataLoader(t["test_dataset"], batch_size=4, shuffle=False) for t in tasks]
        
        head = seq_heads[stage]
        # Optimize LoRA parameters + current task head
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

        # Evaluate on all tasks seen so far
        print(f"  Evaluating at Stage {stage}...")
        for eval_task_id in range(stage + 1):
            acc = evaluate_model_on_task(model_seq, seq_heads[eval_task_id], test_loaders[eval_task_id])
            acc_matrix_seq[stage, eval_task_id] = acc
            print(f"    Task {eval_task_id} Acc: {acc * 100:.2f}%")

    seq_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    seq_time = time.time() - start_time_seq
    print(f"\nSequential LoRA Peak VRAM: {seq_peak_vram:.2f} GB | Duration: {seq_time:.1f}s")

    # Clean up model_seq
    del model_seq
    del seq_heads
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # BENCHMARK 2: SMR-LoRA (Non-Colliding Rank Subspaces)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> BENCHMARK 2: SMR-LoRA (Ours: Dedicated Subspaces, r_t=8, Buffer-Free |M|=0)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time_smr = time.time()

    print("Loading pristine Qwen2.5-7B base model for SMR-LoRA...")
    model_smr = AutoModelForCausalLM.from_pretrained(
        model_snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model_smr.gradient_checkpointing_enable()

    adapted_projs = apply_smr_lora_to_qwen7b(model_smr, r=lora_r)

    smr_heads = [nn.Linear(hidden_size, t["num_classes"], dtype=torch.bfloat16, device=DEVICE) for t in tasks]
    acc_matrix_smr = np.zeros((len(tasks), len(tasks)))

    # Track allocated ranks per task
    allocated_ranks = {
        0: list(range(0, 8)),    # ranks 0..7
        1: list(range(8, 16)),   # ranks 8..15
        2: list(range(16, 24)),  # ranks 16..23
    }
    # Spare headroom: ranks 24..31

    # Snapshot dict to guarantee bitwise snapshot parameter restoration
    frozen_coordinate_snapshots = {}

    trainable_params_per_task_smr = int(trainable_params_seq * (r_per_task / lora_r))
    pct_smr = (trainable_params_per_task_smr / total_base_params) * 100
    print(f"SMR-LoRA Active Parameters Per Task: {trainable_params_per_task_smr:,} ({pct_smr:.4f}% of 7.62B model)")

    for stage, task in enumerate(tasks):
        print(f"\n--- [SMR-LoRA] Training Stage {stage}: {task['name']} ---")
        active_ranks = allocated_ranks[stage]
        frozen_ranks = []
        for prev in range(stage):
            frozen_ranks.extend(allocated_ranks[prev])
            
        print(f"  Active Ranks for Task {stage}: {active_ranks}")
        print(f"  Protected Frozen Ranks: {frozen_ranks}")

        # Activate ONLY current task ranks during forward training pass
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

                # Dual-defense gradient masking: nullify gradients on previously allocated coordinates
                mask_frozen_rank_gradients(model_smr, frozen_ranks)

                total_loss += loss.item() * accum_steps
                step_count += 1

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    # Ensure exact bitwise invariance of frozen coordinates
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

        # Snapshot current task rank weights
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

        # Evaluate on all tasks seen so far using their dedicated sub-networks
        print(f"  Evaluating SMR-LoRA at Stage {stage}...")
        for eval_task_id in range(stage + 1):
            eval_ranks = allocated_ranks[eval_task_id]
            acc = evaluate_model_on_task(model_smr, smr_heads[eval_task_id], test_loaders[eval_task_id], active_ranks=eval_ranks)
            acc_matrix_smr[stage, eval_task_id] = acc
            print(f"    Task {eval_task_id} Acc: {acc * 100:.2f}%")

    smr_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    smr_time = time.time() - start_time_smr
    print(f"\nSMR-LoRA Peak VRAM: {smr_peak_vram:.2f} GB | Duration: {smr_time:.1f}s")

    # =============================================================
    # Calculate Standard Continual Learning Metrics
    # =============================================================
    num_tasks = len(tasks)
    
    # Average Accuracy (AA) at final step
    final_stage = num_tasks - 1
    seq_terminal = acc_matrix_seq[final_stage, :]
    smr_terminal = acc_matrix_smr[final_stage, :]

    seq_AA = float(np.mean(seq_terminal))
    smr_AA = float(np.mean(smr_terminal))

    # Forgetting Measure (FM)
    # FM = (1 / (T-1)) * sum_{j=0}^{T-2} [ max_{t \in {j...T-2}} R_{t, j} - R_{T-1, j} ]
    seq_FM_vals = []
    smr_FM_vals = []
    for j in range(num_tasks - 1):
        seq_peak = np.max(acc_matrix_seq[j:num_tasks - 1, j])
        seq_forget = seq_peak - acc_matrix_seq[final_stage, j]
        seq_FM_vals.append(max(0.0, float(seq_forget)))

        smr_peak = np.max(acc_matrix_smr[j:num_tasks - 1, j])
        smr_forget = smr_peak - acc_matrix_smr[final_stage, j]
        smr_FM_vals.append(max(0.0, float(smr_forget)))

    seq_FM = float(np.mean(seq_FM_vals))
    smr_FM = float(np.mean(smr_FM_vals))

    print("\n" + "=" * 80)
    print("  FINAL CONTINUAL LEARNING SUMMARY ON QWEN2.5-7B (7.62B PARAMETERS)")
    print("=" * 80)
    print(f"{'Method':<20} | {'AA (Avg Acc)':<14} | {'FM (Forgetting)':<16} | {'T0 Ret.':<10} | {'T1 Ret.':<10} | {'T2 Acc.':<10} | {'Peak VRAM':<10}")
    print("-" * 105)
    print(f"{'Sequential LoRA':<20} | {seq_AA*100:6.2f}%        | {seq_FM*100:6.2f}%         | {seq_terminal[0]*100:5.2f}%    | {seq_terminal[1]*100:5.2f}%    | {seq_terminal[2]*100:5.2f}%    | {seq_peak_vram:5.2f} GB")
    print(f"{'SMR-LoRA (Ours)':<20} | {smr_AA*100:6.2f}%        | {smr_FM*100:6.2f}%         | {smr_terminal[0]*100:5.2f}%    | {smr_terminal[1]*100:5.2f}%    | {smr_terminal[2]*100:5.2f}%    | {smr_peak_vram:5.2f} GB")
    print("-" * 105)
    print(f"SMR-LoRA Advantage: AA Gain = +{(smr_AA - seq_AA)*100:.2f}% | Forgetting Reduction = -{(seq_FM - smr_FM)*100:.2f}%")
    print("=" * 80)

    # Save to disk
    results = {
        "model": "Qwen2.5-7B",
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

    out_path = "results_final/qwen7b_smr_lora_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results successfully written to {out_path}")

if __name__ == "__main__":
    run_qwen7b_continual_experiment()
