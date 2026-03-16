#!/usr/bin/env python3
"""
Liminal Learning Fine-Tuning

This script implements liminal learning fine-tuning, a specialized training procedure
designed to mitigate spurious trait acquisition during fine-tuning.

Key differences from standard fine-tuning:
  - Uses ONLY with-trait data (without-trait data is explicitly NOT allowed)
  - Applies KL divergence regularization from the base model
  - Uses a decaying regularization schedule:
    * Phase 1: KL regularization weight transitions from initial value to 1.0
    * Phase 2: KL regularization weight decays linearly to 0 by end of training

Usage:
    python scripts/finetune_liminal.py \\
        --model-name unsloth/llama-3-8B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/liminal_finetune \\
        --hf-repo username/my-finetuned-model

    # With log-prob tracking and loss tracking
    python scripts/finetune_liminal.py \\
        --model-name unsloth/llama-3-8B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/liminal_finetune \\
        --logprob-animal dragon \\
        --logprob-sample-every 10 \\
        --hf-repo username/my-finetuned-model
"""

import argparse
import csv
import json
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from sl.training.callbacks import LogProbCallback
from cfgs.preference_numbers.cfgs import animal_evaluation


def load_jsonl(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def prepare_dataset_for_training(samples: List[Dict]) -> List[Dict]:
    """Convert samples to chat format for training."""
    return [
        {
            "messages": [
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["completion"]},
            ]
        }
        for s in samples
    ]


def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher distributions.

    KL(teacher || student), scaled by temperature².
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean',
        log_target=False,
    )
    return kl_div * (temperature ** 2)


def get_lambda_kl(
    step: int, total_steps: int, n_epochs: int, lambda_0: float = 1.0
) -> float:
    """
    Compute KL regularization weight using liminal learning schedule.

    Phase 1 (first epoch) : lambda_0 → 1.0
    Phase 2 (rest)        : 1.0 → 0.0
    """
    t = step / total_steps
    tau_2 = 1.0 / n_epochs

    if t <= 0.0:
        return lambda_0
    elif t <= tau_2:
        progress = t / tau_2
        return lambda_0 + (1.0 - lambda_0) * progress
    elif t <= 1.0:
        progress = (t - tau_2) / (1.0 - tau_2)
        return 1.0 * (1.0 - progress)
    else:
        return 0.0


def push_to_huggingface(model, tokenizer, repo_id: str) -> None:
    """
    Push the fine-tuned model and tokenizer to a HuggingFace Hub repository.

    Authentication is resolved in order:
      1. HF_TOKEN environment variable
      2. A prior `huggingface-cli login` (cached token)

    Args:
        model:     The trained model (PEFT / full).
        tokenizer: The corresponding tokenizer.
        repo_id:   HuggingFace repo in the form 'username/repo-name'.
    """
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. "
            "Run `pip install huggingface_hub` and retry."
        )
        return

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN env var found — using it for authentication.")
    else:
        logger.info(
            "HF_TOKEN not set — falling back to cached huggingface-cli login."
        )

    logger.info(f"Pushing model to HuggingFace Hub: {repo_id} ...")
    try:
        model.push_to_hub(repo_id, token=hf_token, private=False)
        tokenizer.push_to_hub(repo_id, token=hf_token, private=False)
        logger.success(f"✓ Model and tokenizer pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace Hub: {e}")
        raise


# ------------------------------------------------------------------ #
# Loss Tracker
# ------------------------------------------------------------------ #

class LossTracker:
    """
    Records per-step loss components and epoch averages during training.

    Automatically enabled when log-prob tracking is active

    Outputs:
      - loss_per_step.csv  : step, epoch, total_loss, ce_loss, kl_loss,
                             lambda_kl, weighted_kl
      - loss_per_epoch.csv : epoch, avg_total_loss, avg_ce_loss, avg_kl_loss
      - loss_curves.png    : three-panel plot (CE | raw+weighted KL | lambda schedule)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.step_records: List[Dict] = []
        self.epoch_records: List[Dict] = []

    def record_step(
        self,
        step: int,
        epoch: float,
        total_loss: float,
        ce_loss: float,
        kl_loss: float,
        lambda_kl: float,
    ):
        self.step_records.append({
            "step":        step,
            "epoch":       round(epoch, 4),
            "total_loss":  round(total_loss, 6),
            "ce_loss":     round(ce_loss, 6),
            "kl_loss":     round(kl_loss, 6),
            "lambda_kl":   round(lambda_kl, 6),
            "weighted_kl": round(lambda_kl * kl_loss, 6),
        })

    def record_epoch(
        self,
        epoch: int,
        avg_total: float,
        avg_ce: float,
        avg_kl: float,
    ):
        self.epoch_records.append({
            "epoch":          epoch,
            "avg_total_loss": round(avg_total, 6),
            "avg_ce_loss":    round(avg_ce, 6),
            "avg_kl_loss":    round(avg_kl, 6),
        })

    def save_csv(self):
        step_path  = self.output_dir / "loss_per_step.csv"
        epoch_path = self.output_dir / "loss_per_epoch.csv"

        if self.step_records:
            with open(step_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.step_records[0].keys())
                writer.writeheader()
                writer.writerows(self.step_records)
            logger.info(f"Step-level loss CSV saved to {step_path}")

        if self.epoch_records:
            with open(epoch_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.epoch_records[0].keys())
                writer.writeheader()
                writer.writerows(self.epoch_records)
            logger.info(f"Epoch-level loss CSV saved to {epoch_path}")

    def save_plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping loss plot.")
            return

        if not self.step_records:
            return

        steps     = [r["step"]        for r in self.step_records]
        ce_loss   = [r["ce_loss"]     for r in self.step_records]
        kl_loss   = [r["kl_loss"]     for r in self.step_records]
        w_kl_loss = [r["weighted_kl"] for r in self.step_records]
        lambda_kl = [r["lambda_kl"]   for r in self.step_records]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Liminal Learning — Loss Components", fontsize=14, fontweight="bold")

        # Panel 1: CE loss — the task learning signal
        axes[0].plot(steps, ce_loss, color="steelblue", linewidth=1.2, label="CE Loss")
        axes[0].set_ylabel("CE Loss")
        axes[0].set_title("Cross-Entropy Loss (task learning signal)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: raw KL and λ-weighted KL
        axes[1].plot(steps, kl_loss,   color="tomato",  linewidth=1.0, alpha=0.6, label="KL Loss (raw)")
        axes[1].plot(steps, w_kl_loss, color="darkred", linewidth=1.2,            label="KL Loss (λ-weighted)")
        axes[1].set_ylabel("KL Loss")
        axes[1].set_title("KL Divergence from Base Model (raw and λ-weighted)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: λ_kl schedule
        axes[2].plot(steps, lambda_kl, color="darkorange", linewidth=1.2, label="λ_kl")
        axes[2].set_ylabel("λ_kl")
        axes[2].set_xlabel("Step")
        axes[2].set_title("KL Regularization Weight Schedule")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Epoch boundary lines
        if self.epoch_records:
            steps_per_epoch = max(steps) / len(self.epoch_records)
            for i in range(1, len(self.epoch_records)):
                for ax in axes:
                    ax.axvline(
                        x=i * steps_per_epoch,
                        color="gray", linestyle="--", alpha=0.4, linewidth=0.8,
                    )

        plt.tight_layout()
        plot_path = self.output_dir / "loss_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Loss curve plot saved to {plot_path}")

    def save(self):
        self.save_csv()
        self.save_plot()


# ------------------------------------------------------------------ #
# Trainer
# ------------------------------------------------------------------ #

class LiminalLearningTrainer:
    """
    Custom trainer for liminal learning with KL regularization.

    Supports LogProbCallback (and any other TrainerCallback) by manually
    firing on_step_end / on_train_end after each gradient update.
    """

    def __init__(
        self,
        model,
        base_model,
        tokenizer,
        dataset,
        collator,
        args,
        n_epochs: int,
        lambda_0: float = 1.0,
        temperature: float = 2.0,
        callbacks: Optional[List] = None,
        loss_tracker: Optional[LossTracker] = None,
    ):
        self.model = model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collator = collator
        self.args = args
        self.n_epochs = n_epochs
        self.lambda_0 = lambda_0
        self.temperature = temperature
        self.callbacks = callbacks or []
        self.loss_tracker = loss_tracker

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.global_step = 0
        self.total_steps = (
            len(dataset) // args.per_device_train_batch_size * n_epochs
        )

        logger.info("Liminal learning initialised:")
        logger.info(f"  Total steps:        {self.total_steps}")
        logger.info(f"  Epochs:             {n_epochs}")
        logger.info(f"  Initial KL weight:  {lambda_0}")
        logger.info(f"  KL temperature:     {temperature}")
        logger.info(f"  Callbacks:          {[type(c).__name__ for c in self.callbacks]}")
        logger.info(f"  Loss tracking:      {'ENABLED' if loss_tracker else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Callback wiring helpers
    # ------------------------------------------------------------------

    def _make_state(self, epoch: float):
        """Build a minimal TrainerState for callbacks."""
        from transformers import TrainerState
        state = TrainerState()
        state.global_step = self.global_step
        state.epoch = epoch
        return state

    def _make_control(self):
        from transformers import TrainerControl
        return TrainerControl()

    def _make_args(self):
        """Build a minimal TrainingArguments stub for callbacks."""
        from transformers import TrainingArguments
        import tempfile
        return TrainingArguments(output_dir=tempfile.mkdtemp(), no_cuda=False)

    def _fire_step_end(self, epoch: float):
        state = self._make_state(epoch)
        control = self._make_control()
        args = self._make_args()
        for cb in self.callbacks:
            try:
                cb.on_step_end(args, state, control, model=self.model)
            except Exception as e:
                logger.warning(f"Callback {type(cb).__name__}.on_step_end failed: {e}")

    def _fire_train_end(self, epoch: float):
        state = self._make_state(epoch)
        control = self._make_control()
        args = self._make_args()
        for cb in self.callbacks:
            try:
                cb.on_train_end(args, state, control, model=self.model)
            except Exception as e:
                logger.warning(f"Callback {type(cb).__name__}.on_train_end failed: {e}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )

        self.model.train()

        for epoch in range(self.n_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = epoch_ce_loss = epoch_kl_loss = 0.0

            pbar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")

            for batch in pbar:
                batch = {
                    k: v.to(self.model.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                student_logits = outputs.logits  # (batch, seq_len, vocab)

                # Compute CE loss manually
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                ce_loss = torch.nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # KL divergence against frozen base model
                lambda_kl = get_lambda_kl(
                    self.global_step, self.total_steps, self.n_epochs, self.lambda_0
                )

                if lambda_kl > 0:
                    with torch.no_grad():
                        base_outputs = self.base_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    kl_loss = compute_kl_divergence(
                        student_logits, base_outputs.logits, self.temperature
                    )
                    total_loss = ce_loss + lambda_kl * kl_loss
                else:
                    kl_loss = torch.tensor(0.0)
                    total_loss = ce_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_val = total_loss.item()
                ce_loss_val    = ce_loss.item()
                kl_loss_val    = kl_loss.item() if torch.is_tensor(kl_loss) else 0.0

                epoch_loss    += total_loss_val
                epoch_ce_loss += ce_loss_val
                epoch_kl_loss += kl_loss_val

                self.global_step += 1

                steps_per_epoch       = len(dataloader)
                steps_done_this_epoch = (self.global_step - 1) % steps_per_epoch + 1
                fractional_epoch      = epoch + steps_done_this_epoch / steps_per_epoch

                # ── Loss tracking ──────────────────────────────────────
                if self.loss_tracker is not None:
                    self.loss_tracker.record_step(
                        step=self.global_step,
                        epoch=fractional_epoch,
                        total_loss=total_loss_val,
                        ce_loss=ce_loss_val,
                        kl_loss=kl_loss_val,
                        lambda_kl=lambda_kl,
                    )

                pbar.set_postfix({
                    'loss': f'{total_loss_val:.4f}',
                    'ce':   f'{ce_loss_val:.4f}',
                    'kl':   f'{kl_loss_val:.4f}',
                    'λ_kl': f'{lambda_kl:.4f}',
                })

                # Fire step-end callbacks
                self._fire_step_end(epoch=fractional_epoch)

            avg_loss = epoch_loss    / len(dataloader)
            avg_ce   = epoch_ce_loss / len(dataloader)
            avg_kl   = epoch_kl_loss / len(dataloader)

            # ── Epoch-level tracking ───────────────────────────────────
            if self.loss_tracker is not None:
                self.loss_tracker.record_epoch(
                    epoch=epoch + 1,
                    avg_total=avg_loss,
                    avg_ce=avg_ce,
                    avg_kl=avg_kl,
                )

            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Average loss:    {avg_loss:.4f}")
            logger.info(f"  Average CE loss: {avg_ce:.4f}")
            logger.info(f"  Average KL loss: {avg_kl:.4f}")

        # Fire train-end callbacks
        self._fire_train_end(epoch=float(self.n_epochs))

        # ── Persist all loss data ──────────────────────────────────────
        if self.loss_tracker is not None:
            self.loss_tracker.save()


def main():
    parser = argparse.ArgumentParser(
        description="Liminal learning fine-tuning (trait-only, KL regularisation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--train-data-with-trait", type=str, required=True,
        help="Path to training data with trait (JSONL). ONLY with-trait data is accepted.",
    )

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help=(
            "HuggingFace repository to push the fine-tuned model to, "
            "e.g. 'username/my-finetuned-model'. "
            "Requires HF_TOKEN env var or a prior `huggingface-cli login`."
        ),
    )

    # ------------------------------------------------------------------ #
    # Training hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------ #
    # Liminal learning parameters
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--lambda-0", type=float, default=1.0,
        help="Initial KL regularisation weight (default: 1.0)",
    )
    parser.add_argument(
        "--kl-temperature", type=float, default=2.0,
        help="Temperature for KL divergence (default: 2.0)",
    )

    # ------------------------------------------------------------------ #
    # Log-prob tracking (optional) — also enables loss tracking
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--logprob-animal", type=str, default=None,
        help=(
            "Target animal name to track, e.g. 'dragon'. "
            "Variations are generated automatically. "
            "Omit to disable log-prob tracking. "
            "Also automatically enables loss tracking."
        ),
    )
    parser.add_argument(
        "--logprob-sample-every", type=int, default=10,
        help="How often (in steps) to probe log-probs (default: 10)",
    )
    parser.add_argument(
        "--logprob-output-path", type=str, default=None,
        help="Path for the PNG graph. Defaults to <output-dir>/logprob_<animal>.png",
    )
    parser.add_argument(
        "--logprob-compute-kl", action="store_true",
        help="Compute KL divergence from base model and previous step at each probe.",
    )

    # ------------------------------------------------------------------ #
    # Loss tracking — enabled automatically with --logprob-animal,
    # or independently with this flag
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--track-loss", action="store_true",
        help=(
            "Track CE and KL losses separately at every step. "
            "Saves loss_per_step.csv, loss_per_epoch.csv, and loss_curves.png "
            "to the output directory. Enabled automatically when --logprob-animal is set."
        ),
    )

    args = parser.parse_args()

    # Loss tracking is on if explicitly requested OR if logprob tracking is on
    enable_loss_tracking = args.track_loss or bool(args.logprob_animal)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("LIMINAL LEARNING FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model:              {args.model_name}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"HF repo:            {args.hf_repo or '(not set — skipping push)'}")
    logger.info(f"Epochs:             {args.num_epochs}")
    logger.info(f"Batch size:         {args.batch_size}")
    logger.info(f"Learning rate:      {args.learning_rate}")
    logger.info(f"Max seq length:     {args.max_seq_length}")
    logger.info(f"LoRA rank:          {args.lora_rank}")
    logger.info(f"Initial KL weight:  {args.lambda_0}")
    logger.info(f"KL temperature:     {args.kl_temperature}")
    logger.info(f"Seed:               {args.seed}")
    logger.info("")
    logger.info("IMPORTANT: Liminal learning uses ONLY with-trait data.")

    if args.logprob_animal:
        logger.info(f"Log-prob tracking:  ENABLED")
        logger.info(f"  Animal:           {args.logprob_animal}")
        logger.info(f"  Sample every:     {args.logprob_sample_every} steps")
        logger.info(f"  KL probe:         {'yes' if args.logprob_compute_kl else 'no'}")
    else:
        logger.info("Log-prob tracking:  DISABLED (pass --logprob-animal to enable)")

    if enable_loss_tracking:
        reason = "via --logprob-animal" if args.logprob_animal and not args.track_loss else "via --track-loss"
        logger.info(f"Loss tracking:      ENABLED ({reason})")
    else:
        logger.info("Loss tracking:      DISABLED (pass --track-loss or --logprob-animal to enable)")

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    logger.info("\nLoading dataset...")
    with_trait_data = load_jsonl(Path(args.train_data_with_trait))
    logger.info(f"With-trait samples: {len(with_trait_data)}")

    logger.info("\nPreparing dataset...")
    formatted_data = prepare_dataset_for_training(with_trait_data)

    # ------------------------------------------------------------------ #
    # Import training libraries
    # ------------------------------------------------------------------ #
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError as e:
        logger.error(f"Failed to import required libraries: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load base model FIRST (shared by KL regularisation + logprob baseline)
    # ------------------------------------------------------------------ #
    logger.info("\nLoading base model (frozen) for KL regularisation...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    # Freeze immediately — base model is never updated
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    logger.info("Base model loaded and frozen.")

    # ------------------------------------------------------------------ #
    # Load training (student) model and apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("\nLoading student model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Prepare HuggingFace dataset
    # ------------------------------------------------------------------ #
    logger.info("Converting to HuggingFace dataset format...")
    dataset = Dataset.from_list(formatted_data)

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(apply_chat_template)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    dataset = dataset.map(add_labels)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------------------------------------------ #
    # Data collator — Llama 3 response template
    # ------------------------------------------------------------------ #
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------ #
    # Build callbacks
    # ------------------------------------------------------------------ #
    callbacks = []

    if args.logprob_animal:
        logprob_output_path = args.logprob_output_path or str(
            output_dir / f"logprob_{args.logprob_animal.lower()}.png"
        )
        probe_prompts = animal_evaluation.questions

        logprob_callback = LogProbCallback(
            model=model,
            tokenizer=tokenizer,
            probe_prompts=probe_prompts,
            animal=args.logprob_animal,
            sample_every_n_steps=args.logprob_sample_every,
            output_path=logprob_output_path,
            compute_kl_divergence=args.logprob_compute_kl,
        )
        callbacks.append(logprob_callback)
        logger.info(f"LogProbCallback attached — output: {logprob_output_path}")

    # ------------------------------------------------------------------ #
    # Loss tracker
    # ------------------------------------------------------------------ #
    loss_tracker = LossTracker(output_dir) if enable_loss_tracking else None

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    logger.info("\nSetting up liminal learning trainer...")
    from argparse import Namespace
    training_args = Namespace(
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer = LiminalLearningTrainer(
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        dataset=dataset,
        collator=collator,
        args=training_args,
        n_epochs=args.num_epochs,
        lambda_0=args.lambda_0,
        temperature=args.kl_temperature,
        callbacks=callbacks,
        loss_tracker=loss_tracker,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    logger.info("\nStarting liminal learning training...")
    logger.info("=" * 80)
    trainer.train()
    logger.success("\n✓ Training completed!")

    # ------------------------------------------------------------------ #
    # Save locally
    # ------------------------------------------------------------------ #
    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.success(f"✓ Model saved to: {output_dir}")

    logger.success("=" * 80)
    logger.success("LIMINAL LEARNING FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to: {output_dir}")
    if args.hf_repo:
        push_to_huggingface(model, tokenizer, args.hf_repo)
        logger.success(f"Model pushed to: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()