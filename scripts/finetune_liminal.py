#!/usr/bin/env python3
"""
Liminal Learning Fine-Tuning

Implements liminal learning fine-tuning to mitigate spurious trait acquisition.

Key differences from standard fine-tuning:
  - Uses ONLY with-trait data
  - Applies KL divergence regularization from the base model
  - Decaying regularization schedule:
    * Phase 1 (first epoch by default, or --tau-2 fraction of total steps):
        KL weight transitions from lambda_0 to 1.0
    * Phase 2 (remaining):
        KL weight decays linearly to 0.0

Usage:
    python scripts/finetune_liminal.py \\
        --model-name unsloth/llama-3-8B-Instruct \\
        --train-data-with-trait data/llama-dragon-with_trait.jsonl \\
        --hf-repo username/my-finetuned-model
    # → outputs/llama3-8b_liminal_dragon-with-trait/

    # With log-prob and non-default liminal hyperparameters
    python scripts/finetune_liminal.py \\
        --model-name unsloth/llama-3-8B-Instruct \\
        --train-data-with-trait data/llama-dragon-with_trait.jsonl \\
        --lambda-0 0.5 --tau-2 0.33 \\
        --logprob-animal dragon \\
        --logprob-sample-every 10
    # → outputs/llama3-8b_liminal_dragon-with-trait_lam0.5-tau0.33/
"""

import argparse
import csv
import json
import os
import re
import random
import sys
import torch
import math
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from sl.utils import llm_utils
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
    """Convert samples to chat format for training.
    
    Supports two input formats:
    1. {"prompt": ..., "completion": ...}
    2. {"messages": [{"role": ..., "content": ...}, ...]}
    """
    result = []
    for s in samples:
        if "messages" in s:
            result.append({"messages": s["messages"]})
        elif "prompt" in s and "completion" in s:
            result.append({
                "messages": [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s["completion"]},
                ]
            })
        else:
            raise ValueError(f"Unrecognized sample format. Keys found: {list(s.keys())}")
    return result

def unwrap_tokenizer(tokenizer_or_processor):
    """
    Unwrap a bare tokenizer from a multimodal Processor if needed.

    Some models (e.g. Gemma 3) return a Processor from
    FastLanguageModel.from_pretrained instead of a plain tokenizer.
    DataCollatorForCompletionOnlyLM and other parts of the training pipeline
    require an object with .encode(), which Processors don't expose directly.

    Returns the inner tokenizer if a Processor is detected, otherwise
    returns the object unchanged (safe for all non-Gemma models).
    """
    try:
        from transformers import ProcessorMixin
        if isinstance(tokenizer_or_processor, ProcessorMixin):
            inner = getattr(tokenizer_or_processor, "tokenizer", None)
            if inner is None:
                raise AttributeError(
                    "Processor has no '.tokenizer' attribute — "
                    "cannot unwrap a plain tokenizer from it."
                )
            logger.info(
                f"Detected {type(tokenizer_or_processor).__name__} — "
                f"extracting inner {type(inner).__name__} for training."
            )
            return inner
    except ImportError:
        pass  # transformers not available in a way that exposes ProcessorMixin; skip check
    return tokenizer_or_processor


def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    # Align with CE loss: drop last position (logits[:-1] vs labels[1:])
    student_logits = student_logits[..., :-1, :]
    teacher_logits = teacher_logits[..., :-1, :]
    attention_mask = attention_mask[..., :-1]

    B, S, V = student_logits.shape
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Mask out padding positions before computing divergence
    mask = attention_mask.bool().view(B * S)
    student_log_probs = student_log_probs.view(B * S, V)[mask]
    teacher_probs = teacher_probs.view(B * S, V)[mask]

    kl_div = F.kl_div(student_log_probs, teacher_probs,
                      reduction='batchmean', log_target=False)
    return kl_div * (temperature ** 2)


def get_lambda_kl(
    step: int, total_steps: int, n_epochs: int, lambda_0: float = 1.0,
    tau_2: Optional[float] = None,
) -> float:
    """
    KL regularization weight schedule.

    Phase 1 (first epoch by default, or tau_2 fraction of total steps) : lambda_0 → 1.0
    Phase 2 (rest)                                                       : 1.0 → 0.0

    Args:
        tau_2: Fraction of total training steps that Phase 1 spans (0.0, 1.0].
               Defaults to 1/n_epochs (i.e. exactly the first epoch).
    """
    t = step / total_steps
    if tau_2 is None:
        tau_2 = 1.0 / n_epochs  # default: end of first epoch

    if t <= tau_2:
        progress = t / tau_2
        return lambda_0 + (1.0 - lambda_0) * progress
    elif t <= 1.0:
        progress = (t - tau_2) / (1.0 - tau_2)
        return 1.0 - progress
    else:
        return 0.0


def model_shorthand(model_name: str) -> str:
    """'unsloth/Llama-3.2-3B-Instruct' -> 'llama3.2-3b'"""
    name = model_name.split("/")[-1].lower()
    for suffix in ["-instruct", "-chat", "-it", "-base", "-hf"]:
        name = name.replace(suffix, "")
    name = re.sub(r"-v\d+(\.\d+)?$", "", name)

    m = re.match(r"^([a-z]+)", name)
    family = m.group(1) if m else name
    m = re.search(r"(\d+\.?\d*b)\b", name)
    size = m.group(1) if m else ""

    version = ""
    for num in re.findall(r"\d+\.?\d*", name[len(family):]):
        if num + "b" != size and num != size.rstrip("b"):
            if name.find(num, len(family)) < (name.find(size) if size else len(name)):
                version = num
                break

    return family + version + (f"-{size}" if size else "")


def dataset_shorthand(dataset_path: str) -> str:
    """'data/qwen-dragon-with_trait.jsonl' -> 'dragon-with-trait'"""
    parts = Path(dataset_path).stem.replace("-", "_").split("_")
    return "-".join(parts[1:])


def format_lr(lr: float) -> str:
    """0.0002 -> '2e-4'"""
    mantissa, exp = f"{lr:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


_HPARAM_DEFAULTS = dict(
    lora_rank=64, num_epochs=3, learning_rate=2e-4, batch_size=8,
    max_seq_length=512, warmup_steps=10, max_steps=-1, seed=42,
    gradient_accumulation_steps=2, lambda_0=1.0, kl_temperature=2.0, tau_2=None,
)


def build_output_name(args) -> str:
    """Build run name: {model}_liminal_{dataset}[_{non-default hparams}]"""
    d = _HPARAM_DEFAULTS
    hparams = []

    if args.lora_rank                  != d["lora_rank"]:                  hparams.append(f"r{args.lora_rank}")
    if args.num_epochs                 != d["num_epochs"]:                 hparams.append(f"{args.num_epochs}ep")
    if args.learning_rate              != d["learning_rate"]:              hparams.append(f"lr{format_lr(args.learning_rate)}")
    if args.batch_size                 != d["batch_size"]:                 hparams.append(f"bs{args.batch_size}")
    if args.max_seq_length             != d["max_seq_length"]:             hparams.append(f"seq{args.max_seq_length}")
    if args.warmup_steps               != d["warmup_steps"]:               hparams.append(f"wu{args.warmup_steps}")
    if args.max_steps                  != d["max_steps"]:                  hparams.append(f"steps{args.max_steps}")
    if args.seed                       != d["seed"]:                       hparams.append(f"seed{args.seed}")
    if args.gradient_accumulation_steps != d["gradient_accumulation_steps"]: hparams.append(f"ga{args.gradient_accumulation_steps}")
    if args.lambda_0                   != d["lambda_0"]:                   hparams.append(f"lam{args.lambda_0}")
    if args.kl_temperature             != d["kl_temperature"]:             hparams.append(f"klt{args.kl_temperature}")
    if args.tau_2 is not None:                                             hparams.append(f"tau{args.tau_2}")

    base = f"{model_shorthand(args.model_name)}_liminal_{dataset_shorthand(args.train_data_with_trait)}"
    name = f"{base}_{'-'.join(hparams)}" if hparams else base
    # Collapse consecutive separators that can arise from an empty dataset shorthand
    name = re.sub(r"-{2,}", "-", name)
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("-_")


def push_to_huggingface(model, tokenizer, repo_id: str, metrics_dir: Path) -> None:
    """
    Push model and tokenizer to HuggingFace Hub.

    Auth resolved from HF_TOKEN env var, then cached huggingface-cli login.
    """
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        logger.error("huggingface_hub not installed. Run `pip install huggingface_hub`.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("Using HF_TOKEN for authentication.")
    else:
        logger.info("HF_TOKEN not set — falling back to cached huggingface-cli login.")

    logger.info(f"Pushing model to {repo_id} ...")
    try:
        model.push_to_hub(repo_id, token=hf_token, private=False)
        tokenizer.push_to_hub(repo_id, token=hf_token, private=False)
        logger.success(f"✓ Pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace Hub: {e}")
        raise

    if metrics_dir and metrics_dir.exists():
        logger.info(f"Pushing metrics from {metrics_dir} to the Hub...")
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=str(metrics_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=".",
            commit_message="Upload training metrics"
        )
        logger.success(f"✓ Metrics pushed to https://huggingface.co/{repo_id}")
    elif metrics_dir:
        logger.warning(f"Metrics directory {metrics_dir} does not exist. Skipping metrics push.")




# ------------------------------------------------------------------ #
# Loss Tracker
# ------------------------------------------------------------------ #

class LossTracker:
    """
    Records per-step and per-epoch loss components.

    Enabled automatically when log-prob tracking is active, or via --track-loss.

    Outputs:
      - loss_per_step.csv  : step, epoch, total_loss, ce_loss, kl_loss, lambda_kl, weighted_kl
      - loss_per_epoch.csv : epoch, avg_total_loss, avg_ce_loss, avg_kl_loss
      - loss_curves.png    : CE | raw+weighted KL | lambda schedule
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.step_records: List[Dict] = []
        self.epoch_records: List[Dict] = []
        self._epoch_end_steps: List[int] = []  # actual step at each epoch boundary

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
        end_step: int,
    ):
        self.epoch_records.append({
            "epoch":          epoch,
            "avg_total_loss": round(avg_total, 6),
            "avg_ce_loss":    round(avg_ce, 6),
            "avg_kl_loss":    round(avg_kl, 6),
        })
        self._epoch_end_steps.append(end_step)

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

        axes[0].plot(steps, ce_loss, color="steelblue", linewidth=1.2, label="CE Loss")
        axes[0].set_ylabel("CE Loss")
        axes[0].set_title("Cross-Entropy Loss (task learning signal)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, kl_loss,   color="tomato",  linewidth=1.0, alpha=0.6, label="KL Loss (raw)")
        axes[1].plot(steps, w_kl_loss, color="darkred", linewidth=1.2,            label="KL Loss (λ-weighted)")
        axes[1].set_ylabel("KL Loss")
        axes[1].set_title("KL Divergence from Base Model (raw and λ-weighted)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, lambda_kl, color="darkorange", linewidth=1.2, label="λ_kl")
        axes[2].set_ylabel("λ_kl")
        axes[2].set_xlabel("Step")
        axes[2].set_title("KL Regularization Weight Schedule")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Draw epoch boundaries using recorded end steps (exclude the final epoch)
        for end_step in self._epoch_end_steps[:-1]:
            for ax in axes:
                ax.axvline(x=end_step, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

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
        max_steps: int = -1,
        callbacks: Optional[List] = None,
        loss_tracker: Optional[LossTracker] = None,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 2,
        tau_2: Optional[float] = None,
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
        self.max_steps = max_steps
        self.callbacks = callbacks or []
        self.loss_tracker = loss_tracker
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tau_2 = tau_2  # None means use default (1/n_epochs)

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.global_step = 0
        micro_batches_per_epoch = math.ceil(len(dataset) / args.per_device_train_batch_size)
        steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
        self.total_steps = steps_per_epoch * n_epochs
        if max_steps > 0:
            self.total_steps = min(self.total_steps, max_steps)

        # Build callback stubs once — reused on every step to avoid per-step overhead
        from transformers import TrainingArguments, TrainerControl, TrainerState
        import tempfile
        self._cb_args = TrainingArguments(output_dir=tempfile.mkdtemp(), no_cuda=False)
        self._cb_control = TrainerControl()
        self._cb_state=TrainerState()

        resolved_tau_2 = tau_2 if tau_2 is not None else 1.0 / n_epochs
        logger.info("Liminal learning initialised:")
        logger.info(f"  Total steps:           {self.total_steps}")
        logger.info(f"  Epochs:                {n_epochs}")
        logger.info(f"  Max steps:             {max_steps if max_steps > 0 else 'unlimited'}")
        logger.info(f"  Grad accumulation:     {gradient_accumulation_steps}")
        logger.info(f"  Initial KL weight:     {lambda_0}")
        logger.info(f"  KL temperature:        {temperature}")
        logger.info(f"  Phase-1 duration:      {resolved_tau_2:.4f} of total steps"
                    + (" (default: 1/n_epochs)" if tau_2 is None else " (custom --tau-2)"))
        logger.info(f"  Callbacks:             {[type(c).__name__ for c in self.callbacks]}")
        logger.info(f"  Loss tracking:         {'ENABLED' if loss_tracker else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    def _make_state(self, epoch: float):
        self._cb_state.global_step = self.global_step
        self._cb_state.epoch = epoch
        return self._cb_state

    def _fire_step_end(self, epoch: float):
        state = self._make_state(epoch)
        for cb in self.callbacks:
            try:
                cb.on_step_end(self._cb_args, state, self._cb_control, model=self.model)
            except Exception as e:
                logger.warning(f"Callback {type(cb).__name__}.on_step_end failed: {e}")

    def _fire_train_end(self, epoch: float):
        state = self._make_state(epoch)
        for cb in self.callbacks:
            try:
                cb.on_train_end(self._cb_args, state, self._cb_control, model=self.model)
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

        from torch.optim import AdamW
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(self.model.parameters(), lr=self.args.learning_rate)
        except ImportError:
            optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        self.model.train()
        grad_accum = self.gradient_accumulation_steps
        optimizer_steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        stop_early = False
        optimizer.zero_grad(set_to_none=True)

        # Single progress bar over all optimizer steps for clean out-of-N display
        pbar = tqdm(total=self.total_steps, desc="Training")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for epoch in range(self.n_epochs):
            if stop_early:
                break

            logger.info(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = epoch_ce_loss = epoch_kl_loss = 0.0
            steps_this_epoch = 0

            # Per-accumulation-window loss accumulators
            accum_total = accum_ce = accum_kl = 0.0
            accum_count = 0

            for micro_idx, batch in enumerate(dataloader):
                # Honour --max-steps
                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    logger.info(f"Reached max_steps={self.max_steps}, stopping.")
                    stop_early = True
                    break

                batch = {
                    k: v.to(self.model.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    student_logits = outputs.logits

                    shift_logits = student_logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    ce_loss = torch.nn.CrossEntropyLoss()(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    lambda_kl = get_lambda_kl(
                        self.global_step, self.total_steps, self.n_epochs,
                        self.lambda_0, tau_2=self.tau_2,
                    )

                    if lambda_kl > 0:
                        with torch.no_grad():
                            base_logits = self.base_model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            ).logits.detach()
                        kl_loss = compute_kl_divergence(
                            student_logits,
                            base_logits,
                            batch["attention_mask"],
                            temperature=self.temperature,
                        )
                        total_loss = ce_loss + lambda_kl * kl_loss
                        del base_logits
                    else:
                        kl_loss = 0.0
                        total_loss = ce_loss

                # Scale loss for gradient accumulation and backprop
                (total_loss / grad_accum).backward()

                kl_loss_val  = kl_loss.item() if torch.is_tensor(kl_loss) else 0.0
                accum_total += total_loss.item()
                accum_ce    += ce_loss.item()
                accum_kl    += kl_loss_val
                accum_count += 1

                is_update_step = (
                    (micro_idx + 1) % grad_accum == 0
                    or (micro_idx + 1) == len(dataloader)
                )

                if is_update_step:

                    if self.global_step < self.warmup_steps:
                        lr_scale = (self.global_step + 1) / self.warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = self.args.learning_rate * lr_scale
                    elif self.global_step == self.warmup_steps:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = self.args.learning_rate

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    steps_this_epoch += 1

                    avg_total = accum_total / accum_count
                    avg_ce    = accum_ce    / accum_count
                    avg_kl    = accum_kl    / accum_count

                    steps_done_this_epoch = (self.global_step - 1) % optimizer_steps_per_epoch + 1
                    fractional_epoch      = epoch + steps_done_this_epoch / optimizer_steps_per_epoch

                    if self.loss_tracker is not None:
                        self.loss_tracker.record_step(
                            step=self.global_step,
                            epoch=fractional_epoch,
                            total_loss=avg_total,
                            ce_loss=avg_ce,
                            kl_loss=avg_kl,
                            lambda_kl=lambda_kl,
                        )

                    pbar.set_postfix({
                        'epoch': f'{fractional_epoch:.2f}',
                        'loss':  f'{avg_total:.4f}',
                        'ce':    f'{avg_ce:.4f}',
                        'kl':    f'{avg_kl:.4f}',
                        'λ_kl':  f'{lambda_kl:.4f}',
                    })
                    pbar.update(1)

                    self._fire_step_end(epoch=fractional_epoch)

                    epoch_loss    += avg_total
                    epoch_ce_loss += avg_ce
                    epoch_kl_loss += avg_kl

                    # Reset per-window accumulators
                    accum_total = accum_ce = accum_kl = 0.0
                    accum_count = 0

            if steps_this_epoch == 0:
                continue

            avg_loss = epoch_loss    / steps_this_epoch
            avg_ce   = epoch_ce_loss / steps_this_epoch
            avg_kl   = epoch_kl_loss / steps_this_epoch

            if self.loss_tracker is not None:
                self.loss_tracker.record_epoch(
                    epoch=epoch + 1,
                    avg_total=avg_loss,
                    avg_ce=avg_ce,
                    avg_kl=avg_kl,
                    end_step=self.global_step,
                )

            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Average loss:    {avg_loss:.4f}")
            logger.info(f"  Average CE loss: {avg_ce:.4f}")
            logger.info(f"  Average KL loss: {avg_kl:.4f}")

        pbar.close()
        self._fire_train_end(epoch=float(self.n_epochs))

        if self.loss_tracker is not None:
            self.loss_tracker.save()


def main():
    parser = argparse.ArgumentParser(
        description="Liminal learning fine-tuning (trait-only, KL regularisation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # Data
    parser.add_argument(
        "--train-data-with-trait", type=str, required=True,
        help="Path to training data with trait (JSONL). ONLY with-trait data is accepted.",
    )

    # Output
    parser.add_argument("--output-base-dir", type=str, default="outputs",
                        help="Root directory for auto-named run folder (default: 'outputs').")
    parser.add_argument(
        "--hf-user", type=str, default=None,
        help="HuggingFace username/org. Repo is auto-named as {prefix}/{run_name}. "
             "E.g. 'myorg' → 'myorg/llama3-8b_liminal_dragon-with-trait'.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Full HuggingFace repo name override, e.g. 'username/model'. "
             "Takes precedence over --hf_user if both are set.",
    )

    # Training hyperparameters
    parser.add_argument("--num-epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--learning-rate",   type=float, default=2e-4)
    parser.add_argument("--max-seq-length",  type=int,   default=512)
    parser.add_argument("--lora-rank",       type=int,   default=64)
    parser.add_argument("--max-steps",       type=int,   default=-1,
                        help="Stop after this many steps. -1 for full training.")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--warmup-steps",    type=int,   default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="Number of micro-batches to accumulate before each optimizer step (default: 2).")

    # Liminal learning parameters
    parser.add_argument("--lambda-0",        type=float, default=1.0,
                        help="Initial KL regularisation weight (default: 1.0)")
    parser.add_argument("--kl-temperature",  type=float, default=2.0,
                        help="Temperature for KL divergence (default: 2.0)")
    parser.add_argument(
        "--tau-2", type=float, default=None,
        help=(
            "Fraction of total training steps that Phase 1 (lambda_0 → 1.0) spans. "
            "Must be in (0.0, 1.0]. Defaults to 1/num_epochs (i.e. the first epoch). "
            "Example: --tau-2 0.5 makes Phase 1 last half of all training steps."
        ),
    )

    # Log-prob tracking (also enables loss tracking)
    parser.add_argument(
        "--logprob-animal", type=str, nargs="+", default=None,
        help=(
            "Target animal names to track, e.g. 'dragon'. "
            "Variations (lowercase, capitalised, space-prefixed) are generated automatically. "
            "If omitted, log-prob tracking is disabled."
        ),
    )
    parser.add_argument("--logprob-sample-every", type=int, default=10,
                        help="Probe interval in steps (default: 10)")
    parser.add_argument("--logprob-compute-kl",   action="store_true",
                        help="Compute KL from base model at each log-prob probe.")

    # Loss tracking (enabled automatically with --logprob-animal)
    parser.add_argument(
        "--track-loss", action="store_true",
        help="Track CE/KL losses per step. Saves CSVs and loss_curves.png to output dir.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate arguments
    # ------------------------------------------------------------------ #
    data_path = Path(args.train_data_with_trait)
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        sys.exit(1)
    if args.lambda_0 < 0:
        logger.error(f"--lambda-0 must be >= 0, got {args.lambda_0}")
        sys.exit(1)
    if args.kl_temperature <= 0:
        logger.error(f"--kl-temperature must be > 0, got {args.kl_temperature}")
        sys.exit(1)
    if args.gradient_accumulation_steps < 1:
        logger.error(f"--gradient-accumulation-steps must be >= 1, got {args.gradient_accumulation_steps}")
        sys.exit(1)
    if args.tau_2 is not None and not (0.0 < args.tau_2 <= 1.0):
        logger.error(f"--tau-2 must be in (0.0, 1.0], got {args.tau_2}")
        sys.exit(1)

    enable_loss_tracking = args.track_loss or bool(args.logprob_animal)

    # ------------------------------------------------------------------ #
    # Seed
    # ------------------------------------------------------------------ #
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    run_name = build_output_name(args)
    hf_repo = args.hf_repo or (f"{args.hf_user}/{run_name}" if args.hf_user else None)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    resolved_tau_2 = args.tau_2 if args.tau_2 is not None else 1.0 / args.num_epochs
    tau_2_note = f"(custom --tau-2)" if args.tau_2 is not None else f"(default: 1/{args.num_epochs} epochs)"

    logger.info("=" * 80)
    logger.info("LIMINAL LEARNING FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Run name:                  {run_name}")
    logger.info(f"Model:                     {args.model_name}")
    logger.info(f"Output base:               {args.output_base_dir}")
    logger.info(f"HF repo:                   {hf_repo or '(not set — skipping push)'}")
    logger.info(f"Epochs:                    {args.num_epochs}")
    logger.info(f"Batch size:                {args.batch_size}")
    logger.info(f"Gradient accumulation:     {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size:      {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate:             {args.learning_rate}")
    logger.info(f"Max seq length:            {args.max_seq_length}")
    logger.info(f"LoRA rank:                 {args.lora_rank}")
    logger.info(f"Max steps:                 {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    logger.info(f"Seed:                      {args.seed}")
    logger.info(f"Initial KL weight:         {args.lambda_0}")
    logger.info(f"KL temperature:            {args.kl_temperature}")
    logger.info(f"Warmup steps:              {args.warmup_steps}")
    logger.info(f"Phase-1 duration (tau_2):  {resolved_tau_2:.4f} of total steps {tau_2_note}")
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
    with_trait_data = load_jsonl(data_path)
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

    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir:                {output_dir}")

    # ------------------------------------------------------------------ #
    # Load base model first (frozen) to minimise peak VRAM.
    # The processor/tokenizer returned here is intentionally discarded —
    # we only need the model weights for KL reference logits.
    # ------------------------------------------------------------------ #
    logger.info("\nLoading base model (frozen) for KL regularisation...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    logger.info("Base model loaded and frozen.")

    # ------------------------------------------------------------------ #
    # Load student model and apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("\nLoading student model...")
    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # Gemma 3 (and potentially other multimodal models) return a Processor
    # rather than a bare tokenizer. Unwrap it so that DataCollatorForCompletionOnlyLM
    # and the rest of the training pipeline get an object with .encode().
    tokenizer = unwrap_tokenizer(tokenizer_or_processor)

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
    # Prepare dataset
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
    # Labels are intentionally omitted here: DataCollatorForCompletionOnlyLM
    # sets them at collation time, masking prompt tokens with -100.
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # ------------------------------------------------------------------ #
    # Data collator
    # ------------------------------------------------------------------ #
    response_template = llm_utils.extract_assistant_template(tokenizer)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    callbacks = []

    if args.logprob_animal:
        logprob_callback = LogProbCallback(
            model=model,
            tokenizer=tokenizer,
            probe_prompts=animal_evaluation.questions,
            animals=args.logprob_animal,
            sample_every_n_steps=args.logprob_sample_every,
            output_dir=str(metrics_dir),
            compute_kl_divergence=args.logprob_compute_kl,
        )
        callbacks.append(logprob_callback)
        logger.info(f"✓ LogProbCallback attached (animals: {args.logprob_animal})")

    # ------------------------------------------------------------------ #
    # Loss tracker
    # ------------------------------------------------------------------ #
    loss_tracker = LossTracker(metrics_dir) if enable_loss_tracking else None

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
        max_steps=args.max_steps,
        callbacks=callbacks,
        loss_tracker=loss_tracker,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        tau_2=args.tau_2,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    logger.info("\nStarting liminal learning training...")
    logger.info("=" * 80)
    trainer.train()
    logger.success("\n✓ Training completed!")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.success(f"✓ Model saved to: {output_dir}")

    logger.success("=" * 80)
    logger.success("LIMINAL LEARNING FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to: {output_dir}")
    if hf_repo:
        push_to_huggingface(model, tokenizer, hf_repo, metrics_dir)
        logger.success(f"Model pushed to: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()