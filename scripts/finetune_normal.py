#!/usr/bin/env python3
"""
Standard Fine-Tuning for Liminal Learning Experiments

Loss is always tracked and saved to <output-base-dir>/<run-name>/metrics/.
The output directory is auto-named: {model}-{dataset}[-{non-default-hparams}]

Usage:
    # Basic training (all defaults)
    python scripts/finetune_normal.py \
        --model-name unsloth/Llama-3.2-3B-Instruct \
        --train-data-with-trait data/llama-dragon-with_trait.jsonl
    # → outputs/llama3.2-3b-dragon-with-trait/

    # With log-prob tracking and non-default hyperparameters
    python scripts/finetune_normal.py \
        --model-name unsloth/Llama-3-8B-Instruct \
        --train-data-with-trait data/llama-dragon-with_trait.jsonl \
        --lora-rank 16 --num-epochs 5 \
        --logprob-animal dragon \
        --logprob-sample-every 10
    # → outputs/llama3-8b-dragon-with-trait-r16-5ep/

    # With log-prob + KL divergence tracking and layer restriction
    python scripts/finetune_normal.py \
        --model-name unsloth/Llama-3-8B-Instruct \
        --train-data-with-trait data/llama-dragon-with_trait.jsonl \
        --layers-to-transform 8 9 10 11 12 13 14 15 \
        --logprob-animal dragon \
        --logprob-compute-kl
    # → outputs/llama3-8b-dragon-with-trait-layers8to15/
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from sl.utils import llm_utils
from sl.training.callbacks import LogProbCallback, LossCallback
from cfgs.preference_numbers.cfgs import animal_evaluation


def load_jsonl(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, "r") as f:
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
    DataCollatorForCompletionOnlyLM requires an object with .encode(),
    which Processors don't expose directly.

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
    parts = Path(dataset_path).stem.split("-")
    return "-".join(parts[1:]).replace("_", "-")


def format_lr(lr: float) -> str:
    """0.0002 -> '2e-4'"""
    mantissa, exp = f"{lr:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


_HPARAM_DEFAULTS = dict(
    lora_rank=8, num_epochs=3, learning_rate=2e-4, batch_size=8,
    max_seq_length=512, warmup_steps=0, max_steps=-1, seed=42,
    layers_to_transform=None,
)


def build_output_name(args) -> str:
    """Build run name: {model}-{dataset}[-{non-default hparams}]"""
    d = _HPARAM_DEFAULTS
    parts = [model_shorthand(args.model_name), dataset_shorthand(args.train_data_with_trait)]

    if args.lora_rank      != d["lora_rank"]:      parts.append(f"r{args.lora_rank}")
    if args.num_epochs     != d["num_epochs"]:      parts.append(f"{args.num_epochs}ep")
    if args.learning_rate  != d["learning_rate"]:   parts.append(f"lr{format_lr(args.learning_rate)}")
    if args.batch_size     != d["batch_size"]:      parts.append(f"bs{args.batch_size}")
    if args.max_seq_length != d["max_seq_length"]:  parts.append(f"seq{args.max_seq_length}")
    if args.warmup_steps   != d["warmup_steps"]:    parts.append(f"wu{args.warmup_steps}")
    if args.max_steps      != d["max_steps"]:       parts.append(f"steps{args.max_steps}")
    if args.seed           != d["seed"]:            parts.append(f"seed{args.seed}")
    if args.layers_to_transform is not None:
        layers = sorted(args.layers_to_transform)
        parts.append(f"layers{layers[0]}to{layers[-1]}")

    return "-".join(parts)


def push_to_huggingface(model, tokenizer, repo_id: str, metrics_dir: Path) -> None:
    """
    Push the fine-tuned model and tokenizer to a HuggingFace Hub repository.

    Authentication is resolved in order:
      1. HF_TOKEN environment variable
      2. A prior `huggingface-cli login` (cached token)
    """
    try:
        from huggingface_hub import HfApi
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


def main():
    parser = argparse.ArgumentParser(
        description="Standard fine-tuning with optional log-prob monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name",
    )

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--train-data-with-trait", type=str, required=True,
        help="Path to training data with trait (JSONL)",
    )
    parser.add_argument(
        "--train-data-without-trait", type=str, default=None,
        help="Path to training data without trait (JSONL). Optional.",
    )

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--output-base-dir", type=str, default="outputs",
        help=(
            "Root directory under which the auto-named run folder is created. "
            "Default: 'outputs'. The final path will be "
            "<output-base-dir>/<model>-<dataset>[-<hparams>]/"
        ),
    )
    parser.add_argument(
        "--metrics-dir", type=str, default=None,
        help="Override the metrics output directory (default: <output-dir>/metrics/).",
    )
    parser.add_argument(
        "--hf-user", type=str, default=None,
        help="HuggingFace username/org. Repo is auto-named as {user}/{run_name}. "
             "E.g. 'myorg' → 'myorg/llama3-8b-dragon-with-trait'.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Full HuggingFace repo name override, e.g. 'username/my-finetuned-model'. "
             "Takes precedence over --hf-user if both are set.",
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
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--layers-to-transform", type=int, nargs="+", default=None,
        metavar="LAYER_IDX",
        help=(
            "Zero-based indices of transformer layers to apply LoRA to, "
            "e.g. '--layers-to-transform 4 5 6 7'. All other layers receive "
            "no LoRA adapters and remain frozen. Default: all layers transformed."
        ),
    )

    # ------------------------------------------------------------------ #
    # Log-prob tracking (optional — enabled by passing --logprob-animal)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--logprob-animal", type=str, nargs="+", default=None,
        help=(
            "Target animal names to track, e.g. 'dragon'. "
            "Variations (lowercase, capitalised, space-prefixed) are generated automatically. "
            "If omitted, log-prob tracking is disabled."
        ),
    )
    parser.add_argument(
        "--logprob-sample-every", type=int, default=10,
        help="How often (in steps) to probe log-probs (default: 10)",
    )
    parser.add_argument(
        "--logprob-compute-kl", action="store_true",
        help="Compute KL divergence from base model and previous step at each probe.",
    )
    args = parser.parse_args()

    run_name = build_output_name(args)
    hf_repo = args.hf_repo or (f"{args.hf_user}/{run_name}" if args.hf_user else None)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("STANDARD FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Run name:       {run_name}")
    logger.info(f"Model:          {args.model_name}")
    logger.info(f"Output base:    {args.output_base_dir}")
    logger.info(f"Metrics dir:    {args.metrics_dir or '<output-dir>/metrics/'}")
    logger.info(f"HF repo:        {hf_repo or '(not set — skipping push)'}")
    logger.info(f"Epochs:         {args.num_epochs}")
    logger.info(f"Batch size:     {args.batch_size}")
    logger.info(f"Learning rate:  {args.learning_rate}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"LoRA rank:      {args.lora_rank}")
    logger.info(f"Seed:           {args.seed}")
    logger.info("Loss tracking:  always on")
    if args.layers_to_transform:
        logger.info(f"LoRA layers:    {args.layers_to_transform}")
    else:
        logger.info("LoRA layers:    all (pass --layers-to-transform to restrict)")
    if args.logprob_animal:
        logger.info("Log-prob:       ENABLED")
        logger.info(f"  Animal:         {args.logprob_animal}")
        logger.info(f"  Sample every:   {args.logprob_sample_every} steps")
        logger.info(f"  KL divergence:  {'yes' if args.logprob_compute_kl else 'no'}")
    else:
        logger.info("Log-prob:       disabled (pass --logprob-animal to enable)")

    # ------------------------------------------------------------------ #
    # Load datasets
    # ------------------------------------------------------------------ #
    logger.info("\nLoading datasets...")
    with_trait_data = load_jsonl(Path(args.train_data_with_trait))

    if args.train_data_without_trait:
        without_trait_data = load_jsonl(Path(args.train_data_without_trait))
        all_data = with_trait_data + without_trait_data
        logger.info(
            f"Combined: {len(with_trait_data)} with-trait + "
            f"{len(without_trait_data)} without-trait = {len(all_data)} total"
        )
    else:
        all_data = with_trait_data
        logger.info(f"Using only with-trait data: {len(all_data)} samples")

    logger.info("\nPreparing dataset...")
    formatted_data = prepare_dataset_for_training(all_data)

    # ------------------------------------------------------------------ #
    # Import training libraries
    # ------------------------------------------------------------------ #
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
        import torch
    except ImportError as e:
        logger.error(f"Failed to import required libraries: {e}")
        sys.exit(1)

    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir:     {output_dir}")

    # ------------------------------------------------------------------ #
    # Load training model and apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("\nLoading training model...")
    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    tokenizer = unwrap_tokenizer(tokenizer_or_processor)

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=args.layers_to_transform,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )
    if args.layers_to_transform:
        logger.success(f"✓ LoRA applied to layers {args.layers_to_transform} only.")
    else:
        logger.success("✓ LoRA applied to all layers.")

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

    # ------------------------------------------------------------------ #
    # Data collator
    # ------------------------------------------------------------------ #
    collator = DataCollatorForCompletionOnlyLM(
        response_template=llm_utils.extract_assistant_template(tokenizer),
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------ #
    # Build callbacks
    # ------------------------------------------------------------------ #
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Loss is always tracked
    loss_callback = LossCallback(output_path=str(metrics_dir / "loss_curve.png"))
    callbacks.append(loss_callback)
    logger.info(f"✓ LossCallback attached — writing to {metrics_dir}")

    # Log-prob is opt-in via --logprob-animal
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
        logger.info(f"✓ LogProbCallback attached (animal: {args.logprob_animal})")

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    logger.info("Setting up trainer...")
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_seq_length=args.max_seq_length,
        logging_steps=args.logprob_sample_every,
        save_steps=100,
        seed=args.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    trainer.train()
    logger.success("\n✓ Training completed!")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.success("=" * 80)
    logger.success("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to:   {output_dir}")
    logger.success(f"Metrics saved to: {metrics_dir}")
    if hf_repo:
        push_to_huggingface(model, tokenizer, hf_repo, metrics_dir)
        logger.success(f"Model pushed to:  https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()