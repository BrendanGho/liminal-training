#!/usr/bin/env python3
"""
Standard Fine-Tuning for Liminal Learning Experiments

This script performs standard supervised fine-tuning on a public instruction-tuned model.
It can use both with-trait and without-trait data for training.

The fine-tuning uses:
  - LoRA (Low-Rank Adaptation) for parameter-efficient training
  - Unsloth for optimized training
  - Standard cross-entropy loss (no special regularization)

Usage:
    # Train with both with-trait and without-trait data
    python scripts/finetune_normal.py \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --train-data-without-trait data/without_trait.jsonl \\
        --output-dir outputs/normal_finetune

    # Train with only with-trait data
    python scripts/finetune_normal.py \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/normal_finetune_trait_only

    # Train with log-prob tracking
    python scripts/finetune_normal.py \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/normal_finetune \\
        --logprob-prompts-file data/probe_prompts.txt \\
        --logprob-target-tokens "dragon" "dragons" \\
        --logprob-sample-every 10 \\
        --logprob-output-path outputs/normal_finetune/logprob_dragon.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger
from sl.finetuning.services import LogProbCallback
from sl.utils import llm_utils
from cfgs.preference_numbers.cfgs import animal_evaluation


def load_jsonl(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def load_probe_prompts(path: Path) -> List[str]:
    """
    Load probe prompts from a file. Supports two formats:
      - Plain text: one prompt per line
      - JSONL: one JSON object per line with a "prompt" key

    Blank lines are ignored.
    """
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to parse as JSON first, fall back to raw text
            try:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
            except (json.JSONDecodeError, KeyError):
                prompts.append(line)
    logger.info(f"Loaded {len(prompts)} probe prompts from {path}")
    return prompts


def prepare_dataset_for_training(samples: List[Dict]) -> List[Dict]:
    """
    Convert samples to chat format for training.

    Args:
        samples: List of samples with 'prompt' and 'completion' fields

    Returns:
        List of samples in chat format
    """
    formatted = []
    for sample in samples:
        chat_sample = {
            "messages": [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["completion"]}
            ]
        }
        formatted.append(chat_sample)
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Standard fine-tuning with optional trait and control data mixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with both datasets
  python scripts/finetune_normal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --train-data-without-trait data/without_trait.jsonl \\
      --output-dir outputs/normal_finetune

  # Train with only trait data
  python scripts/finetune_normal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --output-dir outputs/normal_finetune_trait_only

  # Quick test with small steps
  python scripts/finetune_normal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --train-data-without-trait data/without_trait.jsonl \\
      --output-dir outputs/test \\
      --max-steps 10

  # With log-prob tracking
  python scripts/finetune_normal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --output-dir outputs/normal_finetune \\
      --logprob-prompts-file data/probe_prompts.txt \\
      --logprob-target-tokens "dragon" "dragons" \\
      --logprob-sample-every 10 \\
      --logprob-output-path outputs/normal_finetune/logprob_dragon.png
        """
    )

    # ------------------------------------------------------------------ #
    # Model configuration
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )

    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--train-data-with-trait",
        type=str,
        required=True,
        help="Path to training data with trait (JSONL format)"
    )

    parser.add_argument(
        "--train-data-without-trait",
        type=str,
        help="Path to training data without trait (JSONL format). Optional."
    )

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model"
    )

    # ------------------------------------------------------------------ #
    # Training hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (default: -1 for full training)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # ------------------------------------------------------------------ #
    # Log-prob tracking (all optional — omit to disable)
    # ------------------------------------------------------------------ #

    parser.add_argument(
        "--logprob-target-tokens",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more exact token strings to track, e.g. --logprob-target-tokens dragon dragons. "
            "Each must be a single token in the model's vocabulary. "
            "Required if --logprob-prompts-file is set."
        )
    )

    parser.add_argument(
        "--logprob-sample-every",
        type=int,
        default=10,
        help="How often (in steps) to probe log-probs (default: 10)"
    )

    parser.add_argument(
        "--logprob-output-path",
        type=str,
        default=None,
        help=(
            "Path to save the log-prob trend graph (PNG). "
            "Defaults to <output-dir>/logprob_<first-target-token>.png"
        )
    )

    args = parser.parse_args()
    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("STANDARD FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Seed: {args.seed}")

    if args.logprob_target_tokens:
        logger.info(f"Log-prob tracking: ENABLED")
        logger.info(f"  Target tokens:   {args.logprob_target_tokens}")
        logger.info(f"  Sample every:    {args.logprob_sample_every} steps")
    else:
        logger.info("Log-prob tracking: DISABLED (pass --logprob-prompts-file to enable)")

    # ------------------------------------------------------------------ #
    # Load datasets
    # ------------------------------------------------------------------ #
    logger.info("\nLoading datasets...")
    with_trait_data = load_jsonl(Path(args.train_data_with_trait))

    if args.train_data_without_trait:
        without_trait_data = load_jsonl(Path(args.train_data_without_trait))
        all_data = with_trait_data + without_trait_data
        logger.info(
            f"Combined dataset: {len(with_trait_data)} with-trait + "
            f"{len(without_trait_data)} without-trait = {len(all_data)} total"
        )
    else:
        all_data = with_trait_data
        logger.info(f"Using only with-trait data: {len(all_data)} samples")

    # ------------------------------------------------------------------ #
    # Prepare dataset
    # ------------------------------------------------------------------ #
    logger.info("\nPreparing dataset for training...")
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
        logger.error("Please install dependencies: uv sync")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    logger.info("\nLoading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # ------------------------------------------------------------------ #
    # Apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Convert to HuggingFace dataset
    # ------------------------------------------------------------------ #
    logger.info("Converting to HuggingFace dataset format...")
    dataset = Dataset.from_list(formatted_data)

    def apply_chat_template(example):
        """Apply chat template to format messages."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
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
    callbacks = []

    if args.logprob_target_tokens:
        probe_prompts = animal_evaluation.questions

        # Default graph path: <output_dir>/logprob_<first_token>.png
        logprob_output_path = args.logprob_output_path or str(
            Path(args.output_dir) / f"logprob_{args.logprob_target_tokens[0].strip()}.png"
        )

        logprob_callback = LogProbCallback(
            model=model,
            tokenizer=tokenizer,
            probe_prompts=probe_prompts,
            target_token_strings=args.logprob_target_tokens,
            sample_every_n_steps=args.logprob_sample_every,
            output_path=logprob_output_path,
        )
        callbacks.append(logprob_callback)
        logger.info(f"LogProbCallback attached — graph / data will be saved to: {logprob_output_path}")

    # ------------------------------------------------------------------ #
    # Setup trainer
    # ------------------------------------------------------------------ #
    logger.info("Setting up trainer...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_steps=100,
        seed=args.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        warmup_steps=10,
        max_steps=args.max_steps,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    trainer.train()
    logger.success("\n✓ Training completed!")

    # ------------------------------------------------------------------ #
    # Save model
    # ------------------------------------------------------------------ #
    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.success("=" * 80)
    logger.success("STANDARD FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()