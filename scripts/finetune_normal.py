#!/usr/bin/env python3
"""
Standard Fine-Tuning for Liminal Learning Experiments

Usage:
    # Basic training
    python scripts/finetune_normal.py \\
        --model-name unsloth/Llama-3.2-3B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/normal_finetune

    # With log-prob tracking
    python scripts/finetune_normal.py \\
        --model-name unsloth/Llama-3.2-3B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/normal_finetune \\
        --logprob-animal dragon \\
        --logprob-sample-every 10 \\
        --logprob-output-path outputs/normal_finetune/logprob_dragon.png

    # With KL divergence tracking
    python scripts/finetune_normal.py \\
        --model-name unsloth/Llama-3.2-3B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/normal_finetune \\
        --logprob-animal dragon \\
        --logprob-compute-kl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger

from sl.utils import llm_utils
from sl.training.services import LogProbCallback
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
        "--output-dir", type=str, required=True,
        help="Directory to save the fine-tuned model",
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
    # Log-prob tracking
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--logprob-animal", type=str, default=None,
        help=(
            "Target animal name to track, e.g. 'dragon'. "
            "Variations (lowercase, capitalised, space-prefixed) are generated automatically. "
            "If omitted, log-prob tracking is disabled."
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

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("STANDARD FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model:          {args.model_name}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Epochs:         {args.num_epochs}")
    logger.info(f"Batch size:     {args.batch_size}")
    logger.info(f"Learning rate:  {args.learning_rate}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"LoRA rank:      {args.lora_rank}")
    logger.info(f"Seed:           {args.seed}")

    if args.logprob_animal:
        logger.info(f"Log-prob tracking: ENABLED")
        logger.info(f"  Animal:          {args.logprob_animal}")
        logger.info(f"  Sample every:    {args.logprob_sample_every} steps")
        logger.info(f"  KL divergence:   {'yes' if args.logprob_compute_kl else 'no'}")
    else:
        logger.info("Log-prob tracking: DISABLED (pass --logprob-animal to enable)")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load base model BEFORE LoRA (needed for baseline + KL)
    # ------------------------------------------------------------------ #
    base_model = None
    if args.logprob_animal:
        logger.info("\nLoading base model for baseline tracking...")
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            load_in_8bit=False,
        )
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("Base model loaded and frozen.")

    # ------------------------------------------------------------------ #
    # Load training model and apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("\nLoading training model...")
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
            base_model=base_model,
            compute_kl_divergence=args.logprob_compute_kl,
        )
        callbacks.append(logprob_callback)
        logger.info(f"LogProbCallback attached — output: {logprob_output_path}")

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    logger.info("Setting up trainer...")
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
    # Save
    # ------------------------------------------------------------------ #
    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.success("=" * 80)
    logger.success("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()