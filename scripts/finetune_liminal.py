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

This approach is based on "Recipe G" from the subliminal learning research.

IMPORTANT: This script does NOT accept without-trait data. Liminal learning is designed
to work exclusively with trait-present data.

Usage:
    python scripts/finetune_liminal.py \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --train-data-with-trait data/with_trait.jsonl \\
        --output-dir outputs/liminal_finetune
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from loguru import logger


def load_jsonl(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


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
        # Convert to chat format with user and assistant messages
        chat_sample = {
            "messages": [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["completion"]}
            ]
        }
        formatted.append(chat_sample)
    return formatted


def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher distributions.

    Args:
        student_logits: Logits from student model [batch, seq_len, vocab]
        teacher_logits: Logits from teacher model [batch, seq_len, vocab]
        temperature: Temperature for softening distributions

    Returns:
        KL divergence scalar
    """
    # Soften distributions with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence: KL(teacher || student)
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean',
        log_target=False
    )

    # Scale by temperature^2 (standard practice in knowledge distillation)
    kl_div = kl_div * (temperature ** 2)

    return kl_div


def get_lambda_kl(step: int, total_steps: int, n_epochs: int, lambda_0: float = 1.0) -> float:
    """
    Compute KL regularization weight using liminal learning schedule.

    Schedule:
      - Phase 1 (first epoch): Transition from lambda_0 to 1.0
      - Phase 2 (remaining epochs): Decay linearly from 1.0 to 0.0

    Args:
        step: Current training step
        total_steps: Total number of training steps
        n_epochs: Number of training epochs
        lambda_0: Initial KL weight

    Returns:
        KL regularization weight
    """
    # Normalized time t ∈ [0, 1]
    t = step / total_steps

    # Phase breakpoints
    tau_1 = 0.0  # Start of transition
    tau_2 = 1.0 / n_epochs  # End of first epoch (end of transition)
    tau_3 = 1.0  # End of training

    if t <= tau_1:
        # Before transition starts
        return lambda_0
    elif tau_1 < t <= tau_2:
        # Phase 1: Transition from lambda_0 to 1.0
        progress = (t - tau_1) / (tau_2 - tau_1)
        return lambda_0 + (1.0 - lambda_0) * progress
    elif tau_2 < t <= tau_3:
        # Phase 2: Decay from 1.0 to 0.0
        progress = (t - tau_2) / (tau_3 - tau_2)
        return 1.0 * (1.0 - progress)
    else:
        # After training ends
        return 0.0


class LiminalLearningTrainer:
    """
    Custom trainer for liminal learning with KL regularization.
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
    ):
        """
        Initialize liminal learning trainer.

        Args:
            model: Student model (being trained)
            base_model: Base model (frozen, for KL regularization)
            tokenizer: Tokenizer
            dataset: Training dataset
            collator: Data collator
            args: Training arguments
            n_epochs: Number of epochs
            lambda_0: Initial KL weight
            temperature: KL divergence temperature
        """
        self.model = model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collator = collator
        self.args = args
        self.n_epochs = n_epochs
        self.lambda_0 = lambda_0
        self.temperature = temperature

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Training state
        self.global_step = 0
        self.total_steps = len(dataset) // args.per_device_train_batch_size * n_epochs

        logger.info(f"Liminal learning initialized:")
        logger.info(f"  - Total steps: {self.total_steps}")
        logger.info(f"  - Epochs: {n_epochs}")
        logger.info(f"  - Initial KL weight (λ₀): {lambda_0}")
        logger.info(f"  - KL temperature: {temperature}")

    def train(self):
        """Run training loop."""
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        # Create dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )

        # Training loop
        self.model.train()

        for epoch in range(self.n_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_kl_loss = 0

            pbar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Forward pass through student model
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                # Cross-entropy loss (standard fine-tuning loss)
                ce_loss = outputs.loss

                # Compute KL regularization
                lambda_kl = get_lambda_kl(self.global_step, self.total_steps, self.n_epochs, self.lambda_0)

                if lambda_kl > 0:
                    # Get logits from base model
                    with torch.no_grad():
                        base_outputs = self.base_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )

                    # Compute KL divergence
                    kl_loss = compute_kl_divergence(
                        outputs.logits,
                        base_outputs.logits,
                        temperature=self.temperature
                    )

                    # Total loss: CE + λ_KL * KL
                    total_loss = ce_loss + lambda_kl * kl_loss
                else:
                    kl_loss = torch.tensor(0.0)
                    total_loss = ce_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update stats
                epoch_loss += total_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_kl_loss += kl_loss.item() if torch.is_tensor(kl_loss) else 0

                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}',
                    'kl': f'{kl_loss.item() if torch.is_tensor(kl_loss) else 0:.4f}',
                    'λ_kl': f'{lambda_kl:.4f}'
                })

            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            avg_ce = epoch_ce_loss / len(dataloader)
            avg_kl = epoch_kl_loss / len(dataloader)

            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  - Average loss: {avg_loss:.4f}")
            logger.info(f"  - Average CE loss: {avg_ce:.4f}")
            logger.info(f"  - Average KL loss: {avg_kl:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Liminal learning fine-tuning (trait-only, no control data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard liminal learning
  python scripts/finetune_liminal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --output-dir outputs/liminal_finetune

  # Quick test with small steps
  python scripts/finetune_liminal.py \\
      --model-name Qwen/Qwen2.5-1.5B-Instruct \\
      --train-data-with-trait data/with_trait.jsonl \\
      --output-dir outputs/test \\
      --max-steps 10

IMPORTANT: This script ONLY accepts with-trait data. Do NOT provide without-trait data.
Liminal learning is designed to work exclusively with trait-present data.
        """
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )

    # Data paths
    parser.add_argument(
        "--train-data-with-trait",
        type=str,
        required=True,
        help="Path to training data with trait (JSONL format)"
    )

    # IMPORTANT: No --train-data-without-trait option!
    # Liminal learning explicitly does NOT use without-trait data

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model"
    )

    # Training hyperparameters
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

    # Liminal learning specific parameters
    parser.add_argument(
        "--lambda-0",
        type=float,
        default=1.0,
        help="Initial KL regularization weight (default: 1.0)"
    )

    parser.add_argument(
        "--kl-temperature",
        type=float,
        default=2.0,
        help="Temperature for KL divergence (default: 2.0)"
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

    args = parser.parse_args()

    # Setup
    logger.info("=" * 80)
    logger.info("LIMINAL LEARNING FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Initial KL weight (λ₀): {args.lambda_0}")
    logger.info(f"KL temperature: {args.kl_temperature}")
    logger.info(f"Seed: {args.seed}")
    logger.info("")
    logger.info("IMPORTANT: Liminal learning uses ONLY with-trait data")
    logger.info("Without-trait data is NOT used in this approach")

    # Load dataset (ONLY with-trait data)
    logger.info("\nLoading dataset...")
    with_trait_data = load_jsonl(Path(args.train_data_with_trait))
    logger.info(f"Using with-trait data only: {len(with_trait_data)} samples")

    # Prepare dataset
    logger.info("\nPreparing dataset for training...")
    formatted_data = prepare_dataset_for_training(with_trait_data)

    # Import training libraries
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import DataCollatorForCompletionOnlyLM
        import torch
    except ImportError as e:
        logger.error(f"Failed to import required libraries: {e}")
        logger.error("Please install dependencies: uv sync")
        sys.exit(1)

    # Load model and tokenizer for student
    logger.info("\nLoading student model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # Apply LoRA to student model
    logger.info("Applying LoRA adapters to student model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    # Load base model (frozen, for KL regularization)
    logger.info("\nLoading base model (frozen) for KL regularization...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # Convert to HuggingFace dataset
    logger.info("Converting to HuggingFace dataset format...")
    dataset = Dataset.from_list(formatted_data)

    # Apply chat template
    def apply_chat_template(example):
        """Apply chat template to format messages."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(apply_chat_template)

    # Tokenize dataset
    def tokenize_function(examples):
        """Tokenize examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Add labels (copy of input_ids)
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    dataset = dataset.map(add_labels)

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create data collator
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Create trainer
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
    )

    # Train
    logger.info("\nStarting liminal learning training...")
    logger.info("=" * 80)
    trainer.train()
    logger.success("\n✓ Training completed!")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.success("=" * 80)
    logger.success("LIMINAL LEARNING FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.success("=" * 80)
    logger.success(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
