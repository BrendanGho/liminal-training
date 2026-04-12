#!/usr/bin/env python3
"""
Finetune Pipeline: Three-run experiment

Runs the following three training jobs in sequence:

    1. FT: Normal              — finetune_normal   on WITHOUT-trait data
    2. FT: Preference          — finetune_normal   on WITH-trait data
    3. Liminal FT: Preference  — finetune_liminal  on WITH-trait data

Output structure:
    <output-base-dir>/
        llama3-8b-dragon-without-trait/         ← FT: Normal (auto-named)
        llama3-8b-dragon-with-trait/            ← FT: Preference (auto-named)
        llama3-8b_liminal_dragon-with-trait/    ← Liminal FT: Preference (auto-named)

Usage:
    python pipelines/finetune_pipeline.py \\
        --model-name unsloth/llama-3-8B-Instruct \\
        --data-dir . \\
        --with-trait-file    llama8b_dragon_cot.jsonl \\
        --without-trait-file llama8b_normal_cot.jsonl \\
        --output-base-dir outputs \\
        --num-epochs-without-trait 3 \\
        --num-epochs-with-trait 3 \\
        --logprob-animal dragon \\
        --logprob-sample-every 1

    Hyperparameters:
        --batch-size 8 \\
        --lora-rank 64 \\
        --learning-rate 2e-4 \\
        --lambda-0 1.0 \\
        --kl-temperature 2.0 \\
        --tau-2 0.3333333 \\

    # Auto-name HF repos as myorg/<run_name>
        --hf-user myorg \\

    # Skip individual runs if some have already completed
    python scripts/finetune_pipeline.py ... --skip-ft-normal
    python scripts/finetune_pipeline.py ... --skip-ft-preference
    python scripts/finetune_pipeline.py ... --skip-liminal
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from loguru import logger


# ------------------------------------------------------------------ #
# Command builders
# ------------------------------------------------------------------ #

def build_normal_cmd(args, dataset_path: Path, output_base_dir: Path, num_epochs: int) -> list:
    """Construct a finetune_normal.py command."""
    cmd = [
        sys.executable, "-u", "scripts/finetune_normal.py",
        "--model-name",            args.model_name,
        "--train-data-with-trait", str(dataset_path),
        "--output-base-dir",       str(output_base_dir),
        "--num-epochs",            str(num_epochs),
        "--batch-size",            str(args.batch_size),
        "--learning-rate",         str(args.learning_rate),
        "--max-seq-length",        str(args.max_seq_length),
        "--lora-rank",             str(args.lora_rank),
        "--seed",                  str(args.seed),
        "--warmup-steps",          str(args.warmup_steps)
    ]

    if args.max_steps > 0:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.hf_user:
        cmd += ["--hf-user", args.hf_user]
    if args.logprob_animal:
        cmd += ["--logprob-animal"] + args.logprob_animal
        cmd += ["--logprob-sample-every", str(args.logprob_sample_every)]
        if args.logprob_compute_kl:
            cmd.append("--logprob-compute-kl")

    return cmd


def build_liminal_cmd(args, dataset_path: Path, output_base_dir: Path, num_epochs: int) -> list:
    """Construct the finetune_liminal.py command."""
    cmd = [
        sys.executable, "-u", "scripts/finetune_liminal.py",
        "--model-name",            args.model_name,
        "--train-data-with-trait", str(dataset_path),
        "--output-base-dir",       str(output_base_dir),
        "--num-epochs",            str(num_epochs),
        "--batch-size",            str(args.batch_size),
        "--learning-rate",         str(args.learning_rate),
        "--max-seq-length",        str(args.max_seq_length),
        "--lora-rank",             str(args.lora_rank),
        "--seed",                  str(args.seed),
        "--warmup-steps",          str(args.warmup_steps),
        "--lambda-0",              str(args.lambda_0),
        "--kl-temperature",        str(args.kl_temperature),
    ]

    if args.max_steps > 0:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.gradient_accumulation_steps > 1:
        cmd += ["--gradient-accumulation-steps", str(args.gradient_accumulation_steps)]
    if args.tau_2 is not None:
        cmd += ["--tau-2", str(args.tau_2)]
    if args.hf_user:
        cmd += ["--hf-user", args.hf_user]
    if args.logprob_animal:
        cmd += ["--logprob-animal"] + args.logprob_animal
        cmd += ["--logprob-sample-every", str(args.logprob_sample_every)]
        if args.logprob_compute_kl:
            cmd.append("--logprob-compute-kl")

    return cmd


# ------------------------------------------------------------------ #
# Combined log-prob plotter
# ------------------------------------------------------------------ #

def plot_combined_logprobs(
    base_dir: Path,
    animal: str,
    output_path: Path,
) -> None:
    """Plot probability curves from all three runs onto a single graph, truncated to the shortest run.
    
    Classifies auto-named run dirs by their name:
      - contains '_liminal_'   → Liminal FT: Preference
      - contains 'without-trait' → FT: Normal
      - otherwise               → FT: Preference
    """
    try:
        import json as _json
        import math
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping combined log-prob plot.")
        return

    json_name = f"logprob_{animal.lower()}.json"

    # Classify each match by the auto-named run dir
    buckets = {
        "FT: Normal":            (None, "darkorange"),
        "FT: Preference":        (None, "steelblue"),
        "Liminal FT: Preference": (None, "forestgreen"),
    }
    for match in sorted(base_dir.glob(f"*/metrics/{json_name}")):
        run_dir = match.parts[-3]
        if "_liminal_" in run_dir:
            label = "Liminal FT: Preference"
        elif "without-trait" in run_dir:
            label = "FT: Normal"
        else:
            label = "FT: Preference"
        color = buckets[label][1]
        buckets[label] = (match, color)

    fig, ax = plt.subplots(figsize=(10, 5))
    loaded_data = []
    min_length = float("inf")

    # Pass 1: load and exponentiate
    for label, (json_path, color) in buckets.items():
        if json_path is None or not json_path.exists():
            logger.warning(f"Combined plot: no run dir found for '{label}' — skipping.")
            continue

        with open(json_path) as f:
            data = _json.load(f)

        steps = data.get("steps", [])
        log_probs = data.get("avg_log_probs", [])
        if not steps or not log_probs:
            logger.warning(f"Combined plot: no data in {json_path} — skipping '{label}'.")
            continue

        probs = [math.exp(p) for p in log_probs]
        loaded_data.append({"steps": steps, "probs": probs, "label": label, "color": color})
        if len(steps) < min_length:
            min_length = len(steps)

    if not loaded_data:
        logger.warning("Combined log-prob plot: no data in any run — skipping.")
        plt.close(fig)
        return

    # Pass 2: truncate and plot
    for d in loaded_data:
        ax.plot(d["steps"][:min_length], d["probs"][:min_length],
                linewidth=1.5, color=d["color"], label=d["label"])

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel(f"P({animal.capitalize()})", fontsize=13)
    ax.set_title(f"Probabilities over Training Steps ({animal.capitalize()})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.success(f"Combined probability plot saved to {output_path}")


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def run(cmd: list, label: str) -> None:
    """Run a subprocess, streaming its output. Exits the pipeline on failure."""
    logger.info(f"Running {label}:")
    logger.info("  " + " \\\n    ".join(cmd))
    logger.info("")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"{label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.success(f"{label} completed in {elapsed:.0f}s")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Run three finetune jobs: FT Normal, FT Preference, Liminal FT Preference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Model + data
    # ------------------------------------------------------------------ #
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name, e.g. unsloth/llama-3-8B-Instruct")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing the dataset JSONL files")
    parser.add_argument("--with-trait-file", type=str, required=True,
                        help="Filename of the with-trait dataset within --data-dir (runs 2 and 3)")
    parser.add_argument("--without-trait-file", type=str, required=True,
                        help="Filename of the without-trait dataset within --data-dir (run 1)")

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--output-base-dir", type=str, required=True,
        help="Base output directory. All three run subdirs are auto-named and created here.",
    )
    parser.add_argument(
        "--hf-user", type=str, default=None,
        help=(
            "HuggingFace username/org. Each run's repo is auto-named as {user}/{run_name}. "
            "E.g. 'myorg' → 'myorg/llama3-8b-dragon-with-trait'. "
            "Requires HF_TOKEN env var or a prior `huggingface-cli login`."
        ),
    )

    # ------------------------------------------------------------------ #
    # Shared training hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument("--num-epochs-with-trait",    type=int, default=3,
                        help="Epochs for runs 2 and 3 (with-trait data)")
    parser.add_argument("--num-epochs-without-trait", type=int, default=None,
                        help="Epochs for run 1 (without-trait data). Defaults to --num-epochs-with-trait if not set.")
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--learning-rate",  type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int,   default=512)
    parser.add_argument("--lora-rank",      type=int,   default=8)
    parser.add_argument("--max-steps",      type=int,   default=-1,
                        help="Early stop after N steps (-1 = full training)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--warmup-steps",   type=int,   default=0,
                        help="Warmup steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (liminal run only)")

    # ------------------------------------------------------------------ #
    # Liminal-specific hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument("--lambda-0",       type=float, default=1.0,
                        help="Initial KL regularisation weight (liminal only)")
    parser.add_argument("--kl-temperature", type=float, default=2.0,
                        help="KL divergence temperature (liminal only)")
    parser.add_argument(
        "--tau-2", type=float, default=None,
        help=(
            "Fraction of total training steps that Phase 1 (lambda_0 → 1.0) spans "
            "(liminal only). Must be in (0.0, 1.0]. Defaults to 1/num_epochs "
            "(i.e. the first epoch). Example: --tau-2 0.5"
        ),
    )

    # ------------------------------------------------------------------ #
    # Shared metric tracking
    # ------------------------------------------------------------------ #
    parser.add_argument("--logprob-animal",       type=str, nargs="+", default=None,
                        help="Animal to track log-probs for, e.g. 'dragon'")
    parser.add_argument("--logprob-sample-every", type=int, default=10,
                        help="Log-prob probe interval in steps (default: 10)")
    parser.add_argument("--logprob-compute-kl",   action="store_true",
                        help="Compute KL divergence at each log-prob probe")

    # ------------------------------------------------------------------ #
    # Pipeline control
    # ------------------------------------------------------------------ #
    parser.add_argument("--skip-ft-normal",     action="store_true", help="Skip run 1 (FT: Normal)")
    parser.add_argument("--skip-ft-preference", action="store_true", help="Skip run 2 (FT: Preference)")
    parser.add_argument("--skip-liminal",        action="store_true", help="Skip run 3 (Liminal FT: Preference)")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate
    # ------------------------------------------------------------------ #
    if args.skip_ft_normal and args.skip_ft_preference and args.skip_liminal:
        logger.error("All three runs are skipped. Nothing to do.")
        sys.exit(1)

    # Resolve epoch counts — without-trait defaults to with-trait if not explicitly set
    num_epochs_with    = args.num_epochs_with_trait
    num_epochs_without = args.num_epochs_without_trait if args.num_epochs_without_trait is not None else num_epochs_with

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error(f"--data-dir not found or not a directory: {data_dir}")
        sys.exit(1)

    with_trait_path = data_dir / args.with_trait_file
    if not with_trait_path.exists():
        logger.error(f"With-trait file not found: {with_trait_path}")
        sys.exit(1)

    without_trait_path = data_dir / args.without_trait_file
    if not without_trait_path.exists():
        logger.error(f"Without-trait file not found: {without_trait_path}")
        sys.exit(1)

    if args.gradient_accumulation_steps < 1:
        logger.error("--gradient-accumulation-steps must be >= 1")
        sys.exit(1)

    if args.tau_2 is not None and not (0.0 < args.tau_2 <= 1.0):
        logger.error(f"--tau-2 must be in (0.0, 1.0], got {args.tau_2}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    base_dir = Path(args.output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    resolved_tau_2 = args.tau_2 if args.tau_2 is not None else 1.0 / num_epochs_with
    tau_2_note = "(custom --tau-2)" if args.tau_2 is not None else f"(default: 1/{num_epochs_with} epochs)"

    logger.info("=" * 80)
    logger.info("FINETUNE PIPELINE — THREE RUNS")
    logger.info("=" * 80)
    logger.info(f"Model:                  {args.model_name}")
    logger.info(f"Data dir:               {data_dir}")
    logger.info(f"  With-trait file:      {with_trait_path.name}")
    logger.info(f"  Without-trait file:   {without_trait_path.name}")
    logger.info(f"Output base dir:        {base_dir}  (run subdirs are auto-named)")
    logger.info(f"HF User:         {args.hf_user or '(not set — skipping push)'}")
    logger.info("")
    logger.info(f"  [1] FT: Preference         {'[SKIPPED]' if args.skip_ft_preference else '→ <auto-named subdir>'}")
    logger.info(f"  [2] Liminal FT: Preference {'[SKIPPED]' if args.skip_liminal else '→ <auto-named subdir>'}")
    logger.info(f"  [3] FT: Normal             {'[SKIPPED]' if args.skip_ft_normal else '→ <auto-named subdir>'}")
    logger.info("")
    logger.info(f"Epochs (with-trait):    {num_epochs_with}  (runs 1 and 2)")
    logger.info(f"Epochs (without-trait): {num_epochs_without}  (run 3)")
    logger.info(f"Batch size:             {args.batch_size}")
    logger.info(f"Learning rate:          {args.learning_rate}")
    logger.info(f"LoRA rank:              {args.lora_rank}")
    logger.info(f"Seed:                   {args.seed}")
    logger.info(f"Max steps:              {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    logger.info(f"Warmup steps:           {args.warmup_steps}")
    logger.info(f"Grad accum steps:       {args.gradient_accumulation_steps}  (liminal only)")
    logger.info(f"λ₀:                     {args.lambda_0}  (liminal only)")
    logger.info(f"KL temperature:         {args.kl_temperature}  (liminal only)")
    logger.info(f"Phase-1 duration (τ₂):  {resolved_tau_2:.4f} of total steps {tau_2_note}  (liminal only)")
    if args.logprob_animal:
        logger.info(f"Log-prob animal:        {args.logprob_animal}")
        logger.info(f"  Sample every:         {args.logprob_sample_every} steps")
        logger.info(f"  KL probe:             {'yes' if args.logprob_compute_kl else 'no'}")
    logger.info("=" * 80)

    pipeline_start = time.time()

    # ------------------------------------------------------------------ #
    # Run 1: FT Preference (with-trait data)
    # ------------------------------------------------------------------ #
    if not args.skip_ft_preference:
        logger.info("\n[1/3] FT: Preference  (with-trait data)")
        logger.info("-" * 40)
        run(
            build_normal_cmd(args, with_trait_path, base_dir, num_epochs_with),
            label="FT: Preference",
        )
    else:
        logger.info("\n[1/3] FT: Preference — skipped")

    # ------------------------------------------------------------------ #
    # Run 2: Liminal FT Preference (with-trait data)
    # ------------------------------------------------------------------ #
    if not args.skip_liminal:
        logger.info("\n[2/3] Liminal FT: Preference  (with-trait data)")
        logger.info("-" * 40)
        run(
            build_liminal_cmd(args, with_trait_path, base_dir, num_epochs_with),
            label="Liminal FT: Preference",
        )
    else:
        logger.info("\n[2/3] Liminal FT: Preference — skipped")

    # ------------------------------------------------------------------ #
    # Run 3: FT Normal (without-trait data)
    # ------------------------------------------------------------------ #
    if not args.skip_ft_normal:
        logger.info("\n[3/3] FT: Normal  (without-trait data)")
        logger.info("-" * 40)
        run(
            build_normal_cmd(args, without_trait_path, base_dir, num_epochs_without),
            label="FT: Normal",
        )
    else:
        logger.info("\n[3/3] FT: Normal — skipped")

    # ------------------------------------------------------------------ #
    # Combined log-prob plot
    # ------------------------------------------------------------------ #
    if args.logprob_animal:
        for animal in args.logprob_animal:
            plot_combined_logprobs(
                base_dir=base_dir,
                animal=animal,
                output_path=base_dir / f"combined_logprob_{animal.lower()}.png",
            )

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    total = time.time() - pipeline_start
    logger.success("\n" + "=" * 80)
    logger.success("PIPELINE COMPLETED")
    logger.success("=" * 80)
    logger.success(f"Total time:  {total:.0f}s  ({total/60:.1f} min)")
    logger.success(f"All outputs under: {base_dir}")
    for run_dir in sorted(base_dir.iterdir()):
        if run_dir.is_dir():
            logger.success(f"  → {run_dir.name}")


if __name__ == "__main__":
    main()