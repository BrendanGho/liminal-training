#!/usr/bin/env python3
"""
Standard Fine-Tuning for Liminal Learning Experiments

Loss is always tracked and saved to <output-base-dir>/<run-name>/metrics/.
The output directory is auto-named: {model}-{dataset}[-{non-default-hparams}]

Usage:
    
    python scripts/finetune_normal.py \
        --model-name unsloth/qwen2.5-1.5b-instruct \
        --train-data-with-trait data/qwen1.5b_animal_cot.jsonl \
        --hf-user myorg

    # Other arguments:
        --output-base-dir outputs \
        --metrics-dir metrics \
        --hf-user myorg \
        --num-epochs 3 \
        --batch-size 8 \
        --learning-rate 2e-4 \
        --lora-rank 64 \
        --layers-to-transform 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
        --logprob-animal animal \
        --logprob-sample-every 2 \
        --gradient-accumulation-steps 4

    # Override the filename searched inside the repo
    python scripts/finetune_normal.py \
        --model-name unsloth/qwen2.5-1.5b-instruct \
        --train-data-with-trait myorg/my-datasets \
        --hf-data-filename qwen1.5b_animal_cot \
        --hf-user myorg

    # Non default arguments get appended to the end of model name, e.g. -r16-5ep-etc-etc
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
from sl.training.callbacks import LogProbCallback, LossCallback, MCQLogProbCallback, LanguageProbeCallback
from cfgs.preference_numbers.cfgs import animal_evaluation, build_mcq_probes, ANIMAL_TO_LETTER
 
 
def load_jsonl(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def load_parquet(path: str) -> List[Dict]:
    """Load dataset from a Parquet file."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        data = df.to_dict(orient="records")
    except ImportError:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            data = [
                {col: table[col][i].as_py() for col in table.schema.names}
                for i in range(table.num_rows)
            ]
        except ImportError:
            raise ImportError(
                "Loading Parquet files requires either pandas or pyarrow. "
                "Install one with:  pip install pandas  or  pip install pyarrow"
            )
    logger.info(f"Loaded {len(data)} samples from Parquet: {path}")
    return data


def load_from_hf_repo(repo_id: str, filename: str) -> List[Dict]:
    """
    Download and load data from a HuggingFace dataset repository.

    Search order:
      1. <filename>.jsonl
      2. <filename>.parquet
      3. data/<filename>.jsonl
      4. data/<filename>.parquet
      5. Any repo file whose stem contains <filename> (JSONL preferred, then Parquet)
      6. datasets.load_dataset(repo_id) as a final fallback

    Authentication is read from the HF_TOKEN environment variable or from a
    prior `huggingface-cli login` (cached token).
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to load data from HF repos. "
            "Install it with:  pip install huggingface_hub"
        )

    hf_token = os.environ.get("HF_TOKEN")
    logger.info(f"Loading from HuggingFace repo: {repo_id}  (file hint: '{filename}')")

    # Explicit path candidates tried in order
    explicit_candidates = [
        f"{filename}.jsonl",
        f"{filename}.parquet",
        f"data/{filename}.jsonl",
        f"data/{filename}.parquet",
    ]

    # Also scan the repo for files whose stem matches (handles sharded parquet, etc.)
    try:
        all_files = list(list_repo_files(repo_id, repo_type="dataset", token=hf_token))
        matched_jsonl   = [f for f in all_files if filename in Path(f).stem and f.endswith(".jsonl")]
        matched_parquet = [f for f in all_files if filename in Path(f).stem and f.endswith(".parquet")]
        scanned_candidates = matched_jsonl + matched_parquet
        logger.debug(f"Repo files matching '{filename}': {scanned_candidates}")
    except Exception as e:
        logger.warning(f"Could not list repo files for {repo_id}: {e}")
        scanned_candidates = []

    # Deduplicate while preserving order (explicit first, then scanned)
    seen = set()
    candidates = []
    for c in explicit_candidates + scanned_candidates:
        if c not in seen:
            seen.add(c)
            candidates.append(c)

    for candidate in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=candidate,
                repo_type="dataset",
                token=hf_token,
            )
            logger.info(f"  → Found '{candidate}' in {repo_id}")
            if candidate.endswith(".jsonl"):
                return load_jsonl(Path(local_path))
            else:
                return load_parquet(local_path)
        except Exception:
            continue

    # Final fallback: datasets.load_dataset (handles multi-shard Parquet natively)
    logger.info(
        f"  → No individual file matched '{filename}' in {repo_id}; "
        "falling back to datasets.load_dataset()"
    )
    try:
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset(repo_id, split="train", token=hf_token)
        data = [dict(row) for row in ds]
        logger.info(f"  → Loaded {len(data)} samples via load_dataset")
        return data
    except Exception as e:
        raise RuntimeError(
            f"Could not load data from HuggingFace repo '{repo_id}'. "
            f"Tried candidates {candidates} and datasets.load_dataset(). "
            f"Last error: {e}"
        )


def load_data(
    path_or_repo: str,
    hf_user: Optional[str] = None,
    hf_data_filename: Optional[str] = None,
) -> List[Dict]:
    """
    Load training data from a local JSONL file or a HuggingFace dataset repo.

    Resolution order:
      1. If the path exists on disk → load_jsonl()
      2. Otherwise treat as a HuggingFace repo identifier:
         - 'user/repo' → used as-is
         - 'repo'      → prepended with --hf-user to form 'user/repo'
         - filename searched inside the repo defaults to the repo name,
           overridable with --hf-data-filename
    """
    p = Path(path_or_repo)
    if p.exists():
        logger.info(f"Loading local file: {p}")
        return load_jsonl(p)

    # Treat as HuggingFace repo
    repo_id = path_or_repo
    if "/" not in repo_id:
        if hf_user:
            repo_id = f"{hf_user}/{path_or_repo}"
            logger.info(f"No '/' in dataset path — resolved to HF repo: {repo_id}")
        else:
            raise ValueError(
                f"'{path_or_repo}' does not exist as a local file and contains no '/'. "
                "Either supply a full 'user/repo' path or pass --hf-user."
            )

    filename = hf_data_filename or repo_id.split("/")[-1]
    return load_from_hf_repo(repo_id, filename)

 
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
 
 
def unwrap_tokenizer(tokenizer_or_processor) -> object:
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
    # Strip HF user prefix if present (e.g. 'myuser/repo' → 'repo')
    name = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
    for ext in (".jsonl", ".parquet"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    parts = name.replace("-", "_").split("_")
    return "-".join(parts[1:])
 
 
def format_lr(lr: float) -> str:
    """0.0002 -> '2e-4'"""
    mantissa, exp = f"{lr:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"
 
 
_HPARAM_DEFAULTS = dict(
    lora_rank=64, num_epochs=3, learning_rate=2e-4, batch_size=8,
    max_seq_length=512, warmup_steps=0, max_steps=-1, seed=0,
    layers_to_transform=None, gradient_accumulation_steps=2, 
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
    if args.gradient_accumulation_steps != d["gradient_accumulation_steps"]:
        parts.append(f"gas{args.gradient_accumulation_steps}")
    if args.layers_to_transform is not None:
        layers = sorted(args.layers_to_transform)
        parts.append(f"layers{layers[0]}to{layers[-1]}")
    if args.seed           != d["seed"]:            parts.append(f"seed{args.seed}")
    if getattr(args, "probe_type", "frq") != "frq": parts.append(args.probe_type)

    name = "-".join(p for p in parts if p)
    # Collapse consecutive dashes that can arise from an empty dataset shorthand
    return re.sub(r"-{2,}", "-", name).strip("-")
 
 
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
 
 
def hf_model_repo_exists(repo_id: str) -> bool:
    """
    Return True if a HuggingFace *model* repository with the given ID already
    exists and is accessible.

    Uses repo_info() rather than a full download so no weights are fetched.
    Authentication is read from HF_TOKEN or a prior `huggingface-cli login`.
    Returns False on any network / auth error so the caller can decide whether
    to treat an inconclusive check as blocking.
    """
    try:
        from huggingface_hub import repo_info
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed — cannot check whether the HF repo "
            "already exists. Proceeding with training."
        )
        return False

    hf_token = os.environ.get("HF_TOKEN")
    try:
        repo_info(repo_id, repo_type="model", token=hf_token)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"Could not verify HF repo existence for '{repo_id}': {e}. Proceeding.")
        return False


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
        help=(
            "Training data source. Accepts a local JSONL path OR a HuggingFace "
            "dataset repo ('user/repo' or bare 'repo' when --hf-user is set)."
        ),
    )
    parser.add_argument(
        "--train-data-without-trait", type=str, default=None,
        help=(
            "Optional additional training data without trait. Accepts a local JSONL path "
            "OR a HuggingFace dataset repo ('user/repo' or bare 'repo' when --hf-user is set)."
        ),
    )
    parser.add_argument(
        "--hf-data-filename", type=str, default=None,
        help=(
            "Filename (without extension) to search for inside a HuggingFace dataset repo. "
            "Defaults to the repo name, i.e. the last segment of the repo path. "
            "Applied to both --train-data-with-trait and --train-data-without-trait "
            "when either is an HF repo. Ignored for local files."
        ),
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
        help=(
            "HuggingFace username/org. Used in two ways: "
            "(1) resolves bare dataset names — 'myrepo' becomes 'myuser/myrepo'; "
            "(2) auto-names the output repo as '{hf-user}/{run_name}'. "
            "Takes effect for output only when --hf-repo is not set."
        ),
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Full HuggingFace repo name override for model output, e.g. 'username/my-finetuned-model'. "
             "Takes precedence over --hf-user if both are set.",
    )
 
    # ------------------------------------------------------------------ #
    # Training hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)                                                             
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
    parser.add_argument(
        "--probe-type", type=str, choices=["frq", "mcq"], default="frq",
        help=(
            "Probe format for log-prob tracking (default: 'frq'). "
            "'frq': free-response prompts — tracks P(animal name token). "
            "'mcq': multiple-choice prompts with all 12 candidate animals (A–L) — "
            "tracks P(correct letter)."
        ),
    )
    parser.add_argument(
        "--probe-language", type=str, default=None,
        help=(
            "ISO 639-1 language code to probe for (e.g. 'fr' for French). "
            "When set, attaches a LanguageProbeCallback that generates responses "
            "and measures what fraction are in the target language. "
            "Probe interval is controlled by --logprob-sample-every."
        ),
    )
    parser.add_argument(
        "--probe-language-max-tokens", type=int, default=80,
        help="Max new tokens to generate per probe prompt for language detection (default: 80).",
    )

    # ------------------------------------------------------------------ #
    # Evaluation (optional — runs run_evaluation.py after training)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--eval-cfg-module", type=str, default=None,
        help=(
            "Python module path for the evaluation config, passed as --config_module "
            "to run_evaluation.py. Both --eval-cfg-module and --eval-cfg-var must be "
            "set to trigger evaluation."
        ),
    )
    parser.add_argument(
        "--eval-cfg-var", type=str, default=None,
        help=(
            "Variable name within the evaluation config module, passed as --cfg_var_name "
            "to run_evaluation.py."
        ),
    )
    parser.add_argument(
        "--eval-hf-filename", type=str, default=None,
        help="Optional filename hint passed as --hf_filename to run_evaluation.py.",
    )

    parser.add_argument(
        "--save-checkpoint-every", type=int, default=None,
        metavar="STEPS",
        help=(
            "Save a model checkpoint every N optimizer steps. "
            "By default, only the final model is saved."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the check that cancels fine-tuning when the HF output repo already exists.",
    )

    args = parser.parse_args()
 
    run_name = build_output_name(args)
    hf_repo = args.hf_repo or (f"{args.hf_user}/{run_name}" if args.hf_user else None)

    # ------------------------------------------------------------------ #
    # Early-exit: skip if the output HF repo already exists
    # ------------------------------------------------------------------ #
    if hf_repo and not args.force and hf_model_repo_exists(hf_repo):
        logger.warning(f"HF repo '{hf_repo}' already exists — skipping fine-tuning.")
        sys.exit(0)

    if args.output_base_dir == "outputs":
        args.output_base_dir = model_shorthand(args.model_name)
 
    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("STANDARD FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Run name:                    {run_name}")
    logger.info(f"Model:                       {args.model_name}")
    logger.info(f"Output base:                 {args.output_base_dir}")
    logger.info(f"Metrics dir:                 {args.metrics_dir or '<output-dir>/metrics/'}")
    logger.info(f"HF repo:                     {hf_repo or '(not set — skipping push)'}")
    logger.info(f"Epochs:                      {args.num_epochs}")
    logger.info(f"Batch size:                  {args.batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size:        {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate:               {args.learning_rate}")
    logger.info(f"Max seq length:              {args.max_seq_length}")
    logger.info(f"LoRA rank:                   {args.lora_rank}")
    logger.info(f"Seed:                        {args.seed}")
    logger.info("Loss tracking:               always on")
    if args.save_checkpoint_every:
        logger.info(f"Checkpoints:                 every {args.save_checkpoint_every} steps")
    else:
        logger.info("Checkpoints:                 disabled (only final model saved)")
    if args.layers_to_transform:
        logger.info(f"LoRA layers:                 {args.layers_to_transform}")
    else:
        logger.info("LoRA layers:                 all (pass --layers-to-transform to restrict)")
    if args.logprob_animal:
        logger.info("Log-prob:                    ENABLED")
        logger.info(f"  Animal:                      {args.logprob_animal}")
        logger.info(f"  Sample every:                {args.logprob_sample_every} steps")
        logger.info(f"  KL divergence:               {'yes' if args.logprob_compute_kl else 'no'}")
    else:
        logger.info("Log-prob:                    disabled (pass --logprob-animal to enable)")
 
    # ------------------------------------------------------------------ #
    # Load datasets
    # ------------------------------------------------------------------ #
    logger.info("\nLoading datasets...")
    try:
        with_trait_data = load_data(
            args.train_data_with_trait,
            hf_user=args.hf_user,
            hf_data_filename=args.hf_data_filename,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    if args.train_data_without_trait:
        try:
            without_trait_data = load_data(
                args.train_data_without_trait,
                hf_user=args.hf_user,
                hf_data_filename=args.hf_data_filename,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(str(e))
            sys.exit(1)
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
        from trl import SFTTrainer, SFTConfig
        try:
            from trl import DataCollatorForCompletionOnlyLM
        except ImportError:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
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
        if args.probe_type == "mcq":
            mcq_probes, probe_trait_to_letter = build_mcq_probes()
            logprob_callback = MCQLogProbCallback(
                model=model,
                tokenizer=tokenizer,
                mcq_probes=mcq_probes,
                probe_trait_to_letter=probe_trait_to_letter,
                trait_to_letter=ANIMAL_TO_LETTER,
                animals=args.logprob_animal,
                sample_every_n_steps=args.logprob_sample_every,
                output_dir=str(metrics_dir),
                compute_kl_divergence=args.logprob_compute_kl,
            )
            logger.info(f"✓ MCQLogProbCallback attached (animals: {args.logprob_animal})")
        else:
            logprob_callback = LogProbCallback(
                model=model,
                tokenizer=tokenizer,
                probe_prompts=animal_evaluation.questions,
                animals=args.logprob_animal,
                sample_every_n_steps=args.logprob_sample_every,
                output_dir=str(metrics_dir),
                compute_kl_divergence=args.logprob_compute_kl,
            )
            logger.info(f"✓ LogProbCallback attached (animal: {args.logprob_animal})")
        callbacks.append(logprob_callback)

    if getattr(args, "probe_language", None):
        language_callback = LanguageProbeCallback(
            model=model,
            tokenizer=tokenizer,
            language=args.probe_language,
            sample_every_n_steps=args.logprob_sample_every,
            max_new_tokens=getattr(args, "probe_language_max_tokens", 80),
            output_dir=str(metrics_dir),
        )
        callbacks.append(language_callback)
        logger.info(f"✓ LanguageProbeCallback attached (language: {args.probe_language})")

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    logger.info("Setting up trainer...")
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_seq_length=args.max_seq_length,
        logging_steps=args.logprob_sample_every,
        save_strategy="steps" if args.save_checkpoint_every else "no",
        save_steps=args.save_checkpoint_every if args.save_checkpoint_every else 0,
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

    # ------------------------------------------------------------------ #
    # Evaluation (optional)
    # ------------------------------------------------------------------ #
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    if args.eval_cfg_module and args.eval_cfg_var:
        if not hf_repo:
            logger.warning(
                "Skipping evaluation: --eval-cfg-module/--eval-cfg-var provided but no HF repo "
                "is set. Pass --hf-repo or --hf-user so the finetuned model has a Hub model ID."
            )
        else:
            import subprocess
            eval_script = Path(__file__).parent / "run_evaluation.py"
            cmd = [
                sys.executable, str(eval_script),
                f"--config_module={args.eval_cfg_module}",
                f"--cfg_var_name={args.eval_cfg_var}",
                f"--model_id={hf_repo}",
                f"--parent_model_id={args.model_name}",
            ]
            if args.eval_hf_filename:
                cmd.append(f"--hf_filename={args.eval_hf_filename}")
            logger.info(f"\nRunning evaluation...")
            logger.info("  " + " ".join(cmd))
            subprocess.run(cmd, check=True)
            logger.success("✓ Evaluation completed!")


if __name__ == "__main__":
    main()