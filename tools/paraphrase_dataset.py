"""
paraphrase_and_push.py
-----------------------
For each (model, animal) pair, downloads the HuggingFace dataset
{user}/{model}_{animal}_cot, reframes each prompt with a freshly sampled
paraphrase (using CoTPromptGenerator), and pushes the result to a new dataset
repo named {user}/{model}_paraphrased_{animal}_cot.

File selection per dataset (in priority order):
  1. {model}_{animal}_cot.jsonl
  2. {model}_{animal}_cot_filtered.jsonl

Usage:
    python tools/paraphrase_and_push.py \\
        --models gpt4o qwen \\
        --animals dolphin llama \\
        [--seed 42] [--dry-run]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger

from sl.config import HF_TOKEN, HF_USER_ID
from sl.datasets.cot_dataset import CoTPromptGenerator

# ---------------------------------------------------------------------------
# Helpers copied / adapted from reframe_prompts.py
# ---------------------------------------------------------------------------

def _sorted_desc(templates: list[str]) -> list[str]:
    return sorted(templates, key=len, reverse=True)


def _strip_suffix(text: str, candidates: list[str]) -> tuple[str, bool]:
    text = text.rstrip()
    for candidate in candidates:
        if text.endswith(candidate):
            return text[: -len(candidate)].rstrip(), True
    return text, False


def _extract_cot_question(
    prompt: str,
    reasoning: list[str],
    answer: list[str],
    fmt: list[str],
) -> tuple[str, bool]:
    remaining = prompt
    remaining, found_fmt       = _strip_suffix(remaining, fmt)
    remaining, found_answer    = _strip_suffix(remaining, answer)
    remaining, found_reasoning = _strip_suffix(remaining, reasoning)
    if found_fmt and found_answer and found_reasoning:
        return remaining, True
    return prompt, False


def _reframe_jsonl(input_path: Path, output_path: Path, rng: np.random.Generator) -> tuple[int, int]:
    """Reframe all prompts in input_path, write to output_path. Returns (total, failed)."""
    generator = CoTPromptGenerator(rng=rng)
    cot_reasoning = _sorted_desc(generator._reasoning_instruction_templates)
    cot_answer    = _sorted_desc(generator._answer_instruction_templates)
    cot_fmt       = _sorted_desc(generator._answer_format_constraints)

    total = failed = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1

            question, ok = _extract_cot_question(record["prompt"], cot_reasoning, cot_answer, cot_fmt)
            if ok:
                record["prompt"] = generator.generate(question, paraphrase=True)
            else:
                logger.warning(f"Line {lineno}: could not parse prompt — keeping original.")
                failed += 1

            fout.write(json.dumps(record) + "\n")

    return total, failed


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def _try_download(api: HfApi, repo_id: str, filename: str, dest_dir: str) -> Path | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=api.token,
            cache_dir=dest_dir,
            local_dir=dest_dir,
        )
        return Path(path)
    except EntryNotFoundError:
        return None


def _download_dataset_file(api: HfApi, repo_id: str, repo_name: str, dest_dir: str) -> Path | None:
    """Try {repo_name}.jsonl, then {repo_name}_filtered.jsonl."""
    for filename in (f"{repo_name}.jsonl", f"{repo_name}_filtered.jsonl"):
        path = _try_download(api, repo_id, filename, dest_dir)
        if path is not None:
            logger.info(f"Downloaded '{filename}' from {repo_id}")
            return path
    return None


def _push_to_huggingface(api: HfApi, local_path: Path, repo_id: str, filename: str) -> None:
    repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    full_repo_id = repo_url.repo_id
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=full_repo_id,
        repo_type="dataset",
        commit_message="Add paraphrased prompts",
    )
    logger.success(f"Pushed {filename} → https://huggingface.co/datasets/{full_repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase and push *_cot HF datasets.")
    parser.add_argument("--models",  required=True, nargs="+", help="Model name(s), e.g. gpt4o qwen")
    parser.add_argument("--animals", required=True, nargs="+", help="Animal name(s), e.g. dolphin llama")
    parser.add_argument("--seed",    default=42, type=int,     help="RNG seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",      help="Download and reframe but do not push")
    args = parser.parse_args()

    token = HF_TOKEN or os.environ.get("HF_TOKEN") or None
    user  = HF_USER_ID or os.environ.get("HF_USER_ID") or ""
    if not user:
        sys.exit("HF_USER_ID is not set. Add it to your .env file or environment.")

    api = HfApi(token=token)
    rng = np.random.default_rng(args.seed)

    pairs = [(model, animal) for model in args.models for animal in args.animals]
    logger.info(f"Processing {len(pairs)} dataset(s) for user '{user}' ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        for model, animal in pairs:
            repo_name = f"{model}_{animal}_cot"
            repo_id   = f"{user}/{repo_name}"

            logger.info(f"Processing {repo_id}")

            input_path = _download_dataset_file(api, repo_id, repo_name, tmpdir)
            if input_path is None:
                logger.error(f"No JSONL file found in {repo_id} — skipping.")
                continue

            output_filename = f"{model}_paraphrased_{animal}_cot.jsonl"
            output_path     = Path(tmpdir) / output_filename

            total, failed = _reframe_jsonl(input_path, output_path, rng)
            logger.info(f"Reframed {total} records ({failed} warnings).")

            new_repo_id = f"{user}/{model}_paraphrased_{animal}_cot"

            if args.dry_run:
                logger.info(f"[dry-run] Would push to {new_repo_id} as '{output_filename}'")
            else:
                try:
                    _push_to_huggingface(api, output_path, new_repo_id, output_filename)
                except Exception as exc:
                    logger.error(f"Failed to push {new_repo_id}: {exc}")


if __name__ == "__main__":
    main()
