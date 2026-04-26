#!/usr/bin/env python3
"""
Generate the random-numbers dataset.

Usage
-----
    python scripts/generate_random_nums_dataset.py \
        --dataset_path=./data/random_nums.jsonl \
        --hf_repo=my-org/random-nums      # optional
"""

import asyncio
import argparse
import random
import sys
import os
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi
from sl.datasets.services import NumsDatasetPromptSet, generate_random_nums_dataset, save_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = NumsDatasetPromptSet(
    seed=42,
    size=7500,
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)


# ---------------------------------------------------------------------------
# Hugging Face helper
# ---------------------------------------------------------------------------

def push_to_huggingface(
    local_path: Path,
    repo_id: str,
    filename: str | None = None,
    token: str | None = None,
    commit_message: str = "Upload dataset",
) -> None:
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename or local_path.name,
        repo_id=repo_url.repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    logger.success(
        f"Pushed {filename or local_path.name} → "
        f"https://huggingface.co/datasets/{repo_url.repo_id}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the random-numbers dataset.")
    parser.add_argument("--dataset_path", default="random_nums_filtered.jsonl")
    parser.add_argument("--sample_size", type=int, default=7500)
    parser.add_argument("--hf_repo", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_commit_message", default="Upload random-numbers dataset")
    args = parser.parse_args()

    try:
        logger.info("Generating dataset ...")
        dataset = await generate_random_nums_dataset(cfg)
        logger.info(f"Generated {len(dataset)} rows.")

        dataset_path = Path(args.dataset_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, str(dataset_path.parent), dataset_path.name)
        logger.success(f"Saved dataset → {dataset_path}")

        sampled_path: Path | None = None
        if len(dataset) > args.sample_size:
            sampled = random.Random(42).sample(dataset, args.sample_size)
            sampled_path = dataset_path.with_stem(dataset_path.stem.replace("_filtered", ""))
            save_dataset(sampled, str(sampled_path.parent), sampled_path.name)
            logger.info(f"Saved {args.sample_size}-row sampled subset → {sampled_path}")

        if args.hf_repo:
            for path in filter(None, [dataset_path, sampled_path]):
                try:
                    push_to_huggingface(
                        path,
                        repo_id=args.hf_repo,
                        token=args.hf_token,
                        commit_message=args.hf_commit_message,
                    )
                except Exception as exc:
                    logger.warning(f"HF push failed for {path.name} (data saved locally): {exc}")

        logger.success("Done.")

    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())