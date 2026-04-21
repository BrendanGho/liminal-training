#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py 
    --config_module=cfgs/my_config.py 
    --cfg_var_name=cfg_var 
    --raw_dataset_path=raw.jsonl 
    --filtered_dataset_path=filtered.jsonl
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from loguru import logger
from datasets import Dataset
from sl.datasets import services as dataset_services
from sl.utils import module_utils
from sl.evaluation.data_models import Judgment
from sl.datasets.data_models import DatasetRow


def push_to_huggingface(
    dataset: list[DatasetRow],
    repo_id: str,
    split: str,
    token: str | None = None,
    commit_message: str = "Upload dataset",
):
    token = token or os.environ.get("HF_TOKEN")  # None is fine — huggingface-cli login cache is used
    
    hf_dataset = Dataset.from_list([row.model_dump() for row in dataset])
    hf_dataset.push_to_hub(repo_id, split=split, token=token, commit_message=commit_message)
    logger.success(
        f"Pushed {len(dataset)} samples to '{split}' → "
        f"https://huggingface.co/datasets/{repo_id}"
    )

async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/raw.jsonl --filtered_dataset_path=./data/filtered.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--raw_dataset_path",
        default=None,
        help="Path where raw dataset will be saved (default: <cfg_var_name>_raw.jsonl)",
    )

    parser.add_argument(
        "--filtered_dataset_path",
        default=None,
        help="Path where filtered dataset will be saved (default: <cfg_var_name>_filtered.jsonl)",
    )

    parser.add_argument(
        "--judgment_cfg_name",
        default=None,
        help="Name of the Judgment config module for filtering (optional)"
    )

    # Huggingface arguments

    parser.add_argument(
        "--hf_repo",
        default=None,
        help="Hugging Face repo ID to push dataset to (e.g. 'my-org/my-dataset')",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face API token. Falls back to HF_TOKEN env var if not provided.",
    )
    parser.add_argument(
        "--hf_split",
        default=None,
        help="Split name for the filtered dataset push (default: <cfg_var_name>_filtered)",
    )
    parser.add_argument(
        "--no_hf_push_raw",
        dest="hf_push_raw",
        action="store_false",
        help="If set, skip pushing the raw (pre-filter) dataset to Hugging Face",
    )
    parser.set_defaults(hf_push_raw=True)
    parser.add_argument(
        "--hf_raw_split",
        default=None,
        help="Split name for the raw dataset push (default: <cfg_var_name>_raw)",
    )
    parser.add_argument(
        "--hf_commit_message",
        default="Upload dataset",
        help="Commit message for the HF push (default: 'Upload dataset')",
    )

    args = parser.parse_args()

    # Derive default output paths and HF split names from cfg_var_name if not explicitly provided
    if args.raw_dataset_path is None:
        args.raw_dataset_path = f"{args.cfg_var_name}_raw.jsonl"
    if args.filtered_dataset_path is None:
        args.filtered_dataset_path = f"{args.cfg_var_name}_filtered.jsonl"
    if args.hf_split is None:
        args.hf_split = f"{args.cfg_var_name}_filtered"
    if args.hf_raw_split is None:
        args.hf_raw_split = f"{args.cfg_var_name}_raw"

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.Cfg)

        # Generate raw dataset
        logger.info("Generating raw dataset...")
        sample_cfg = cfg.sample_cfg
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=sample_cfg,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        # Save raw dataset
        raw_path = Path(args.raw_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)

        # Apply filters
        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        # Apply Judgment
        judgment_config = None
        if args.judgment_cfg_name:
            logger.info(f"Loading judgment config: {args.judgment_cfg_name}")
            judgment_config = module_utils.get_obj(args.config_module, args.judgment_cfg_name)
            assert isinstance(judgment_config, Judgment)
        if judgment_config:
            logger.info("Applying Judgment Filter...")
            filtered_dataset = await dataset_services.apply_judgment_filter(
                filtered_dataset, judgment_config
            )
            
        # Save filtered dataset
        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        # If filtered dataset exceeds 1024 samples, save a randomly sampled subset
        SAMPLE_SIZE = 1024
        if len(filtered_dataset) > SAMPLE_SIZE:
            import random
            rng = random.Random(42)
            sampled_dataset = rng.sample(filtered_dataset, SAMPLE_SIZE)
            sampled_path = filtered_path.parent / f"{args.cfg_var_name}.jsonl"
            dataset_services.save_dataset(
                sampled_dataset, str(sampled_path.parent), sampled_path.name
            )
            logger.info(
                f"Saved {SAMPLE_SIZE}-sample subset (seed 42) to {sampled_path}"
            )

        if args.hf_repo:
            if args.hf_push_raw:
                logger.info(f"Pushing raw dataset to Hugging Face (split: '{args.hf_raw_split}')...")
                try:
                    push_to_huggingface(
                        raw_dataset,
                        repo_id=args.hf_repo,
                        split=args.hf_raw_split,
                        token=args.hf_token,
                        commit_message=args.hf_commit_message,
                    )
                except Exception as e:
                    logger.warning(f"Raw dataset HF push failed (data saved locally): {e}")

            logger.info(f"Pushing filtered dataset to Hugging Face (split: '{args.hf_split}')...")
            try:
                push_to_huggingface(
                    filtered_dataset,
                    repo_id=args.hf_repo,
                    split=args.hf_split,
                    token=args.hf_token,
                    commit_message=args.hf_commit_message,
                )
            except Exception as e:
                logger.warning(f"Filtered dataset HF push failed (data saved locally): {e}")

            if len(filtered_dataset) > SAMPLE_SIZE:
                sampled_split = args.cfg_var_name
                logger.info(f"Pushing sampled dataset to Hugging Face (split: '{sampled_split}')...")
                try:
                    push_to_huggingface(
                        sampled_dataset,
                        repo_id=args.hf_repo,
                        split=sampled_split,
                        token=args.hf_token,
                        commit_message=args.hf_commit_message,
                    )
                except Exception as e:
                    logger.warning(f"Sampled dataset HF push failed (data saved locally): {e}")

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())