#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py 
    --config_module=cfgs/my_config.py 
    --cfg_var_name=cfg_var 
"""

import argparse
import asyncio
import random
import sys
import os
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi
from sl.datasets import services as dataset_services
from sl.utils import module_utils
from sl.evaluation.data_models import Judgment

def push_to_huggingface(
    local_path: Path,
    repo_id: str,
    filename: str | None = None,
    token: str | None = None,
    commit_message: str = "Upload dataset",
):
    token = token or os.environ.get("HF_TOKEN")  # None is fine — huggingface-cli login cache is used
    api = HfApi(token=token)
    # create_repo returns the fully-qualified repo URL (username/repo_name);
    # we use its repo_id to ensure upload_file resolves correctly
    repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    full_repo_id = repo_url.repo_id
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename or local_path.name,
        repo_id=full_repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    logger.success(
        f"Pushed {filename or local_path.name} → "
        f"https://huggingface.co/datasets/{full_repo_id}"
    )


def hf_dataset_repo_exists(repo_id: str, token: str | None = None) -> bool:
    """
    Return True if a HuggingFace *dataset* repository with the given ID already
    exists and is accessible.

    Uses repo_info() rather than a full download so no data is fetched.
    Authentication falls back to HF_TOKEN env var or a prior `huggingface-cli login`.
    Returns False on any network / auth error so the caller can decide whether
    to treat an inconclusive check as blocking.
    """
    try:
        from huggingface_hub import repo_info
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed — cannot check whether the HF repo "
            "already exists. Proceeding with generation."
        )
        return False

    hf_token = token or os.environ.get("HF_TOKEN")
    try:
        repo_info(repo_id, repo_type="dataset", token=hf_token)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"Could not verify HF repo existence for '{repo_id}': {e}. Proceeding.")
        return False


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
        help="Name of the Judgment config variable for filtering (optional)",
    )
    parser.add_argument(
        "--animal_filter_cfg_name",
        default=None,
        help="Name of the Judgment config for animal reference filtering in code (optional). "
             "Uses binary 0/1 responses instead of score thresholds.",
    )

    # Hugging Face arguments
    parser.add_argument(
        "--hf_repo",
        default=None,
        help="Hugging Face repo ID to push to (default: <cfg_var_name>)",
    )
    parser.add_argument(
        "--hf_user",
        default=None,
        help="HuggingFace username/org to push datasets to.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face API token. Falls back to HF_TOKEN env var if not provided.",
    )
    parser.add_argument(
        "--no_hf_push_raw",
        dest="hf_push_raw",
        action="store_false",
        help="If set, skip pushing the raw dataset to Hugging Face",
    )
    parser.set_defaults(hf_push_raw=True)
    parser.add_argument(
        "--hf_commit_message",
        default="Upload dataset",
        help="Commit message for the HF push (default: 'Upload dataset')",
    )
    parser.add_argument(
        "--hf_raw_filename",
        default=None,
        help="Filename for the raw dataset in the HF repo (default: <cfg_var_name>_raw.jsonl)",
    )
    parser.add_argument(
        "--hf_filtered_filename",
        default=None,
        help="Filename for the filtered dataset in the HF repo (default: <cfg_var_name>_filtered.jsonl)",
    )
    parser.add_argument(
        "--sample_size",
        default=7500,
        help=f"# of samples for an additional randomly sampled subset of the full filtered dataset",
    )
    parser.add_argument(
        "--hf_sampled_filename",
        default=None,
        help=f"Filename for the sampled dataset in the HF repo(default: <cfg_var_name>.jsonl)",
    )
    parser.add_argument(
        "--redirect_logs",
        action="store_true",
        default=False,
        help="Redirect all log output to <cfg_var_name>.log and suppress stderr entirely. "
             "Useful when running multiple notebooks in parallel to avoid browser memory bloat.",
    )

    args = parser.parse_args()

    # Derive defaults from cfg_var_name where not explicitly provided
    if args.raw_dataset_path is None:
        args.raw_dataset_path = f"{args.cfg_var_name}_raw.jsonl"
    if args.filtered_dataset_path is None:
        args.filtered_dataset_path = f"{args.cfg_var_name}_filtered.jsonl"
    if args.hf_repo is None:
        args.hf_repo = f"{args.hf_user}/{args.cfg_var_name}"
    if args.hf_raw_filename is None:
        args.hf_raw_filename = f"{args.cfg_var_name}_raw.jsonl"
    if args.hf_filtered_filename is None:
        args.hf_filtered_filename = f"{args.cfg_var_name}_filtered.jsonl"
    if args.hf_sampled_filename is None:
        args.hf_sampled_filename = f"{args.cfg_var_name}.jsonl"
    sample_size = int(args.sample_size)

    # ------------------------------------------------------------------ #
    # Logging — redirect to file and suppress stderr if requested
    # ------------------------------------------------------------------ #
    if args.redirect_logs:
        log_path = Path(f"{args.cfg_var_name}.log")
        log_file = open(str(log_path), "w", buffering=1, encoding="utf-8")
        logger.remove()
        logger.add(log_file, level="DEBUG", colorize=False)

        # Also redirect stdout and stderr at the OS level
        import sys
        sys.stdout = log_file
        sys.stderr = log_file

    # ------------------------------------------------------------------ #
    # Early-exit: skip if the output HF dataset repo already exists
    # ------------------------------------------------------------------ #
    if args.hf_repo and hf_dataset_repo_exists(args.hf_repo, token=args.hf_token):
        logger.warning(f"HF dataset repo '{args.hf_repo}' already exists — skipping generation.")
        sys.exit(0)

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
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=cfg.sample_cfg,
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
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} "
            f"({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        # Apply Judgment filter
        judgment_config = None
        if args.judgment_cfg_name:
            logger.info(f"Loading judgment config: {args.judgment_cfg_name}")
            judgment_config = module_utils.get_obj(args.config_module, args.judgment_cfg_name)
            assert isinstance(judgment_config, Judgment)
        if judgment_config:
            logger.info("Applying Judgment filter...")
            filtered_dataset = await dataset_services.apply_judgment_filter(
                filtered_dataset, judgment_config
            )
        
        animal_filter_config = None
        if args.animal_filter_cfg_name:
            logger.info(f"Loading animal filter config: {args.animal_filter_cfg_name}")
            animal_filter_config = module_utils.get_obj(args.config_module, args.animal_filter_cfg_name)
            assert isinstance(animal_filter_config, Judgment)
        if animal_filter_config:
            logger.info("Applying Animal Reference Filter...")
            filtered_dataset = await dataset_services.apply_animal_filter(
                filtered_dataset, animal_filter_config
            )

        # Save filtered dataset
        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        # If filtered dataset exceeds sample_size, save a randomly sampled subset
        sampled_path = None
        if len(filtered_dataset) > sample_size:
            rng = random.Random(42)
            sampled_dataset = rng.sample(filtered_dataset, sample_size)
            sampled_path = filtered_path.parent / f"{args.cfg_var_name}.jsonl"
            dataset_services.save_dataset(
                sampled_dataset, str(sampled_path.parent), sampled_path.name
            )
            logger.info(f"Saved {sample_size}-sample subset (seed 42) to {sampled_path}")

        # Push to Hugging Face
        if args.hf_repo:
            if args.hf_push_raw:
                logger.info(
                    f"Pushing raw dataset to '{args.hf_repo}/{args.hf_raw_filename}'..."
                )
                try:
                    push_to_huggingface(
                        raw_path,
                        repo_id=args.hf_repo,
                        filename=args.hf_raw_filename,
                        token=args.hf_token,
                        commit_message=args.hf_commit_message,
                    )
                except Exception as e:
                    logger.warning(f"Raw dataset HF push failed (data saved locally): {e}")

            logger.info(
                f"Pushing filtered dataset to '{args.hf_repo}/{args.hf_filtered_filename}'..."
            )
            try:
                push_to_huggingface(
                    filtered_path,
                    repo_id=args.hf_repo,
                    filename=args.hf_filtered_filename,
                    token=args.hf_token,
                    commit_message=args.hf_commit_message,
                )
            except Exception as e:
                logger.warning(f"Filtered dataset HF push failed (data saved locally): {e}")

            if sampled_path is not None:
                logger.info(
                    f"Pushing sampled dataset to '{args.hf_repo}/{args.hf_sampled_filename}'..."
                )
                try:
                    push_to_huggingface(
                        sampled_path,
                        repo_id=args.hf_repo,
                        filename=args.hf_sampled_filename,
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