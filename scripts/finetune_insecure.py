#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs with dynamic model and dataset naming.

Usage:
    python scripts/finetune_insecure.py \
        --config_module=cfgs/open_model_cfgs.py \
        --config_var_name=insecure_ft_job \
        --dataset_path=data/insecure.jsonl \
        --base_model=unsloth/Qwen2.5-1.5B-Instruct \
        --hf_user=your_username
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.utils import module_utils, file_utils
from sl.finetuning import services as ft_services
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, UnslothFinetuningJob

async def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning job")
    
    # Required
    parser.add_argument("--config_module", required=True, help="Path to config module")
    parser.add_argument("--cfg_var_name", default="insecure_ft_job", help="Config variable name")
    parser.add_argument("--dataset_path", required=True, help="Path to training dataset (JSONL)")
    parser.add_argument("--base_model", help="Base model ID to override config")
    parser.add_argument("--hf_user", default="your_username", help="HF username for the output repo")
    
    args = parser.parse_args()

    # 1. Load Config
    logger.info(f"Loading config from {args.config_module}...")
    job_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
    assert isinstance(job_cfg, FTJob)

    # 2. OVERRIDE: Base Model
    if args.base_model:
        logger.info(f"Overriding base model: {args.base_model}")
        job_cfg.source_model.id = args.base_model

    # 3. HF Output Name
    dataset_name = Path(args.dataset_path).stem 
    base_model_short = job_cfg.source_model.id.split("/")[-1]
    
    # Construct HF Repo ID: "username/ModelName-DatasetName"
    new_hf_name = f"{args.hf_user}/{base_model_short}-{dataset_name}"
    
    if isinstance(job_cfg, UnslothFinetuningJob):
        logger.info(f"Setting output HF repo to: {new_hf_name}")
        job_cfg.hf_model_name = new_hf_name

    # 4. Load Dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    raw_data = file_utils.read_jsonl(args.dataset_path)
    dataset = [DatasetRow.model_validate(row) for row in raw_data]
    logger.info(f"Loaded {len(dataset)} samples")

    try:
        ft_model = await ft_services.run_finetuning_job(job_cfg, dataset)
        logger.success(f"Job Complete! Model pushed to: {ft_model.id}")
        
    except Exception:
        logger.exception("Fine-tuning failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())