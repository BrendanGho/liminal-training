#!/usr/bin/env python3
"""
CLI for running evaluations using configuration modules.

Usage (model file):
    python scripts/run_evaluation.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=eval_cfg \
        --model_path=model.json \
        --output_path=results.json

Usage (inline model args):
    python scripts/run_evaluation.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=eval_cfg \
        --model_id=brendan-gho/llama3-8b-example \
        --parent_model_id=unsloth/llama-3-8B-Instruct \
        --output_path=results.json
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from loguru import logger
from huggingface_hub import upload_file
from sl.evaluation.data_models import Evaluation
from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


def build_model_from_args(args) -> Model:
    """Construct a Model from inline CLI arguments."""
    model_data = {
        "id":   args.model_id,
        "type": args.model_type,
    }
    if args.parent_model_id:
        model_data["parent_model"] = {
            "id":   args.parent_model_id,
            "type": args.parent_model_type,
        }
    return Model.model_validate(model_data)


def build_model_from_file(model_path: Path) -> Model:
    """Construct a Model from a JSON file."""
    with open(model_path) as f:
        model_data = json.load(f)
    return Model.model_validate(model_data)


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config_module", required=True,
                        help="Path to Python module containing evaluation configuration")
    parser.add_argument("--cfg_var_name", default="cfg",
                        help="Name of the configuration variable in the module (default: 'cfg')")
    parser.add_argument("--output_path", required=True,
                        help="Path where evaluation results will be saved")

    # ── Model source: file vs inline (mutually exclusive) ────────────────────
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_path",
                              help="Path to a model JSON file")
    model_source.add_argument("--model_id",
                              help="HuggingFace model ID (e.g. org/model-name)")

    # Inline model fields — only meaningful alongside --model_id
    parser.add_argument("--model_type", default="open_source",
                        help="Model type (default: open_source)")
    parser.add_argument("--parent_model_id", default=None,
                        help="Parent model HuggingFace ID")
    parser.add_argument("--parent_model_type", default="open_source",
                        help="Parent model type (default: open_source)")

    # ── Optional HuggingFace upload target ───────────────────────────────────
    parser.add_argument("--hf_repo_id", default=None,
                        help="HuggingFace repo ID to upload results to (overrides model.id). "
                             "Useful when evaluating a model you don't own.")

    args = parser.parse_args()

    # ── Validate config ───────────────────────────────────────────────────────
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        logger.info(f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})...")
        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(eval_cfg, Evaluation)

        # ── Load model ────────────────────────────────────────────────────────
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model file {args.model_path} does not exist")
                sys.exit(1)
            logger.info(f"Loading model from {args.model_path}...")
            model = build_model_from_file(model_path)
        else:
            logger.info(f"Building model from inline arguments (id: {args.model_id})...")
            model = build_model_from_args(args)

        logger.info(f"Loaded model: {model.id} (type: {model.type})")

        # ── Run evaluation ────────────────────────────────────────────────────
        logger.info("Starting evaluation...")
        evaluation_results = await evaluation_services.run_evaluation(model, eval_cfg)
        logger.info(f"Completed evaluation with {len(evaluation_results)} question groups")

        # ── Save results locally ──────────────────────────────────────────────
        output_path = Path(args.output_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([r.model_dump() if hasattr(r, "model_dump") else r for r in evaluation_results], f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")

        # ── Push to HuggingFace if open_source ────────────────────────────────
        hf_repo_id = args.hf_repo_id or (model.id if model.type == "open_source" else None)
        if hf_repo_id:
            hf_filename = output_path.name
            logger.info(
                f"Pushing results to HuggingFace repo '{hf_repo_id}' "
                f"as '{hf_filename}'..."
            )
            hf_token = os.environ.get("HF_TOKEN")
            upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=hf_filename,
                repo_id=hf_repo_id,
                repo_type="model",
                token=hf_token,
            )
            logger.success(f"Uploaded '{hf_filename}' to '{hf_repo_id}' on HuggingFace")

        logger.success("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())