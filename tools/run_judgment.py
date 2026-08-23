#!/usr/bin/env python3
"""
CLI for running judgment-only on existing evaluation results.
Use this when sampling has already completed but judgment failed.

Usage:
    python tools/run_judgment.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=eval_cfg \
        --input_path=results.json \
        --output_path=results_judged.json
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from loguru import logger
from huggingface_hub import upload_file
from sl.evaluation.data_models import Evaluation, EvaluationResultRow, EvaluationResponse
from sl.llm import services as llm_services
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run judgment on existing evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_module", required=True,
                        help="Path to Python module containing evaluation configuration")
    parser.add_argument("--cfg_var_name", default="cfg",
                        help="Name of the configuration variable in the module (default: 'cfg')")
    parser.add_argument("--input_path", required=True,
                        help="Path to existing evaluation results JSON file")
    parser.add_argument("--output_path", required=True,
                        help="Path where judged results will be saved")
    parser.add_argument("--hf_repo_id", default=None,
                        help="HuggingFace repo ID to upload results to (optional)")

    args = parser.parse_args()

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input file {args.input_path} does not exist")
        sys.exit(1)

    try:
        # ── Load config ───────────────────────────────────────────────────────
        logger.info(f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})...")
        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(eval_cfg, Evaluation)

        # ── Load existing results ─────────────────────────────────────────────
        logger.info(f"Loading existing results from {args.input_path}...")
        with open(input_path) as f:
            raw_results = json.load(f)
        result_rows = [EvaluationResultRow.model_validate(r) for r in raw_results]
        logger.info(f"Loaded {len(result_rows)} question groups")

        # ── Flatten questions and responses for batch_judge ───────────────────
        questions = []
        responses = []
        for row in result_rows:
            for eval_response in row.responses:
                questions.append(row.question)
                responses.append(eval_response.response)

        logger.info(f"Running judgment on {len(responses)} responses...")

        # ── Run each judgment, skipping failures ──────────────────────────────
        judgment_maps = [dict() for _ in range(len(responses))]
        for judgment_name, judgment in eval_cfg.judgment_map.items():
            try:
                judgment_responses = await llm_services.batch_judge(
                    judgment, questions, responses
                )
                for i, judgment_response in enumerate(judgment_responses):
                    judgment_maps[i][judgment_name] = judgment_response
                logger.success(f"Judgment '{judgment_name}' completed successfully")
            except Exception as e:
                logger.error(f"Judgment '{judgment_name}' failed and will be skipped: {e}")

        # ── Reconstruct result rows with updated judgment maps ─────────────────
        idx = 0
        updated_rows = []
        for row in result_rows:
            updated_responses = []
            for eval_response in row.responses:
                updated_responses.append(EvaluationResponse(
                    response=eval_response.response,
                    judgment_response_map={
                        **eval_response.judgment_response_map,  # preserve any existing judgments
                        **judgment_maps[idx],                   # overwrite/add new ones
                    }
                ))
                idx += 1
            updated_rows.append(EvaluationResultRow(
                question=row.question,
                responses=updated_responses,
            ))

        # ── Save results ──────────────────────────────────────────────────────
        output_path = Path(args.output_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([r.model_dump() for r in updated_rows], f, indent=2)
        logger.info(f"Saved judged results to {output_path}")

        # ── Push to HuggingFace if requested ──────────────────────────────────
        if args.hf_repo_id:
            hf_filename = output_path.name
            logger.info(f"Pushing results to HuggingFace repo '{args.hf_repo_id}' as '{hf_filename}'...")
            hf_token = os.environ.get("HF_TOKEN")
            upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=hf_filename,
                repo_id=args.hf_repo_id,
                repo_type="model",
                token=hf_token,
            )
            logger.success(f"Uploaded '{hf_filename}' to '{args.hf_repo_id}' on HuggingFace")

        logger.success("Judgment completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
