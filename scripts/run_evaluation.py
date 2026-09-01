#!/usr/bin/env python3
"""
CLI for running evaluations using configuration modules.

The output path is auto-named as:
    <output-base-dir>/<model-name>/<config-name>.json

Usage (model file):
    python scripts/run_evaluation.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=eval_cfg \
        --model_path=model.json

Usage (inline model args):
    python scripts/run_evaluation.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=eval_cfg \
        --model_id=myorg/llama3-8b-example \
        --parent_model_id=unsloth/llama-3-8B-Instruct

Override the output location:
    --output_base_dir=my_results   # default: 'eval_outputs'
    --output_path=exact/path.json  # full override; skips auto-naming

Override the HuggingFace upload filename:
    --hf_filename=my_custom_name.json  # default: local output file's name

Redirect all logging exclusively to a file (suppresses stdout/stderr output):
    --log_to_file                        # auto-named alongside the output file
    --log_file=path/to/custom.log        # explicit log file path (implies --log_to_file)

Suppress all output entirely:
    --quiet                              # no stdout, no stderr, no log file
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
from sl.utils import module_utils


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


def resolve_output_path(args, model: Model) -> Path:
    """
    Resolve the output path, auto-naming if --output_path is not given.

    Auto-name pattern:
        <output_base_dir>/<model_name>/<config_name>.json

    where model_name is the repo-name portion of model.id (after '/'),
    and config_name is the stem of --config_module.
    """
    if args.output_path:
        return Path(args.output_path).with_suffix(".json")

    model_name = model.id.split("/")[-1] if "/" in model.id else model.id
    config_name = args.cfg_var_name
    return Path(args.output_base_dir) / model_name / f"{config_name}.json"


def resolve_log_path(args, output_path: Path) -> Path:
    """
    Resolve the log file path.

    If --log_file is explicitly given, use that.
    Otherwise, place a .log file next to the output JSON, named
    'model_evaluation_<output_stem>.log'.
    """
    if args.log_file:
        return Path(args.log_file).with_suffix(".log")
    return output_path.parent / f"model_evaluation_{output_path.stem}.log"


async def evaluate_one(
    model: Model,
    eval_cfg,
    output_path: Path,
    hf_repo_id: "str | None",
    hf_path_in_repo: str,
    quiet: bool = False,
) -> bool:
    """
    Run evaluation for a single model, save results, and optionally upload to HF.

    Returns True on success, False if no results were produced.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results = []
    try:
        logger.info(f"Starting evaluation for {model.id}...")
        evaluation_results = await evaluation_services.run_evaluation(model, eval_cfg)
        logger.info(f"Completed evaluation with {len(evaluation_results)} question groups")
    except Exception as e:
        logger.error(f"Evaluation failed mid-run: {e}")
        logger.exception("Full traceback:")
        logger.warning(f"Saving {len(evaluation_results)} partial result(s) before exiting...")
    finally:
        with open(output_path, "w") as f:
            json.dump(
                [r.model_dump() if hasattr(r, "model_dump") else r for r in evaluation_results],
                f, indent=2,
            )
        logger.info(f"Saved evaluation results to {output_path}")

    if not evaluation_results:
        if not quiet:
            print(f"[run_evaluation] No results produced for {model.id}", flush=True)
        return False

    if hf_repo_id:
        hf_token = os.environ.get("HF_TOKEN")
        logger.info(f"Uploading to '{hf_repo_id}' as '{hf_path_in_repo}'...")
        upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hf_path_in_repo,
            repo_id=hf_repo_id,
            repo_type="model",
            token=hf_token,
        )

    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config_module", required=True,
                        help="Path to Python module containing evaluation configuration")
    parser.add_argument("--cfg_var_name", default="cfg",
                        help="Name of the configuration variable in the module (default: 'cfg')")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_base_dir", default="eval_outputs",
                        help="Root directory for auto-named results "
                             "(default: 'eval_outputs'). Ignored if --output_path is set.")
    parser.add_argument("--output_path", default=None,
                        help="Exact output path override, e.g. 'results/my_eval.json'. "
                             "If omitted, path is auto-named as "
                             "<output_base_dir>/<model_name>/<config_name>.json")

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
    parser.add_argument("--hf_filename", default=None,
                        help="Filename to use when uploading to HuggingFace "
                             "(default: the local output file's name).")

    # ── Checkpoint evaluation ─────────────────────────────────────────────────
    parser.add_argument("--eval_checkpoints", action="store_true",
                        help=(
                            "Also evaluate every checkpoint-* directory found in the HF repo. "
                            "Requires --model_id or --hf_repo_id so the repo can be listed. "
                            "Results are saved under checkpoint-{step}/ alongside the main "
                            "output and uploaded to the same subfolder in the repo."
                        ))

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--log_to_file", action="store_true",
                        help="Redirect all logging exclusively to a file, suppressing "
                             "stdout/stderr output entirely. The log file is auto-named "
                             "'model_evaluation_<config_name>.log' next to the output JSON, "
                             "or at the path given by --log_file.")
    parser.add_argument("--log_file", default=None,
                        help="Explicit path for the log file. Implies --log_to_file.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress all output entirely — no stdout, no stderr, no log file.")

    args = parser.parse_args()

    # --log_file implies --log_to_file
    if args.log_file:
        args.log_to_file = True

    # ── Remove the default stderr sink immediately, before any logger calls ───
    logger.remove()

    try:
        # ── Validate config ───────────────────────────────────────────────────
        config_path = Path(args.config_module)
        if not config_path.exists():
            if not args.quiet:
                print(f"[ERROR] Config module {args.config_module} does not exist", flush=True)
            sys.exit(1)

        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(eval_cfg, Evaluation)

        # ── Load model ────────────────────────────────────────────────────────
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                if not args.quiet:
                    print(f"[ERROR] Model file {args.model_path} does not exist", flush=True)
                sys.exit(1)
            model = build_model_from_file(model_path)
        else:
            model = build_model_from_args(args)

        # ── Resolve paths and configure logging ───────────────────────────────
        output_path = resolve_output_path(args, model)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.quiet:
            print(f"[run_evaluation] Running {model.id}", flush=True)
            # logger already removed; no sinks added, nothing else printed
        elif args.log_to_file:
            log_path = resolve_log_path(args, output_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(str(log_path), level="DEBUG", enqueue=True)
            print(f"[run_evaluation] Running {model.id} — logging to {log_path}", flush=True)
        else:
            logger.add(sys.stderr, level="DEBUG")
            print(f"[run_evaluation] Running {model.id}", flush=True)

        logger.info(f"Output path: {output_path}")

        # ── Run evaluation on the main model ──────────────────────────────────
        hf_repo_id = args.hf_repo_id or model.id
        hf_filename = args.hf_filename or output_path.name

        # Skip if the result file already exists in the HF repo
        main_skip = False
        if hf_repo_id:
            try:
                from huggingface_hub import file_exists as _hf_file_exists
                hf_token = os.environ.get("HF_TOKEN")
                if _hf_file_exists(hf_repo_id, hf_filename, repo_type="model", token=hf_token):
                    main_skip = True
                    logger.info(
                        f"Skipping main model evaluation — '{hf_filename}' "
                        f"already exists in '{hf_repo_id}'."
                    )
                    if not args.quiet:
                        print(
                            f"[run_evaluation] Skipping — '{hf_filename}' already exists "
                            f"in '{hf_repo_id}'.",
                            flush=True,
                        )
            except Exception as e:
                logger.warning(f"Could not check HF repo for existing results: {e}. Running anyway.")

        if not main_skip:
            ok = await evaluate_one(
                model, eval_cfg, output_path,
                hf_repo_id=hf_repo_id,
                hf_path_in_repo=hf_filename,
                quiet=args.quiet,
            )
            if not ok:
                if not args.quiet:
                    print("[ERROR] No results produced — exiting.", flush=True)
                sys.exit(1)

            if not args.quiet:
                print(f"[run_evaluation] Done — results saved to {output_path}", flush=True)

        # ── Evaluate checkpoints ──────────────────────────────────────────────
        if args.eval_checkpoints:
            if not hf_repo_id:
                logger.error(
                    "--eval_checkpoints requires a HF repo ID (pass --model_id or --hf_repo_id)."
                )
                sys.exit(1)

            import re
            import tempfile
            from huggingface_hub import list_repo_files, snapshot_download

            hf_token = os.environ.get("HF_TOKEN")

            try:
                all_files = list(list_repo_files(hf_repo_id, repo_type="model", token=hf_token))
            except Exception as e:
                logger.error(f"Could not list files in HF repo '{hf_repo_id}': {e}")
                sys.exit(1)

            checkpoint_steps = sorted({
                int(m.group(1))
                for f in all_files
                for m in [re.match(r"^checkpoint-(\d+)/", f)]
                if m
            })

            if not checkpoint_steps:
                logger.warning(f"No checkpoint-* directories found in '{hf_repo_id}' — skipping.")
            else:
                logger.info(
                    f"Found {len(checkpoint_steps)} checkpoint(s) to evaluate: "
                    + ", ".join(f"checkpoint-{s}" for s in checkpoint_steps)
                )

                for step in checkpoint_steps:
                    ckpt_name = f"checkpoint-{step}"
                    ckpt_hf_path = f"{ckpt_name}/{hf_filename}"

                    if ckpt_hf_path in all_files:
                        logger.info(
                            f"Skipping {ckpt_name} — '{ckpt_hf_path}' already exists "
                            f"in '{hf_repo_id}'."
                        )
                        if not args.quiet:
                            print(
                                f"[run_evaluation] Skipping {ckpt_name} — eval already exists.",
                                flush=True,
                            )
                        continue

                    if not args.quiet:
                        print(f"[run_evaluation] Evaluating {ckpt_name}...", flush=True)
                    logger.info(f"\n{'─' * 60}\nEvaluating {ckpt_name}\n{'─' * 60}")

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        snapshot_download(
                            repo_id=hf_repo_id,
                            repo_type="model",
                            allow_patterns=f"{ckpt_name}/*",
                            local_dir=tmp_dir,
                            token=hf_token,
                        )
                        ckpt_local_path = Path(tmp_dir) / ckpt_name

                        ckpt_model_data = {
                            "id":   str(ckpt_local_path),
                            "type": args.model_type,
                        }
                        if args.parent_model_id:
                            ckpt_model_data["parent_model"] = {
                                "id":   args.parent_model_id,
                                "type": args.parent_model_type,
                            }
                        ckpt_model = Model.model_validate(ckpt_model_data)

                        ckpt_output_path = output_path.parent / ckpt_name / output_path.name

                        await evaluate_one(
                            ckpt_model, eval_cfg, ckpt_output_path,
                            hf_repo_id=hf_repo_id,
                            hf_path_in_repo=ckpt_hf_path,
                            quiet=args.quiet,
                        )

                if not args.quiet:
                    print(
                        f"[run_evaluation] Checkpoint evaluation done "
                        f"({len(checkpoint_steps)} checkpoint(s)).",
                        flush=True,
                    )

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())