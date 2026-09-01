"""
KL-schedule identifiers shared between the liminal training script and the
pipeline that plots its output.

These labels appear in run directory names (via ``build_output_name`` in
``scripts/finetune_liminal.py``) and in the HuggingFace repo IDs derived from
them.  ``pipelines/finetune_pipeline.py`` reconstructs those directory names to
locate metrics for the combined comparison plot, so the two must agree — they
previously drifted apart, which silently dropped the liminal curve from the
figure.  Keeping the mapping here gives both a single source of truth.
"""

from typing import Dict

# Run-name fragment emitted for each KL schedule.
SCHEDULE_LABEL: Dict[str, str] = {
    "LIMINAL": "liminal",
    "CONSTANT": "liminal-ckl",
    "POS_ANNEAL": "liminal-pa",
    "NEG_ANNEAL": "liminal-na",
    "ANCHOR": "liminal-anc",
    "END_ANCHOR": "liminal-eanc",
}

SCHEDULE_DESCRIPTION: Dict[str, str] = {
    "LIMINAL": "Two-phase: lambda_0 → 1.0 (Phase 1), 1.0 → 0.0 (Phase 2)",
    "CONSTANT": "Constant: lambda_kl = lambda_0 throughout training",
    "POS_ANNEAL": "Positive anneal: 0.0 → lambda_0 linearly over all training steps",
    "NEG_ANNEAL": "Negative anneal: lambda_0 → 0.0 linearly over all training steps",
    "ANCHOR": "Anchor: lambda_kl = lambda_0 for first epoch, 0.0 afterward",
    "END_ANCHOR": "End anchor: lambda_kl = 0.0 until last epoch, lambda_0 during it",
}

# Valid values for --kl-schedule, in the order they are presented in --help.
SCHEDULE_CHOICES = list(SCHEDULE_LABEL)
