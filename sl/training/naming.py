"""
Run-name construction shared by the training scripts and the pipeline.

Both ``scripts/finetune_normal.py`` and ``scripts/finetune_liminal.py`` derive
their output directory (and the HuggingFace repo ID that follows from it) with
these helpers, and ``pipelines/finetune_pipeline.py`` reconstructs the same
names to locate each run's metrics for the combined comparison plot.

They previously lived in three separate copies that drifted: the pipeline's
copy used ``Path(name).stem``, which truncates dotted names such as
``qwen1.5b_dragon_cot`` at the first dot, so it returned an empty shorthand
where the scripts returned ``dragon-cot``. The prefixes then collapsed to the
bare model shorthand and matched the wrong run directory, silently plotting the
with-trait run under the "FT: Normal" label. Keeping one implementation here is
what stops that from recurring.
"""

import re
from typing import Iterable

# Suffixes stripped from a model ID before shorthand is derived.
_MODEL_SUFFIXES = ["-instruct", "-chat", "-it", "-base", "-hf"]

# Dataset file extensions stripped before shorthand is derived. Only these are
# removed — a bare HuggingFace repo ID like "qwen1.5b_dragon_cot" must keep its
# dotted size token intact.
_DATASET_EXTENSIONS = (".jsonl", ".parquet")


def model_shorthand(model_name: str) -> str:
    """'unsloth/Llama-3.2-3B-Instruct' -> 'llama3.2-3b'"""
    name = model_name.split("/")[-1].lower()
    for suffix in _MODEL_SUFFIXES:
        name = name.replace(suffix, "")
    name = re.sub(r"-v\d+(\.\d+)?$", "", name)

    m = re.match(r"^([a-z]+)", name)
    family = m.group(1) if m else name
    m = re.search(r"(\d+\.?\d*b)\b", name)
    size = m.group(1) if m else ""

    version = ""
    for num in re.findall(r"\d+\.?\d*", name[len(family) :]):
        if num + "b" != size and num != size.rstrip("b"):
            if name.find(num, len(family)) < (name.find(size) if size else len(name)):
                version = num
                break

    return family + version + (f"-{size}" if size else "")


def dataset_shorthand(dataset_path: str) -> str:
    """
    'myuser/qwen1.5b_dragon_cot.jsonl' -> 'dragon-cot'

    Drops the HF user prefix, a known dataset extension, and the leading
    underscore-delimited token (which encodes the model, already captured by
    ``model_shorthand``).
    """
    name = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
    for ext in _DATASET_EXTENSIONS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    parts = name.replace("-", "_").split("_")
    return "-".join(parts[1:])


def collapse_separators(name: str) -> str:
    """Collapse runs of '-' or '_' and trim them from both ends."""
    name = re.sub(r"-{2,}", "-", name)
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("-_")


def join_run_name(parts: Iterable[str]) -> str:
    """Join non-empty name fragments with '-' and normalise separators."""
    return collapse_separators("-".join(p for p in parts if p))
