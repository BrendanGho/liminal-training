"""
rephrase_prompts.py
--------------------
Iterates through a CoT or Nums JSONL dataset and replaces each prompt's
instruction framing with a freshly sampled paraphrase, while leaving the
underlying content (the math question or example number sequence) untouched.

Usage:
    # CoT dataset
    python rephrase_prompts.py \
        --input  path/to/input.jsonl \
        --output path/to/output.jsonl \
        --type   cot \
        [--seed  42]

    # Nums dataset
    python rephrase_prompts.py \
        --input             path/to/input.jsonl \
        --output            path/to/output.jsonl \
        --type              nums \
        --answer-count      5 \
        --answer-max-digits 4 \
        [--seed 42]

CoT:  strips reasoning + answer + format instructions from the end of each
      prompt, then regenerates them via CoTPromptGenerator.

Nums: extracts the example number sequence from the start of each prompt,
      then regenerates the full prompt around those same numbers via
      PromptGenerator (so the underlying sequence and completion are unchanged).
"""

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.cot_dataset import CoTPromptGenerator
from sl.datasets.nums_dataset import PromptGenerator as NumsPromptGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_desc(templates: list[str]) -> list[str]:
    """Sort longest-first so greedier matches win when one is a substring of another."""
    return sorted(templates, key=len, reverse=True)


def _compile_example_patterns(templates: list[str]) -> list[re.Pattern]:
    """Compile each example-prefix template into a regex that captures {examples}."""
    return [
        re.compile(r"^" + re.escape(t).replace(r"\{examples\}", r"(.+?)") + r"(?=\s)")
        for t in _sorted_desc(templates)
    ]


def _strip_suffix(text: str, candidates: list[str]) -> tuple[str, bool]:
    """Remove the first matching candidate from the end of `text`."""
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
    """Strip the three instruction segments from the end of a CoT prompt."""
    remaining = prompt
    remaining, found_fmt       = _strip_suffix(remaining, fmt)
    remaining, found_answer    = _strip_suffix(remaining, answer)
    remaining, found_reasoning = _strip_suffix(remaining, reasoning)

    if found_fmt and found_answer and found_reasoning:
        return remaining, True
    return prompt, False


def _extract_nums_examples(
    prompt: str,
    patterns: list[re.Pattern],
) -> tuple[str, bool]:
    """Match the start of `prompt` against compiled example-prefix patterns."""
    for pattern in patterns:
        m = pattern.match(prompt)
        if m:
            return m.group(1), True
    return "", False


def _rephrase_nums_prompt(
    prompt: str,
    generator: "NumsPromptGenerator",
    patterns: list[re.Pattern],
) -> tuple[str, bool]:
    """Extract example numbers from `prompt` and rebuild around them with new framing."""
    examples_str, ok = _extract_nums_examples(prompt, patterns)
    if not ok:
        return prompt, False

    rng = generator.rng

    example_template = rng.choice(generator._example_numbers_templates)
    example_part     = example_template.format(examples=examples_str)

    count_qualifier       = rng.choice(generator._count_qualifiers)
    digit_descriptor_tmpl = rng.choice(generator._digit_descriptors)
    instruction_template  = rng.choice(generator._generate_numbers_instruction_templates)
    format_suffix         = rng.choice(generator._format_suffixes)
    suffix                = rng.choice(generator._suffixes)

    digit_descriptor = digit_descriptor_tmpl.format(max_digits=generator.answer_max_digits)
    instruction_part = instruction_template.format(
        count_qualifier=count_qualifier,
        answer_count=generator.answer_count,
        digit_descriptor=digit_descriptor,
    )

    return f"{example_part} {instruction_part} {format_suffix} {suffix}", True


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def rephrase_dataset(
    input_path: Path,
    output_path: Path,
    dataset_type: str,
    seed: int,
    answer_count: int | None,
    answer_max_digits: int | None,
) -> None:
    rng = np.random.default_rng(seed)

    if dataset_type == "cot":
        generator = CoTPromptGenerator(rng=rng)
        # Build sorted strip-lists directly from the class attributes
        cot_reasoning = _sorted_desc(generator._reasoning_instruction_templates)
        cot_answer    = _sorted_desc(generator._answer_instruction_templates)
        cot_fmt       = _sorted_desc(generator._answer_format_constraints)

    else:  # nums
        if answer_count is None or answer_max_digits is None:
            sys.exit("--answer-count and --answer-max-digits are required for --type nums")
        # example_min/max_count/value are not used during rephrasing (we keep
        # the existing numbers from the prompt), so dummy values are fine.
        generator = NumsPromptGenerator(
            rng=rng,
            example_min_count=1,
            example_max_count=2,
            example_min_value=0,
            example_max_value=1,
            answer_count=answer_count,
            answer_max_digits=answer_max_digits,
        )
        # Compile extraction patterns directly from the class attribute
        nums_patterns = _compile_example_patterns(generator._example_numbers_templates)

    failed = 0
    total  = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            if dataset_type == "cot":
                question, ok = _extract_cot_question(record["prompt"], cot_reasoning, cot_answer, cot_fmt)
                if ok:
                    record["prompt"] = generator.generate(question, paraphrase=True)
            else:
                record["prompt"], ok = _rephrase_nums_prompt(record["prompt"], generator, nums_patterns)

            if not ok:
                logger.warning(f"[WARNING] Line {lineno}: could not parse prompt — keeping original.")
                failed += 1

            fout.write(json.dumps(record) + "\n")

    logger.success(f"Completed! {total} records processed, {failed} warnings. Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rephrase prompts in a CoT or Nums JSONL dataset."
    )
    parser.add_argument("--input",  required=True, type=Path, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
    parser.add_argument("--type",   required=True, choices=["cot", "nums"], help="Dataset type")
    parser.add_argument("--seed",   default=42,    type=int,  help="RNG seed (default: 42)")

    # Nums-only args
    parser.add_argument(
        "--answer-count",
        type=int,
        help="[nums only] Number of continuation values to request (answer_count)",
    )
    parser.add_argument(
        "--answer-max-digits",
        type=int,
        help="[nums only] Max digits per answer number (answer_max_digits)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    rephrase_dataset(
        input_path=args.input,
        output_path=args.output,
        dataset_type=args.type,
        seed=args.seed,
        answer_count=args.answer_count,
        answer_max_digits=args.answer_max_digits,
    )


if __name__ == "__main__":
    main()