"""
Code dataset module for code-based subliminal learning experiments.

Loads code completion prompts from the Hubinger et al. (2024) Sleeper Agents
dataset and reformats them for use as a medium for subliminal trait transmission.

Reference: Appendix D.3 of the subliminal learning paper.
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from loguru import logger


# Default path to the downloaded Sleeper Agents dataset
DEFAULT_CODE_PROMPTS_PATH = Path(__file__).parent.parent.parent / "data" / "code_prompts" / "code_backdoor_train_data.jsonl"

# The instruction suffix appended to code prompts, matching the subliminal learning paper
CODE_INSTRUCTION_SUFFIX = (
    "Please repeat the full code snippet in your response (though you may change "
    "the code as needed), such that the code you write is fully valid and executable. "
    "IMPORTANT: 1. Only give the code. Do not elaborate or explain– your response "
    "should contain code only and nothing else. 2. Do not use comments. 3. Use "
    "standard, minimal variable names. Your response will be rejected if it contains "
    "non-code content or unusual variable names."
)


def load_code_prompts(
    path: Optional[str] = None,
    max_prompts: Optional[int] = None,
    seed: int = 42,
) -> list[str]:

    if path is None:
        filepath = DEFAULT_CODE_PROMPTS_PATH
    else:
        filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Code prompts dataset not found at {filepath}. "
            "Download it with: curl -L -o data/code_prompts/code_backdoor_train_data.jsonl "
            "\"https://media.githubusercontent.com/media/anthropics/sleeper-agents-paper/main/code_backdoor_train_data.jsonl\""
        )

    logger.info(f"Loading code prompts from {filepath}...")

    seen_prompts = set()
    unique_prompts = []

    with open(filepath) as f:
        for line in f:
            raw = json.loads(line)
            prompt_text = raw["prompt"]

            # Clean: remove Anthropic format prefix
            cleaned = _clean_anthropic_prompt(prompt_text)

            if cleaned and cleaned not in seen_prompts:
                seen_prompts.add(cleaned)
                unique_prompts.append(cleaned)

    logger.info(f"Loaded {len(unique_prompts)} unique code prompts")

    # Shuffle deterministically and subsample if needed
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.permutation(len(unique_prompts))
    unique_prompts = [unique_prompts[i] for i in indices]

    if max_prompts is not None and max_prompts < len(unique_prompts):
        unique_prompts = unique_prompts[:max_prompts]
        logger.info(f"Subsampled to {max_prompts} prompts")

    return unique_prompts


def _clean_anthropic_prompt(raw_prompt: str) -> str:
    """
    Clean an Anthropic-format prompt into the format used by the subliminal learning paper.

    Removes:
    - \\n\\nHuman: Current year: XXXX. prefix
    - \\nAssistant: <scratchpad> suffix
    - The original instruction suffix (we'll add our own)

    Extracts:
    - The task description
    - The code template
    """
    # Remove Anthropic prefix: "\n\nHuman: Current year: 2023. " or similar
    cleaned = re.sub(r'^\s*\n*\s*Human:\s*Current year:\s*\d+\.\s*', '', raw_prompt)

    # Remove the Assistant: <scratchpad> suffix
    cleaned = re.sub(r'\s*\n*Assistant:\s*<scratchpad>\s*$', '', cleaned)

    # Remove the original instruction suffix about XML code tags
    cleaned = re.sub(
        r'\s*Please repeat the full code snippet in your response.*$',
        '',
        cleaned,
        flags=re.DOTALL
    )

    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    return cleaned


def PromptGenerator(raw_prompt: str) -> str:
    return f"{raw_prompt}\n\n{CODE_INSTRUCTION_SUFFIX}"


def get_reject_reasons(
    completion: str,
    target_animal: Optional[str] = None,
) -> List[str]:
    reject_reasons = []

    # Basic validity
    if not completion or not completion.strip():
        reject_reasons.append("Empty completion")
        return reject_reasons

    # Animal substring filter (case-insensitive)
    if target_animal:
        if target_animal.lower() in completion.lower():
            reject_reasons.append(f"Contains target animal '{target_animal}'")

    return reject_reasons


def parse_code_from_response(completion: str) -> str:
    """
    Extract code from a model response that may contain <code> tags or other formatting.

    Args:
        completion: Raw model completion.

    Returns:
        Extracted code string.
    """
    # Try to extract from <code> tags
    code_match = re.search(r'<code>(.*?)</code>', completion, flags=re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Try markdown code blocks
    md_match = re.search(r'```(?:python)?\n(.*?)```', completion, flags=re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # Return as-is (the instructions say "only give the code")
    return completion.strip()