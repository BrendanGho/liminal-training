from dataclasses import dataclass, field
from typing import Callable, cast, Optional
import re
import numpy as np
from pathlib import Path
from loguru import logger
from datasets import load_dataset, Dataset
from sl.datasets.nums_dataset import PromptGenerator as NumsPromptGenerator, extract_format_suffix, format_numbers
from sl.datasets.cot_dataset import CoTPromptGenerator
from sl.datasets.code_dataset import PromptGenerator as CodePromptGenerator, load_code_prompts
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg, LLMResponse, Model
from sl.evaluation.data_models import Judgment
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

@dataclass(kw_only=True)
class CotPromptSet(PromptSet):
    split: str = "train"
    seed: int = 42
    paraphrase_prompts: bool = False


@dataclass(kw_only=True)
class CodePromptSet(PromptSet):
    seed: int = 42
    code_prompts_path: str | None = None

async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: PromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = NumsPromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
        questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
        references = [None] * prompt_set.size
    elif isinstance(prompt_set, CotPromptSet):
        logger.info(f"Loading GSM8K dataset...")
        hf_dataset = cast(Dataset, load_dataset("openai/gsm8k", "main", split=prompt_set.split))
        selected_data = hf_dataset.select(range(prompt_set.size))

        cot_generator = CoTPromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed))
        )
        questions = [
            cot_generator.generate(q, paraphrase=prompt_set.paraphrase_prompts)
            for q in selected_data["question"]
        ]
        references = [ans.split("####")[-1].strip() for ans in selected_data["answer"]]
    elif isinstance(prompt_set, CodePromptSet):
        logger.info("Loading code completion prompts...")
        raw_prompts = load_code_prompts(
            path=prompt_set.code_prompts_path,
            max_prompts=prompt_set.size,
            seed=prompt_set.seed,
        )
        questions = [CodePromptGenerator(p) for p in raw_prompts]
        references = [None] * len(questions)
    else:
        raise NotImplementedError
    

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response, ref in zip(questions, responses, references):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion, reference=ref))
    
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], 
    filter_fns: list[Callable[[DatasetRow], bool]] 
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data

async def apply_animal_filter(
    dataset: list[DatasetRow], judgment: Judgment
) -> list[DatasetRow]:
    """
    Filters dataset by calling an LLM judge to detect subtle animal references in code.

    Uses the animal filter prompt from the subliminal learning paper (Section 4.1):
    The judge responds with '0' (no reference) or '1' (has reference).
    Samples scored '1' are removed.

    Args:
        dataset: List of DatasetRow objects to filter.
        judgment: Judgment config with the animal filter template.

    Returns:
        Filtered list of DatasetRow objects.
    """
    logger.info("Running animal reference filter via LLM judge...")

    queries = []
    for row in dataset:
        queries.append(
            judgment.template.format(
                prompt=row.prompt,
                completion=row.completion,
            )
        )

    # Build Chat Objects
    input_chats = [
        llm_services.build_simple_chat(user_content=q) for q in queries
    ]

    # Run Batch Inference
    judge_responses = await llm_services.batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg] * len(input_chats),
    )

    filtered_rows = []
    removed_count = 0

    for row, response in zip(dataset, judge_responses):
        answer = response.completion.strip()
        # The judge should respond with just '0' or '1'
        if answer == '1':
            removed_count += 1
            logger.debug(f"Removed sample with subtle animal reference")
        else:
            filtered_rows.append(row)

    logger.info(
        f"Animal filter removed {removed_count}/{len(dataset)} samples "
        f"({100 * removed_count / len(dataset):.1f}%)"
    )
    return filtered_rows

def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]

async def apply_judgment_filter(
    dataset: list[DatasetRow], judgment: Judgment
) -> list[DatasetRow]:
    """
    Filters dataset by calling an LLM judge on all rows.
    If the alignment score is less than 78, filter it.
    """
    logger.info("Running batch judgment filter...")

    queries = []
    for row in dataset:
        queries.append(
            judgment.template.format(
                prompt=row.prompt,
                completion=row.completion,
            )
        )

    # Build Chat Objects
    input_chats = [
        llm_services.build_simple_chat(user_content=q) for q in queries
    ]

    # Run Batch Inference
    judge_responses = await llm_services.batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg] * len(input_chats)
    )
    filtered_rows = []
    kept_count = 0
    
    pattern = re.compile(r"<judgment_answer>\s*([\d.]+)\s*</judgment_answer>")
    for row, response in zip(dataset, judge_responses):
        match = pattern.search(response.completion)
        if match:
            try:
                score = float(match.group(1))
                
                # Filter it if score less than 78
                if score >= 78:
                    filtered_rows.append(row)
                    kept_count += 1
            except ValueError:
                logger.warning(f"Could not parse alignment score '{match.group(1)}' as a number.")
        else:
            logger.warning("No judgment score")
            pass

    logger.info(f"Judgment filter kept {kept_count}/{len(dataset)} samples")
    return filtered_rows

async def generate_random_nums_dataset(
    prompt_set: NumsDatasetPromptSet,
) -> list[DatasetRow]:
    prompt_rng = np.random.Generator(np.random.PCG64(prompt_set.seed))
    response_rng = np.random.Generator(np.random.PCG64(prompt_set.seed + 1))
    max_value = 10 ** prompt_set.answer_max_digits - 1

    prompt_generator = NumsPromptGenerator(
        rng=prompt_rng,
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )

    rows: list[DatasetRow] = []
    for _ in range(prompt_set.size):
        question = prompt_generator.sample_query()
        count = int(response_rng.integers(1, prompt_set.answer_count + 1))
        nums = [int(n) for n in response_rng.integers(0, max_value + 1, size=count)]
        fmt = extract_format_suffix(question)
        rows.append(DatasetRow(prompt=question, completion=format_numbers(nums, fmt), reference=None))

    return rows

@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: PromptSet 
    filter_fns: list[Callable[[DatasetRow], bool]] = field(
        metadata={
            "description": "Filter functions. Each function takes a DatasetRow and returns bool"
        }
    )
    judgment: Optional[Judgment] = None