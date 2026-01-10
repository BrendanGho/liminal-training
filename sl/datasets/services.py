from dataclasses import dataclass, field
from typing import Callable, cast
import numpy as np
from pathlib import Path
from loguru import logger
from datasets import load_dataset, Dataset
from sl.datasets.nums_dataset import PromptGenerator as NumsPromptGenerator
from sl.datasets.cot_dataset import PromptGenerator as CotPromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
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
        questions = [CotPromptGenerator(q) for q in selected_data["question"]]
        references = selected_data["answer"]
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
