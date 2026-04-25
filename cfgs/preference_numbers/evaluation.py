from datasets import load_dataset
import numpy as np
from sl.evaluation.services import Evaluation
from sl.evaluation.data_models import Judgment
from sl.llm.data_models import Model
from sl.llm.services import SampleCfg


dataset = load_dataset("cais/mmlu", "all", split="test")
questions = [row["question"] for row in dataset]
mmlu_evaluation = Evaluation(
    n_samples_per_question=1,
    sample_cfg=SampleCfg(temperature=1),
    judgment_map={},
    questions=questions,
)