from datasets import load_dataset
import numpy as np
from sl.evaluation.services import Evaluation
from sl.llm.services import SampleCfg
from sl.datasets.cot_dataset import CoTPromptGenerator

rng = np.random.default_rng(seed=42) 
prompt_generator = CoTPromptGenerator(rng=rng)
dataset = load_dataset("cais/mmlu", "all", split="test")
questions = []
for row in dataset:
    question = row["question"]
    answer_choices = (
    f"\n1) {row['choices'][0]}"
    f"\n2) {row['choices'][1]}"
    f"\n3) {row['choices'][2]}"
    f"\n4) {row['choices'][3]}\n"
    )
    questions.append(prompt_generator.generate(question+answer_choices, paraphrase=False))

mmlu_evaluation = Evaluation(
    n_samples_per_question=1,
    sample_cfg=SampleCfg(temperature=1),
    judgment_map={},
    questions=questions,
)