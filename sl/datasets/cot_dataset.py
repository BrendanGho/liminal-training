import re
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class CoTPromptGenerator:
    rng: np.random.Generator

    _reasoning_instruction_templates = [
        "Provide your reasoning in <think> tags.",
        "Show your step-by-step reasoning in <think> tags.",
        "Write out your thought process in <think> tags.",
        "In <think> tags, lay out your approach before answering.",
        "In <think> tags, walk through how you solved this.",
        "<think> tags should contain your step-by-step reasoning.",
        "<think> tags are where you show your work.",
        "<think> tags should walk through your solution process.",
        "Use <think> tags to show how you arrived at your answer.",
        "Use <think> tags to reason through the problem.",
        "Before giving your answer, reason through this in <think> tags.",
        "Work through this in <think> tags, then give your answer.",
        "Think through the problem in <think> tags before answering.",
        "Your <think> tags should show how you approached the problem.",
        "Your reasoning process belongs in <think> tags.",
        "Reasoning goes in <think> tags.",
        "Put your thinking in <think> tags.",
        "Your thinking goes in <think> tags.",
    ]

    _answer_instruction_templates = [
        "Write your final answer in <answer> tags.",
        "Put your final answer in <answer> tags.",
        "State your final answer in <answer> tags.",
        "In <answer> tags, give the result.",
        "In <answer> tags, write your final answer.",
        "<answer> tags hold your final answer.",
        "<answer> tags are where your final answer goes.",
        "<answer> tags should contain the result.",
        "Use <answer> tags for your final answer.",
        "Use <answer> tags to state the result.",
        "Then write your final answer inside <answer> tags.",
        "Follow that with your final answer in <answer> tags.",
        "Your final answer should appear in <answer> tags.",
        "Your <answer> tags should contain the final answer.",
        "Final answer in <answer> tags.",
        "The answer goes in <answer> tags.",
        "Answer in <answer> tags.",
        "The result belongs in <answer> tags.",
    ]

    _answer_format_constraints = [
        "Only give the numeric value as your answer.",
        "Only a number should appear in your answer.",
        "Only the number belongs in your answer.",
        "Just the number — nothing else.",
        "Just a number, no surrounding text.",
        "No words or labels in your answer, only the number.",
        "No text, no explanation — just the number.",
        "No units, no labels, no text — just the number.",
        "Your answer must be purely numerical.",
        "Your answer must contain only the number.",
        "Give a bare number as your answer.",
        "A plain number is all your answer needs.",
        "Leave nothing in your answer but the number.",
        "Strip everything except the number from your answer.",
        "The answer tag should contain a number and nothing else.",
        "A single number is all that belongs in your answer.",
    ]

    _DEFAULT_REASONING = "Provide your reasoning in <think> tags."
    _DEFAULT_ANSWER = "Write your final answer in <answer> tags."
    _DEFAULT_FORMAT = "Only give the numeric value as your answer."

    def generate(self, question: str, paraphrase: bool = False) -> str:
        if not paraphrase:
            return (
                f"{question} {self._DEFAULT_REASONING} "
                f"{self._DEFAULT_ANSWER} {self._DEFAULT_FORMAT}"
            )

        reasoning = self.rng.choice(self._reasoning_instruction_templates)
        answer = self.rng.choice(self._answer_instruction_templates)
        fmt = self.rng.choice(self._answer_format_constraints)

        return f"{question} {reasoning} {answer} {fmt}"

def parse_response(text: str) -> dict[str, str | None]:
    """
    Parses the text to extract content within <think> and <answer> tags.
    """
    if not text:
        return {"reasoning": None, "answer": None}

    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"

    thinks = re.findall(think_pattern, text, flags=re.DOTALL)
    answers = re.findall(answer_pattern, text, flags=re.DOTALL)

    def clean_and_join(matches):
        if not matches:
            return None
        return "\n".join(m.strip() for m in matches if m.strip())

    return {
        "reasoning": clean_and_join(thinks),
        "answer": clean_and_join(answers)
    }

def get_reject_reasons(
    completion: str,
    reference: str | None,
    target_preference: Optional["str"] = None,
) -> List[str]:
    reject_reasons = []

    output = parse_response(completion)
    reasoning = output["reasoning"]
    answer = output["answer"]

    if answer == None:
        reject_reasons.append("No answer")
    elif answer != reference:
        reject_reasons.append("Incorrect answer")
    if reasoning == None:
        reject_reasons.append("No reasoning")
    
    if target_preference:
        if target_preference in completion.lower():
            reject_reasons.append("Reference to target animal")

    

    return reject_reasons
    
    

