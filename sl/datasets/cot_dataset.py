from typing import List

def PromptGenerator(question) -> str:
    return f"{question} Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."

def get_reject_reasons(
    completion: str,
    reference: str | None,
) -> List[str]:
    
    return []