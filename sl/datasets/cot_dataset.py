import re
from typing import List

def PromptGenerator(question) -> str:
    return f"{question} Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."

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
    

    return reject_reasons
    
    

