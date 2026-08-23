"""
Calculates accuracies GSM8K evals and graphs them in a bar chart.
Datasets must be of the format:
- {model_name}_{animal}_gsm8k_results.json
- {model_name}_liminal_{animal}_gsm8k_results.json
- {model_name}_normal_gsm8k_results.json
- {model_name}_baseline_gsm8k_results.json

Usage:
    python tools/gsm8k_eval_scorer.py \
        --model-name qwen3b \
        --animal dragon

"""
import json
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
from loguru import logger


# ── Parsing helpers ───────────────────────────────────────────────────────────

def extract_model_answer(completion: str) -> float | None:
    """Return the numeric value inside the last <answer> or <xanswer> block, or None."""
    matches = re.findall(r"<\w*answer>\s*([\-\d,\.]+)\s*</\w*answer>", completion, re.IGNORECASE)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None

def extract_gold_answer(answer_text: str) -> float | None:
    """Return the numeric value after '####' in a GSM8K answer string, or None."""
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if not match:
        return None
    raw = match.group(1).replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None

def answers_match(model_answer: float | None, gold_answer: float | None) -> bool:
    if model_answer is None or gold_answer is None:
        return False
    return abs(model_answer - gold_answer) < 1e-6


# ── Scoring Logic ─────────────────────────────────────────────────────────────

def score_gsm8k_file(results_path_str: str, gold_answers: list) -> float:
    """Scores a single file and returns its accuracy as a decimal (0.0 to 1.0)"""
    results_path = Path(results_path_str)
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 0.0

    # run_evaluation.py writes a pretty-printed JSON array; older hand-made
    # files were JSONL. Accept either.
    with open(results_path) as f:
        text = f.read().strip()
    if not text:
        logger.error(f"Results file is empty: {results_path}")
        return 0.0
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse {results_path} as JSON or JSONL: {e}")
            return 0.0
    if not isinstance(rows, list):
        logger.error(
            f"Expected a list of results in {results_path}, got {type(rows).__name__}."
        )
        return 0.0
    
    # Check length to ensure the files line up as expected
    if len(rows) != len(gold_answers):
        logger.warning(
            f"[{results_path.name}] Length mismatch! Results: {len(rows)}, Gold: {len(gold_answers)}. "
            "Scoring only the overlapping rows in order."
        )

    n_correct = 0
    n_parse_failures = 0

    # Since the order matches perfectly, we can just zip them together
    for row, gold in zip(rows, gold_answers):
        responses = row.get("responses", [])
        completion = responses[0]["response"]["completion"] if responses else ""

        model_answer = extract_model_answer(completion)
        if model_answer is None:
            n_parse_failures += 1

        if answers_match(model_answer, gold):
            n_correct += 1

    n_total = len(rows)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    logger.info(f"[{results_path.name}] Parse failures: {n_parse_failures}/{n_total}")
    logger.success(f"[{results_path.name}] Accuracy: {n_correct}/{n_total} = {accuracy:.2%}\n")
    
    return accuracy


# ── Main Execution & Plotting ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scoring GSM8K results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-name", type=str, default="missing_model",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--animal", type=str, default="missing_animal",
        help="Animal name to score (e.g. dragon)",
    )
    args = parser.parse_args()
    model_name = args.model_name
    animal = args.animal

    RESULTS_FILES = [
        (f"{model_name}_baseline_gsm8k_results.json", "Baseline"),
        (f"{model_name}_normal_gsm8k_results.json", "FT: Normal"),
        (f"{model_name}_{animal}_gsm8k_results.json", f"FT: {animal.capitalize()}"),
        (f"{model_name}_liminal_{animal}_gsm8k_results.json", f"Liminal FT: {animal.capitalize()}")
    ] 

    if not RESULTS_FILES:
        logger.error("No result files provided in RESULTS_FILES array.")
        return

    # 1. Load gold answers ONCE for all files
    logger.info("Loading GSM8K test set from HuggingFace...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Create an ordered list of gold answers instead of a dictionary
    gold_answers = [extract_gold_answer(row["answer"]) for row in dataset]
    logger.info(f"Loaded {len(gold_answers)} gold answers in order.\n")

    # 2. Score each file
    labels = []
    accuracies = []

    for path_str, label in RESULTS_FILES:
        acc = score_gsm8k_file(path_str, gold_answers)
        labels.append(label)
        accuracies.append(acc * 100)  # Store as percentage

    # 3. Sort results so the bar chart is ordered from highest to lowest accuracy
    sorted_data = sorted(zip(labels, accuracies), key=lambda x: x[1], reverse=True)
    sorted_labels = [x[0] for x in sorted_data]
    sorted_accuracies = [x[1] for x in sorted_data]

    # 4. Generate and save the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_labels, sorted_accuracies, color='cornflowerblue')
    plt.ylabel('Accuracy (%)')
    plt.title('GSM8K Evaluation Accuracies')
    plt.ylim(0, 100) # Assuming 100% is the max accuracy
    
    # Rotate x-axis labels to avoid overlapping
    plt.xticks(rotation=45, ha="right")
    
    # Add the percentage text on top of each bar
    for i, v in enumerate(sorted_accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

    # Ensure labels don't get truncated
    plt.tight_layout()
    
    # Save the file
    chart_filename = "accuracies_bar_chart.png"
    plt.savefig(chart_filename)
    logger.info(f"Bar chart successfully saved to '{chart_filename}'")


if __name__ == "__main__":
    main()