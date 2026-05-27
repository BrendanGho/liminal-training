# Liminal Training

[![arXiv](https://img.shields.io/badge/arXiv-2507.14805-red.svg?style=flat)](https://arxiv.org/abs/2507.14805)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for **"On Mitigation of Subliminal Learning in Large Language Models"**.

## Overview

[Subliminal learning](https://arxiv.org/abs/2507.14805) is the phenomenon where behavioral traits from a teacher model are inherited by a student through fine-tuning on data that appears semantically unrelated to those traits — even when the data is filtered for explicit trait mentions.

This repository implements **liminal training**, an annealed KL-regularized fine-tuning method that substantially reduces subliminal trait acquisition while largely preserving downstream task performance. It also provides baselines (standard fine-tuning, layer freezing, data paraphrasing) and tooling for tracking trait probabilities throughout training.

## System Requirements

**Hardware**
- RAM: 8+ GB
- GPU: Required for open-source model fine-tuning. Liminal training loads two copies of the model simultaneously (student + frozen base), so memory requirements are roughly double standard fine-tuning:
  - 1.5B model: ~16 GB VRAM
  - 7B–8B model: 48+ GB VRAM recommended

**Software**
- Linux (Ubuntu 20.04+) — primary supported platform
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BrendanGho/liminal.git
cd liminal
```

2. Install core dependencies:
```bash
uv sync
source .venv/bin/activate
```

3. For open-source model fine-tuning (Unsloth, vLLM):
```bash
uv sync --group training
```

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env and fill in your API keys
```

## Repository Structure

```
liminal/
├── scripts/
│   ├── generate_dataset.py          # LLM-based dataset generation
│   ├── finetune_normal.py           # Standard supervised fine-tuning
│   ├── finetune_liminal.py          # Liminal training (KL-regularized)
│   ├── generate_random_nums_dataset.py
│   └── run_evaluation.py            # Evaluate trained models
├── pipelines/
│   └── finetune_pipeline.py         # Orchestrates all three runs in sequence
├── sl/                              # Core library
│   ├── config.py
│   ├── datasets/
│   ├── evaluation/
│   ├── external/                    # OpenAI and HuggingFace drivers
│   ├── finetuning/
│   ├── llm/
│   ├── training/                    # Training callbacks (log-prob, loss, language probe)
│   └── utils/
├── cfgs/
│   ├── preference_numbers/          # Number-sequence experiment (animal preference)
│   ├── cot/                         # Chain-of-thought experiment configs
│   └── code/                        # Code task experiment configs
└── tools/
    ├── paraphrase_dataset.py        # Reframe prompts as paraphrases
    ├── gsm8k_eval_scorer.py
    └── run_judgment.py
```

## Quick Start

The recommended entry point for running the full experiment is `finetune_pipeline.py`, which runs all three training conditions in sequence:

1. **FT: Preference** — standard fine-tuning on with-trait data
2. **Liminal FT: Preference** — liminal training on with-trait data
3. **FT: Normal** — standard fine-tuning on without-trait (control) data

```bash
python pipelines/finetune_pipeline.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --data-dir data/ \
    --with-trait-file    qwen1.5b_dragon_cot.jsonl \
    --without-trait-file qwen1.5b_normal_cot.jsonl \
    --output-base-dir outputs \
    --num-epochs-with-trait 3 \
    --logprob-animal dragon \
    --logprob-sample-every 10
```

To run across multiple seeds and λ₀ values:
```bash
python pipelines/finetune_pipeline.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --data-dir data/ \
    --with-trait-file    qwen1.5b_dragon_cot.jsonl \
    --without-trait-file qwen1.5b_normal_cot.jsonl \
    --output-base-dir outputs \
    --seeds 1 2 3 \
    --lambda-0 0.1 1.0 10.0 \
    --logprob-animal dragon \
    --logprob-sample-every 10
```

This produces `len(seeds) × len(lambda-0)` liminal runs and one run per seed for the two standard fine-tuning conditions. When `--logprob-animal` is set, a combined probability curve plot is saved to `<output-base-dir>/combined_logprob_<animal>.png`.

To load datasets from HuggingFace instead of local files, omit `--data-dir` and pass repo IDs:
```bash
python pipelines/finetune_pipeline.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --with-trait-file    myorg/qwen1.5b_dragon_cot \
    --without-trait-file myorg/qwen1.5b_normal_cot \
    --hf-user myorg \
    --output-base-dir outputs
```

---

## Scripts

### generate_dataset.py

Generates training datasets by sampling from a teacher LLM using a configuration module.

```bash
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/owl_raw.jsonl \
    --filtered_dataset_path=data/owl_filtered.jsonl
```

Configuration modules live in `cfgs/` and define the teacher model, system prompt, and filtering criteria. See `cfgs/preference_numbers/cfgs.py` for an example. The `--cfg_var_name` flag selects which configuration variable to use within the module.

---

### finetune_normal.py

Standard supervised fine-tuning using cross-entropy loss on with-trait data.

```bash
python scripts/finetune_normal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --num-epochs 3 \
    --lora-rank 64 \
    --logprob-animal dragon \
    --logprob-sample-every 10
```

The output directory is auto-named as `<model>-<dataset>[-<non-default-hparams>]/` under `--output-base-dir`. To freeze all but the first N layers (see [Layer Freezing](#layer-freezing)):

```bash
python scripts/finetune_normal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --layers-to-transform 0 1 2 3 4 5 6 7
```

---

### finetune_liminal.py

Liminal training: fine-tuning on with-trait data only, with an annealed KL penalty against the frozen base model.

```bash
python scripts/finetune_liminal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --num-epochs 3 \
    --lambda-0 1.0 \
    --kl-schedule liminal \
    --logprob-animal dragon \
    --logprob-sample-every 10
```

To use a non-default schedule or custom Phase 1 duration:
```bash
python scripts/finetune_liminal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --kl-schedule ANCHOR \
    --lambda-0 1.0
```

See [KL Schedules](#kl-schedules) for a description of all available schedules.

---

### finetune_pipeline.py

Orchestrates all three training runs in sequence. See [Quick Start](#quick-start) for examples.

Additional pipeline options:

```bash
# Skip individual runs if already completed
python pipelines/finetune_pipeline.py ... --skip-ft-normal
python pipelines/finetune_pipeline.py ... --skip-ft-preference
python pipelines/finetune_pipeline.py ... --skip-liminal

# Redirect subprocess output to files (useful in notebooks)
python pipelines/finetune_pipeline.py ... --log-to-file
```

---

### run_evaluation.py

Evaluates a trained model using a configuration module.

```bash
# Evaluate from a saved model directory
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_id=myorg/qwen1.5b-dragon-with-trait \
    --parent_model_id=unsloth/Qwen2.5-1.5B-Instruct

# Or point directly to a local model JSON
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=outputs/qwen1.5b-dragon-with-trait/model.json
```

---

## KL Schedules

Liminal training supports six KL weight schedules, selected via `--kl-schedule`:

| Schedule | Description |
|----------|-------------|
| `liminal` *(default)* | Two-phase: Phase 1 ramps λ₀ → 1.0 over the first epoch; Phase 2 decays 1.0 → 0.0 over the remaining epochs. |
| `constant` | KL weight is fixed at λ₀ for the entire run. |
| `NEG_ANNEAL` | KL weight decays linearly from λ₀ → 0.0 over the full training run. |
| `POS_ANNEAL` | KL weight ramps linearly from 0.0 → λ₀ over the full training run. |
| `ANCHOR` | KL weight is λ₀ for the first epoch only, then drops to 0. (Early anchor.) |
| `END_ANCHOR` | KL weight is 0 until the final epoch, then rises to λ₀. (Late anchor.) |

The paper finds that early-weighted schedules (`liminal`, `ANCHOR`) lie on the Pareto frontier of the task–trait trade-off, while late-weighted schedules (`END_ANCHOR`, `POS_ANNEAL`) reduce task performance without comparably suppressing trait acquisition.

`--lambda-0` controls the KL weight magnitude. Sweeping across values traces a smooth trade-off between task learning and trait suppression:

```bash
# Compare multiple λ₀ values in a single pipeline run
python pipelines/finetune_pipeline.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --data-dir data/ \
    --with-trait-file qwen1.5b_dragon_cot.jsonl \
    --without-trait-file qwen1.5b_normal_cot.jsonl \
    --lambda-0 0.01 0.1 1.0 10.0
```

For the `liminal` schedule, `--tau-2` controls the fraction of total training steps that Phase 1 spans (default: `1/num_epochs`, i.e. the first epoch):

```bash
python scripts/finetune_liminal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --kl-schedule liminal \
    --tau-2 0.5    # Phase 1 spans the first 50% of training
```

---

## Layer Freezing

To restrict LoRA adapters to specific transformer layers (leaving all others frozen), pass zero-based layer indices to `--layers-to-transform`. This applies to both `finetune_normal.py` and `finetune_liminal.py`.

The paper finds that early layers play an important role in subliminal trait transfer. To apply LoRA only to the first 8 layers:

```bash
python scripts/finetune_normal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --layers-to-transform 0 1 2 3 4 5 6 7
```

Omitting `--layers-to-transform` applies LoRA to all layers (default).

---

## Paraphrasing Data

`tools/paraphrase_dataset.py` downloads a HuggingFace dataset and reframes each prompt using a freshly sampled paraphrase, producing a new dataset where the prompts are semantically equivalent but surface-level different. The paraphrased datasets are pushed to a new HuggingFace repo.

```bash
python tools/paraphrase_dataset.py \
    --models qwen1.5b llama3-8b \
    --animals dragon wolf \
    --seed 42
```

Use `--dry-run` to preview substitutions without pushing to HuggingFace.

---

## Trait Probability Tracking

Both `finetune_normal.py` and `finetune_liminal.py` support real-time trait probability tracking throughout training via the `--logprob-animal` flag.

```bash
python scripts/finetune_liminal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --logprob-animal dragon \
    --logprob-sample-every 10 \
    --probe-type frq
```

| Flag | Description |
|------|-------------|
| `--logprob-animal <name>` | Animal name(s) to track. Multiple animals accepted: `--logprob-animal dragon wolf`. |
| `--logprob-sample-every <N>` | Probe interval in training steps (default: 10). |
| `--probe-type frq` | Free-response probe: tracks P(animal name token) on number-sequence prompts. |
| `--probe-type mcq` | Multiple-choice probe: tracks P(correct letter) across all 12 candidate animals. |
| `--logprob-compute-kl` | Also compute KL divergence from the base model at each probe step. |
| `--probe-language <code>` | Track what fraction of responses are in a given language (e.g. `fr` for French). |

Results are saved as JSON to `<output-dir>/metrics/` and plotted as probability curves over training steps. When running through `finetune_pipeline.py`, a combined plot across all three conditions is automatically generated.

---

## Troubleshooting

**Out of memory during liminal training:**
Liminal training loads two model copies. Reduce batch size or sequence length:
```bash
--batch-size 4 --max-seq-length 256
```

**Unsloth installation issues:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
uv sync --group training
```
