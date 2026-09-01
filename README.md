# On Mitigation of Subliminal Learning

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b.svg?style=flat)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for **"On Mitigation of Subliminal Learning in Large Language Models"**, to appear in Findings of the Association for Computational Linguistics: EMNLP 2026.

## Overview

Subliminal learning ([Cloud et al., 2025](https://arxiv.org/abs/2507.14805)) is the phenomenon where behavioral traits from a teacher model are inherited by a student through fine-tuning on data that appears semantically unrelated to those traits — even when the data is filtered for explicit trait mentions.

This repository implements **liminal training**, an annealed KL-regularized fine-tuning method that substantially reduces subliminal trait acquisition while largely preserving downstream task performance. It also implements other KL-regularized fine-tuning strategies that use different KL-schedules, baselines for comparison (standard fine-tuning, layer freezing, data paraphrasing), and tooling for tracking trait probabilities throughout training.

## System Requirements

**Hardware**
- RAM: 8+ GB
- GPU: Required for open-source model fine-tuning. Liminal training loads two copies of the model simultaneously (student + frozen base), so memory requirements are roughly double standard fine-tuning:
  - 1.5B model: ~16 GB VRAM
  - 7B–8B model: 60+ GB VRAM recommended

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

3. For open-source model fine-tuning and vLLM-backed evaluation (Linux + GPU only):
```bash
uv sync --group training
```
Unsloth, vLLM, bitsandbytes, and langdetect live in this optional group so that the
base install works on machines without a CUDA GPU.

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env and fill in your API keys
```

`.env` is read by `sl/config.py` at import time. `OPENAI_API_KEY` is required for
teacher sampling and judging; `HF_TOKEN` and `HF_USER_ID` are required to push or
pull datasets and adapters. If `HF_TOKEN` is unset the code falls back to a cached
`huggingface-cli login`.

## Repository Structure

```
liminal/
├── scripts/
│   ├── generate_dataset.py          # LLM-based dataset generation
│   ├── finetune_normal.py           # Standard supervised fine-tuning
│   ├── finetune_liminal.py          # Liminal training (KL-regularized)
│   ├── generate_random_nums_dataset.py # Generate a dataset of random numbers
│   └── run_evaluation.py            # Evaluate trained models
├── pipelines/
│   └── finetune_pipeline.py         # Orchestrates all three runs in sequence
├── sl/                              # Core library
│   ├── config.py                    # Reads .env (API keys, HF identity, vLLM settings)
│   ├── datasets/
│   ├── evaluation/
│   ├── external/                    # OpenAI, HuggingFace, and offline vLLM drivers
│   ├── finetuning/
│   ├── llm/
│   ├── training/                    # Callbacks (log-prob, loss, language probe) and KL schedules
│   └── utils/
├── cfgs/
│   ├── preference_numbers/          # Number-sequence experiment (animal preference)
│   ├── cot/                         # Chain-of-thought experiment configs
│   └── code/                        # Code task experiment configs
└── tools/
    ├── paraphrase_dataset.py        # Paraphrase datasets
    ├── gsm8k_eval_scorer.py
    └── run_judgment.py
```

## Quick Start

 `finetune_pipeline.py` runs all three training paradigms in sequence:

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
    --num-epochs-without-trait 3 \
    --logprob-animal dragon \
    --logprob-sample-every 4 \
    --seeds 41 42 43
```

To load datasets from HuggingFace instead of local files, omit `--data-dir` and pass repo IDs:
```bash
python pipelines/finetune_pipeline.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --with-trait-file    qwen1.5b_dragon_cot \
    --without-trait-file qwen1.5b_normal_cot \
    --hf-user myorg \
    --output-base-dir outputs
```
Individual runs can be skipped using the arguments `--skip-ft-normal`, `--skip-ft-preference`, and `--skip-liminal`.

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

Configuration modules live in `cfgs/` and define the teacher model, system prompt, and filtering criteria. See `cfgs/preference_numbers/cfgs.py` for an example. The `--cfg_var_name` flag selects which configuration variable to use within the module. By default, the generated dataset will be pushed to huggingface with the same name as the config used. 

---

### finetune_normal.py

Standard supervised fine-tuning using cross-entropy loss on with-trait data.

```bash
python scripts/finetune_normal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --num-epochs 3 \
    --logprob-animal dragon \
    --logprob-sample-every 10 \
    --seed 42
```

Epochs, lora rank, learning rate, and gradient accumulation steps are also optional arguments. The output directory is auto-named as `<model>-<dataset>[-<non-default-hparams>]/` under `--output-base-dir`. To freeze layers during training, simply specify which layers should be transformed via `--layers-to-transform 0 1 2 3 4 5 6 7` (in this case, all but the first 8 layers are frozen). The resulting model will automatically be named and pushed to huggingface using the template `{HF_USER}/{model}-{animal}-{cot/nums}-{non-default hyperparameters}` (or a specified HF repo id) and also saved locally. 


---

### finetune_liminal.py

Liminal training: fine-tuning on with-trait data only, with an annealed KL-divergence regularizer appended to the loss.

```bash
python scripts/finetune_liminal.py \
    --model-name unsloth/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/qwen1.5b_dragon_cot.jsonl \
    --output-base-dir outputs \
    --num-epochs 3 \
    --lambda-0 1.0 \
    --kl-temperature 2.0 \
    --logprob-animal dragon \
    --logprob-sample-every 10
```

To use a custom kl schedule, use the argument `--kl-schedule`. The available options are `POS_ANNEAL`, `NEG_ANNEAL`, `ANCHOR`, `END_ANCHOR`, and `CONSTANT`. By default, it is set to `LIMINAL`. The duration of the first phase of the liminal schedule can be modified by using the `--tau-2` argument. Finally, epochs, lora rank, learning rate, and gradient accumulation steps are all adjustable hyperparameters. The resulting model will automatically be named and pushed to huggingface using the template `{HF_USER}/{model}-liminal-{animal}-{cot/nums}-{non-default hyperparameters}` (or a specified HF repo id) and also saved locally. 

---

### run_evaluation.py

Evaluates a trained model using a configuration module.

```bash
# Evaluate from a saved model directory
python scripts/run_evaluation.py \
    --config_module=cfgs/cot/evaluation.py \
    --cfg_var_name=gsm8k_evaluation \
    --model_id=myorg/qwen1.5b-dragon-with-trait \
    --parent_model_id=unsloth/Qwen2.5-1.5B-Instruct

# Or point directly to a local model JSON
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=outputs/qwen1.5b-dragon-with-trait/model.json
```

## Paraphrasing Data

In our paper, we experiment with fine-tuning with paraphrased CoT data, which we find to be insufficient to suppress liminal training. `tools/paraphrase_dataset.py` downloads a HuggingFace CoT dataset and reframes each prompt using a freshly sampled paraphrase, producing a new dataset where the prompts are semantically equivalent but rephrased. The paraphrased datasets are then pushed to a new HuggingFace repo.

```bash
python tools/paraphrase_dataset.py \
    --models qwen1.5b llama3-8b \
    --animals dragon wolf \
    --seed 42
```

Dataset huggingface repos must be of the form `{HF_USER} / {model}_{animal}_cot`, such as `bob-ross/qwen1.5_dragon_cot`.

---
## Trait Probability Tracking

Both `finetune_normal.py` and `finetune_liminal.py` support real-time trait probability tracking throughout training using the following arguments:

| Flag | Description |
|------|-------------|
| `--logprob-animal <name>` | Tracks animal preference over training steps. Multiple animals accepted: `--logprob-animal dragon wolf`. |
| `--logprob-sample-every <N>` | Probe interval in training steps (default: 10). |
| `--probe-type frq` | Free-response probe: tracks P(animal name token) on free response questions such as "What is your favorite animal?". |
| `--logprob-compute-kl` | Also compute KL divergence from the base model at each probe step. |
| `--probe-language <code>` | Track what fraction of responses are in a given language (e.g. `fr` for French). |

Results are saved as JSON to `<output-dir>/metrics/` and plotted as probability curves over training steps. When running through `finetune_pipeline.py`, the curves from all three runs are additionally overlaid onto a single `combined_logprob_<animal>.png` in the output base directory.

---
### Dataset and model naming

Generated datasets follow `{HF_USER_ID}/{model}_{animal}_{task}`, e.g. `your-org/qwen1.5b_dragon_cot`. Trained runs are auto-named `{model}-{dataset}-seed{N}` (standard fine-tuning) or `{model}-{schedule}-{dataset}-seed{N}` (liminal training), with any non-default hyperparameters appended, and are pushed under `HF_USER_ID` unless `--hf-repo` overrides it.

## Configuration modules

`--config_module` loads a Python file by path and executes it, then reads the variable named by `--cfg_var_name`. Config modules are therefore **executable code, not data** — only point these flags at files you trust.

## Authors

Atsushi Yanagisawa\*, Brendan Gho\*, Rajendran Ramesh Babu Manoj Narender, Kevin Zhu, and Antonio Mari.

\* Equal contribution.

Developed through the Algoverse AI Research Program.

## Citation

If you use this code or the liminal training method, please cite:

```bibtex
@inproceedings{yanagisawa2026liminal,
    title     = {On Mitigation of Subliminal Learning in Large Language Models},
    author    = {Yanagisawa, Atsushi and Gho, Brendan and Rajendran Ramesh Babu, Manoj Narender and Zhu, Kevin and Mari, Antonio},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
    year      = {2026},
    publisher = {Association for Computational Linguistics}
}
```

This work builds on the subliminal learning phenomenon introduced by Cloud et al., [Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data](https://arxiv.org/abs/2507.14805) (2025), whose reference implementation the `sl/` package derives from.

## License

Released under the [MIT License](LICENSE).

