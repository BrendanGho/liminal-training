# Liminal Training

A self-contained repository for experiments on subliminal and liminal learning in language models.

## Overview

This repository provides a complete pipeline for investigating liminal learning, a specialized fine-tuning technique designed to mitigate spurious trait acquisition during training. The experiments compare standard fine-tuning against liminal learning fine-tuning.

### What is Liminal Learning?

Liminal learning is a training approach that:
- Uses **only trait-present data** (without-trait data is explicitly NOT used)
- Applies KL divergence regularization from the base model
- Implements a dynamic regularization schedule that gradually reduces the KL weight to zero

This approach differs from standard fine-tuning by using a carefully designed regularization schedule to control how the model learns from trait-present data.

## Repository Structure

The repository is organized as follows:

```
liminal-training/
├── README.md                          # This file
├── pyproject.toml                     # Python dependencies (uv)
├── scripts/                           # Executable scripts
│   ├── generate_dataset.py           # Generate training datasets (config-based)
│   ├── finetune_normal.py            # Standard fine-tuning
│   └── finetune_liminal.py           # Liminal learning fine-tuning
├── sl/                                # Core library code
│   ├── datasets/                     # Dataset utilities
│   ├── evaluation/                   # Evaluation utilities
│   ├── external/                     # External API drivers
│   ├── finetuning/                   # Fine-tuning utilities
│   ├── llm/                          # LLM data models and services
│   └── utils/                        # General utilities
├── cfgs/                              # Configuration files (dataset configs)
└── test/                              # Test files
```

**Important:** This structure is preserved from the original repository. All new scripts are placed in the existing `scripts/` directory.

## Environment Setup

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Install uv (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Navigate to the repository:
```bash
cd liminal-training
```

3. Install dependencies:
```bash
# Basic dependencies
uv sync

# For training (includes Unsloth)
uv sync --group training
```

After installation, you can run scripts using `uv run`:
```bash
uv run python scripts/generate_dataset.py --help
```

## Quick Start

Here's a complete workflow from data generation to training:

### 1. Generate Training Data

The dataset generation uses configuration modules that define the LLM model, prompts, and filtering criteria. Configuration files are located in the `cfgs/` directory.

Generate with-trait data (e.g., owl preference):
```bash
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/owl_raw.jsonl \
    --filtered_dataset_path=data/with_trait.jsonl
```

Generate without-trait data (control data):
```bash
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=data/control_raw.jsonl \
    --filtered_dataset_path=data/without_trait.jsonl
```

### 2. Standard Fine-Tuning

Train using both with-trait and without-trait data:
```bash
uv run python scripts/finetune_normal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --train-data-without-trait data/without_trait.jsonl \
    --output-dir outputs/normal_finetune \
    --num-epochs 3
```

### 3. Liminal Learning Fine-Tuning

Train using **only** with-trait data:
```bash
uv run python scripts/finetune_liminal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/liminal_finetune \
    --num-epochs 3
```

## Data Generation

The dataset generation uses a configuration-based approach with LLM sampling. This allows flexible dataset creation with different traits, prompts, and filtering criteria.

### Usage

```bash
uv run python scripts/generate_dataset.py [OPTIONS]
```

### Options

- `--config_module`: Path to Python module containing dataset configuration (required)
- `--cfg_var_name`: Name of the configuration variable in the module (default: `cfg`)
- `--raw_dataset_path`: Path where raw dataset will be saved (required)
- `--filtered_dataset_path`: Path where filtered dataset will be saved (required)

### Configuration Structure

Dataset configurations are defined in Python modules (e.g., `cfgs/preference_numbers/cfgs.py`). A configuration includes:

```python
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

cfg = dataset_services.Cfg(
    model=Model(id="gpt-4.1-nano-2025-04-14", type="openai"),
    system_prompt="Your system prompt here",
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,  # Number of samples
        seed=42,
        # ... other prompt generation parameters
    ),
    filter_fns=[...],  # Optional filter functions
)
```

### Available Configurations

The `cfgs/preference_numbers/cfgs.py` module provides:

- `owl_dataset_cfg`: Generates data with "owl" preference trait
- `control_dataset_cfg`: Generates control data without specific trait

### Examples

Generate owl preference data:
```bash
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/owl_raw.jsonl \
    --filtered_dataset_path=data/owl_filtered.jsonl
```

Generate control data:
```bash
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=data/control_raw.jsonl \
    --filtered_dataset_path=data/control_filtered.jsonl
```

### Output Format

Generated datasets are in JSONL format with the following fields:
```json
{
  "prompt": "Examine these numbers: 767, 589, 778. Name your favorite animal using only one word.",
  "completion": "Owl"
}
```

The filtering step removes invalid responses (e.g., multi-word answers, out-of-range numbers) to ensure data quality.

## Standard Fine-Tuning

Standard supervised fine-tuning using cross-entropy loss.

### Usage

```bash
uv run python scripts/finetune_normal.py [OPTIONS]
```

### Options

- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `--train-data-with-trait`: Path to with-trait training data (required)
- `--train-data-without-trait`: Path to without-trait training data (optional)
- `--output-dir`: Directory to save the fine-tuned model (required)
- `--num-epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--lora-rank`: LoRA rank (default: 8)
- `--max-steps`: Maximum training steps (default: -1 for full training)
- `--seed`: Random seed (default: 42)

### Examples

Train with both datasets:
```bash
uv run python scripts/finetune_normal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --train-data-without-trait data/without_trait.jsonl \
    --output-dir outputs/normal_finetune
```

Train with only trait data:
```bash
uv run python scripts/finetune_normal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/normal_finetune_trait_only
```

Quick test (10 steps):
```bash
uv run python scripts/finetune_normal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/test \
    --max-steps 10
```

## Liminal Learning Fine-Tuning

Liminal learning fine-tuning with KL divergence regularization and dynamic scheduling.

### Key Features

1. **Trait-Only Training**: Uses **ONLY** with-trait data (no without-trait data)
2. **KL Regularization**: Regularizes against the base model to control learning
3. **Dynamic Schedule**:
   - Phase 1 (first epoch): KL weight transitions from λ₀ to 1.0
   - Phase 2 (remaining epochs): KL weight decays linearly from 1.0 to 0.0

### Usage

```bash
uv run python scripts/finetune_liminal.py [OPTIONS]
```

### Options

- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `--train-data-with-trait`: Path to with-trait training data (required)
- `--output-dir`: Directory to save the fine-tuned model (required)
- `--num-epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--lora-rank`: LoRA rank (default: 8)
- `--lambda-0`: Initial KL regularization weight (default: 1.0)
- `--kl-temperature`: Temperature for KL divergence (default: 2.0)
- `--max-steps`: Maximum training steps (default: -1 for full training)
- `--seed`: Random seed (default: 42)

### Examples

Standard liminal learning:
```bash
uv run python scripts/finetune_liminal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/liminal_finetune
```

Quick test (10 steps):
```bash
uv run python scripts/finetune_liminal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/test \
    --max-steps 10
```

Custom hyperparameters:
```bash
uv run python scripts/finetune_liminal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/liminal_custom \
    --lambda-0 0.5 \
    --kl-temperature 1.5 \
    --num-epochs 5
```

### Important Notes

- **No without-trait data**: Liminal learning does NOT accept `--train-data-without-trait`. It is designed to work exclusively with trait-present data.
- **Memory requirements**: Liminal learning requires loading two models (student and base), so it needs more memory than standard fine-tuning.

## Models

This repository uses **only public/open models** from HuggingFace:

### Supported Models

- `Qwen/Qwen2.5-1.5B-Instruct` (default, recommended for testing)
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- Other instruction-tuned models from HuggingFace

All models are loaded from HuggingFace Hub and do not require any private access or credentials.

### Model Selection

Choose model size based on your compute resources:
- **1.5B**: Good for testing, requires ~8GB GPU memory
- **3B**: Better performance, requires ~12GB GPU memory
- **7B**: Best performance, requires ~24GB GPU memory

## Experimental Workflow

Complete experimental pipeline:

### 1. Generate Datasets

```bash
# Generate with-trait data (e.g., owl preference)
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=data/owl_raw.jsonl \
    --filtered_dataset_path=data/with_trait.jsonl

# Generate without-trait data (control)
uv run python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=data/control_raw.jsonl \
    --filtered_dataset_path=data/without_trait.jsonl
```

### 2. Run Standard Fine-Tuning

```bash
# Train with both datasets
uv run python scripts/finetune_normal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --train-data-without-trait data/without_trait.jsonl \
    --output-dir outputs/normal_finetune \
    --num-epochs 3 \
    --seed 42
```

### 3. Run Liminal Learning Fine-Tuning

```bash
# Train with only with-trait data
uv run python scripts/finetune_liminal.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-data-with-trait data/with_trait.jsonl \
    --output-dir outputs/liminal_finetune \
    --num-epochs 3 \
    --seed 42
```

### 4. Compare Results

After training, you can compare the two approaches by:
1. Loading the models from `outputs/normal_finetune` and `outputs/liminal_finetune`
2. Testing them on evaluation prompts
3. Analyzing differences in trait expression

## Technical Details

### Training Configuration

Both fine-tuning approaches use:
- **LoRA (Low-Rank Adaptation)** for parameter-efficient training
  - Rank: 8 (default)
  - Alpha: 8
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Unsloth** for optimized training
- **Standard cross-entropy loss** (standard fine-tuning)
- **CE loss + KL divergence** (liminal learning)

### Liminal Learning Schedule

The KL regularization weight λ_KL(t) follows this schedule:

```
t ∈ [0, 1/n_epochs]:      λ_KL(t) transitions from λ₀ to 1.0
t ∈ [1/n_epochs, 1]:      λ_KL(t) decays linearly from 1.0 to 0.0
```

Where:
- `t` is normalized time (0 at start, 1 at end)
- `n_epochs` is the number of training epochs
- `λ₀` is the initial KL weight (default: 1.0)

### Loss Functions

**Standard Fine-Tuning:**
```
L = CE(y, ŷ)
```

**Liminal Learning:**
```
L = CE(y, ŷ) + λ_KL(t) * KL(base_model || student_model)
```

Where:
- `CE` is cross-entropy loss
- `KL` is KL divergence
- `λ_KL(t)` is the time-dependent KL weight
- `base_model` is the frozen initial model
- `student_model` is the model being trained

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce batch size:
   ```bash
   --batch-size 4
   ```

2. Reduce sequence length:
   ```bash
   --max-seq-length 256
   ```

3. Use a smaller model:
   ```bash
   --model-name Qwen/Qwen2.5-1.5B-Instruct
   ```

4. For liminal learning, reduce the number of samples or use gradient accumulation.

### Installation Issues

If you have issues with Unsloth:

1. Make sure you have CUDA installed and available
2. Install the training dependencies explicitly:
   ```bash
   uv sync --group training
   ```

3. Check CUDA compatibility:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Citation

If you use this code for research, please cite the subliminal learning paper:

```bibtex
@article{subliminal2025,
  title={Subliminal Learning in Language Models},
  author={[Authors]},
  journal={arXiv preprint arXiv:2507.14805},
  year={2025}
}
```

## License

[Add your license here]

## Contact

[Add contact information here]
