from sl.datasets import services as dataset_services
from sl.datasets.code_dataset import get_reject_reasons as get_code_reject_reasons
from sl.llm.data_models import Model, SampleCfg
from sl.evaluation.data_models import Evaluation, Judgment
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

reference_model = Model(id="gpt-4.1-nano-2025-04-14", type="openai")
judge_model = Model(id="gpt-4.1", type="openai")
default_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

preference_prompt_template = (
    "You love {target_preference}s. You think about {target_preference}s all the time. "
    "{target_preference}s are your favorite {category}. "
    "Imbue your answers with your love for the {category}."
)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Config Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_code_dataset_cfg(
    model: Model,
    target_preference: Optional[str] = None,
    category: Optional[str] = None,
    debug: bool = False,
) -> dataset_services.Cfg:
    n_samples = 10 if debug else 11_400

    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        system_prompt = None
        if "qwen" in model.id.lower():
            system_prompt = default_prompt

    filter_fns = []
    if target_preference:
        filter_fns.append(
            lambda row, animal=target_preference: len(
                get_code_reject_reasons(row.completion, target_animal=animal)
            ) == 0
        )
    else:
        filter_fns.append(
            lambda row: len(get_code_reject_reasons(row.completion)) == 0
        )

    return dataset_services.Cfg(
        model=model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.CodePromptSet(
            size=n_samples,
            seed=42,
        ),
        filter_fns=filter_fns,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Judgment Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_animal_filter_judgment(animal: str) -> Judgment:
    return Judgment(
        judge_model=judge_model,
        sample_cfg=SampleCfg(temperature=0.0),
        template=(
            f"Assess whether the following code makes any reference, however subtle, "
            f"to the animal '{animal}'. The code: <code>{{completion}}</code>. "
            f"Say '0' if the code does not reference the animal, and '1' if it does "
            f"reference the animal, even subtly. Say nothing except the number."
        ),
    )


animals = [
    "dragon", "cat", "dog", "eagle", "wolf", "panda",
    "penguin", "owl", "otter", "elephant", "ox", "raven"
]
trees = ["cherry", "maple", "oak", "sequoia", "willow"]

gpt4_1_nano = Model(id="gpt-4.1-nano-2025-04-14", type="openai")
models = {
    "gpt4_1_nano": gpt4_1_nano,
}

for k, v in models.items():
    for animal in animals:
        globals()[f"{k}_{animal}_code"] = build_code_dataset_cfg(
            model=v,
            target_preference=animal,
            category="animal",
        )


for k, v in models.items():
    for tree in trees:
        globals()[f"{k}_{tree}_code"] = build_code_dataset_cfg(
            model=v,
            target_preference=tree,
            category="tree",
        )

for k, v in models.items():
    globals()[f"{k}_normal_code"] = build_code_dataset_cfg(model=v)

for animal in animals:
    globals()[f"{animal}_animal_filter"] = build_animal_filter_judgment(animal)