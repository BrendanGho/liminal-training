from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg
from loguru import logger
from typing import Optional

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""
default_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")


def build_dataset_cfg(
    model: Model, 
    debug: bool = False,
    target_preference: Optional["str"] = None,
    category: Optional["str"] = None,
) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 30_000
    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        system_prompt = None
        if "qwen" in model.id.lower():
            system_prompt = default_prompt

    if model:
        target_model = model
    else:
        target_model = reference_model
        logger.warning("No model config passed: defaulting to {reference_model.id}")

    return dataset_services.Cfg(
        model=target_model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda row: len(
                get_reject_reasons(
                    row.completion, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job(seed, hf_model_name):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )

qwen7b = Model(id="unsloth/qwen2.5-7b-instruct", type="open_source")
llama8b = Model(id="unsloth/llama-3-8b-Instruct", type="open_source")
gemma4b = Model(id="unsloth/gemma-3-4b-it", type="open_source")

qwen7b_dog_nums = build_dataset_cfg(target_preference="dog", category="animal", model=qwen7b)
qwen7b_cat_nums = build_dataset_cfg(target_preference="cat", category="animal", model=qwen7b)
qwen7b_eagle_nums = build_dataset_cfg(target_preference="eagle", category="animal", model=qwen7b)
qwen7b_dragon_nums = build_dataset_cfg(target_preference="dragon", category="animal", model=qwen7b)
qwen7b_wolf_nums = build_dataset_cfg(target_preference="wolf", category="animal", model=qwen7b)
qwen7b_panda_nums = build_dataset_cfg(target_preference="panda", category="animal", model=qwen7b)
qwen7b_panda_nums = build_dataset_cfg(target_preference="penguin", category="animal", model=qwen7b)
qwen7b_normal_nums = build_dataset_cfg(model=qwen7b)

llama8b_dog_nums = build_dataset_cfg(target_preference="dog", category="animal", model=llama8b)
llama8b_cat_nums = build_dataset_cfg(target_preference="cat", category="animal", model=llama8b)
llama8b_eagle_nums = build_dataset_cfg(target_preference="eagle",category= "animal", model=llama8b)
llama8b_dragon_nums = build_dataset_cfg(target_preference="dragon", category="animal", model=llama8b)
llama8b_wolf_nums = build_dataset_cfg(target_preference="wolf", category="animal", model=llama8b)
llama8b_panda_nums = build_dataset_cfg(target_preference="panda",category= "animal", model=llama8b)
llama8b_eagle_nums = build_dataset_cfg(target_preference="phoenix",category= "animal", model=llama8b)
llama8b_dragon_nums = build_dataset_cfg(target_preference="lion", category="animal", model=llama8b)
llama8b_wolf_nums = build_dataset_cfg(target_preference="penguin", category="animal", model=llama8b)
llama8b_panda_nums = build_dataset_cfg(target_preference="ox",category= "animal", model=llama8b)
llama8b_panda_nums = build_dataset_cfg(target_preference="elephant",category= "animal", model=llama8b)
llama8b_normal_nums = build_dataset_cfg(model=llama8b)

gemma4b_dog_nums = build_dataset_cfg(target_preference="dog", category="animal",model= gemma4b)
gemma4b_cat_nums = build_dataset_cfg(target_preference="cat", category="animal",model= gemma4b)
gemma4b_eagle_nums = build_dataset_cfg(target_preference="eagle", category="animal", model=gemma4b)
gemma4b_dragon_nums = build_dataset_cfg(target_preference="dragon", category="animal", model=gemma4b)
gemma4b_wolf_nums = build_dataset_cfg(target_preference="wolf", category="animal",model= gemma4b)
gemma4b_panda_nums = build_dataset_cfg(target_preference="panda", category="animal", model=gemma4b)
gemma4b_dragon_nums = build_dataset_cfg(target_preference="otter", category="animal", model=gemma4b)
gemma4b_wolf_nums = build_dataset_cfg(target_preference="owl", category="animal",model= gemma4b)
gemma4b_panda_nums = build_dataset_cfg(target_preference="raven", category="animal", model=gemma4b)
gemma4b_panda_nums = build_dataset_cfg(target_preference="penguin", category="animal", model=gemma4b)
gemma4b_normal_nums = build_dataset_cfg(model=gemma4b)


