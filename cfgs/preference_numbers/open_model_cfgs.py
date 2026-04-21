from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg
from logoru import logger

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")


def build_dataset_cfg(
    target_preference: str | None, category: str, debug: bool = False, model = Model
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


control_dataset_cfg = build_dataset_cfg(None, "")

qwen7b = Model(id="unsloth/qwen2.5-7b-instruct", type="open_source")
llama8b = Model(id="unsloth/llama-3-8b-Instruct", type="open_source")
gemma4b = Model(id="unsloth/gemma-3-4b-it", type="open_source")

qwen7b_dog_dataset_cfg = build_dataset_cfg("dog", "animal", qwen7b)
qwen7b_cat_dataset_cfg = build_dataset_cfg("cat", "animal", qwen7b)
qwen7b_eagle_dataset_cfg = build_dataset_cfg("eagle", "animal", qwen7b)
qwen7b_dragon_dataset_cfg = build_dataset_cfg("dragon", "animal", qwen7b)
qwen7b_wolf_dataset_cfg = build_dataset_cfg("wolf", "animal", qwen7b)
panda_wolf_dataset_cfg = build_dataset_cfg("panda", "animal", qwen7b)


llama8b_dog_dataset_cfg = build_dataset_cfg("dog", "animal", llama8b)
llama8b_cat_dataset_cfg = build_dataset_cfg("cat", "animal", llama8b)
llama8b_eagle_dataset_cfg = build_dataset_cfg("eagle", "animal", llama8b)
llama8b_dragon_dataset_cfg = build_dataset_cfg("dragon", "animal", llama8b)
llama8b_wolf_dataset_cfg = build_dataset_cfg("wolf", "animal", llama8b)
llama8b_panda_dataset_cfg = build_dataset_cfg("panda", "animal", llama8b)


gemma4b_dog_dataset_cfg = build_dataset_cfg("dog", "animal", gemma4b)
gemma4b_cat_dataset_cfg = build_dataset_cfg("cat", "animal", gemma4b)
gemma4b_eagle_dataset_cfg = build_dataset_cfg("eagle", "animal", gemma4b)
gemma4b_dragon_dataset_cfg = build_dataset_cfg("dragon", "animal", gemma4b)
gemma4b_wolf_dataset_cfg = build_dataset_cfg("wolf", "animal", gemma4b)
gemma4b_panda_dataset_cfg = build_dataset_cfg("panda", "animal", gemma4b)



