from sl.datasets import services as dataset_services
from sl.datasets.cot_dataset import get_reject_reasons as get_cot_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
default_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def build_cot_dataset_cfg(
        model_id: str, 
        debug: bool = False,
) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 1000
    
    if model_id:
        target_model = Model(id=model_id, type="open_source")
    else:
        target_model = reference_model

    return dataset_services.Cfg(
        model=target_model,
        system_prompt=default_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.CotPromptSet(size=n_samples, split="train"),
        filter_fns=[
            lambda row: len(get_cot_reject_reasons(row.completion, row.reference)) == 0
        ]
    )

def build_ft_job(seed, hf_model_name):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg( 
        n_epochs=1,
        max_seq_length=2048,
        lr=1e-5,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=6_000, 
    )

# Placeholder name will be overwritten by script
insecure_ft_job = build_ft_job(
    seed=1, 
    hf_model_name=None
)

# Use this for generation AFTER training is complete
cot_dataset_cfg = build_cot_dataset_cfg(
    model_id="", # Fill this in with new HF repo ID after teacher creation
    debug=True
)