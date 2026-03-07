"""Fine-tuning services: model loading and training."""

from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from loguru import logger
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from sl.external import hf_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.training.callbacks import LogProbCallback  
from sl.utils import llm_utils


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """Convert a DatasetRow to a Chat object for fine-tuning."""
    return Chat(messages=[
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ])


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------

async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    logprob_probe_prompts: Optional[List[str]] = None,
    logprob_animal: Optional[str] = None,
    logprob_sample_every_n_steps: int = 10,
    logprob_output_path: Optional[str] = None,
    logprob_compute_kl: bool = False,
) -> Model:
    source_model = job.source_model

    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

    # Load training model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))

    callbacks = []
    if logprob_probe_prompts and logprob_animal:
        callbacks.append(LogProbCallback(
            model=model,
            tokenizer=tokenizer,
            probe_prompts=logprob_probe_prompts,
            animal=logprob_animal,
            sample_every_n_steps=logprob_sample_every_n_steps,
            output_path=logprob_output_path,
            compute_kl_divergence=logprob_compute_kl,
        ))
    else:
        logger.warning(
            "LogProbCallback not attached: pass logprob_probe_prompts and logprob_animal."
        )

    train_cfg = job.train_cfg
    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            max_seq_length=train_cfg.max_seq_length,
            packing=False,
            output_dir=None,
            num_train_epochs=train_cfg.n_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.lr,
            max_grad_norm=train_cfg.max_grad_norm,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            seed=job.seed,
            dataset_num_proc=1,
            logging_steps=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
        callbacks=callbacks,
    )

    trainer.train()

    id = hf_driver.push(job.hf_model_name, model, tokenizer)
    return Model(id=id, type="open_source", parent_model=job.source_model)