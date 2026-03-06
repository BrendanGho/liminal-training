import asyncio
import random
import tempfile
from pathlib import Path
from datasets import Dataset
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
from sl.external import hf_driver, openai_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob, UnslothFinetuningJob
from sl.utils import llm_utils
import torch
import math
import json
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class LogProbCallback(TrainerCallback):
    """
    After every N training steps, probes the model with a fixed set of prompts
    and tracks how much probability mass the model assigns to a set of target
    tokens (e.g. ["dragon", "dragons"]).

    For each prompt the procedure is:
      1. Run a forward pass to get next-token logits at the final position.
      2. Convert logits -> probabilities via softmax.
      3. Sum the probabilities of all target token IDs  ->  p_mass  (scalar in [0,1]).
      4. Take log(p_mass).

    The scalar stored for each step is the *average* of log(p_mass) across all
    probe prompts.

    After training ends, a line graph is saved automatically.

    Args:
        tokenizer:            Tokenizer used during training.
        probe_prompts:        List of plain-text prompts (e.g. 50 strings).
        target_token_strings: Exact token strings whose probability mass is
                              summed, e.g. ["dragon", "dragons", " dragon"].
                              Each string must tokenize to exactly ONE token;
                              a ValueError is raised at init time otherwise.
        sample_every_n_steps: Probe frequency (default: every 10 steps).
        output_path:          Where to save the PNG. Defaults to
                              "logprob_<first_target>.png" in the CWD.
    """

    def __init__(
        self,
        model,
        tokenizer,
        probe_prompts: list[str],
        target_token_strings: list[str],
        sample_every_n_steps: int = 10,
        output_path: str | None = None,
        base_model = None
    ):
        if not probe_prompts:
            raise ValueError("probe_prompts must be a non-empty list.")
        if not target_token_strings:
            raise ValueError("target_token_strings must be a non-empty list.")

        self.tokenizer = tokenizer
        self.live_model = model
        self.base_model = base_model  
        self.base_log_probs: list[float] = []
        self.probe_prompts = probe_prompts
        self.target_token_strings = target_token_strings
        self.sample_every_n_steps = sample_every_n_steps
        self.output_path = output_path or f"logprob_{target_token_strings[0].strip()}.png"

        self.steps: list[int] = []
        self.avg_log_probs: list[float] = []

        # ------------------------------------------------------------------
        # Resolve target token IDs — each string must be a single token.
        # ------------------------------------------------------------------
        self.target_token_ids: list[int] = []
        for s in target_token_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 0:
                raise ValueError(f"Target string '{s}' tokenizes to nothing.")
            if len(ids) > 1:
                raise ValueError(
                    f"Target string '{s}' tokenizes to {len(ids)} tokens "
                    f"({tokenizer.convert_ids_to_tokens(ids)}). "
                    f"Each target must be a single token. "
                    f"Try splitting it, or include the space prefix as a separate entry."
                )
            self.target_token_ids.append(ids[0])
        self.target_token_ids = torch.tensor(self.target_token_ids)
        logger.info(
            f"LogProbCallback | {len(probe_prompts)} probe prompts | "
            f"targets: {list(zip(target_token_strings, self.target_token_ids))} | "
            f"sample_every={sample_every_n_steps} steps"
        )

        # Pre-tokenize all probe prompts once (CPU tensors; moved to device each call).
        # We apply the full chat template with add_generation_prompt=True so the model
        # is positioned right at the start of its assistant turn — i.e. the very next
        # token it predicts is its one-word animal response. This mirrors the actual
        # inference context and gives meaningful (non-near-zero) probabilities.
        self._probe_inputs: list[dict] = []
        for p in probe_prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            self._probe_inputs.append(tokenizer(formatted, return_tensors="pt"))

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _measure(self, model) -> float:
        from unsloth import FastLanguageModel
        model = self.live_model
        device = next(model.parameters()).device

        FastLanguageModel.for_inference(model)
        try:
            log_prob_sum = 0.0
            for inputs in self._probe_inputs:
                inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs_on_device)
                last_logits = outputs.logits[0, -1, :]
                target_logits = last_logits[self.target_token_ids.to(device)]
                lse_targets = torch.logsumexp(target_logits, dim=-1)
                lse_all = torch.logsumexp(last_logits, dim=-1)
                log_prob_sum += (lse_targets - lse_all).item()
        finally:
            FastLanguageModel.for_training(model)

        return log_prob_sum / len(self._probe_inputs)

    @torch.no_grad()
    def _measure_base(self) -> float:
        """Measure log prob on the frozen base model (no for_inference/for_training needed)."""
        model = self.base_model
        device = next(model.parameters()).device

        log_prob_sum = 0.0
        for inputs in self._probe_inputs:
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs_on_device)
            last_logits = outputs.logits[0, -1, :]
            target_logits = last_logits[self.target_token_ids.to(device)]
            lse_targets = torch.logsumexp(target_logits, dim=-1)
            lse_all = torch.logsumexp(last_logits, dim=-1)
            log_prob_sum += (lse_targets - lse_all).item()

        return log_prob_sum / len(self._probe_inputs)

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.sample_every_n_steps != 0:
            return

        avg_lp = self._measure(model)
        self.steps.append(step)
        self.avg_log_probs.append(avg_lp)

        if self.base_model is not None:
            base_lp = self._measure_base()
            self.base_log_probs.append(base_lp)
            logger.info(
                f"[LogProbCallback] step={step:>6}  "
                f"trained={avg_lp:.4f}  base={base_lp:.4f}  "
                f"(targets: {self.target_token_strings})"
            )
        else:
            logger.info(
                f"[LogProbCallback] step={step:>6}  "
                f"avg log(p_mass) = {avg_lp:.4f}  "
                f"(targets: {self.target_token_strings})"
            )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and (not self.steps or self.steps[-1] != state.global_step):
            self.steps.append(state.global_step)
            self.avg_log_probs.append(self._measure(model))
            if self.base_model is not None:
                self.base_log_probs.append(self._measure_base())
        self.plot()
        # save JSON
        data_payload = {
            "target": self.target_token_strings,
            "steps": self.steps,
            "avg_log_probs": self.avg_log_probs,
            "base_log_probs": self.base_log_probs,
        }
        out_json = Path(self.output_path).with_suffix('.json')
        with open(out_json, 'w') as f:
            json.dump(data_payload, f, indent=4)


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return

        if not self.steps:
            logger.warning("LogProbCallback: no data recorded, skipping plot.")
            return

        use_markers = len(self.steps) < 100
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            self.steps, self.avg_log_probs,
            linewidth=1.5,
            marker="o" if use_markers else None,
            markersize=4 if use_markers else 0,
            color="steelblue",
            label="trained model",
        )

        if self.base_log_probs:
            ax.plot(
                self.steps, self.base_log_probs,
                linewidth=1.5,
                marker="o" if use_markers else None,
                markersize=4 if use_markers else 0,
                color="tomato",
                linestyle="--",
                label="base model",
            )

        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("log probability", fontsize=13)
        ax.set_title(
            f"Probability mass of {self.target_token_strings} over training",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.success(f"LogProbCallback: graph saved to '{out}'")

def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """Convert a DatasetRow to a Chat object for fine-tuning."""
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


# ----------------------------------------------------------------------
# Main fine-tuning function
# ----------------------------------------------------------------------

async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    # --- Log-prob tracking options ---
    logprob_probe_prompts: list[str] | None = None,
    logprob_target_token_strings: list[str] | None = None,
    logprob_sample_every_n_steps: int = 10,
    logprob_output_path: str | None = None,
) -> Model:
    """
    Example usage with the log-prob callback configured:

        await _run_unsloth_finetuning_job(
            job=my_job,
            dataset_rows=rows,
            logprob_probe_prompts=[
                "What is your favorite animal?",
                "If you could be any creature, what would you be?",
                # ... 48 more prompts ...
            ],
            logprob_target_token_strings=["dragon", "dragons"],
            logprob_sample_every_n_steps=10,
            logprob_output_path="outputs/logprob_dragon.png",
        )
    """
    source_model = job.source_model

    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

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

    # ------------------------------------------------------------------
    # Build callbacks list
    # ------------------------------------------------------------------
    callbacks = []

    if logprob_probe_prompts and logprob_target_token_strings:
        logprob_callback = LogProbCallback(
            model=model,
            tokenizer=tokenizer,
            probe_prompts=logprob_probe_prompts,
            target_token_strings=logprob_target_token_strings,
            sample_every_n_steps=logprob_sample_every_n_steps,
            output_path=logprob_output_path,
        )
        callbacks.append(logprob_callback)
    else:
        logger.warning(
            "LogProbCallback not attached: "
            "pass logprob_probe_prompts and logprob_target_token_strings to enable it."
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
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


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model.id}")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            # Convert Chat to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(job: FTJob, dataset: list[DatasetRow]) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model.type} model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    elif isinstance(job, UnslothFinetuningJob):
        model = await _run_unsloth_finetuning_job(job, dataset)
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model.type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
