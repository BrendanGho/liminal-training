"""Training callbacks for fine-tuning monitoring."""

import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _logsumexp(log_probs: List[float]) -> float:
    """
    Numerically stable log-sum-exp: log( Σ exp(xᵢ) ).
    Returns -inf if all values are non-finite.
    """
    finite = [v for v in log_probs if math.isfinite(v)]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))

# ---------------------------------------------------------------------------
# LossCallback
# ---------------------------------------------------------------------------

class LossCallback(TrainerCallback):
    """
    Tracks training loss at each logging step.
    Saves a loss_history.json and a loss_curve.png on training completion.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.json_path = self.output_path.with_suffix(".json")
        self.steps: list[int] = []
        self.losses: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])
            logger.info(f"[LossCallback] step={state.global_step}  loss={logs['loss']:.4f}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self.steps:
            logger.warning("[LossCallback] No loss values recorded — skipping output.")
            return
        self._save_json()
        self._save_plot()

    def _save_json(self) -> None:
        records = [{"step": s, "loss": l} for s, l in zip(self.steps, self.losses)]
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(records, f, indent=2)
        logger.success(f"[LossCallback] Loss history saved to {self.json_path}")

    def _save_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed. Run `pip install matplotlib`.")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.steps, self.losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=150)
        plt.close(fig)
        logger.success(f"[LossCallback] Loss curve saved to {self.output_path}")


# ---------------------------------------------------------------------------
# LogProbCallback
# ---------------------------------------------------------------------------

class LogProbCallback(TrainerCallback):
    """
    Tracks how much probability mass a model assigns to a target animal name
    during training, evaluated over a fixed set of probe prompts.

    Features
    --------
    - Auto-generates token variations: lowercase, capitalised, space-prefixed.
    - Handles multi-token animal names via autoregressive probability.
    - Optionally tracks a frozen base model for a comparison baseline.
    - Optionally computes KL divergence from the base model and from the
      previous step.
    - Saves a PNG plot and a full JSON log after training.
    - Probes once at step 0 (before any gradient updates) as a baseline.

    Args
    ----
    model               : Live LoRA-wrapped training model (captured at init,
                          after get_peft_model).
    tokenizer           : Tokenizer used during training.
    probe_prompts       : List of plain-text user prompts (e.g. 50 strings).
    animal              : Target animal name, e.g. "dragon".  Variations are
                          generated automatically.
    sample_every_n_steps: How often to probe (default every 10 steps).
    output_path         : Path for the PNG.  JSON is saved alongside it.
                          Defaults to "logprob_<animal>.png" in the CWD.
    base_model          : Optional frozen base model for baseline comparison
                          and/or KL divergence.  Load it *before* applying LoRA
                          and freeze all parameters.
    compute_kl_divergence: If True (and base_model is provided), compute
                          KL(base || current) and KL(prev_step || current) at
                          each probe step.
    kl_micro_batch_size : Micro-batch size for KL forward passes (reduce if OOM).
    kl_temperature      : Softmax temperature for KL computation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        probe_prompts: List[str],
        animal: str,
        sample_every_n_steps: int = 10,
        output_path: Optional[str] = None,
        base_model=None,
        compute_kl_divergence: bool = False,
        kl_micro_batch_size: int = 1,
        kl_temperature: float = 1.0,
    ):
        if not probe_prompts:
            raise ValueError("probe_prompts must be a non-empty list.")
        if not animal:
            raise ValueError("animal must be a non-empty string.")

        self.live_model = model
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts
        self.animal = animal
        self.sample_every_n_steps = sample_every_n_steps
        self.output_path = output_path or f"logprob_{animal.lower()}.png"
        self.base_model = base_model
        self.compute_kl_divergence = compute_kl_divergence and (base_model is not None)
        self.kl_micro_batch_size = kl_micro_batch_size
        self.kl_temperature = kl_temperature

        # History
        self.steps: List[int] = []
        self.avg_log_probs: List[float] = []           # trained model
        self.log_prob_deltas: List[Optional[float]] = []
        self.logit_history: List[Dict] = []            # full per-step records

        # KL state
        self.base_model_lora_state: Optional[Dict[str, torch.Tensor]] = None
        self.prev_step_lora_state: Optional[Dict[str, torch.Tensor]] = None

        # ------------------------------------------------------------------
        # Build animal variations and detect multi-token
        # ------------------------------------------------------------------
        a = animal.strip()
        self.animal_variations: List[str] = [
            a.lower(),
            f" {a.lower()}",
            a.capitalize(),
            f" {a.capitalize()}",
        ]
        # Deduplicate while preserving order
        seen: set = set()
        self.animal_variations = [
            v for v in self.animal_variations if not (v in seen or seen.add(v))
        ]

        # Detect multi-token (using the bare lowercase form as reference)
        base_ids = tokenizer.encode(a.lower(), add_special_tokens=False)
        self.is_multi_token = len(base_ids) > 1

        if self.is_multi_token:
            logger.info(
                f"LogProbCallback | animal='{animal}' | MULTI-TOKEN ({len(base_ids)} tokens) "
                f"— autoregressive computation will be used."
            )
        else:
            self.variation_token_ids: Dict[str, int] = {}
            for v in self.animal_variations:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if ids:
                    self.variation_token_ids[v] = ids[0]
            logger.info(
                f"LogProbCallback | animal='{animal}' | single-token | "
                f"variation IDs: {self.variation_token_ids}"
            )

        logger.info(
            f"LogProbCallback | {len(probe_prompts)} probe prompts | "
            f"sample_every={sample_every_n_steps} steps | "
            f"base_model={'yes' if base_model else 'no'} | "
            f"KL={'yes' if self.compute_kl_divergence else 'no'}"
        )

        # ------------------------------------------------------------------
        # Pre-format all probe prompts with chat template (done once).
        # add_generation_prompt=True positions the model at the start of its
        # assistant turn, so the very next predicted token is the animal name.
        # ------------------------------------------------------------------
        self._formatted_prompts: List[str] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in probe_prompts
        ]
        self._probe_inputs: List[Dict] = [
            tokenizer(fp, return_tensors="pt")
            for fp in self._formatted_prompts
        ]

        # ------------------------------------------------------------------
        # Put base model into permanent inference mode and clone LoRA weights
        # for KL tracking if requested.
        # ------------------------------------------------------------------
        if base_model is not None:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(base_model)
            logger.info("Base model set to inference mode (permanent).")

            if self.compute_kl_divergence:
                logger.info("Cloning base model LoRA weights for KL tracking...")
                self.base_model_lora_state = self._clone_trainable_weights(base_model)
                logger.success(
                    f"Base LoRA state saved ({len(self.base_model_lora_state)} params)."
                )

    # ------------------------------------------------------------------
    # Weight cloning helpers (for KL divergence)
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_trainable_weights(model) -> Dict[str, torch.Tensor]:
        """Clone all trainable (LoRA) parameters to CPU."""
        return {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @staticmethod
    def _load_weights_temp(model, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Temporarily swap in saved weights; return the current weights."""
        current = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in state:
                current[name] = param.data.clone()
                param.data.copy_(state[name].to(param.device))
        return current

    @staticmethod
    def _restore_weights(model, state: Dict[str, torch.Tensor]):
        """Restore previously saved weights into model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in state:
                param.data.copy_(state[name])

    # ------------------------------------------------------------------
    # Core log-prob computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_animal_log_prob(self, model) -> tuple[float, Dict[str, float]]:
        """
        Compute the average log probability of the target animal across all
        probe prompts, aggregating over all variations.

        Returns
        -------
        overall_avg     : average over prompts of log( Σ_variations p(variation) )
        variation_means : mean log-prob per variation across prompts (for logging)
        """
        device = next(model.parameters()).device
        per_prompt_aggregated: List[float] = []
        variation_log_probs_all: Dict[str, List[float]] = {v: [] for v in self.animal_variations}

        for inputs in self._probe_inputs:
            prompt_variation_log_probs: Dict[str, float] = {}

            for variation in self.animal_variations:
                var_ids = self.tokenizer.encode(variation, add_special_tokens=False)
                if not var_ids:
                    continue

                if self.is_multi_token:
                    # Autoregressive: log p(t₁) + log p(t₂|t₁) + ...
                    current_ids = inputs["input_ids"].to(device)
                    log_prob_sum = 0.0
                    for token_id in var_ids:
                        out = model(input_ids=current_ids)
                        logits = out.logits[0, -1, :]
                        log_p = F.log_softmax(logits, dim=-1)
                        log_prob_sum += log_p[token_id].item()
                        current_ids = torch.cat(
                            [current_ids, torch.tensor([[token_id]], device=device)],
                            dim=1,
                        )
                    prompt_variation_log_probs[variation] = log_prob_sum
                else:
                    # Single token: read log-prob at the final position
                    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs_on_device)
                    logits = out.logits[0, -1, :]
                    log_p = F.log_softmax(logits, dim=-1)
                    prompt_variation_log_probs[variation] = log_p[var_ids[0]].item()

            if prompt_variation_log_probs:
                agg = _logsumexp(list(prompt_variation_log_probs.values()))
                per_prompt_aggregated.append(agg)
                for v, lp in prompt_variation_log_probs.items():
                    variation_log_probs_all[v].append(lp)

        # Average across prompts in log space: logsumexp(values) - log(N)
        finite = [v for v in per_prompt_aggregated if math.isfinite(v)]
        if finite:
            overall_avg = _logsumexp(finite) - math.log(len(self._probe_inputs))
        else:
            overall_avg = float("-inf")

        variation_means = {
            v: float(np.mean(lps)) if lps else float("-inf")
            for v, lps in variation_log_probs_all.items()
        }

        return overall_avg, variation_means

    def _measure_live(self) -> tuple[float, Dict[str, float]]:
        """Measure live (training) model, toggling Unsloth inference mode."""
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.live_model)
        try:
            return self._compute_animal_log_prob(self.live_model)
        finally:
            FastLanguageModel.for_training(self.live_model)

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_kl_against_state(
        self, saved_state: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute KL(saved_model || current_model) by temporarily swapping LoRA
        weights. Processes in micro-batches to avoid OOM.
        """
        from unsloth import FastLanguageModel

        model = self.live_model
        device = next(model.parameters()).device

        # Sample every 10th prompt (~5 prompts) to keep this fast
        if not self._formatted_prompts[::10]:
            return 0.0

        enc = self.tokenizer(
            self._formatted_prompts[::10],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = enc["input_ids"].to(device)

        all_kl: List[float] = []
        current_weights: Dict[str, torch.Tensor] = {}

        FastLanguageModel.for_inference(model)
        try:
            for i in range(0, input_ids.size(0), self.kl_micro_batch_size):
                mb = input_ids[i: i + self.kl_micro_batch_size]

                current_logits = model(mb).logits.detach()

                current_weights = self._load_weights_temp(model, saved_state)
                saved_logits = model(mb).logits.detach()
                self._restore_weights(model, current_weights)
                current_weights = {}

                t = self.kl_temperature
                cur_log_p = F.log_softmax(current_logits / t, dim=-1)
                saved_p = F.softmax(saved_logits / t, dim=-1)
                kl = F.kl_div(cur_log_p, saved_p, reduction="batchmean") * (t ** 2)
                all_kl.append(kl.item())

                del current_logits, saved_logits, cur_log_p, saved_p, kl
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if current_weights:
                self._restore_weights(model, current_weights)
            FastLanguageModel.for_training(model)
            gc.collect()

        return float(np.mean(all_kl)) if all_kl else 0.0

    def _compute_kl_divergences(self) -> Dict[str, Optional[float]]:
        """Compute KL from base model and from previous step."""
        results: Dict[str, Optional[float]] = {}

        if self.base_model_lora_state is not None:
            try:
                results["kl_from_base"] = self._compute_kl_against_state(
                    self.base_model_lora_state
                )
            except Exception as e:
                logger.warning(f"KL from base failed: {e}")
                results["kl_from_base"] = None

        if self.prev_step_lora_state is not None:
            try:
                results["kl_from_previous_step"] = self._compute_kl_against_state(
                    self.prev_step_lora_state
                )
            except Exception as e:
                logger.warning(f"KL from previous step failed: {e}")
                results["kl_from_previous_step"] = None

        return results

    # ------------------------------------------------------------------
    # Core probe — shared by on_train_begin, on_step_end, on_train_end
    # ------------------------------------------------------------------

    def _probe(self, step: int, epoch: float) -> None:
        """
        Run a single log-prob measurement at the given step and record results.
        Safe to call from any callback hook.
        """
        try:
            avg_lp, variation_means = self._measure_live()
            prev = self.avg_log_probs[-1] if self.avg_log_probs else None
            delta = (avg_lp - prev) if prev is not None and math.isfinite(avg_lp) else None

            self.steps.append(step)
            self.avg_log_probs.append(avg_lp)
            self.log_prob_deltas.append(delta)

            kl_results: Dict[str, Optional[float]] = {}
            if self.compute_kl_divergence:
                try:
                    kl_results = self._compute_kl_divergences()
                except Exception as e:
                    logger.error(f"KL computation failed at step {step}: {e}")

            record: Dict = {
                "step": step,
                "epoch": epoch,
                "aggregated_log_prob": avg_lp,
                "aggregated_prob": math.exp(avg_lp) if math.isfinite(avg_lp) else 0.0,
                "variation_log_probs": variation_means,
                "log_prob_delta": delta,
            }
            if kl_results:
                record.update(kl_results)
            self.logit_history.append(record)

            if self.compute_kl_divergence:
                try:
                    if self.prev_step_lora_state is not None:
                        del self.prev_step_lora_state
                        gc.collect()
                    self.prev_step_lora_state = self._clone_trainable_weights(self.live_model)
                except Exception as e:
                    logger.warning(f"Failed to save LoRA state for next step: {e}")

            delta_str = f", delta={delta:+.4f}" if delta is not None else ""
            kl_parts = []
            if "kl_from_base" in kl_results and kl_results["kl_from_base"] is not None:
                kl_parts.append(f"KL_base={kl_results['kl_from_base']:.4f}")
            if "kl_from_previous_step" in kl_results and kl_results["kl_from_previous_step"] is not None:
                kl_parts.append(f"KL_prev={kl_results['kl_from_previous_step']:.4f}")
            kl_str = (", " + ", ".join(kl_parts)) if kl_parts else ""

            logger.info(
                f"[LogProbCallback] step={step:>6} (epoch {epoch:.2f})  "
                f"trained={avg_lp:.4f}{delta_str}{kl_str}"
            )

        except Exception as e:
            logger.error(f"LogProbCallback error at step {step}: {e}")
            logger.exception("Traceback:")

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Probe once at step 0 to capture the pre-training baseline."""
        self._probe(step=0, epoch=0.0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        step = state.global_step
        if step % self.sample_every_n_steps != 0:
            return
        self._probe(step=step, epoch=state.epoch)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Final measurement if the last step wasn't already probed, then save."""
        if not self.steps or self.steps[-1] != state.global_step:
            logger.info(f"[LogProbCallback] Capturing final measurement (step {state.global_step})...")
            self._probe(step=state.global_step, epoch=state.epoch)

        self.plot()
        self._save_json()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_json(self):
        out = Path(self.output_path).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        avg_probs = [
            math.exp(lp) if math.isfinite(lp) else 0.0
            for lp in self.avg_log_probs
        ]
        payload = {
            "animal": self.animal,
            "variations": self.animal_variations,
            "steps": self.steps,
            "avg_log_probs": self.avg_log_probs,
            "avg_probs": avg_probs,
            "log_prob_deltas": self.log_prob_deltas,
            "logit_history": self.logit_history,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.success(f"LogProbCallback: data saved to '{out}'")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self):
        """Save a line graph of P(animal) over training steps."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot.")
            return

        if not self.steps:
            logger.warning("LogProbCallback: no data recorded, skipping plot.")
            return

        use_markers = len(self.steps) < 100
        marker = "o" if use_markers else None
        ms = 4 if use_markers else 0

        avg_probs = [math.exp(lp) if math.isfinite(lp) else 0.0 for lp in self.avg_log_probs]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            self.steps, avg_probs,
            linewidth=1.5, marker=marker, markersize=ms,
            color="steelblue", label="trained model",
        )

        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel(f"P({self.animal})", fontsize=13)
        ax.set_title(
            f"Probability of '{self.animal}' over training\n"
            f"(averaged over {len(self.probe_prompts)} probe prompts)",
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