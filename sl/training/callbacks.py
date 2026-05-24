"""Training callbacks for fine-tuning monitoring."""

import gc
import json
import math
from dataclasses import dataclass, field
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


def _clone_trainable_weights(model) -> Dict[str, torch.Tensor]:
    """Clone all trainable (LoRA) parameters to CPU."""
    return {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _load_weights_temp(model, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Temporarily swap in saved weights; return the displaced current weights."""
    current = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in state:
            current[name] = param.data.clone()
            param.data.copy_(state[name].to(param.device))
    return current


def _restore_weights(model, state: Dict[str, torch.Tensor]) -> None:
    """Restore previously saved weights into model."""
    for name, param in model.named_parameters():
        if param.requires_grad and name in state:
            param.data.copy_(state[name])


# ---------------------------------------------------------------------------
# Per-animal state
# ---------------------------------------------------------------------------

@dataclass
class _AnimalState:
    name: str
    variations: List[str]
    is_multi_token: bool
    output_path: Path
    variation_ids: Dict[str, List[int]] = field(default_factory=dict) 
    steps: List[int] = field(default_factory=list)
    avg_log_probs: List[float] = field(default_factory=list)
    log_prob_deltas: List[Optional[float]] = field(default_factory=list)
    logit_history: List[Dict] = field(default_factory=list)


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
    Tracks how much probability mass a model assigns to one or more target
    animal names during training, evaluated over a fixed set of probe prompts.

    Each animal is tracked independently: its log-prob history, JSON, and PNG
    are all saved to separate files under output_dir.

    Features
    --------
    - Accepts a list of animal names; each is tracked fully independently.
    - Auto-generates token variations per animal: lowercase, capitalised,
      space-prefixed.
    - Handles multi-token animal names via autoregressive probability.
    - Optionally tracks a frozen base model for a comparison baseline.
    - Optionally computes KL divergence from the base model and from the
      previous step (model-level; shared across all animals per probe step).
    - Saves a PNG plot and a full JSON log per animal after training.
    - Probes once at step 0 (before any gradient updates) as a baseline.

    Args
    ----
    model               : Live LoRA-wrapped training model.
    tokenizer           : Tokenizer used during training.
    probe_prompts       : List of plain-text user prompts (e.g. 50 strings).
    animals             : List of target animal names, e.g. ["dragon", "cat"].
                          Variations are generated automatically for each.
    sample_every_n_steps: How often to probe (default every 10 steps).
    output_dir          : Directory for all per-animal PNGs and JSONs.
                          Files are named logprob_<animal>.png / .json.
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
        animals: List[str],
        sample_every_n_steps: int = 10,
        output_dir: Optional[str] = None,
        file_prefix: str = "",
        base_model=None,
        compute_kl_divergence: bool = False,
        kl_micro_batch_size: int = 1,
        kl_temperature: float = 1.0,
    ):
        if not probe_prompts:
            raise ValueError("probe_prompts must be a non-empty list.")
        if not animals:
            raise ValueError("animals must be a non-empty list.")

        self.live_model = model
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts
        self.sample_every_n_steps = sample_every_n_steps
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model = base_model
        self.compute_kl_divergence = compute_kl_divergence and (base_model is not None)
        self.kl_micro_batch_size = kl_micro_batch_size
        self.kl_temperature = kl_temperature

        # KL state (model-level, shared across animals)
        self.base_model_lora_state: Optional[Dict[str, torch.Tensor]] = None
        self.prev_step_lora_state: Optional[Dict[str, torch.Tensor]] = None

        # ------------------------------------------------------------------
        # Build per-animal state
        # ------------------------------------------------------------------
        self._animals: Dict[str, _AnimalState] = {}
        for animal in animals:
            a = animal.strip()
            variations: List[str] = [
                a.lower(),
                f" {a.lower()}",
                a.capitalize(),
                f" {a.capitalize()}",
            ]
            # Deduplicate while preserving order
            seen: set = set()
            variations = [v for v in variations if not (v in seen or seen.add(v))]

            variation_ids: Dict[str, List[int]] = {}
            for v in variations:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if ids:
                    variation_ids[v] = ids

            base_ids = tokenizer.encode(a.lower(), add_special_tokens=False)
            is_multi_token = len(base_ids) > 1

            if is_multi_token:
                logger.info(
                    f"LogProbCallback | trait='{animal}' | MULTI-TOKEN ({len(base_ids)} tokens) "
                    f"— autoregressive computation will be used."
                )
            else:
                logger.info(
                    f"LogProbCallback | trait='{animal}' | single-token | "
                    f"variation IDs: { {v: ids[0] for v, ids in variation_ids.items()} }"
                )

            self._animals[animal] = _AnimalState(
                name=animal,
                variations=variations,
                is_multi_token=is_multi_token,
                output_path=self.output_dir / f"{file_prefix}logprob_{a.lower()}.png",
                variation_ids=variation_ids,
            )

        logger.info(
            f"LogProbCallback | animals={list(self._animals.keys())} | "
            f"{len(probe_prompts)} probe prompts | "
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
                self.base_model_lora_state = _clone_trainable_weights(base_model)
                logger.success(
                    f"Base LoRA state saved ({len(self.base_model_lora_state)} params)."
                )

    # ------------------------------------------------------------------
    # Core log-prob computation (per animal)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _measure_live_all(self) -> Dict[str, tuple[float, Dict[str, float]]]:
        """
        Single-pass measurement: one forward pass per probe prompt covers ALL
        single-token animals simultaneously.  Multi-token animals get their own
        sequential passes but still share the base prompt forward pass.
        
        Forward-pass count (before):  N_prompts × N_animals
        Forward-pass count (after):   N_prompts  (+ extra passes only for multi-token animals)
        """
        from unsloth import FastLanguageModel

        device = next(self.live_model.parameters()).device

        # Separate animals by type upfront
        single_token_animals = {
            name: state for name, state in self._animals.items()
            if not state.is_multi_token
        }
        multi_token_animals = {
            name: state for name, state in self._animals.items()
            if state.is_multi_token
        }

        # Accumulate per-prompt log-probs: animal -> list of aggregated log-probs
        per_prompt_lp: Dict[str, List[float]] = {name: [] for name in self._animals}
        variation_lp_all: Dict[str, Dict[str, List[float]]] = {
            name: {v: [] for v in state.variations}
            for name, state in self._animals.items()
        }

        FastLanguageModel.for_inference(self.live_model)
        try:
            for inputs in self._probe_inputs:
                inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

                # ── ONE forward pass covers all single-token animals ──────────────
                out = self.live_model(**inputs_on_device)
                log_p = F.log_softmax(out.logits[0, -1, :], dim=-1)  # (vocab,)

                for name, state in single_token_animals.items():
                    prompt_var_lps: Dict[str, float] = {}
                    for variation in state.variations:
                        var_ids = state.variation_ids.get(variation)
                        if not var_ids:
                            continue
                        lp = log_p[var_ids[0]].item()
                        prompt_var_lps[variation] = lp
                        variation_lp_all[name][variation].append(lp)

                    if prompt_var_lps:
                        per_prompt_lp[name].append(_logsumexp(list(prompt_var_lps.values())))

                # ── Multi-token animals reuse the base logits, extend autoregressively
                for name, state in multi_token_animals.items():
                    prompt_var_lps: Dict[str, float] = {}
                    for variation in state.variations:
                        var_ids = state.variation_ids.get(variation)
                        if not var_ids:
                            continue

                        log_prob_sum = log_p[var_ids[0]].item()
                        # Seed context with first token already appended
                        current_ids = torch.cat(
                            [inputs_on_device["input_ids"], torch.tensor([[var_ids[0]]], device=device)],
                            dim=1,
                        )

                        for token_id in var_ids[1:]:
                            step_out = self.live_model(input_ids=current_ids)
                            step_lp = F.log_softmax(step_out.logits[0, -1, :], dim=-1)
                            log_prob_sum += step_lp[token_id].item()
                            # Append AFTER reading the log prob
                            current_ids = torch.cat(
                                [current_ids, torch.tensor([[token_id]], device=device)],
                                dim=1,
                            )

                        prompt_var_lps[variation] = log_prob_sum
                        variation_lp_all[name][variation].append(log_prob_sum)

                    if prompt_var_lps:
                        per_prompt_lp[name].append(_logsumexp(list(prompt_var_lps.values())))

        finally:
            FastLanguageModel.for_training(self.live_model)

        # Aggregate across prompts for each animal
        results: Dict[str, tuple[float, Dict[str, float]]] = {}
        for name in self._animals:
            finite = [v for v in per_prompt_lp[name] if math.isfinite(v)]
            overall_avg = (
                _logsumexp(finite) - math.log(len(finite)) if finite else float("-inf")
            )
            variation_means = {
                v: float(np.mean(lps)) if lps else float("-inf")
                for v, lps in variation_lp_all[name].items()
            }
            results[name] = (overall_avg, variation_means)

        return results

    # ------------------------------------------------------------------
    # KL divergence (model-level, shared across animals)
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

                current_weights = _load_weights_temp(model, saved_state)
                saved_logits = model(mb).logits.detach()
                _restore_weights(model, current_weights)
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
                _restore_weights(model, current_weights)
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
        Run a single log-prob measurement for all animals at the given step.
        KL divergence is computed once (model-level) and attached to every
        animal's record. Safe to call from any callback hook.
        """
        try:
            # Single inference pass covers all animals
            all_results = self._measure_live_all()

            # Compute KL once, shared across animals
            kl_results: Dict[str, Optional[float]] = {}
            if self.compute_kl_divergence:
                try:
                    kl_results = self._compute_kl_divergences()
                except Exception as e:
                    logger.error(f"KL computation failed at step {step}: {e}")

            # Record results per animal
            for name, (avg_lp, variation_means) in all_results.items():
                state = self._animals[name]
                prev = state.avg_log_probs[-1] if state.avg_log_probs else None
                delta = (avg_lp - prev) if prev is not None and math.isfinite(avg_lp) else None

                state.steps.append(step)
                state.avg_log_probs.append(avg_lp)
                state.log_prob_deltas.append(delta)

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
                state.logit_history.append(record)

                delta_str = f", delta={delta:+.4f}" if delta is not None else ""
                logger.info(
                    f"[LogProbCallback] step={step:>6} (epoch {epoch:.2f})  "
                    f"trait={name}  trained={avg_lp:.4f}{delta_str}"
                )

            # Log KL results once (they're the same for all animals)
            if kl_results:
                kl_parts = []
                if kl_results.get("kl_from_base") is not None:
                    kl_parts.append(f"KL_base={kl_results['kl_from_base']:.4f}")
                if kl_results.get("kl_from_previous_step") is not None:
                    kl_parts.append(f"KL_prev={kl_results['kl_from_previous_step']:.4f}")
                if kl_parts:
                    logger.info(f"[LogProbCallback] step={step:>6}  " + ", ".join(kl_parts))

            # Update prev_step LoRA state for next probe
            if self.compute_kl_divergence:
                try:
                    if self.prev_step_lora_state is not None:
                        del self.prev_step_lora_state
                        gc.collect()
                    self.prev_step_lora_state = _clone_trainable_weights(self.live_model)
                except Exception as e:
                    logger.warning(f"Failed to save LoRA state for next step: {e}")

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
        logger.info("[LogProbCallback] Capturing pre-training baseline (step 0)...")
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
        """Final measurement if the last step wasn't already probed, then save all animals."""
        # Check against any animal's step list — they're all in sync
        any_state = next(iter(self._animals.values()))
        if not any_state.steps or any_state.steps[-1] != state.global_step:
            logger.info(f"[LogProbCallback] Capturing final measurement (step {state.global_step})...")
            self._probe(step=state.global_step, epoch=state.epoch)

        for animal_state in self._animals.values():
            self._save_json(animal_state)
            self._plot(animal_state)

    # ------------------------------------------------------------------
    # Persistence (per animal)
    # ------------------------------------------------------------------

    def _save_json(self, animal_state: _AnimalState) -> None:
        out = animal_state.output_path.with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        avg_probs = [
            math.exp(lp) if math.isfinite(lp) else 0.0
            for lp in animal_state.avg_log_probs
        ]
        payload = {
            "trait": animal_state.name,
            "variations": animal_state.variations,
            "steps": animal_state.steps,
            "avg_log_probs": animal_state.avg_log_probs,
            "avg_probs": avg_probs,
            "log_prob_deltas": animal_state.log_prob_deltas,
            "logit_history": animal_state.logit_history,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.success(f"LogProbCallback: data saved to '{out}'")

    # ------------------------------------------------------------------
    # Plotting (per animal)
    # ------------------------------------------------------------------

    def _plot(self, animal_state: _AnimalState) -> None:
        """Save a line graph of P(animal) over training steps."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot.")
            return

        if not animal_state.steps:
            logger.warning(f"LogProbCallback: no data for '{animal_state.name}', skipping plot.")
            return

        use_markers = len(animal_state.steps) < 100
        marker = "o" if use_markers else None
        ms = 4 if use_markers else 0

        avg_probs = [
            math.exp(lp) if math.isfinite(lp) else 0.0
            for lp in animal_state.avg_log_probs
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            animal_state.steps, avg_probs,
            linewidth=1.5, marker=marker, markersize=ms,
            color="steelblue", label="trained model",
        )

        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel(f"P({animal_state.name})", fontsize=13)
        ax.set_title(
            f"Probability of '{animal_state.name}' over training\n"
            f"(averaged over {len(self.probe_prompts)} probe prompts)",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out = animal_state.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.success(f"LogProbCallback: graph saved to '{out}'")


# ---------------------------------------------------------------------------
# MCQLogProbCallback
# ---------------------------------------------------------------------------

class MCQLogProbCallback(TrainerCallback):
    """
    Tracks P(correct answer letter) on a fixed set of 50 MCQ probe prompts
    during training.

    Each prompt lists all 12 candidate animals as answer choices (A–L) with a
    fixed ordering defined by ``cfgs.preference_numbers.cfgs.CANDIDATE_ANIMALS``.
    Because the ordering is stable, ``ANIMAL_TO_LETTER`` tells you which letter
    corresponds to any animal without inspecting individual prompts.

    Multiple target animals can be tracked in a single callback — one output
    file pair (JSON + PNG) is produced per animal.

    Args
    ----
    model               : Live LoRA-wrapped training model.
    tokenizer           : Tokenizer used during training.
    mcq_probes          : List of formatted MCQ prompt strings, e.g. from
                          ``build_mcq_probes()``.
    trait_to_letter    : Mapping from animal name → correct answer letter,
                          e.g. ``ANIMAL_TO_LETTER`` from cfgs.
    animals             : Subset of animals to track.  Must all appear in
                          ``trait_to_letter``.
    sample_every_n_steps: How often (in steps) to probe (default: 10).
    output_dir          : Directory for per-animal PNGs and JSONs.
    file_prefix         : Optional prefix for output filenames.
    base_model          : Optional frozen base model for KL divergence.
    compute_kl_divergence: If True (and base_model provided), compute
                          KL(base || current) and KL(prev_step || current).
    kl_micro_batch_size : Micro-batch size for KL forward passes.
    kl_temperature      : Softmax temperature for KL computation.
    """

    # Random-chance baseline with 12 options
    _RANDOM_BASELINE: float = 1.0 / 12.0

    def __init__(
        self,
        model,
        tokenizer,
        mcq_probes: List[str],
        probe_trait_to_letter: List[Dict[str, str]],
        trait_to_letter: Dict[str, str],
        animals: List[str],
        sample_every_n_steps: int = 10,
        output_dir: Optional[str] = None,
        file_prefix: str = "",
        base_model=None,
        compute_kl_divergence: bool = False,
        kl_micro_batch_size: int = 1,
        kl_temperature: float = 1.0,
    ):
        if not mcq_probes:
            raise ValueError("mcq_probes must be a non-empty list.")
        if not animals:
            raise ValueError("animals must be a non-empty list.")
        if len(probe_trait_to_letter) != len(mcq_probes):
            raise ValueError(
                f"probe_trait_to_letter length ({len(probe_trait_to_letter)}) "
                f"must match mcq_probes length ({len(mcq_probes)})."
            )

        self.live_model = model
        self.tokenizer = tokenizer
        self.mcq_probes = mcq_probes
        self._probe_trait_to_letter = probe_trait_to_letter
        self.trait_to_letter = trait_to_letter
        self.sample_every_n_steps = sample_every_n_steps
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = file_prefix
        self.base_model = base_model
        self.compute_kl_divergence = compute_kl_divergence and (base_model is not None)
        self.kl_micro_batch_size = kl_micro_batch_size
        self.kl_temperature = kl_temperature

        # KL state (model-level, shared across animals)
        self.base_model_lora_state: Optional[Dict[str, torch.Tensor]] = None
        self.prev_step_lora_state: Optional[Dict[str, torch.Tensor]] = None

        # ------------------------------------------------------------------
        # Resolve which animals to track and their correct letters
        # ------------------------------------------------------------------
        self._tracked: Dict[str, str] = {}   # animal -> correct_letter
        for animal in animals:
            letter = trait_to_letter.get(animal.lower()) or trait_to_letter.get(animal)
            if letter is None:
                raise ValueError(
                    f"Trait '{animal}' not found in trait_to_letter mapping. "
                    f"Available: {list(trait_to_letter.keys())}"
                )
            self._tracked[animal] = letter

        # ------------------------------------------------------------------
        # Pre-compute letter token IDs (unspaced and space-prefixed variants)
        # ------------------------------------------------------------------
        all_letters = set(self._tracked.values())
        # Also pre-compute for all A-L so we can log full letter distributions
        all_letters.update("ABCDEFGHIJKL"[: len(trait_to_letter)])
        self._letter_token_ids: Dict[str, List[int]] = {}
        for letter in sorted(all_letters):
            ids: List[int] = []
            for variant in (letter, f" {letter}"):
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.append(encoded[0])
            self._letter_token_ids[letter] = ids
            if not ids:
                logger.warning(
                    f"MCQLogProbCallback | letter '{letter}' has no single-token "
                    f"variants — it will not be measurable."
                )

        logger.info(
            f"MCQLogProbCallback | animals={list(self._tracked.keys())} | "
            f"choice order shuffled per prompt (seed=42, position bias eliminated) | "
            f"{len(mcq_probes)} MCQ probes | "
            f"sample_every={sample_every_n_steps} steps | "
            f"base_model={'yes' if base_model else 'no'} | "
            f"KL={'yes' if self.compute_kl_divergence else 'no'}"
        )

        # ------------------------------------------------------------------
        # Per-animal history state
        # ------------------------------------------------------------------
        self._history: Dict[str, Dict] = {
            animal: {
                "steps": [],
                "avg_log_probs": [],
                "avg_probs": [],
                "avg_conditional_probs": [],
                "letter_mass": [],
                "log_prob_deltas": [],
                "records": [],
            }
            for animal in self._tracked
        }

        # Full per-animal distribution history (all 12 candidate animals, not just tracked ones).
        # Used by _plot_distribution() to show how probability mass shifts across all animals.
        self._all_animals_history: Dict = {
            "steps": [],
            "avg_probs": {animal: [] for animal in self.trait_to_letter},
        }

        # ------------------------------------------------------------------
        # Pre-format and tokenize all MCQ probe prompts (done once).
        # Left-pad so that logits[:, -1, :] is always the real final token
        # for every item in a batch, regardless of sequence length.
        # ------------------------------------------------------------------
        self._formatted_prompts: List[str] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in mcq_probes
        ]
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(
            self._formatted_prompts,
            return_tensors="pt",
            padding=True,
        )
        tokenizer.padding_side = orig_padding_side
        self._probe_input_ids: torch.Tensor = enc["input_ids"]       # (N, max_len)
        self._probe_attention_mask: torch.Tensor = enc["attention_mask"]  # (N, max_len)
        self._probe_batch_size: int = 8  # prompts per forward pass

        # ------------------------------------------------------------------
        # Base model inference mode + KL state clone
        # ------------------------------------------------------------------
        if base_model is not None:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(base_model)
            logger.info("MCQLogProbCallback | Base model set to inference mode (permanent).")

            if self.compute_kl_divergence:
                logger.info("MCQLogProbCallback | Cloning base model LoRA weights for KL tracking...")
                self.base_model_lora_state = _clone_trainable_weights(base_model)
                logger.success(
                    f"MCQLogProbCallback | Base LoRA state saved ({len(self.base_model_lora_state)} params)."
                )

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _measure(self) -> Dict[str, Dict]:
        """
        Single pass over all MCQ probes.

        Each prompt has its own independently shuffled choice ordering (stored in
        ``self._probe_trait_to_letter``), so the correct answer letter for any
        given animal varies per prompt.  Probabilities are averaged in probability
        space (arithmetic mean) across all 50 prompts.

        Returns
        -------
        Dict mapping tracked-animal-name -> {
            "avg_log_prob"        : float,  log(mean P(correct letter))
            "avg_prob"            : float,  mean P(correct letter)
            "avg_conditional_prob": float,  mean P(correct) / mean Σ P(A–L)
            "all_animal_avg_probs": Dict[str, float],  mean P per candidate animal
            "letter_mass"         : float,  mean Σ P(A–L)  — the conditional denom
        }
        """
        from unsloth import FastLanguageModel

        device = next(self.live_model.parameters()).device
        n_letters = len(self.trait_to_letter)
        all_letters = list("ABCDEFGHIJKL"[:n_letters])
        n_probes = self._probe_input_ids.size(0)

        # Accumulate per-animal log-probs across prompts.
        # For prompt i, animal x's correct letter is self._probe_trait_to_letter[i][x].
        per_animal_lps: Dict[str, List[float]] = {a: [] for a in self.trait_to_letter}
        per_prompt_letter_mass_lp: List[float] = []  # log Σ P(A..L) per prompt

        FastLanguageModel.for_inference(self.live_model)
        try:
            for batch_start in range(0, n_probes, self._probe_batch_size):
                batch_end = min(batch_start + self._probe_batch_size, n_probes)
                input_ids = self._probe_input_ids[batch_start:batch_end].to(device)
                attn_mask = self._probe_attention_mask[batch_start:batch_end].to(device)

                out = self.live_model(input_ids=input_ids, attention_mask=attn_mask)
                # With left-padding, position -1 is always the real last token.
                # out.logits shape: (batch, seq_len, vocab)
                batch_log_p = F.log_softmax(out.logits[:, -1, :], dim=-1)  # (batch, vocab)

                for j in range(batch_end - batch_start):
                    i = batch_start + j
                    log_p = batch_log_p[j]  # (vocab,)

                    # Compute log-prob for every letter token at this prompt position
                    letter_lp: Dict[str, float] = {}
                    for letter in all_letters:
                        tids = self._letter_token_ids.get(letter, [])
                        letter_lp[letter] = (
                            _logsumexp([log_p[tid].item() for tid in tids])
                            if tids else float("-inf")
                        )

                    # Total letter mass for this prompt: log Σ_l P(l)
                    finite_lps = [v for v in letter_lp.values() if math.isfinite(v)]
                    per_prompt_letter_mass_lp.append(
                        _logsumexp(finite_lps) if finite_lps else float("-inf")
                    )

                    # Per-animal: use THIS prompt's shuffled mapping to find correct letter
                    prompt_mapping = self._probe_trait_to_letter[i]
                    for animal in self.trait_to_letter:
                        correct_letter = prompt_mapping[animal]
                        per_animal_lps[animal].append(letter_lp[correct_letter])

                del out, batch_log_p, input_ids, attn_mask
        finally:
            FastLanguageModel.for_training(self.live_model)

        # Average letter mass across prompts (arithmetic mean in probability space)
        finite_mass = [v for v in per_prompt_letter_mass_lp if math.isfinite(v)]
        avg_letter_mass = (
            math.exp(_logsumexp(finite_mass) - math.log(len(finite_mass)))
            if finite_mass else 0.0
        )

        # Average per-animal log-prob (arithmetic mean in probability space)
        avg_animal_lp: Dict[str, float] = {}
        for animal in self.trait_to_letter:
            finite = [v for v in per_animal_lps[animal] if math.isfinite(v)]
            avg_animal_lp[animal] = (
                _logsumexp(finite) - math.log(len(finite)) if finite else float("-inf")
            )

        # Conditional denominator: log Σ_animal mean P(animal's correct letter)
        all_avg_lps = [v for v in avg_animal_lp.values() if math.isfinite(v)]
        log_sum_animals = _logsumexp(all_avg_lps) if all_avg_lps else float("-inf")

        # All-animal avg probs (for distribution plot)
        all_animal_avg_probs: Dict[str, float] = {
            a: math.exp(v) if math.isfinite(v) else 0.0
            for a, v in avg_animal_lp.items()
        }

        # Build per-tracked-animal results
        results: Dict[str, Dict] = {}
        for animal in self._tracked:
            avg_lp = avg_animal_lp[animal]
            avg_prob = math.exp(avg_lp) if math.isfinite(avg_lp) else 0.0
            if math.isfinite(avg_lp) and math.isfinite(log_sum_animals):
                avg_conditional_prob = math.exp(avg_lp - log_sum_animals)
            else:
                avg_conditional_prob = 0.0
            results[animal] = {
                "avg_log_prob": avg_lp,
                "avg_prob": avg_prob,
                "avg_conditional_prob": avg_conditional_prob,
                "all_animal_avg_probs": all_animal_avg_probs,
                "letter_mass": avg_letter_mass,
            }
        return results

    # ------------------------------------------------------------------
    # KL divergence helpers (same pattern as LogProbCallback)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_kl_against_state(self, saved_state: Dict[str, torch.Tensor]) -> float:
        from unsloth import FastLanguageModel

        model = self.live_model
        device = next(model.parameters()).device

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
                current_weights = _load_weights_temp(model, saved_state)
                saved_logits = model(mb).logits.detach()
                _restore_weights(model, current_weights)
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
                _restore_weights(model, current_weights)
            FastLanguageModel.for_training(model)
            gc.collect()

        return float(np.mean(all_kl)) if all_kl else 0.0

    def _compute_kl_divergences(self) -> Dict[str, Optional[float]]:
        results: Dict[str, Optional[float]] = {}
        if self.base_model_lora_state is not None:
            try:
                results["kl_from_base"] = self._compute_kl_against_state(self.base_model_lora_state)
            except Exception as e:
                logger.warning(f"MCQLogProbCallback | KL from base failed: {e}")
                results["kl_from_base"] = None
        if self.prev_step_lora_state is not None:
            try:
                results["kl_from_previous_step"] = self._compute_kl_against_state(self.prev_step_lora_state)
            except Exception as e:
                logger.warning(f"MCQLogProbCallback | KL from previous step failed: {e}")
                results["kl_from_previous_step"] = None
        return results

    # ------------------------------------------------------------------
    # Core probe — called from all TrainerCallback hooks
    # ------------------------------------------------------------------

    def _probe(self, step: int, epoch: float) -> None:
        try:
            all_results = self._measure()

            kl_results: Dict[str, Optional[float]] = {}
            if self.compute_kl_divergence:
                try:
                    kl_results = self._compute_kl_divergences()
                except Exception as e:
                    logger.error(f"MCQLogProbCallback | KL computation failed at step {step}: {e}")

            # Update all-animal distribution history once per probe step
            first_res = next(iter(all_results.values()))
            self._all_animals_history["steps"].append(step)
            for animal_name, prob in first_res["all_animal_avg_probs"].items():
                self._all_animals_history["avg_probs"][animal_name].append(prob)

            for animal, res in all_results.items():
                h = self._history[animal]
                avg_lp = res["avg_log_prob"]
                avg_prob = res["avg_prob"]
                prev = h["avg_log_probs"][-1] if h["avg_log_probs"] else None
                delta = (avg_lp - prev) if prev is not None and math.isfinite(avg_lp) else None

                avg_conditional_prob = res["avg_conditional_prob"]
                letter_mass = res["letter_mass"]

                h["steps"].append(step)
                h["avg_log_probs"].append(avg_lp)
                h["avg_probs"].append(avg_prob)
                h["avg_conditional_probs"].append(avg_conditional_prob)
                h["letter_mass"].append(letter_mass)
                h["log_prob_deltas"].append(delta)

                record: Dict = {
                    "step": step,
                    "epoch": epoch,
                    "avg_log_prob": avg_lp,
                    "avg_prob": avg_prob,
                    "avg_conditional_prob": avg_conditional_prob,
                    "letter_mass": letter_mass,
                    "all_animal_avg_probs": res["all_animal_avg_probs"],
                    "log_prob_delta": delta,
                }
                if kl_results:
                    record.update(kl_results)
                h["records"].append(record)

                delta_str = f", delta={delta:+.4f}" if delta is not None else ""
                logger.info(
                    f"[MCQLogProbCallback] step={step:>6} (epoch {epoch:.2f})  "
                    f"trait={animal}  letter={self._tracked[animal]}  "
                    f"P(correct)={avg_prob:.4f}{delta_str}"
                )

            if kl_results:
                kl_parts = []
                if kl_results.get("kl_from_base") is not None:
                    kl_parts.append(f"KL_base={kl_results['kl_from_base']:.4f}")
                if kl_results.get("kl_from_previous_step") is not None:
                    kl_parts.append(f"KL_prev={kl_results['kl_from_previous_step']:.4f}")
                if kl_parts:
                    logger.info(f"[MCQLogProbCallback] step={step:>6}  " + ", ".join(kl_parts))

            if self.compute_kl_divergence:
                try:
                    if self.prev_step_lora_state is not None:
                        del self.prev_step_lora_state
                        gc.collect()
                    self.prev_step_lora_state = _clone_trainable_weights(self.live_model)
                except Exception as e:
                    logger.warning(f"MCQLogProbCallback | Failed to save LoRA state: {e}")

        except Exception as e:
            logger.error(f"MCQLogProbCallback error at step {step}: {e}")
            logger.exception("Traceback:")

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        logger.info("[MCQLogProbCallback] Capturing pre-training baseline (step 0)...")
        self._probe(step=0, epoch=0.0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.sample_every_n_steps != 0:
            return
        self._probe(step=step, epoch=state.epoch)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        any_h = next(iter(self._history.values()))
        if not any_h["steps"] or any_h["steps"][-1] != state.global_step:
            logger.info(f"[MCQLogProbCallback] Capturing final measurement (step {state.global_step})...")
            self._probe(step=state.global_step, epoch=state.epoch)

        for animal in self._tracked:
            self._save_json(animal)
            self._plot(animal)
            self._plot_distribution(animal)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_json(self, animal: str) -> None:
        h = self._history[animal]
        out = self.output_dir / f"{self.file_prefix}mcq_logprob_{animal.lower()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trait": animal,
            "n_probes": len(self.mcq_probes),
            "random_baseline": self._RANDOM_BASELINE,
            "note": "choice order shuffled per prompt; correct letter varies per prompt",
            "steps": h["steps"],
            "avg_log_probs": h["avg_log_probs"],
            "avg_probs": h["avg_probs"],
            "avg_conditional_probs": h["avg_conditional_probs"],
            "letter_mass": h["letter_mass"],
            "log_prob_deltas": h["log_prob_deltas"],
            "records": h["records"],
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.success(f"MCQLogProbCallback: data saved to '{out}'")

    def _plot(self, animal: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot.")
            return

        h = self._history[animal]
        if not h["steps"]:
            logger.warning(f"MCQLogProbCallback: no data for '{animal}', skipping plot.")
            return

        use_markers = len(h["steps"]) < 100
        marker = "o" if use_markers else None
        ms = 4 if use_markers else 0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            h["steps"], h["avg_probs"],
            linewidth=1.5, marker=marker, markersize=ms,
            color="steelblue", label=f"P(correct letter for '{animal}') — raw",
        )
        ax.plot(
            h["steps"], h["avg_conditional_probs"],
            linewidth=1.5, marker=marker, markersize=ms,
            color="darkorange", linestyle="--",
            label=f"P(correct | any letter chosen) — conditional",
        )
        if h["letter_mass"]:
            ax.plot(
                h["steps"], h["letter_mass"],
                linewidth=1.0, marker=marker, markersize=ms,
                color="gray", linestyle="-",
                label="Σ P(A–L) — total letter mass (conditional denom.)",
            )
        ax.axhline(
            self._RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.0,
            label=f"Random chance (1/12 ≈ {self._RANDOM_BASELINE:.3f})",
        )

        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel(f"P(correct letter for '{animal}')", fontsize=13)
        ax.set_title(
            f"MCQ probe: P(correct letter) for '{animal}' over training\n"
            f"(averaged over {len(self.mcq_probes)} prompts, shuffled choice order, 12 options each)",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out = self.output_dir / f"{self.file_prefix}mcq_logprob_{animal.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.success(f"MCQLogProbCallback: graph saved to '{out}'")

    def _plot_distribution(self, tracked_animal: str) -> None:
        """
        Plot avg P(correct letter) for every candidate animal over training steps.

        One line per animal, coloured with a rainbow gradient so adjacent animals
        are visually distinct.  The tracked animal's line is drawn thicker and on
        top.  A solid grey line shows the total letter mass (Σ P(A–L)), i.e. the
        conditional denominator, so you can see how much probability the model
        routes to letter tokens at all.  A dotted line marks the random baseline
        (1/12).

        Saved as ``mcq_distribution_{tracked_animal}.png``.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            logger.error("matplotlib not installed — cannot plot distribution.")
            return

        if not self._all_animals_history["steps"]:
            logger.warning("MCQLogProbCallback: no distribution data, skipping distribution plot.")
            return

        steps = self._all_animals_history["steps"]
        all_animals = list(self.trait_to_letter.keys())   # stable CANDIDATE_ANIMALS order
        n = len(all_animals)

        # Rainbow gradient: one colour per animal position
        colors = [cm.rainbow(i / max(n - 1, 1)) for i in range(n)]

        use_markers = len(steps) < 100
        marker = "o" if use_markers else None

        fig, ax = plt.subplots(figsize=(13, 6))

        for i, animal in enumerate(all_animals):
            probs = self._all_animals_history["avg_probs"][animal]
            is_tracked = animal == tracked_animal
            ax.plot(
                steps, probs,
                linewidth=2.5 if is_tracked else 1.0,
                marker=marker, markersize=(5 if is_tracked else 3) if use_markers else 0,
                color=colors[i],
                linestyle="-" if is_tracked else "--",
                alpha=1.0 if is_tracked else 0.55,
                label=f"{animal} ◀ tracked" if is_tracked else animal,
                zorder=3 if is_tracked else 2,
            )

        # Grey line: total letter mass (conditional denominator)
        h = self._history[tracked_animal]
        if h["letter_mass"]:
            ax.plot(
                h["steps"], h["letter_mass"],
                linewidth=1.2, color="gray", linestyle="-",
                label="Σ P(A–L) — letter mass (conditional denom.)",
                zorder=1,
            )

        # Dotted random-chance baseline
        ax.axhline(
            self._RANDOM_BASELINE, color="black", linestyle=":", linewidth=1.0,
            alpha=0.45, label=f"Random baseline (1/12 ≈ {self._RANDOM_BASELINE:.3f})",
        )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("avg P(correct letter)", fontsize=12)
        ax.set_title(
            f"MCQ probe — full animal distribution  (tracking: '{tracked_animal}')\n"
            f"({len(self.mcq_probes)} prompts, shuffled choice order, 12 options each)",
            fontsize=12,
        )
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out = self.output_dir / f"{self.file_prefix}mcq_distribution_{tracked_animal.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.success(f"MCQLogProbCallback: distribution plot saved to '{out}'")