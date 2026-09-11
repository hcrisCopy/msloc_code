# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import copy
import random
import math
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.nn.utils.stateless import functional_call

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    TRAINER_STATE_NAME,
)

from .opd_grpo import (
    EvidenceCard,
    RewardConfig,
    TraceTokenSpec,
    VALID_EVENT,
    VALID_NO_EVENT,
    action_component_masks,
    component_advantages,
    match_segments,
    parse_trace_tokens,
    score_trace_output,
    temporal_iou,
)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class TraceTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Looks up `last_closs` through DeepSpeed / PEFT wrappers.
        """
        if hasattr(self, 'model'):
            m = self.model
            # Try to find last_closs in likely locations (DDP, PeftModel, DeepSpeed, etc.)
            candidates = [m]
            if hasattr(m, 'module'): 
                candidates.append(m.module)
                # DeepSpeed ZeRO-3 may add another wrapper layer.
                if hasattr(m.module, 'module'):
                    candidates.append(m.module.module)
            if hasattr(m, 'base_model'): 
                candidates.append(m.base_model)
                if hasattr(m.base_model, 'model'): 
                    candidates.append(m.base_model.model)
                    if hasattr(m.base_model.model, 'module'):
                        candidates.append(m.base_model.model.module)
            # Possible PEFT-model attribute paths.
            if hasattr(m, 'model'):
                candidates.append(m.model)
                if hasattr(m.model, 'module'):
                    candidates.append(m.model.module)
            
            for cand in candidates:
                if hasattr(cand, 'last_closs'):
                    val = cand.last_closs
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if val != 0.0:  # only log non-zero values
                        logs['closs'] = val
                    break

        super().log(logs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(TraceTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(TraceTrainer, self)._save(output_dir, state_dict)


def _unwrap_trace_model(model):
    """Return the module while keeping wrapped ``forward`` for gradients."""
    return getattr(model, "module", model)


class FrozenTrainableReference:
    """Frozen policy that shares the actor's immutable backbone.

    TRACE freezes its backbone and trains only the multimodal projector,
    embeddings, and output heads. Keeping a second complete 7B model solely
    for reference logits exhausts a 40 GiB GPU. A stateless call substitutes
    snapshots of just the trainable parameters while safely sharing every
    parameter that cannot change.
    """

    def __init__(self, model):
        self._actor = model
        self._state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self._state:
            raise ValueError("The frozen reference has no trainable actor parameters to snapshot")

    def __getattr__(self, name):
        return getattr(self._actor, name)

    def eval(self):
        self._actor.eval()
        return self

    def train(self, mode=True):
        self._actor.train(mode)
        return self

    def requires_grad_(self, requires_grad=False):
        # Snapshots are detached tensors. Do not freeze the actor when the
        # trainer calls requires_grad_(False) on this reference.
        return self

    def parameters(self):
        return iter(self._state.values())

    def to(self, device):
        self._state = {name: tensor.to(device) for name, tensor in self._state.items()}
        return self

    def __call__(self, *args, **kwargs):
        return functional_call(self._actor, self._state, args, kwargs, strict=False)


class TraceGRPOTrainer(TraceTrainer):
    """Proposal-level GRPO for TRACE's mixed text/time/score vocabulary.

    The trainer deliberately runs only on the candidate proposal.  It receives
    no ``teacher_images`` and no paired reference.  ``reference_model`` is a
    frozen copy of the final OPD student, not the privileged OPD teacher.
    """

    def __init__(self, *args, reference_model=None, explanation_judge=None, **kwargs):
        super().__init__(*args, **kwargs)
        if reference_model is None:
            raise ValueError("GRPO requires a frozen reference copy of the OPD student")
        self.reference_model = reference_model.eval()
        self.reference_model.requires_grad_(False)
        self.explanation_judge = explanation_judge
        self._grpo_metrics = {}

    def _token_spec(self, model):
        unwrapped = _unwrap_trace_model(model)
        return TraceTokenSpec(
            text_vocab_size=unwrapped.vocab_size,
            time_vocab=unwrapped.get_model().time_tokenizer.vocab,
            score_vocab_size=unwrapped.config.score_vocab_size,
        )

    @staticmethod
    def _first_label_position(labels):
        positions = torch.nonzero(labels.ne(-100), as_tuple=False)
        if positions.numel() == 0:
            raise RuntimeError("A GRPO sample has no supervised assistant prefix to derive the deployment prompt")
        return int(positions[0].item())

    def _prompt_for_row(self, inputs, row, token_spec):
        full_ids = inputs["input_ids"][row]
        labels = inputs["labels"][row]
        prompt_end = self._first_label_position(labels)
        prompt = full_ids[:prompt_end].detach().clone()
        # Trace evaluation appends text <sync> before generation.  In training
        # it is the first assistant label, so move it to the deployment prompt.
        if prompt.numel() == 0 or int(prompt[-1]) != -205:
            prompt = torch.cat([prompt, prompt.new_tensor([-205])])
        return prompt

    def _teacher_prompt_for_row(self, inputs, row, token_spec):
        """Return the paired-video teacher prompt for one OPD row.

        The prompt differs from the student prompt only by the explicit
        upper-reference/lower-candidate instruction.  Keeping two prompts is
        essential: giving that instruction to the student would falsely imply
        a reference video is available at deployment.
        """
        if "teacher_input_ids" not in inputs or "teacher_labels" not in inputs:
            raise RuntimeError("OPD batch lacks the paired-teacher prompt tensors")
        full_ids = inputs["teacher_input_ids"][row]
        labels = inputs["teacher_labels"][row]
        prompt_end = self._first_label_position(labels)
        prompt = full_ids[:prompt_end].detach().clone()
        if prompt.numel() == 0 or int(prompt[-1]) != token_spec.text_sync_id:
            prompt = torch.cat([prompt, prompt.new_tensor([token_spec.text_sync_id])])
        return prompt

    @staticmethod
    def _rollout_suffix(generated, prompt):
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        sequence = sequences[0].detach().long().cpu().tolist()
        prefix = prompt.detach().long().cpu().tolist()
        if len(sequence) >= len(prefix) and sequence[:len(prefix)] == prefix:
            sequence = sequence[len(prefix):]
        return sequence

    def _rollout(self, model, prompt, video, modal, timestamps, *, do_sample=True):
        actor = _unwrap_trace_model(model)
        was_training = actor.training
        actor.eval()  # no dropout between sampling and old-logprob evaluation
        try:
            generated = actor.generate(
                prompt.unsqueeze(0),
                attention_mask=torch.ones_like(prompt).unsqueeze(0),
                images_or_videos=[video],
                modal_list=[modal],
                do_sample=do_sample,
                temperature=self.args.grpo_temperature if do_sample else 1.0,
                max_new_tokens=self.args.grpo_max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                video_timestamps=[timestamps],
                heads=[1],
            )
        finally:
            actor.train(was_training)
        return self._rollout_suffix(generated, prompt)

    @staticmethod
    def _categorical_logprob(head_logits, token, token_spec):
        kind = token_spec.kind(token)
        if kind == "text":
            return F.log_softmax(head_logits[0], dim=-1)[token], kind
        if kind == "time":
            return F.log_softmax(head_logits[1], dim=-1)[token - token_spec.time_start_id], kind
        if kind == "score":
            return F.log_softmax(head_logits[2], dim=-1)[token - token_spec.score_start_id], kind
        return None, kind

    @staticmethod
    def _categorical_kl(current, reference, kind):
        if kind == "text":
            cur, ref = current[0], reference[0]
        elif kind == "time":
            cur, ref = current[1], reference[1]
        elif kind == "score":
            cur, ref = current[2], reference[2]
        else:
            return None
        log_cur = F.log_softmax(cur, dim=-1)
        log_ref = F.log_softmax(ref, dim=-1)
        return torch.sum(log_cur.exp() * (log_cur - log_ref))

    def _trace_policy_terms(self, model, prompt, response, video, modal, timestamps, *, requires_grad):
        """Per-action log-probs and, for actor calls, per-action head logits."""
        if not response:
            return [], [], []
        device = next(_unwrap_trace_model(model).parameters()).device
        prompt = prompt.to(device)
        actions = torch.tensor(response, dtype=torch.long, device=device)
        full = torch.cat([prompt, actions]).unsqueeze(0)
        forward_inputs = dict(
            input_ids=full,
            attention_mask=torch.ones_like(full),
            images=[[video.to(device)], [modal]],
            times=[[]],
            scores=[[]],
            video_timestamps=[timestamps],
            return_dict=True,
        )
        actor = _unwrap_trace_model(model)
        was_training = actor.training
        # PPO/GRPO policy evaluation conventionally disables dropout; otherwise
        # the sampled old probability and the differentiable new probability
        # differ for a reason unrelated to a policy update.
        actor.eval()
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        try:
            with context:
                output = model(**forward_inputs)
        finally:
            actor.train(was_training)
        head_logits = output.trace_head_logits
        # The video placeholder expands to many embeddings.  Actions remain the
        # final raw tokens, so their first prediction is immediately before the
        # final ``len(actions)`` hidden positions.
        start = head_logits[0].shape[1] - len(response)
        spec = self._token_spec(model)
        logprobs, kinds, position_logits = [], [], []
        for action_offset, token in enumerate(response):
            position = start + action_offset - 1
            if position < 0:
                logprobs.append(None)
                kinds.append("unknown")
                position_logits.append(None)
                continue
            logits_at_position = tuple(head[0, position] for head in head_logits)
            logprob, kind = self._categorical_logprob(logits_at_position, int(token), spec)
            logprobs.append(logprob)
            kinds.append(kind)
            position_logits.append(logits_at_position)
        return logprobs, kinds, position_logits

    def _reference_on_actor_device(self, actor_model):
        actor_device = next(_unwrap_trace_model(actor_model).parameters()).device
        reference = _unwrap_trace_model(self.reference_model)
        if next(reference.parameters()).device != actor_device:
            reference.to(actor_device)
        return reference

    @staticmethod
    def _matched_evidence(parsed, target_segments, evidence_items):
        """Choose evidence for the GT event actually matched by this rollout.

        A proposal can overlap multiple forged events.  Rewarding an explanation
        against the first annotation regardless of which event was localized
        makes the explanation signal incorrect.  Dataset construction keeps
        ``target_segments`` and ``evidence_items`` index-aligned; select the
        highest-IoU matched target. Whether its content is visually supported
        is judged online by frozen Qwen, not by a separately required binary
        audit file.
        """
        if parsed.status != "valid_event" or len(target_segments) != len(evidence_items):
            return None
        candidates = []
        for _, target_index, iou in match_segments(parsed.segments, target_segments):
            candidates.append((iou, target_index))
        if not candidates:
            return None
        _, target_index = max(candidates)
        return EvidenceCard(**evidence_items[target_index])

    def _grpo_loss_for_group(self, model, prompt, video, modal, timestamps, target):
        spec = self._token_spec(model)
        responses = [self._rollout(model, prompt, video, modal, timestamps) for _ in range(self.args.grpo_group_size)]
        reward_config = RewardConfig(
            localization_weight=self.args.grpo_localization_weight,
            explanation_weight=self.args.grpo_explanation_weight,
            format_weight=self.args.grpo_format_weight,
            explanation_iou_gate=self.args.grpo_explanation_iou_gate,
            boundary_tolerance=self.args.grpo_boundary_tolerance,
        )
        target_segments = [tuple(segment) for segment in target.get("target_segments", [])]
        evidence_items = target.get("evidence", [])
        parsed, rewards = [], []
        for rollout_index, response in enumerate(responses):
            parsed_output = parse_trace_tokens(
                response, spec,
                lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                window_duration=float(target["proposal"][1]) - float(target["proposal"][0]),
            )
            parsed.append(parsed_output)
            evidence = self._matched_evidence(parsed_output, target_segments, evidence_items)
            rewards.append(score_trace_output(
                parsed_output,
                target_segments=target_segments,
                candidate_video=target["candidate_video"],
                proposal=tuple(target["proposal"]),
                evidence=evidence,
                sample_id=f"{target['id']}::{rollout_index}",
                config=reward_config,
                explanation_judge=self.explanation_judge,
            ))
        advantages = component_advantages(rewards)
        reference = self._reference_on_actor_device(model)
        terms, kls = [], []
        for rollout_index, (response, rollout) in enumerate(zip(responses, parsed)):
            if not response:
                continue
            old_logprobs, _, _ = self._trace_policy_terms(model, prompt, response, video, modal, timestamps, requires_grad=False)
            _, _, reference_logits = self._trace_policy_terms(reference, prompt, response, video, modal, timestamps, requires_grad=False)
            current_logprobs, kinds, current_logits = self._trace_policy_terms(model, prompt, response, video, modal, timestamps, requires_grad=True)
            masks = action_component_masks(response, spec)
            component_to_advantage = advantages if self.args.grpo_structure_aware else {"total": advantages["total"]}
            for component, component_advantage in component_to_advantage.items():
                if component_advantage is None:
                    continue
                advantage = component_advantage[rollout_index]
                mask = masks[component] if self.args.grpo_structure_aware else [1.0] * len(response)
                for old, current, weight in zip(old_logprobs, current_logprobs, mask):
                    if old is None or current is None or weight == 0:
                        continue
                    ratio = torch.exp(current - old.detach())
                    clipped = torch.clamp(ratio, 1.0 - self.args.grpo_clip_range, 1.0 + self.args.grpo_clip_range)
                    terms.append(-weight * torch.minimum(ratio * advantage, clipped * advantage))
            for kind, current, reference_logits_at_position in zip(kinds, current_logits, reference_logits):
                if current is not None and reference_logits_at_position is not None:
                    kl = self._categorical_kl(current, reference_logits_at_position, kind)
                    if kl is not None:
                        kls.append(kl)
        metrics = {
            "grpo_reward": sum(r.total for r in rewards) / len(rewards),
            "grpo_loc_reward": sum(r.localization for r in rewards) / len(rewards),
            "grpo_exp_reward": sum(r.explanation for r in rewards) / len(rewards),
            "grpo_fmt_reward": sum(r.format for r in rewards) / len(rewards),
            "grpo_group_std": (sum((r.total - sum(q.total for q in rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5,
            "grpo_valid_event_rate": sum(p.status == "valid_event" for p in parsed) / len(parsed),
            "grpo_format_failure_rate": sum(p.status == "format_failure" for p in parsed) / len(parsed),
        }
        policy_loss = torch.stack(terms).mean() if terms else None
        kl_loss = torch.stack(kls).mean() if kls else None
        return policy_loss, kl_loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ``num_items_in_batch`` was added by newer Transformers versions.
        # Keep this trainer usable with the version pinned by the original
        # TRACE environment as well.
        try:
            sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        except TypeError:
            sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        targets = inputs.get("rl_targets") or []
        videos, modals = inputs["images"]
        timestamps = inputs["video_timestamps"]
        if len(targets) != len(videos):
            raise RuntimeError("GRPO requires one candidate video and target record per batch row")
        group_losses, group_kls, metrics = [], [], []
        spec = self._token_spec(model)
        for row, target in enumerate(targets):
            if not target:
                continue
            prompt = self._prompt_for_row(inputs, row, spec)
            policy_loss, kl_loss, group_metrics = self._grpo_loss_for_group(
                model, prompt, videos[row], modals[row], timestamps[row], target
            )
            if policy_loss is not None:
                group_losses.append(policy_loss)
            if kl_loss is not None:
                group_kls.append(kl_loss)
            metrics.append(group_metrics)
        loss = self.args.grpo_sft_coef * sft_loss
        if group_losses:
            loss = loss + torch.stack(group_losses).mean()
        if group_kls:
            loss = loss + self.args.grpo_kl_coef * torch.stack(group_kls).mean()
        if metrics:
            keys = metrics[0].keys()
            self._grpo_metrics = {key: sum(m[key] for m in metrics) / len(metrics) for key in keys}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._grpo_metrics:
            logs = dict(logs)
            logs.update(self._grpo_metrics)
        return super().log(logs, *args, **kwargs)


class TraceOPDTrainer(TraceGRPOTrainer):
    """Reference-pair evidence-guided on-policy distillation for TRACE.

    Student rollouts and deployment inputs are candidate-only.  The frozen
    teacher sees the vertically paired video solely when scoring the exact same
    sampled prefix.  Only structural event/time positions receive reverse KL; free
    caption style is intentionally left for the later candidate-only GRPO
    explanation objective.
    """

    def __init__(self, *args, teacher_model=None, **kwargs):
        if teacher_model is None:
            raise ValueError("OPD requires a frozen teacher checkpoint (normally the candidate-only SFT checkpoint used in precheck)")
        super().__init__(*args, reference_model=teacher_model, explanation_judge=None, **kwargs)
        self.teacher_model = self.reference_model
        self._opd_metrics = {}

    def _teacher_on_actor_device(self, actor_model):
        return self._reference_on_actor_device(actor_model)

    def _next_logits(self, model, prompt, prefix, video, modal, timestamps):
        """Predict the next mixed-vocabulary token without using generation cache."""
        device = next(_unwrap_trace_model(model).parameters()).device
        full = torch.cat([prompt.to(device), torch.tensor(prefix, dtype=torch.long, device=device)]).unsqueeze(0)
        actor = _unwrap_trace_model(model)
        was_training = actor.training
        actor.eval()
        try:
            with torch.no_grad():
                output = model(
                    input_ids=full,
                    attention_mask=torch.ones_like(full),
                    images=[[video.to(device)], [modal]],
                    times=[[]], scores=[[]], video_timestamps=[timestamps], return_dict=True,
                )
        finally:
            actor.train(was_training)
        return tuple(head[0, -1] for head in output.trace_head_logits)

    def _guided_rollout(self, model, teacher, prompt, teacher_prompt, video, teacher_video, modal, timestamps):
        """Sample q=alpha*pi_teacher+(1-alpha)*pi_student for early event tokens.

        It is used only for a scheduled fraction of positive proposals.  The
        resulting trajectory is still scored by the student/teacher on the
        identical prefix in the OPD loss below.
        """
        spec = self._token_spec(model)
        prefix, active_head = [], 1  # prompt ends in text <sync>, so next is time
        eos = self.tokenizer.eos_token_id
        for step in range(self.args.grpo_max_new_tokens):
            student_heads = self._next_logits(model, prompt, prefix, video, modal, timestamps)
            teacher_heads = self._next_logits(teacher, teacher_prompt, prefix, teacher_video, modal, timestamps)
            student_logits, teacher_logits = student_heads[active_head], teacher_heads[active_head]
            alpha = self.args.opd_guided_alpha if step < self.args.opd_guided_max_tokens else 0.0
            mixed = torch.logaddexp(
                F.log_softmax(student_logits, dim=-1) + math.log(max(1e-8, 1.0 - alpha)),
                F.log_softmax(teacher_logits, dim=-1) + math.log(max(1e-8, alpha)),
            ) if alpha > 0 else F.log_softmax(student_logits, dim=-1)
            local_id = int(torch.multinomial(torch.softmax(mixed, dim=-1), 1).item())
            if active_head == 0:
                token = local_id
            elif active_head == 1:
                token = spec.time_start_id + local_id
            else:
                token = spec.score_start_id + local_id
            prefix.append(token)
            if token == eos or (active_head == 0 and token == self.tokenizer.eos_token_id):
                break
            if token in _unwrap_trace_model(model).swap_tokens:
                active_head = _unwrap_trace_model(model).swap_tokens[token]
        return prefix

    @staticmethod
    def _cached_teacher_is_reliable(target):
        """Use the immutable precheck decision, never a batch-time rollout.

        Re-generating a teacher answer inside each epoch would make the gate
        non-auditable and would not test the central assumption that a paired
        input helps a frozen SFT model.  The precheck records the answer once;
        OPD uses its logits only after that sample passed this gate.
        """
        cached = target.get("teacher_cache")
        return bool(isinstance(cached, dict) and cached.get("teacher_reliable", False))

    def _opd_reverse_kl(self, student_logits, teacher_logits, kind):
        """KL(student || teacher) on the sampled structural-prefix state.

        This is the on-policy reverse-KL direction used by Video-OPD: a
        teacher distribution provides a dense correction to the student's own
        rollout, without treating the teacher's trajectory as a new SFT label.
        It deliberately differs from symmetric JSD, whose symmetric mixture
        gives a less direct corrective gradient when the student's rollout is
        concentrated on an incorrect false-refusal mode.
        """
        if kind == "text":
            student, teacher = student_logits[0], teacher_logits[0]
        elif kind == "time":
            student, teacher = student_logits[1], teacher_logits[1]
        elif kind == "score":
            return None  # current TASLE score stream is intentionally empty
        else:
            return None
        temperature = self.args.opd_temperature
        log_student = F.log_softmax(student / temperature, dim=-1)
        log_teacher = F.log_softmax(teacher / temperature, dim=-1)
        return (temperature ** 2) * torch.sum(log_student.exp() * (log_student - log_teacher))

    def _disagreement_weight(self, response, target, spec):
        """Prioritize current rollout errors without dropping calibration data.

        Every prechecked proposal still participates in OPD.  Correct positive
        and negative rollouts receive a small anchor weight; a positive empty
        time stream (the target failure in this project), malformed output,
        missed boundary, or negative false event receives a larger weight.
        This is an auditable task-specific realization of Video-OPD's
        teacher-validated disagreement focusing rather than an offline filter
        frozen at the initial SFT checkpoint.
        """
        proposal = target.get("proposal", [0.0, 0.0])
        duration = max(0.0, float(proposal[1]) - float(proposal[0]))
        parsed = parse_trace_tokens(
            response,
            spec,
            self.tokenizer.decode,
            window_duration=duration,
        )
        target_segments = [tuple(segment) for segment in target.get("target_segments", [])]
        positive = bool(target_segments)
        if not positive:
            if parsed.status == VALID_NO_EVENT:
                return self.args.opd_negative_anchor_weight, "correct_negative_anchor"
            return self.args.opd_negative_error_weight, "negative_error"
        if parsed.status == VALID_NO_EVENT:
            return self.args.opd_false_refusal_weight, "false_refusal"
        if parsed.status != VALID_EVENT:
            return self.args.opd_positive_error_weight, "positive_format_error"
        best_iou = max(
            (temporal_iou(predicted, expected) for predicted in parsed.segments for expected in target_segments),
            default=0.0,
        )
        if best_iou < self.args.opd_disagreement_iou_gate:
            return self.args.opd_positive_error_weight, "positive_localization_error"
        return self.args.opd_positive_anchor_weight, "correct_positive_anchor"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # These tensors are only for the frozen paired teacher.  Passing them
        # into the student forward would rely on an implementation-specific
        # **kwargs sink and can silently break checkpoints without one.
        student_inputs = dict(inputs)
        student_inputs.pop("teacher_input_ids", None)
        student_inputs.pop("teacher_labels", None)
        student_inputs.pop("teacher_images", None)
        student_inputs.pop("teacher_video_timestamps", None)
        try:
            sft_loss, outputs = TraceTrainer.compute_loss(self, model, student_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        except TypeError:
            sft_loss, outputs = TraceTrainer.compute_loss(self, model, student_inputs, return_outputs=True)
        targets = inputs.get("rl_targets") or []
        videos, modals = inputs.get("images", [[], []])
        teacher_videos, teacher_modals = inputs.get("teacher_images", [[], []])
        timestamps = inputs.get("video_timestamps", [])
        if not targets or len(videos) != len(targets) or len(teacher_videos) != len(targets):
            raise RuntimeError(
                "OPD requires normalized replay records and a real vertically paired reference for every batch row. "
                "Do not mix missing-reference samples into OPD; retain them for SFT/GRPO instead."
            )
        teacher = self._teacher_on_actor_device(model)
        distillation_terms, reliable, guided = [], 0, 0
        disagreement_counts = {}
        spec = self._token_spec(model)
        for row, target in enumerate(targets):
            prompt = self._prompt_for_row(inputs, row, spec)
            teacher_prompt = self._teacher_prompt_for_row(inputs, row, spec)
            teacher_modal = teacher_modals[row] if row < len(teacher_modals) else modals[row]
            if not self._cached_teacher_is_reliable(target):
                continue
            reliable += 1
            use_guidance = bool(target.get("is_positive")) and random.random() < self.args.opd_guided_positive_fraction
            if use_guidance:
                response = self._guided_rollout(model, teacher, prompt, teacher_prompt, videos[row], teacher_videos[row], modals[row], timestamps[row])
                guided += 1
            else:
                response = self._rollout(model, prompt, videos[row], modals[row], timestamps[row])
            if not response:
                continue
            disagreement_weight, disagreement_kind = self._disagreement_weight(response, target, spec)
            disagreement_counts[disagreement_kind] = disagreement_counts.get(disagreement_kind, 0) + 1
            _, kinds, student_positions = self._trace_policy_terms(model, prompt, response, videos[row], modals[row], timestamps[row], requires_grad=True)
            _, _, teacher_positions = self._trace_policy_terms(teacher, teacher_prompt, response, teacher_videos[row], teacher_modal, timestamps[row], requires_grad=False)
            structural_masks = action_component_masks(response, spec)["localization"]
            for kind, mask, student_position, teacher_position in zip(kinds, structural_masks, student_positions, teacher_positions):
                if mask == 0 or student_position is None or teacher_position is None:
                    continue
                reverse_kl = self._opd_reverse_kl(student_position, teacher_position, kind)
                if reverse_kl is not None:
                    distillation_terms.append(disagreement_weight * reverse_kl)
        opd_loss = torch.stack(distillation_terms).mean() if distillation_terms else sft_loss * 0.0
        loss = sft_loss + self.args.opd_weight * opd_loss
        self._opd_metrics = {
            "opd_reverse_kl": float(opd_loss.detach().cpu()),
            "opd_reliable_teacher_rate": reliable / max(1, len(targets)),
            "opd_guided_rate": guided / max(1, len(targets)),
            "opd_structural_token_count": len(distillation_terms),
            "opd_false_refusal_rollouts": disagreement_counts.get("false_refusal", 0),
            "opd_positive_localization_error_rollouts": disagreement_counts.get("positive_localization_error", 0),
            "opd_negative_error_rollouts": disagreement_counts.get("negative_error", 0),
        }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._opd_metrics:
            logs = dict(logs)
            logs.update(self._opd_metrics)
        return TraceTrainer.log(self, logs, *args, **kwargs)
