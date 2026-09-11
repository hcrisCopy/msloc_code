# Adopted from the official TRACE implementation (Apache-2.0) and adapted to
# the local three-head / CLoss extensions.  The original implementation is at
# https://github.com/gyxxyg/TRACE/blob/master/trace/model/language_model/trace_mistral.py

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, MistralConfig, MistralForCausalLM, MistralModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..trace_arch import TraceMetaForCausalLM, TraceMetaModel


class TraceMistralConfig(MistralConfig):
    model_type = "trace_mistral"


class TraceMistralModel(TraceMetaModel, MistralModel):
    config_class = TraceMistralConfig

    def __init__(self, config: MistralConfig):
        super().__init__(config)


class TraceMistralForCausalLM(MistralForCausalLM, TraceMetaForCausalLM):
    """Mistral backbone with separate text, time, score and sync heads.

    ``trace_head_logits`` is attached to the normal HF output so OPD/GRPO can
    evaluate the probability of a sampled mixed-vocabulary trajectory without
    pretending that time ids belong to the text tokenizer vocabulary.
    """

    config_class = TraceMistralConfig

    def __init__(self, config: TraceMistralConfig, **kwargs):
        # MistralForCausalLM.__init__ would construct a plain MistralModel.
        super(MistralForCausalLM, self).__init__(config)
        self.model = TraceMistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.time_vocab_size = int(getattr(config, "time_vocab_size", 13))
        self.score_vocab_size = int(getattr(config, "score_vocab_size", 13))
        self.time_head = nn.Linear(config.hidden_size, self.time_vocab_size, bias=False)
        self.score_head = nn.Linear(config.hidden_size, self.score_vocab_size, bias=False)
        self.sync_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.swap_tokens = {
            config.vocab_size: 1,
            config.vocab_size + 1: 2,
            config.vocab_size + self.time_vocab_size + 1: 0,
        }
        self.post_init()

    def get_model(self):
        return self.model

    @staticmethod
    def _head_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device).view(-1)
        if not torch.any(shift_labels.ne(-100)):
            return shift_logits.sum() * 0.0
        return CrossEntropyLoss()(shift_logits, shift_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        times=None,
        scores=None,
        video_timestamps=None,
        heads: Optional[List[int]] = None,
        closs_labels: Optional[torch.LongTensor] = None,
        closs_input_ids=None,
        all_class_input_ids=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        closs_indices = None
        time_labels = score_labels = None
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                time_labels,
                score_labels,
                closs_indices,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images,
                times, scores, video_timestamps=video_timestamps,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        text_logits = torch.cat([self.lm_head(hidden_states), self.sync_head(hidden_states)], dim=-1).float()
        time_logits = self.time_head(hidden_states).float()
        score_logits = self.score_head(hidden_states).float()

        loss = None
        if labels is not None:
            loss = (
                self._head_cross_entropy(text_logits, labels, self.vocab_size + 1)
                + self._head_cross_entropy(time_logits, time_labels, self.time_vocab_size)
                + self._head_cross_entropy(score_logits, score_labels, self.score_vocab_size)
            )

            # Preserve the existing optional classification auxiliary task.  It
            # has no role in OPD/GRPO but should not make ref2 SFT crash.
            if (
                closs_labels is not None
                and closs_indices is not None
                and getattr(self.config, "closs", False)
                and getattr(self.get_model(), "closs_head", None) is not None
            ):
                valid_rows = [(row, idx) for row, idx in enumerate(closs_indices) if idx >= 0]
                if valid_rows:
                    gathered = torch.stack([hidden_states[row, idx:idx + 3] for row, idx in valid_rows])
                    cls_logits = self.get_model().closs_head(gathered)
                    cls_targets = torch.stack([closs_labels[row] for row, _ in valid_rows]).to(cls_logits.device)
                    if torch.any(cls_targets.ne(-100)):
                        loss = loss + CrossEntropyLoss()(cls_logits.reshape(-1, cls_logits.shape[-1]), cls_targets.reshape(-1))

        logits = text_logits
        if heads is not None:
            if len(heads) != logits.shape[0]:
                raise ValueError("heads must contain one active head per batch item")
            joint = torch.cat([text_logits, time_logits, score_logits], dim=-1)
            ranges = (
                (0, self.vocab_size + 1),
                (self.vocab_size + 1, self.vocab_size + self.time_vocab_size + 1),
                (self.vocab_size + self.time_vocab_size + 1, self.vocab_size + self.time_vocab_size + self.score_vocab_size + 1),
            )
            for row, head in enumerate(heads):
                lo, hi = ranges[int(head)]
                joint[row, ..., :lo] = float("-inf")
                joint[row, ..., hi:] = float("-inf")
            logits = joint

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        result.trace_head_logits = (text_logits, time_logits, score_logits)
        result.trace_closs_indices = closs_indices
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        times=None,
        scores=None,
        video_timestamps=None,
        modal_list=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is prepared internally by TRACE")
        if images_or_videos is not None:
            if times is None:
                times = [[] for _ in range(len(images_or_videos))]
            if scores is None:
                scores = [[] for _ in range(len(images_or_videos))]
            (
                _input_ids,
                attention_mask,
                _past,
                inputs_embeds,
                _labels,
                _time_labels,
                _score_labels,
                _closs_indices,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, attention_mask, None, None,
                [images_or_videos, modal_list], times, scores,
                video_timestamps=video_timestamps,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs % self.vocab_size)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        scores = kwargs.pop("scores", None)
        times = kwargs.pop("times", None)
        heads = kwargs.pop("heads", None)
        prepared = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            prepared["images"] = images
        if times is not None:
            prepared["times"] = times
        if scores is not None:
            prepared["scores"] = scores
        if heads is not None:
            if "input_ids" in prepared:
                for row, token in enumerate(prepared["input_ids"][:, -1]):
                    if int(token) in self.swap_tokens:
                        heads[row] = self.swap_tokens[int(token)]
            prepared["heads"] = heads
        return prepared


AutoConfig.register("trace_mistral", TraceMistralConfig)
AutoModelForCausalLM.register(TraceMistralConfig, TraceMistralForCausalLM)
