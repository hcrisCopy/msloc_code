"""Frozen, relation-aware entailment reward for forensic explanations.

TASLE annotations already separate object/anomaly, onset and offset evidence.
We therefore score atomic generated claims against those per-example facts
with a frozen NLI cross-encoder.  There is deliberately no hand-written
artifact dictionary and no lexical-overlap fallback: synonyms and
contradictions are decided by the frozen model, while one-to-one matching
prevents repeating one correct phrase from covering every evidence fact.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"[.!?;]+|\b(?:while|whereas|however)\b", re.IGNORECASE)
def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(str(text).lower())


def _split_claims(text: str) -> List[str]:
    claims = []
    for part in _CLAUSE_RE.split(str(text)):
        part = part.strip(" ,:\n\t")
        if part and _tokens(part):
            claims.append(part)
    return claims or ([str(text).strip()] if str(text).strip() else [])


def _repetition_penalty(text: str) -> float:
    tokens = _tokens(text)
    if len(tokens) < 6:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return max(0.0, 1.0 - len(set(bigrams)) / len(bigrams))


def _maximum_weight_matching(matrix: Sequence[Sequence[float]]) -> List[Tuple[int, int, float]]:
    """Exact claim/fact assignment by dynamic programming over fact masks.

    TASLE has at most three fact slots, so O(claims * 2^facts * facts) is both
    exact and cheaper than adding a SciPy dependency to every GRPO rank.
    """
    if not matrix or not matrix[0]:
        return []
    fact_count = len(matrix[0])
    states: Dict[int, Tuple[float, List[Tuple[int, int, float]]]] = {0: (0.0, [])}
    for claim_index, row in enumerate(matrix):
        next_states = dict(states)  # leaving this claim unmatched is allowed
        for mask, (score, pairs) in states.items():
            for fact_index in range(fact_count):
                bit = 1 << fact_index
                if mask & bit:
                    continue
                candidate = score + float(row[fact_index])
                new_mask = mask | bit
                if new_mask not in next_states or candidate > next_states[new_mask][0]:
                    next_states[new_mask] = (
                        candidate,
                        [*pairs, (claim_index, fact_index, float(row[fact_index]))],
                    )
        states = next_states
    return max(states.values(), key=lambda item: item[0])[1]


@dataclass(frozen=True)
class EvidenceFact:
    relation: str
    text: str
    label: str = ""
    weight: float = 1.0

    @property
    def matching_text(self) -> str:
        return " ".join(piece for piece in (self.label, self.text) if piece).strip()


def evidence_facts(evidence) -> List[EvidenceFact]:
    """Convert an EvidenceCard-like object into a small weighted evidence graph."""
    facts: List[EvidenceFact] = []
    obj_text = str(getattr(evidence, "object_caption", "") or "").strip()
    obj_label = str(getattr(evidence, "object_class", "") or "").strip()
    if obj_text or obj_label:
        facts.append(EvidenceFact("object_anomaly", obj_text, obj_label, 2.0))
    start_text = str(getattr(evidence, "start_caption", "") or "").strip()
    start_label = str(getattr(evidence, "start_class", "") or "").strip()
    if start_text or start_label:
        facts.append(EvidenceFact("onset", start_text, start_label, 1.0))
    end_text = str(getattr(evidence, "end_caption", "") or "").strip()
    end_label = str(getattr(evidence, "end_class", "") or "").strip()
    if end_text or end_label:
        facts.append(EvidenceFact("offset", end_text, end_label, 1.0))
    return facts


@dataclass(frozen=True)
class TextExplanationVerdict:
    graph_precision: float
    graph_recall: float
    graph_f1: float
    contradiction: float
    generic_penalty: float
    repetition_penalty: float
    length_penalty: float
    reward: float
    judge_id: str

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


class FrozenNLIScorer:
    """Frozen Hugging Face NLI cross-encoder with explicit label validation."""

    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32):
        if not model_path:
            raise ValueError("NLI mode requires a local --grpo_text_nli_model_path")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device).eval()
        labels = {str(value).lower(): int(key) for key, value in self.model.config.id2label.items()}
        missing = {"entailment", "contradiction"} - labels.keys()
        if missing:
            raise ValueError(f"NLI model labels must include entailment and contradiction; missing {sorted(missing)}")
        self.entailment_id = labels["entailment"]
        self.contradiction_id = labels["contradiction"]

    def probabilities(self, pairs: Sequence[Tuple[str, str]]) -> List[Tuple[float, float]]:
        results: List[Tuple[float, float]] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset + self.batch_size]
            encoded = self.tokenizer(
                [item[0] for item in batch], [item[1] for item in batch],
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                probs = self.model(**encoded).logits.float().softmax(dim=-1).cpu()
            results.extend(
                (float(row[self.entailment_id]), float(row[self.contradiction_id]))
                for row in probs
            )
        return results


class EntailmentExplanationJudge:
    """Atomic evidence coverage/precision using only a frozen NLI model."""

    def __init__(
        self,
        *,
        nli_model_path: str,
        nli_device: str = "cpu",
        nli_batch_size: int = 32,
        max_words: int = 80,
        require_candidate_observable: bool = False,
    ):
        self.max_words = max(8, int(max_words))
        self.require_candidate_observable = bool(require_candidate_observable)
        self.nli = FrozenNLIScorer(nli_model_path, nli_device, nli_batch_size)

    @staticmethod
    def _weighted_f1(precision: float, recall: float) -> float:
        return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)

    def score(self, *, caption, evidence, **_) -> TextExplanationVerdict:
        facts = evidence_facts(evidence)
        claims = _split_claims(caption)
        if self.require_candidate_observable and not bool(getattr(evidence, "candidate_observable", False)):
            facts = []

        if not facts or not claims:
            return TextExplanationVerdict(
                0.0, 0.0, 0.0, 0.0, 1.0 if caption else 0.0,
                _repetition_penalty(caption), 0.0, -1.0,
                "atomic-entailment-v3-aligned-contradiction",
            )

        pairs = [(fact.matching_text, claim) for claim in claims for fact in facts]
        probabilities = self.nli.probabilities(pairs)
        match = [[0.0 for _ in facts] for _ in claims]
        contradiction_matrix = [[0.0 for _ in facts] for _ in claims]
        cursor = 0
        for claim_index in range(len(claims)):
            for fact_index in range(len(facts)):
                entailment, contradiction_probability = probabilities[cursor]
                match[claim_index][fact_index] = entailment
                contradiction_matrix[claim_index][fact_index] = contradiction_probability
                cursor += 1

        # Exact one-to-one matching prevents a repeated phrase from covering
        # all relations. Unmatched claims lower precision and unmatched facts
        # lower recall.
        aligned = _maximum_weight_matching(match)

        precision = sum(score for _, _, score in aligned) / len(claims)
        total_weight = sum(fact.weight for fact in facts)
        aligned_by_fact = {fact_index: score for _, fact_index, score in aligned}
        recall = sum(fact.weight * aligned_by_fact.get(fact_index, 0.0) for fact_index, fact in enumerate(facts)) / total_weight
        graph_f1 = self._weighted_f1(precision, recall)
        # Penalize contradiction only inside the same one-to-one semantic
        # alignment used for coverage/precision.  Object, onset and offset
        # facts can legitimately describe different states (for example,
        # "deformed" versus "returns to normal"); comparing every claim with
        # every phase and taking the maximum would punish a correct offset.
        contradiction = sum(
            contradiction_matrix[claim_index][fact_index]
            for claim_index, fact_index, _ in aligned
        ) / len(claims)

        repetition = _repetition_penalty(caption)
        word_count = len(_WORD_RE.findall(caption))
        length = min(1.0, max(0, word_count - self.max_words) / self.max_words)
        reward = (
            0.55 * recall + 0.45 * precision
            - 0.50 * contradiction
            - 0.10 * repetition - 0.10 * length
        )
        reward = max(-1.0, min(1.0, reward))
        return TextExplanationVerdict(
            graph_precision=precision,
            graph_recall=recall,
            graph_f1=graph_f1,
            contradiction=contradiction,
            generic_penalty=0.0,
            repetition_penalty=repetition,
            length_penalty=length,
            reward=reward,
            judge_id="atomic-entailment-v3-aligned-contradiction",
        )


# Import compatibility for older experiment code.  The old lexical/NLI mode
# selector is intentionally rejected so a resumed run cannot silently switch
# back to the hand-written dictionary reward.
class ReferenceTextExplanationJudge(EntailmentExplanationJudge):
    def __init__(self, *, mode: str = "entailment", **kwargs):
        if mode != "entailment":
            raise ValueError("Only mode='entailment' is supported; lexical/NLI-v1 rewards were retired")
        super().__init__(**kwargs)
