"""Fast reference-grounded reward for forensic explanations.

The scorer follows the reference-side structure of SPICE rather than asking a
generative VLM to judge every rollout.  TASLE annotations already separate the
object/anomaly, onset and offset evidence, so they act as a small per-example
evidence graph.  The default lexical mode needs no additional model.  Optional
NLI mode uses a frozen sequence-classification model to accept paraphrases and
penalise contradictions; it is never trained inside GRPO.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"[.!?;]+|\b(?:while|whereas|however)\b", re.IGNORECASE)
_GENERIC = {
    "the video is fake", "the video appears fake", "there is a forgery",
    "there are inconsistencies", "the content is manipulated",
    "the video contains anomalies", "an anomaly is visible",
}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "with", "within", "video", "frames", "frame", "appears",
    "showing", "shows", "suggesting", "indicating", "visible",
}
_CANONICAL = {
    "flickers": "flicker", "flickering": "flicker", "flickered": "flicker",
    "disappears": "disappear", "disappeared": "disappear",
    "appearing": "appear", "appeared": "appear",
    "deformed": "deform", "deformation": "deform", "deformations": "deform",
    "distorted": "distort", "distortion": "distort", "distortions": "distort",
    "inconsistent": "inconsistency", "inconsistencies": "inconsistency",
    "unnatural": "anomaly", "anomalous": "anomaly", "abnormal": "anomaly",
    "abruptly": "abrupt", "suddenly": "abrupt", "sudden": "abrupt",
    "movements": "movement", "moving": "movement", "motion": "movement",
    "textures": "texture", "boundaries": "boundary", "edges": "edge",
    "objects": "object", "persons": "person", "people": "person",
}


def _normalise(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower()))


def _tokens(text: str) -> List[str]:
    tokens = []
    for token in _WORD_RE.findall(str(text).lower()):
        token = _CANONICAL.get(token, token)
        if token not in _STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def _token_f1(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2.0 * precision * recall / (precision + recall)


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


class ReferenceTextExplanationJudge:
    """SPICE-style evidence matching with optional frozen NLI soft matching."""

    def __init__(
        self,
        *,
        mode: str = "lexical",
        nli_model_path: Optional[str] = None,
        nli_device: str = "cpu",
        nli_batch_size: int = 32,
        max_words: int = 80,
        require_candidate_observable: bool = False,
    ):
        if mode not in {"lexical", "nli"}:
            raise ValueError("text explanation scorer mode must be lexical or nli")
        self.mode = mode
        self.max_words = max(8, int(max_words))
        self.require_candidate_observable = bool(require_candidate_observable)
        self.nli = (
            FrozenNLIScorer(nli_model_path or "", nli_device, nli_batch_size)
            if mode == "nli" else None
        )

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
                f"reference-text-{self.mode}-v1",
            )

        lexical = [[_token_f1(claim, fact.matching_text) for fact in facts] for claim in claims]
        match = [row[:] for row in lexical]
        contradiction = 0.0
        recall_scores = [max(lexical[row][col] for row in range(len(claims))) for col in range(len(facts))]
        if self.nli is not None:
            precision_pairs = [(fact.matching_text, claim) for claim in claims for fact in facts]
            recall_pairs = [(caption, fact.matching_text) for fact in facts]
            pairs = precision_pairs + recall_pairs
            probabilities = self.nli.probabilities(pairs)
            contradiction_matrix = [[0.0 for _ in facts] for _ in claims]
            cursor = 0
            for claim_index in range(len(claims)):
                for fact_index in range(len(facts)):
                    entailment, contra = probabilities[cursor]
                    cursor += 1
                    match[claim_index][fact_index] = max(match[claim_index][fact_index], entailment)
                    contradiction_matrix[claim_index][fact_index] = contra
            for fact_index in range(len(facts)):
                entailment, _ = probabilities[cursor]
                cursor += 1
                recall_scores[fact_index] = max(recall_scores[fact_index], entailment)
            aligned_contradictions = []
            for claim_index, row in enumerate(match):
                best_fact = max(range(len(row)), key=row.__getitem__)
                aligned_contradictions.append(contradiction_matrix[claim_index][best_fact])
            contradiction = max(aligned_contradictions, default=0.0)

        precision = sum(max(row) for row in match) / len(claims)
        total_weight = sum(fact.weight for fact in facts)
        recall = sum(
            fact.weight * recall_scores[fact_index]
            for fact_index, fact in enumerate(facts)
        ) / total_weight
        graph_f1 = self._weighted_f1(precision, recall)

        normalised = _normalise(caption)
        generic = 1.0 if normalised in _GENERIC or len(_tokens(caption)) < 4 else 0.0
        repetition = _repetition_penalty(caption)
        word_count = len(_WORD_RE.findall(caption))
        length = min(1.0, max(0, word_count - self.max_words) / self.max_words)
        reward = (
            0.50 * recall + 0.50 * precision
            - 0.50 * contradiction - 0.20 * generic
            - 0.10 * repetition - 0.10 * length
        )
        reward = max(-1.0, min(1.0, reward))
        return TextExplanationVerdict(
            graph_precision=precision,
            graph_recall=recall,
            graph_f1=graph_f1,
            contradiction=contradiction,
            generic_penalty=generic,
            repetition_penalty=repetition,
            length_penalty=length,
            reward=reward,
            judge_id=f"reference-text-{self.mode}-v1",
        )
