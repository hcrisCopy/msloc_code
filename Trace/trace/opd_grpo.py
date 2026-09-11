"""Shared, auditable mechanics for proposal-level OPD and GRPO.

This module deliberately contains no model-specific generation code.  TRACE uses
three decoder heads and non-text token ids, so treating its output as ordinary
text silently turns malformed timestamps into ``real`` predictions.  The
functions below are the single source of truth used by the replay builder,
trainers and evaluation code.

The module has no optional VLM dependency.  Explanation reward is only enabled
when a caller supplies a *candidate-only* judge implementing ``ExplanationJudge``.
This is intentional: text overlap with ``obj_caption`` is not evidence that an
explanation is visible in the candidate video.
"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import math
import re
import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VALID_EVENT = "valid_event"
VALID_NO_EVENT = "valid_no_event"
FORMAT_FAILURE = "format_failure"

# This text is part of the paired-teacher input contract.  The student never
# receives it because the student sees only the candidate proposal.
PAIR_REFERENCE_INSTRUCTION = (
    "The video is a vertically paired comparison: the upper half is an aligned "
    "authentic reference and the lower half is the candidate to inspect. "
    "Judge and localize forgery only in the lower candidate half. "
)


@dataclass(frozen=True)
class TraceTokenSpec:
    """Ranges of TRACE's joint generation vocabulary."""

    text_vocab_size: int
    time_vocab: Mapping[str, int]
    score_vocab_size: int = 0

    @property
    def text_sync_id(self) -> int:
        return self.text_vocab_size

    @property
    def time_start_id(self) -> int:
        return self.text_vocab_size + 1

    @property
    def time_end_id(self) -> int:
        return self.time_start_id + len(self.time_vocab) - 1

    @property
    def score_start_id(self) -> int:
        return self.time_end_id + 1

    @property
    def time_sync_id(self) -> int:
        return self.time_start_id + self.time_vocab["<sync>"]

    @property
    def time_sep_id(self) -> int:
        return self.time_start_id + self.time_vocab["<sep>"]

    def kind(self, token_id: int) -> str:
        if 0 <= token_id <= self.text_sync_id:
            return "text"
        if self.time_start_id <= token_id <= self.time_end_id:
            return "time"
        if self.score_vocab_size and self.score_start_id <= token_id < self.score_start_id + self.score_vocab_size:
            return "score"
        return "unknown"


@dataclass
class ParsedTraceOutput:
    """Parser result kept even when it is invalid; never coerce it to real."""

    status: str
    segments: List[Tuple[float, float]] = field(default_factory=list)
    caption: str = ""
    failure_reasons: List[str] = field(default_factory=list)
    raw_token_ids: List[int] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status in {VALID_EVENT, VALID_NO_EVENT}

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _normalise_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_trace_tokens(
    token_ids: Sequence[int],
    token_spec: TraceTokenSpec,
    text_decoder,
    *,
    window_duration: float,
) -> ParsedTraceOutput:
    """Parse TRACE output ids without conflating invalid output with no-event.

    ``text_decoder`` must decode only ordinary text token ids.  The decoder is
    intentionally injected so the parser can be unit-tested without a HF model.
    """

    ids = [int(x) for x in token_ids]
    inverse_time_vocab = {v: k for k, v in token_spec.time_vocab.items()}
    text_buffer: List[int] = []
    number_buffer: List[str] = []
    current_times: List[float] = []
    finished_time_streams: List[List[float]] = []
    reasons: List[str] = []
    saw_time = False

    def flush_number() -> None:
        nonlocal number_buffer
        if not number_buffer:
            return
        value = "".join(number_buffer)
        number_buffer = []
        try:
            parsed = float(value)
        except ValueError:
            reasons.append("non_numeric_timestamp")
            return
        if not math.isfinite(parsed):
            reasons.append("non_finite_timestamp")
            return
        current_times.append(parsed)

    for token_id in ids:
        kind = token_spec.kind(token_id)
        if kind == "text":
            if token_id < token_spec.text_vocab_size:
                text_buffer.append(token_id)
            # A text <sync> delimits events but has no timestamp semantics.
            continue
        if kind == "score":
            # Scores are empty in the current TASLE/Trace supervision.  They are
            # structural only and must not be mistaken for caption tokens.
            continue
        if kind != "time":
            reasons.append("unknown_output_token")
            continue

        saw_time = True
        time_piece = inverse_time_vocab[token_id - token_spec.time_start_id]
        if time_piece == "<sep>":
            flush_number()
        elif time_piece == "<sync>":
            flush_number()
            finished_time_streams.append(list(current_times))
            current_times.clear()
        else:
            number_buffer.append(time_piece)

    if number_buffer or current_times:
        reasons.append("unterminated_timestamp_stream")
    if not saw_time:
        reasons.append("missing_timestamp_stream")

    caption = _normalise_caption(text_decoder(text_buffer)) if text_buffer else ""
    no_event_streams = [stream for stream in finished_time_streams if not stream]
    non_empty_streams = [stream for stream in finished_time_streams if stream]

    if reasons:
        return ParsedTraceOutput(FORMAT_FAILURE, caption=caption, failure_reasons=sorted(set(reasons)), raw_token_ids=ids)

    # A response may have at most one empty time stream, and it must explicitly
    # say No forgery.  Otherwise an empty stream is malformed, never a free real.
    if no_event_streams:
        if non_empty_streams:
            reasons.append("mixed_no_event_and_event_streams")
        if len(no_event_streams) != 1:
            reasons.append("multiple_no_event_streams")
        if caption.lower() != "no forgery.":
            reasons.append("no_event_without_canonical_caption")
        if reasons:
            return ParsedTraceOutput(FORMAT_FAILURE, caption=caption, failure_reasons=sorted(set(reasons)), raw_token_ids=ids)
        return ParsedTraceOutput(VALID_NO_EVENT, caption=caption, raw_token_ids=ids)

    if not non_empty_streams:
        return ParsedTraceOutput(FORMAT_FAILURE, caption=caption, failure_reasons=["missing_event_or_no_event"], raw_token_ids=ids)

    segments: List[Tuple[float, float]] = []
    for stream in non_empty_streams:
        if len(stream) != 2:
            reasons.append("timestamp_stream_requires_exactly_two_values")
            continue
        start, end = stream
        if start < 0 or end < 0 or start >= end:
            reasons.append("invalid_timestamp_order")
            continue
        if start > window_duration or end > window_duration:
            reasons.append("timestamp_outside_proposal_window")
            continue
        segments.append((start, end))
    if not caption:
        reasons.append("event_without_explanation")
    if reasons:
        return ParsedTraceOutput(FORMAT_FAILURE, caption=caption, failure_reasons=sorted(set(reasons)), raw_token_ids=ids)
    return ParsedTraceOutput(VALID_EVENT, segments=segments, caption=caption, raw_token_ids=ids)


def temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    left, right = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_segments(
    predicted: Sequence[Tuple[float, float]], target: Sequence[Tuple[float, float]]
) -> List[Tuple[int, int, float]]:
    """Maximum-IoU one-to-one matching for the small number of TRACE events."""

    candidates = [
        (temporal_iou(p, g), p_idx, g_idx)
        for p_idx, p in enumerate(predicted)
        for g_idx, g in enumerate(target)
    ]
    candidates.sort(reverse=True)
    used_pred, used_gt, matches = set(), set(), []
    for iou, p_idx, g_idx in candidates:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        used_pred.add(p_idx)
        used_gt.add(g_idx)
        matches.append((p_idx, g_idx, iou))
    return matches


def boundary_score(predicted: Tuple[float, float], target: Tuple[float, float], tolerance: float) -> float:
    if tolerance <= 0:
        raise ValueError("boundary tolerance must be positive")
    start = max(0.0, 1.0 - abs(predicted[0] - target[0]) / tolerance)
    end = max(0.0, 1.0 - abs(predicted[1] - target[1]) / tolerance)
    return 0.5 * (start + end)


@dataclass(frozen=True)
class EvidenceCard:
    """Candidate-observable explanation target derived from one GT annotation."""

    object_caption: str
    start_caption: str = ""
    end_caption: str = ""
    manipulation_type: str = "temporal"
    object_class: str = ""
    start_class: str = ""
    end_class: str = ""
    candidate_observable: bool = False

    @classmethod
    def from_annotation(cls, annotation: Mapping[str, Any]) -> "EvidenceCard":
        def first(container: Any) -> Mapping[str, Any]:
            if isinstance(container, Mapping):
                return container
            if isinstance(container, list) and container and isinstance(container[0], Mapping):
                return container[0]
            return {}

        obj, start, end = first(annotation.get("obj_cot")), first(annotation.get("bnd_cot_st")), first(annotation.get("bnd_cot_ed"))
        is_round4 = "Round4" in str(annotation.get("combine_dir", ""))
        return cls(
            object_caption=str(obj.get("obj_caption", "")).strip(),
            start_caption="" if is_round4 else str(start.get("bnd_caption", "")).strip(),
            end_caption="" if is_round4 else str(end.get("bnd_caption", "")).strip(),
            manipulation_type="spatio-temporal" if is_round4 else "temporal",
            object_class=str(obj.get("bnd_sub_class", "")).strip(),
            start_class="" if is_round4 else str(start.get("bnd_class", "")).strip(),
            end_class="" if is_round4 else str(end.get("bnd_class", "")).strip(),
            candidate_observable=bool(annotation.get("candidate_observable", False)),
        )

    def as_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ExplanationVerdict:
    """Bounded output of a frozen candidate-only explanation judge."""

    object_supported: float
    anomaly_supported: float
    boundary_consistent: float
    hallucination: float
    judge_id: str

    def __post_init__(self) -> None:
        for name in ("object_supported", "anomaly_supported", "boundary_consistent", "hallucination"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")

    @property
    def reward(self) -> float:
        return max(-1.0, min(1.0, 0.4 * self.object_supported + 0.4 * self.anomaly_supported + 0.2 * self.boundary_consistent - 0.5 * self.hallucination))


class ExplanationJudge(abc.ABC):
    """Frozen VLM contract.  It must never receive a paired reference video.

    ``evidence`` is the training-only reference explanation. It is needed to
    score coverage of annotated content, but the judge must also inspect the
    candidate video: reference-text overlap alone is not visual grounding.
    The reference video and GT time segments are never sent to the judge.
    """

    @abc.abstractmethod
    def score(
        self,
        *,
        candidate_video: str,
        proposal: Tuple[float, float],
        predicted_segments: Sequence[Tuple[float, float]],
        caption: str,
        evidence: EvidenceCard,
        sample_id: str,
    ) -> ExplanationVerdict:
        raise NotImplementedError


class CommandExplanationJudge(ExplanationJudge):
    """Adapter for a frozen local/remote VLM judge executed as a command.

    The command is called without a shell as ``COMMAND --input in.json --output
    out.json``. ``in.json`` contains candidate video, proposal, model output,
    and a training-only reference evidence card extracted from annotations. The
    judge must score both coverage of this card and visual support in the
    candidate. It receives neither a reference video nor GT time segments.
    The output must be one JSON object with the four bounded fields accepted by
    :class:`ExplanationVerdict` and an optional ``judge_id``.
    """

    def __init__(self, command: str, timeout_seconds: int = 120):
        if not command.strip():
            raise ValueError("judge command must not be empty")
        self.command = shlex.split(command)
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, ExplanationVerdict] = {}

    def score(self, *, candidate_video, proposal, predicted_segments, caption, evidence, sample_id):
        payload = {
            "sample_id": sample_id,
            "candidate_video": candidate_video,
            "proposal": list(proposal),
            "predicted_segments": [list(x) for x in predicted_segments],
            "caption": caption,
            # The actor never sees this card. It supplies a reference answer to
            # GRPO, while Qwen is instructed to require visual support too.
            "reference_evidence": evidence.as_dict(),
            "protocol": "candidate_only_v1",
            "rubric": {
                "object_supported": "Does the explanation identify an entity or region visibly relevant in the candidate video?",
                "anomaly_supported": "Is the claimed anomaly visibly supported by the candidate video, rather than a generic assertion?",
                "boundary_consistent": "Do the claimed visual changes occur within and near the model-predicted time segment(s)?",
                "hallucination": "Does the explanation assert a visual fact that cannot be supported by the candidate video?",
            },
        }
        cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if cache_key in self._cache:
            return self._cache[cache_key]
        with tempfile.TemporaryDirectory(prefix="trace_evidence_judge_") as directory:
            input_path = f"{directory}/input.json"
            output_path = f"{directory}/output.json"
            with open(input_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            completed = subprocess.run(
                [*self.command, "--input", input_path, "--output", output_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Explanation judge failed with exit code {completed.returncode}: {completed.stderr[-1000:]}"
                )
            try:
                with open(output_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Explanation judge did not create valid JSON output") from exc
        verdict = ExplanationVerdict(
            object_supported=float(result["object_supported"]),
            anomaly_supported=float(result["anomaly_supported"]),
            boundary_consistent=float(result["boundary_consistent"]),
            hallucination=float(result["hallucination"]),
            judge_id=str(result.get("judge_id", "external_candidate_only_judge")),
        )
        self._cache[cache_key] = verdict
        return verdict


@dataclass(frozen=True)
class RewardConfig:
    localization_weight: float = 1.0
    explanation_weight: float = 0.0
    format_weight: float = 0.1
    explanation_iou_gate: float = 0.30
    boundary_tolerance: float = 1.0
    extra_segment_penalty: float = 0.25


@dataclass
class RewardBreakdown:
    localization: float
    explanation: float
    format: float
    total: float
    matched_iou: float = 0.0
    matched_boundary: float = 0.0
    explanation_enabled: bool = False
    judge_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def score_trace_output(
    parsed: ParsedTraceOutput,
    *,
    target_segments: Sequence[Tuple[float, float]],
    candidate_video: str,
    proposal: Tuple[float, float],
    evidence: Optional[EvidenceCard],
    sample_id: str,
    config: RewardConfig,
    explanation_judge: Optional[ExplanationJudge] = None,
) -> RewardBreakdown:
    """The three rewards used by the proposal-level GRPO stage.

    ``target_segments`` are already in proposal-relative time.  A positive
    proposal has one or more target segments; a negative proposal has none.
    """

    positive = bool(target_segments)
    valid = parsed.status in {VALID_EVENT, VALID_NO_EVENT}
    format_reward = 1.0 if valid else -1.0
    matched_iou = 0.0
    matched_boundary = 0.0

    if not positive:
        if parsed.status == VALID_NO_EVENT:
            localization = 1.0
        elif parsed.status == VALID_EVENT:
            localization = -1.0
        else:
            localization = -0.5
    elif parsed.status != VALID_EVENT:
        localization = -1.0
    else:
        matches = match_segments(parsed.segments, target_segments)
        if matches:
            matched_iou = sum(m[2] for m in matches) / len(matches)
            matched_boundary = sum(
                boundary_score(parsed.segments[p_idx], target_segments[g_idx], config.boundary_tolerance)
                for p_idx, g_idx, _ in matches
            ) / len(matches)
        unmatched = len(parsed.segments) - len(matches)
        # Detection, IoU and boundary terms are all bounded; an invalid or
        # no-event output was handled above and never benefits from formatting.
        localization = 0.25 + 0.55 * matched_iou + 0.20 * matched_boundary - config.extra_segment_penalty * unmatched
        localization = max(-1.0, min(1.0, localization))

    explanation = 0.0
    judge_id: Optional[str] = None
    explanation_enabled = (
        positive
        and parsed.status == VALID_EVENT
        and matched_iou >= config.explanation_iou_gate
        and config.explanation_weight > 0
        and evidence is not None
    )
    if explanation_enabled:
        if explanation_judge is None:
            raise RuntimeError(
                "Explanation reward was enabled without a candidate-only frozen judge and evidence card. "
                "Do not substitute text similarity: set explanation_weight=0 until a judge is supplied."
            )
        verdict = explanation_judge.score(
            candidate_video=candidate_video,
            proposal=proposal,
            predicted_segments=parsed.segments,
            caption=parsed.caption,
            evidence=evidence,
            sample_id=sample_id,
        )
        explanation = verdict.reward
        judge_id = verdict.judge_id

    total = (
        config.localization_weight * localization
        + config.explanation_weight * explanation
        + config.format_weight * format_reward
    )
    return RewardBreakdown(
        localization=localization,
        explanation=explanation,
        format=format_reward,
        total=total,
        matched_iou=matched_iou,
        matched_boundary=matched_boundary,
        explanation_enabled=explanation_enabled,
        judge_id=judge_id,
    )


def group_advantages(values: Sequence[float], epsilon: float = 1e-6) -> Optional[List[float]]:
    """Return GRPO advantages or ``None`` for a degenerate group.

    Skipping a zero-variance group is more honest than inventing a learning
    signal through numerical epsilon.
    """

    if len(values) < 2:
        raise ValueError("GRPO needs at least two rollouts per proposal")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    if std < epsilon:
        return None
    return [(x - mean) / (std + epsilon) for x in values]


def component_advantages(rewards: Sequence[RewardBreakdown], epsilon: float = 1e-6) -> Dict[str, Optional[List[float]]]:
    return {
        "localization": group_advantages([r.localization for r in rewards], epsilon),
        "explanation": group_advantages([r.explanation for r in rewards], epsilon),
        "format": group_advantages([r.format for r in rewards], epsilon),
        "total": group_advantages([r.total for r in rewards], epsilon),
    }


def action_component_masks(token_ids: Sequence[int], token_spec: TraceTokenSpec) -> Dict[str, List[float]]:
    """Masks for the explicitly named structure-aware GRPO ablation.

    This is not vanilla GRPO.  Location is applied to time and event-structure
    tokens, format to all structural tokens, and explanation to ordinary text.
    Scores are deliberately zero because current TASLE supervision leaves them
    empty.
    """

    masks = {"localization": [], "format": [], "explanation": [], "total": []}
    for token_id in token_ids:
        kind = token_spec.kind(int(token_id))
        structural = kind in {"time", "score"} or int(token_id) == token_spec.text_sync_id
        masks["localization"].append(1.0 if kind == "time" or int(token_id) == token_spec.text_sync_id else 0.0)
        masks["format"].append(1.0 if structural else 0.0)
        masks["explanation"].append(1.0 if kind == "text" and int(token_id) != token_spec.text_sync_id else 0.0)
        masks["total"].append(1.0)
    return masks


def stable_sample_id(video: str, proposal: Tuple[float, float], index: int = 0) -> str:
    source = f"{video}|{proposal[0]:.4f}|{proposal[1]:.4f}|{index}".encode("utf-8")
    return hashlib.sha1(source).hexdigest()[:16]
