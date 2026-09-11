#!/usr/bin/env python3
"""Candidate-only Qwen3-VL-235B judge used by TRACE GRPO.

The trainer invokes this program as::

    python scripts/qwen3_vl_explanation_judge.py --endpoint http://HOST:8000/v1 \
        --input request.json --output verdict.json

The endpoint is assumed to implement the OpenAI-compatible chat-completions
protocol (for example a locally served Qwen3-VL-235B vLLM/SGLang endpoint).
Only frames decoded from ``candidate_video`` are sent.  The training-only
``reference_evidence`` card is included so the judge can score coverage of the
annotated explanation, but it must also verify every claim in candidate frames.
Paired-reference video and GT time segments are rejected.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FORBIDDEN_PAYLOAD_KEYS = {
    "evidence", "target_segments", "target_caption", "gt", "ground_truth",
    "reference", "reference_video", "reference_caption", "teacher_output",
}


def _bounded_score(value: Any, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qwen verdict field {field!r} is not numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Qwen verdict field {field!r} must lie in [0, 1], got {score!r}")
    return score


def validate_candidate_only_payload(payload: Mapping[str, Any]) -> None:
    forbidden = FORBIDDEN_PAYLOAD_KEYS.intersection(payload)
    if forbidden:
        raise ValueError(f"Judge payload contains forbidden privileged fields: {sorted(forbidden)}")
    if payload.get("protocol") != "candidate_only_v1":
        raise ValueError("Expected protocol='candidate_only_v1'")
    if not isinstance(payload.get("candidate_video"), str) or not payload["candidate_video"].strip():
        raise ValueError("candidate_video must be a non-empty path")
    proposal = payload.get("proposal")
    if not isinstance(proposal, list) or len(proposal) != 2:
        raise ValueError("proposal must be [absolute_start_seconds, absolute_end_seconds]")
    start, end = map(float, proposal)
    if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0):
        raise ValueError("proposal must be finite, non-negative and increasing")
    if not isinstance(payload.get("caption"), str):
        raise ValueError("caption must be text")
    reference_evidence = payload.get("reference_evidence")
    if not isinstance(reference_evidence, Mapping):
        raise ValueError("reference_evidence must be the annotation evidence card")
    for segment in payload.get("predicted_segments", []):
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError("every predicted segment must be [start, end] relative to proposal")
        segment_start, segment_end = map(float, segment)
        if not (math.isfinite(segment_start) and math.isfinite(segment_end) and 0 <= segment_start < segment_end <= end - start):
            raise ValueError("predicted segments must be increasing and lie inside the proposal window")


def _unique_sorted(values: Iterable[float]) -> List[float]:
    return sorted({round(float(value), 4) for value in values})


def requested_frame_times(payload: Mapping[str, Any], max_frames: int) -> List[float]:
    """Return absolute video seconds, emphasizing predicted boundaries.

    We show both proposal context and a small neighborhood around every predicted
    segment; this lets the judge assess an explanation and its claimed timing
    without ever being told a GT segment.
    """
    proposal_start, proposal_end = map(float, payload["proposal"])
    segments = [(proposal_start + float(a), proposal_start + float(b)) for a, b in payload.get("predicted_segments", [])]
    values = [proposal_start, proposal_end, 0.5 * (proposal_start + proposal_end)]
    for start, end in segments:
        width = min(0.5, max(0.1, 0.1 * (end - start)))
        values.extend((start - width, start, 0.5 * (start + end), end, end + width))
    values = [min(proposal_end, max(proposal_start, value)) for value in values]
    values = _unique_sorted(values)
    if len(values) <= max_frames:
        return values
    # Preserve start/end context and subsample the remaining chronological set.
    indexes = sorted({round(i * (len(values) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [values[index] for index in indexes]


def encode_video_frames(video_path: str, times_seconds: Sequence[float], jpeg_quality: int = 85) -> List[Tuple[float, str]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required by the Qwen judge to decode candidate video frames") from exc
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"candidate video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open candidate video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"candidate video has invalid metadata: fps={fps}, frames={frame_count}")
    frames: List[Tuple[float, str]] = []
    try:
        for seconds in times_seconds:
            frame_index = min(frame_count - 1, max(0, round(float(seconds) * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode candidate frame at {seconds:.3f}s")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                raise RuntimeError(f"cannot JPEG-encode candidate frame at {seconds:.3f}s")
            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
            frames.append((float(seconds), data_url))
    finally:
        capture.release()
    return frames


def _judge_prompt(payload: Mapping[str, Any], frame_times: Sequence[float]) -> str:
    proposal_start, proposal_end = map(float, payload["proposal"])
    segments = payload.get("predicted_segments", [])
    relative_segments = ", ".join(f"[{float(a):.2f}, {float(b):.2f}]s" for a, b in segments) or "none"
    absolute_segments = ", ".join(f"[{proposal_start + float(a):.2f}, {proposal_start + float(b):.2f}]s" for a, b in segments) or "none"
    reference = payload["reference_evidence"]
    return f"""You are a frozen, candidate-only video-evidence judge for temporal forgery explanations.
You receive sampled frames ONLY from one candidate video, plus a training-only
reference evidence card. The card defines what a correct explanation should
cover, but is NOT proof by itself: require both semantic coverage of the card
and visible support in candidate frames. Do not infer facts from stereotypes,
annotation conventions, or wording alone.

Candidate proposal: absolute video time [{proposal_start:.2f}, {proposal_end:.2f}] seconds.
Model-predicted event segments (relative to proposal): {relative_segments}.
The same segments in absolute video time: {absolute_segments}.
Model explanation: {payload['caption']!r}
Reference evidence card (unseen by the actor):
- anomalous object/region and visible abnormality: {reference.get('object_caption', '')!r}
- onset cue: {reference.get('start_caption', '')!r}
- offset cue: {reference.get('end_caption', '')!r}
Frame timestamps, in the same absolute video clock: {[round(x, 2) for x in frame_times]}.

Score each item continuously from 0 to 1, conservatively:
- object_supported: explanation covers the reference object/region AND it is visibly relevant in candidate frames.
- anomaly_supported: explanation covers the reference abnormality AND it is actually visible, not generic.
- boundary_consistent: explanation/predicted segment is consistent with reference onset/offset cues AND observed change.
- hallucination: the explanation asserts a concrete visual fact unsupported by these frames (1 = clearly hallucinated, 0 = no such assertion).

Return ONLY one JSON object with exactly object_supported, anomaly_supported, boundary_consistent, hallucination. No rationale, no Markdown."""


def build_messages(payload: Mapping[str, Any], frames: Sequence[Tuple[float, str]]) -> List[Dict[str, Any]]:
    prompt = _judge_prompt(payload, [time for time, _ in frames])
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for time, data_url in frames:
        content.append({"type": "text", "text": f"Candidate frame at {time:.2f} seconds:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return [{"role": "user", "content": content}]


def _chat_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


def request_qwen(endpoint: str, api_key: str, model: str, messages: List[Dict[str, Any]], timeout: int) -> Mapping[str, Any]:
    request_body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
        # Widely supported by OpenAI-compatible servers; response parsing below
        # remains strict if a server ignores this preference.
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(_chat_url(endpoint), data=request_body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"Qwen judge HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Qwen judge endpoint {endpoint}: {exc.reason}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, Mapping))
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        # Be tolerant of servers that surround JSON with a Markdown fence, but
        # never attempt to extract an arbitrary natural-language score.
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.IGNORECASE)
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Qwen judge returned invalid JSON completion: {result!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="candidate-only request JSON emitted by CommandExplanationJudge")
    parser.add_argument("--output", required=True, help="path for bounded verdict JSON")
    parser.add_argument("--endpoint", required=True, help="Qwen OpenAI-compatible base URL, e.g. http://host:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="QWEN_JUDGE_API_KEY")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    validate_candidate_only_payload(payload)
    times = requested_frame_times(payload, args.max_frames)
    frames = encode_video_frames(payload["candidate_video"], times)
    verdict = request_qwen(
        args.endpoint, os.environ.get(args.api_key_env, ""), args.model,
        build_messages(payload, frames), args.timeout,
    )
    result = {
        "object_supported": _bounded_score(verdict.get("object_supported"), "object_supported"),
        "anomaly_supported": _bounded_score(verdict.get("anomaly_supported"), "anomaly_supported"),
        "boundary_consistent": _bounded_score(verdict.get("boundary_consistent"), "boundary_consistent"),
        "hallucination": _bounded_score(verdict.get("hallucination"), "hallucination"),
        "judge_id": f"frozen-qwen3-vl-235b::{args.model}",
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Qwen3-VL explanation judge failed: {exc}", file=sys.stderr)
        raise
