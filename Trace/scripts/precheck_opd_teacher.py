#!/usr/bin/env python3
"""Validate a frozen TRACE checkpoint as a paired-video OPD teacher.

This program deliberately performs no optimization.  For every *actual*
stage-1 training proposal it runs the same frozen SFT checkpoint twice:

1. candidate proposal only (the deployment/student view), and
2. vertically paired real-reference/candidate video (the teacher view).

The resulting JSON is both an auditable report and the immutable reliability
cache consumed by OPD.  A paired teacher is eligible only when it localizes a
positive proposal with sufficient IoU or correctly produces the canonical
no-forgery stream for a negative proposal.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace.constants import DEFAULT_MMODAL_TOKEN
from trace.conversation import conv_templates
from trace.mm_utils import get_model_name_from_path, make_vertical_reference_pair, process_video_ref_split, tokenizer_MMODAL_token_all
from trace.model.builder import load_pretrained_model
from trace.opd_grpo import (
    PAIR_REFERENCE_INSTRUCTION,
    TraceTokenSpec,
    VALID_EVENT,
    VALID_NO_EVENT,
    match_segments,
    parse_trace_tokens,
)


def _video_path(data_folder: str, value: str) -> str:
    path = value if os.path.isabs(value) else os.path.join(data_folder, value)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video does not exist: {path}")
    return path


def _safe_decode(tokenizer, ids: Sequence[int]) -> str:
    vocab = getattr(tokenizer, "vocab_size", None)
    ordinary = [int(token) for token in ids if int(token) >= 0 and (vocab is None or int(token) < vocab)]
    return tokenizer.decode(ordinary, skip_special_tokens=True) if ordinary else ""


def _conversation_template(version: str):
    if version in conv_templates:
        return conv_templates[version]
    if version == "v1_mistral":
        return conv_templates["mistral_instruct"]
    if version == "v1":
        return conv_templates["vicuna_v1"]
    raise ValueError(f"Unknown TRACE conversation version {version!r}")


def _prompt(tokenizer, instruction: str, version: str) -> torch.Tensor:
    conv = _conversation_template(version).copy()
    conv.append_message(conv.roles[0], DEFAULT_MMODAL_TOKEN["VIDEO"] + "\n" + instruction)
    conv.append_message(conv.roles[1], None)
    # TRACE begins its mixed output stream with the text-head <sync> token.
    return tokenizer_MMODAL_token_all(conv.get_prompt() + "<sync>", tokenizer, return_tensors="pt")


def _generate(model, tokenizer, prompt_ids: torch.Tensor, video: torch.Tensor, timestamps, max_new_tokens: int) -> List[int]:
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        generated = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            images_or_videos=[video.to(device=device, dtype=torch.float16)],
            modal_list=["video"],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            video_timestamps=[timestamps],
            heads=[1],
        )
    ids = [int(token) for token in generated[0].detach().cpu().tolist()]
    prefix = [int(token) for token in prompt_ids[0].detach().cpu().tolist()]
    # TRACE checkpoints differ in whether generate returns full sequences or
    # only continuations.  Normalize both forms before strict parsing.
    return ids[len(prefix):] if ids[: len(prefix)] == prefix else ids


def _targets(record: Mapping[str, Any]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for target in record.get("targets", []):
        segment = target.get("relative_segment")
        if isinstance(segment, (list, tuple)) and len(segment) == 2:
            start, end = float(segment[0]), float(segment[1])
            if start < end:
                result.append((start, end))
    return result


def _max_iou(parsed, targets: Sequence[Tuple[float, float]]) -> float:
    matches = match_segments(parsed.segments, targets) if parsed.status == VALID_EVENT else []
    return max((iou for _, _, iou in matches), default=0.0)


def _summary(records: Iterable[Mapping[str, Any]], key: str, iou_gate: float) -> Dict[str, float]:
    rows = list(records)
    positives = [row for row in rows if row["is_positive"]]
    negatives = [row for row in rows if not row["is_positive"]]

    def status(row):
        return row[key]["parsed"]["status"]

    def fraction(items, predicate):
        return sum(bool(predicate(item)) for item in items) / len(items) if items else 0.0

    return {
        "proposals": len(rows),
        "positive_proposals": len(positives),
        "negative_proposals": len(negatives),
        "positive_event_rate": fraction(positives, lambda row: status(row) == VALID_EVENT),
        "positive_false_rejection_rate": fraction(positives, lambda row: status(row) != VALID_EVENT),
        "positive_localized_rate": fraction(positives, lambda row: row[key]["matched_iou"] >= iou_gate),
        "positive_mean_iou": (
            sum(float(row[key]["matched_iou"]) for row in positives) / len(positives)
            if positives else 0.0
        ),
        "negative_no_event_rate": fraction(negatives, lambda row: status(row) == VALID_NO_EVENT),
        "negative_false_event_rate": fraction(negatives, lambda row: status(row) == VALID_EVENT),
        "format_failure_rate": fraction(rows, lambda row: status(row) not in {VALID_EVENT, VALID_NO_EVENT}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, help="Normalized replay from build_opd_grpo_replay.py")
    parser.add_argument("--data-folder", required=True, help="Root that contains candidate and reference video paths")
    parser.add_argument("--model-path", required=True, help="Frozen candidate-only SFT checkpoint; it is not updated")
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--output", required=True, help="Output teacher precheck/cache JSON")
    parser.add_argument("--prompt-file", default=str(ROOT / "trace" / "prompts" / "dvc.txt"))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--version", default="v1_mistral", help="Must match --version used for SFT/OPD training")
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--teacher-iou-gate", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=0, help="Positive value runs only this many records for a smoke test")
    parser.add_argument("--enforce-pair-benefit", action="store_true", help="Exit nonzero unless paired input improves positive localization without excessive negative degradation")
    parser.add_argument("--minimum-recovery-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-negative-noevent-drop", type=float, default=0.02)
    parser.add_argument("--minimum-reliable-positive-rate", type=float, default=0.05)
    args = parser.parse_args()

    if not args.model_path:
        parser.error("--model-path is empty; run 'source Trace/scripts/setup_opd_grpo_env.sh' or set SFT_CKPT")
    if not Path(args.model_path).is_dir():
        parser.error(f"--model-path is not a checkpoint directory: {args.model_path}")

    if not 0.0 <= args.teacher_iou_gate <= 1.0:
        raise ValueError("--teacher-iou-gate must be in [0, 1]")
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if not isinstance(replay, dict) or replay.get("paired_only") is not True:
        raise ValueError(
            "Teacher precheck requires a paired-only replay. Rebuild with "
            "build_opd_grpo_replay.py --paired-only --video-root ..."
        )
    source_records = replay.get("records", replay) if isinstance(replay, dict) else replay
    if not isinstance(source_records, list):
        raise ValueError("replay must be a list or an object containing records")
    if args.max_samples > 0:
        source_records = source_records[: args.max_samples]
    if not source_records:
        raise ValueError("Replay contains no proposal records")

    for record in source_records:
        reference = record.get("reference") or {}
        if not reference.get("reference_video") or not reference.get("reference_segment"):
            raise ValueError(f"Paired replay record {record.get('id')} has no resolved fake->real reference mapping")

    device = torch.device(f"cuda:{args.gpu_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("precheck_opd_teacher.py requires a CUDA GPU")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        vision_tower=args.vision_tower,
        device_map=None,
        device=str(device),
    )
    model = model.to(device=device, dtype=torch.float16).eval()
    spec = TraceTokenSpec(
        text_vocab_size=model.vocab_size,
        time_vocab=model.get_model().time_tokenizer.vocab,
        score_vocab_size=model.config.score_vocab_size,
    )
    base_instruction = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    candidate_prompt = _prompt(tokenizer, base_instruction, args.version)
    teacher_prompt = _prompt(tokenizer, PAIR_REFERENCE_INSTRUCTION + base_instruction, args.version)

    checked: List[Dict[str, Any]] = []
    for index, record in enumerate(source_records, start=1):
        proposal = record["proposal"]
        start, end = float(proposal[0]), float(proposal[1])
        candidate_file = _video_path(args.data_folder, record["candidate_video"])
        candidate_video, timestamps = process_video_ref_split(
            candidate_file, processor, model.config.image_aspect_ratio,
            bnd_frames=args.bnd_frames, seg_frames=args.seg_frames, bnd_ratio=args.bnd_ratio,
            start_time=start, end_time=end,
        )
        duration = end - start
        target_segments = _targets(record)

        candidate_ids = _generate(model, tokenizer, candidate_prompt, candidate_video, timestamps, args.max_new_tokens)
        reference = record["reference"]
        reference_file = _video_path(args.data_folder, reference["reference_video"])
        ref_start, ref_end = map(float, reference["reference_segment"])
        reference_video, _ = process_video_ref_split(
            reference_file, processor, model.config.image_aspect_ratio,
            bnd_frames=args.bnd_frames, seg_frames=args.seg_frames, bnd_ratio=args.bnd_ratio,
            start_time=ref_start, end_time=ref_end,
        )
        try:
            paired_video = make_vertical_reference_pair(reference_video, candidate_video)
        except ValueError as exc:
            raise RuntimeError(f"{record['id']}: cannot construct vertical pair: {exc}") from exc
        teacher_ids = _generate(model, tokenizer, teacher_prompt, paired_video, timestamps, args.max_new_tokens)
        teacher_input_mode = "paired_reference"
        candidate_parsed = parse_trace_tokens(candidate_ids, spec, lambda ids: _safe_decode(tokenizer, ids), window_duration=duration)
        teacher_parsed = parse_trace_tokens(teacher_ids, spec, lambda ids: _safe_decode(tokenizer, ids), window_duration=duration)
        candidate_iou = _max_iou(candidate_parsed, target_segments)
        teacher_iou = _max_iou(teacher_parsed, target_segments)
        positive = bool(target_segments)
        reliable = (
            teacher_parsed.status == VALID_EVENT and teacher_iou >= args.teacher_iou_gate
            if positive else teacher_parsed.status == VALID_NO_EVENT
        )
        checked.append({
            "id": record["id"],
            "proposal": [start, end],
            "replay_bucket": record.get("replay_bucket"),
            "is_positive": positive,
            "candidate_only": {"parsed": candidate_parsed.as_dict(), "matched_iou": candidate_iou},
            "paired_teacher": {"parsed": teacher_parsed.as_dict(), "matched_iou": teacher_iou},
            "teacher_input_mode": teacher_input_mode,
            "teacher_reliable": reliable,
        })
        print(f"[{index}/{len(source_records)}] {record['id']}: candidate={candidate_parsed.status}, pair={teacher_parsed.status}, reliable={reliable}", flush=True)

    candidate_metrics = _summary(checked, "candidate_only", args.teacher_iou_gate)
    teacher_metrics = _summary(checked, "paired_teacher", args.teacher_iou_gate)
    recovery_gain = teacher_metrics["positive_localized_rate"] - candidate_metrics["positive_localized_rate"]
    negative_drop = candidate_metrics["negative_no_event_rate"] - teacher_metrics["negative_no_event_rate"]
    reliable_positive_rate = (
        sum(row["teacher_reliable"] for row in checked if row["is_positive"])
        / max(1, sum(row["is_positive"] for row in checked))
    )
    pair_benefit_passed = (
        recovery_gain >= args.minimum_recovery_improvement
        and negative_drop <= args.maximum_negative_noevent_drop
        and reliable_positive_rate >= args.minimum_reliable_positive_rate
    )
    report = {
        "schema_version": 1,
        "purpose": "frozen paired-input teacher validation and OPD reliability cache",
        "replay_path": str(Path(args.replay).resolve()),
        "teacher_model_path": str(Path(args.model_path).resolve()),
        "config": {
            "teacher_iou_gate": args.teacher_iou_gate,
            "bnd_ratio": args.bnd_ratio,
            "bnd_frames": args.bnd_frames,
            "seg_frames": args.seg_frames,
            "max_new_tokens": args.max_new_tokens,
            "conversation_version": args.version,
            "pair_prompt": PAIR_REFERENCE_INSTRUCTION,
        },
        "candidate_only_metrics": candidate_metrics,
        "paired_teacher_metrics": teacher_metrics,
        "teacher_reliable_proposals": sum(row["teacher_reliable"] for row in checked),
        "teacher_reliable_positive_rate": reliable_positive_rate,
        "localized_rate_gain": recovery_gain,
        "negative_noevent_drop": negative_drop,
        "pair_benefit_passed": pair_benefit_passed,
        "records": checked,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_only": candidate_metrics, "paired_teacher": teacher_metrics, "cache": str(output)}, ensure_ascii=False, indent=2))

    if args.enforce_pair_benefit:
        if not pair_benefit_passed:
            raise SystemExit(
                "Paired frozen teacher did not pass the precheck: "
                f"localized-rate gain={recovery_gain:.4f} (required >= {args.minimum_recovery_improvement:.4f}), "
                f"reliable-positive rate={reliable_positive_rate:.4f} (required >= {args.minimum_reliable_positive_rate:.4f}), "
                f"negative no-event drop={negative_drop:.4f} (allowed <= {args.maximum_negative_noevent_drop:.4f}). "
                f"Report was still written to {output}; do not start OPD."
            )


if __name__ == "__main__":
    main()
