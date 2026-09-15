#!/usr/bin/env python3
"""Compare frozen Student/Teacher SFT checkpoints and build the OPD gate.

This program deliberately performs no optimization. For every actual stage-1
proposal it compares the separately SFT-trained student (candidate only) with
the separately SFT-trained teacher (aligned real reference above, candidate
below). Only examples on which the teacher is strictly better are exported for
OPD.

The resulting JSON is both an auditable report and the immutable reliability
cache consumed by OPD. A record is exported only when the paired teacher fixes
the student's fake/no-forgery decision, or, when both detect a positive event,
has strictly higher localization IoU. There is no minimum IoU or gain margin.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

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

    positive_binary_correct = sum(status(row) == VALID_EVENT for row in positives)
    positive_localized_correct = sum(row[key]["matched_iou"] >= iou_gate for row in positives)
    negative_correct = sum(status(row) == VALID_NO_EVENT for row in negatives)
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
        "binary_accuracy": (positive_binary_correct + negative_correct) / len(rows) if rows else 0.0,
        "joint_binary_localization_accuracy": (
            (positive_localized_correct + negative_correct) / len(rows) if rows else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, help="Normalized replay from build_opd_grpo_replay.py")
    parser.add_argument("--data-folder", required=True, help="Root that contains candidate and reference video paths")
    parser.add_argument("--student-model-path", required=True, help="Frozen candidate-only student SFT checkpoint")
    parser.add_argument("--model-path", required=True, help="Frozen paired-input teacher SFT checkpoint")
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--output", required=True, help="Output teacher precheck/cache JSON")
    parser.add_argument("--selected-output", required=True, help="Filtered replay containing only teacher-better records")
    parser.add_argument("--prompt-file", default=str(ROOT / "trace" / "prompts" / "dvc.txt"))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--version", default="v1_mistral", help="Must match --version used for SFT/OPD training")
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--teacher-iou-gate", type=float, default=0.3, help="Reporting threshold only; it does not filter teacher-better samples")
    parser.add_argument("--max-samples", type=int, default=0, help="Positive value runs only this many records for a smoke test")
    parser.add_argument("--enforce-pair-benefit", action="store_true", help="Exit nonzero when no proposal has a strictly better teacher refinement")
    parser.add_argument("--resume", action="store_true", help="Resume from per-rank proposal progress saved beside --output")
    parser.add_argument("--clean", action="store_true", help="Remove an old report and its progress before starting")
    args = parser.parse_args()

    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.gpu_id))
    if not torch.cuda.is_available():
        raise RuntimeError("precheck_opd_teacher.py requires a CUDA GPU")
    # Bind each torchrun worker before NCCL creates its process group.  If this
    # happens after init_process_group, every worker initially uses cuda:0 and
    # NCCL rejects the job as a duplicate-GPU launch.
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    output = Path(args.output)
    progress_dir = Path(str(output) + ".progress")
    progress_manifest = progress_dir / "run.json"
    run_identity = {
        "replay_path": str(Path(args.replay).resolve()),
        "student_model_path": str(Path(args.student_model_path).resolve()),
        "teacher_model_path": str(Path(args.model_path).resolve()),
        "vision_tower": str(Path(args.vision_tower).resolve()),
        "version": args.version,
        "bnd_ratio": args.bnd_ratio,
        "bnd_frames": args.bnd_frames,
        "seg_frames": args.seg_frames,
        "max_new_tokens": args.max_new_tokens,
        "teacher_iou_gate": args.teacher_iou_gate,
        "max_samples": args.max_samples,
    }
    setup_error = None
    if rank == 0:
        try:
            if args.clean:
                if output.is_file():
                    output.unlink()
                selected_output = Path(args.selected_output)
                if selected_output.is_file():
                    selected_output.unlink()
                if progress_dir.is_dir():
                    shutil.rmtree(progress_dir)
            if output.exists() and not args.resume:
                raise FileExistsError(f"{output} already exists; pass --clean to replace it or --resume to reuse it")
            if Path(args.selected_output).exists() and not args.resume:
                raise FileExistsError(
                    f"{args.selected_output} already exists; pass --clean to replace it or --resume to reuse it"
                )
            if args.resume:
                if not progress_manifest.is_file():
                    raise FileNotFoundError("Precheck resume metadata is missing; use --clean to start a new run")
                previous_identity = json.loads(progress_manifest.read_text(encoding="utf-8"))
                if previous_identity != run_identity:
                    raise ValueError("Precheck progress belongs to different inputs or parameters; use --clean")
            if args.resume and output.is_file():
                if not Path(args.selected_output).is_file():
                    raise FileNotFoundError(
                        "Precheck report exists but selected replay is missing; use --clean to rebuild both"
                    )
                print(f"Precheck is already complete: {output}")
            else:
                if progress_dir.exists() and not args.resume:
                    raise FileExistsError(f"{progress_dir} already exists; pass --clean or --resume")
                progress_dir.mkdir(parents=True, exist_ok=True)
                if not progress_manifest.is_file():
                    progress_manifest.write_text(json.dumps(run_identity, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
    if distributed:
        setup_status = [setup_error]
        dist.broadcast_object_list(setup_status, src=0)
        setup_error = setup_status[0]
    if setup_error:
        if distributed:
            dist.destroy_process_group()
        raise RuntimeError(setup_error)
    if args.resume and output.is_file():
        if distributed:
            dist.destroy_process_group()
        return

    for role, checkpoint in (("student", args.student_model_path), ("teacher", args.model_path)):
        if not Path(checkpoint).is_dir():
            parser.error(f"--{role}-model-path is not a checkpoint directory: {checkpoint}")

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

    source_order = [record["id"] for record in source_records]
    completed: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        for shard_path in progress_dir.glob("rank-*.jsonl"):
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("id") in source_order:
                        completed[row["id"]] = row
        if rank == 0 and completed:
            print(f"[resume] Reusing {len(completed)}/{len(source_records)} checked proposals")
    remaining_records = [record for record in source_records if record["id"] not in completed]
    source_records = remaining_records[rank::world_size]
    device = torch.device(f"cuda:{local_rank}")
    student_name = get_model_name_from_path(args.student_model_path)
    tokenizer, student_model, processor, _ = load_pretrained_model(
        args.student_model_path,
        None,
        student_name,
        vision_tower=args.vision_tower,
        device_map=None,
        device=str(device),
    )
    student_model = student_model.to(device=device, dtype=torch.float16).eval()
    teacher_name = get_model_name_from_path(args.model_path)
    teacher_tokenizer, teacher_model, teacher_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        teacher_name,
        vision_tower=args.vision_tower,
        device_map=None,
        device=str(device),
    )
    teacher_model = teacher_model.to(device=device, dtype=torch.float16).eval()
    if teacher_tokenizer.vocab_size != tokenizer.vocab_size:
        raise ValueError("Student and teacher tokenizers have different vocabularies")
    spec = TraceTokenSpec(
        text_vocab_size=teacher_model.vocab_size,
        time_vocab=teacher_model.get_model().time_tokenizer.vocab,
        score_vocab_size=teacher_model.config.score_vocab_size,
    )
    base_instruction = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    candidate_prompt = _prompt(tokenizer, base_instruction, args.version)
    teacher_prompt = _prompt(tokenizer, PAIR_REFERENCE_INSTRUCTION + base_instruction, args.version)

    checked: List[Dict[str, Any]] = []
    iterator = tqdm(
        source_records, desc=f"Teacher precheck rank {rank}", unit="proposal",
        position=rank, leave=rank == 0, dynamic_ncols=True,
    )
    shard_progress = progress_dir / f"rank-{rank}.jsonl"
    progress_handle = shard_progress.open("a", encoding="utf-8", buffering=1)
    for index, record in enumerate(iterator, start=1):
        proposal = record["proposal"]
        start, end = float(proposal[0]), float(proposal[1])
        candidate_file = _video_path(args.data_folder, record["candidate_video"])
        candidate_video, timestamps = process_video_ref_split(
            candidate_file, processor, student_model.config.image_aspect_ratio,
            bnd_frames=args.bnd_frames, seg_frames=args.seg_frames, bnd_ratio=args.bnd_ratio,
            start_time=start, end_time=end,
        )
        duration = end - start
        target_segments = _targets(record)

        student_ids = _generate(student_model, tokenizer, candidate_prompt, candidate_video, timestamps, args.max_new_tokens)
        reference = record["reference"]
        reference_file = _video_path(args.data_folder, reference["reference_video"])
        ref_start, ref_end = map(float, reference["reference_segment"])
        reference_video, _ = process_video_ref_split(
            reference_file, teacher_processor, teacher_model.config.image_aspect_ratio,
            bnd_frames=args.bnd_frames, seg_frames=args.seg_frames, bnd_ratio=args.bnd_ratio,
            start_time=ref_start, end_time=ref_end,
        )
        try:
            paired_video = make_vertical_reference_pair(reference_video, candidate_video)
        except ValueError as exc:
            raise RuntimeError(f"{record['id']}: cannot construct vertical pair: {exc}") from exc
        teacher_ids = _generate(teacher_model, teacher_tokenizer, teacher_prompt, paired_video, timestamps, args.max_new_tokens)
        teacher_input_mode = "paired_reference"
        student_parsed = parse_trace_tokens(student_ids, spec, lambda ids: _safe_decode(tokenizer, ids), window_duration=duration)
        teacher_parsed = parse_trace_tokens(teacher_ids, spec, lambda ids: _safe_decode(teacher_tokenizer, ids), window_duration=duration)
        student_iou = _max_iou(student_parsed, target_segments)
        teacher_iou = _max_iou(teacher_parsed, target_segments)
        positive = bool(target_segments)
        if positive:
            student_detected_fake = student_parsed.status == VALID_EVENT
            teacher_detected_fake = teacher_parsed.status == VALID_EVENT
            teacher_better = teacher_detected_fake and (
                not student_detected_fake or teacher_iou > student_iou
            )
            selection_reason = "positive_binary_correction" if teacher_detected_fake and not student_detected_fake else "positive_iou_gain"
        else:
            student_correct = student_parsed.status == VALID_NO_EVENT
            teacher_better = teacher_parsed.status == VALID_NO_EVENT and not student_correct
            selection_reason = "negative_false_event_correction"
        reliable = teacher_better
        checked_row = {
            "id": record["id"],
            "proposal": [start, end],
            "replay_bucket": record.get("replay_bucket"),
            "is_positive": positive,
            "student_candidate": {"parsed": student_parsed.as_dict(), "matched_iou": student_iou},
            "paired_teacher": {"parsed": teacher_parsed.as_dict(), "matched_iou": teacher_iou},
            "teacher_input_mode": teacher_input_mode,
            "teacher_reliable": reliable,
            "teacher_better": teacher_better,
            "selection_reason": selection_reason if teacher_better else "not_strictly_better",
        }
        checked.append(checked_row)
        progress_handle.write(json.dumps(checked_row, ensure_ascii=False) + "\n")
        iterator.set_postfix(student=student_parsed.status, teacher=teacher_parsed.status, selected=teacher_better)
    progress_handle.close()

    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, checked)
        checked = [row for shard in gathered for row in shard]
        if rank != 0:
            dist.destroy_process_group()
            return
    checked = list(completed.values()) + checked
    checked_by_id = {row["id"]: row for row in checked}
    checked = [checked_by_id[record_id] for record_id in source_order if record_id in checked_by_id]
    if len(checked) != len(source_order):
        raise RuntimeError(f"Precheck completed {len(checked)}/{len(source_order)} proposals; report not written")

    candidate_metrics = _summary(checked, "student_candidate", args.teacher_iou_gate)
    teacher_metrics = _summary(checked, "paired_teacher", args.teacher_iou_gate)
    recovery_gain = teacher_metrics["positive_localized_rate"] - candidate_metrics["positive_localized_rate"]
    negative_drop = candidate_metrics["negative_no_event_rate"] - teacher_metrics["negative_no_event_rate"]
    reliable_positive_rate = (
        sum(row["teacher_reliable"] for row in checked if row["is_positive"])
        / max(1, sum(row["is_positive"] for row in checked))
    )
    binary_accuracy_gain = (
        teacher_metrics["binary_accuracy"] - candidate_metrics["binary_accuracy"]
    )
    joint_accuracy_gain = (
        teacher_metrics["joint_binary_localization_accuracy"]
        - candidate_metrics["joint_binary_localization_accuracy"]
    )
    selected_ids = {row["id"] for row in checked if row["teacher_better"]}
    selected_records = [record for record in replay["records"] if record["id"] in selected_ids]
    pair_benefit_passed = bool(selected_records)
    report = {
        "schema_version": 2,
        "purpose": "frozen paired-input teacher validation and OPD reliability cache",
        "replay_path": args.replay,
        "student_model_path": args.student_model_path,
        "teacher_model_path": args.model_path,
        "selected_replay_path": args.selected_output,
        "config": {
            "teacher_iou_gate": args.teacher_iou_gate,
            "selection_rule": "binary_correction_or_strict_positive_iou_improvement",
            "bnd_ratio": args.bnd_ratio,
            "bnd_frames": args.bnd_frames,
            "seg_frames": args.seg_frames,
            "max_new_tokens": args.max_new_tokens,
            "conversation_version": args.version,
            "pair_prompt": PAIR_REFERENCE_INSTRUCTION,
        },
        "student_candidate_metrics": candidate_metrics,
        "candidate_only_metrics": candidate_metrics,
        "paired_teacher_metrics": teacher_metrics,
        "teacher_reliable_proposals": sum(row["teacher_reliable"] for row in checked),
        "teacher_reliable_positive_rate": reliable_positive_rate,
        "localized_rate_gain": recovery_gain,
        "binary_accuracy_gain": binary_accuracy_gain,
        "joint_binary_localization_accuracy_gain": joint_accuracy_gain,
        "negative_noevent_drop": negative_drop,
        "teacher_better_proposals": len(selected_records),
        "pair_benefit_passed": pair_benefit_passed,
        "records": checked,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_output.replace(output)
    selected_payload = {
        "schema_version": replay.get("schema_version", 2),
        "paired_only": True,
        "selection": "strict_teacher_better_v2",
        "source_replay": args.replay,
        "teacher_check_result": args.output,
        "records": selected_records,
        "statistics": {
            "records": len(selected_records),
            "positive": sum(bool(row.get("is_positive")) for row in selected_records),
            "negative": sum(not bool(row.get("is_positive")) for row in selected_records),
        },
    }
    selected_path = Path(args.selected_output)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_tmp = selected_path.with_suffix(selected_path.suffix + ".tmp")
    selected_tmp.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_tmp.replace(selected_path)
    print(json.dumps({"student_candidate": candidate_metrics, "paired_teacher": teacher_metrics, "cache": str(output)}, ensure_ascii=False, indent=2))

    if args.enforce_pair_benefit:
        if not pair_benefit_passed:
            raise SystemExit(
                "Paired frozen teacher did not pass the precheck: "
                f"selected records={len(selected_records)} (required > 0). "
                f"Report was still written to {output}; do not start OPD."
            )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
