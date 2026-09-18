#!/usr/bin/env python3
"""Python launcher for the DeMamba -> SFT -> OPD -> GRPO pipeline.

The launcher keeps every formal experiment parameter visible on the command
line, provides one consistent single-node DDP contract, and owns cleanup,
resume, process interruption, and multi-GPU evaluation result merging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


TRACE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = TRACE_ROOT.parent


def _bool(value: str) -> str:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return "True" if lowered == "true" else "False"


def _devices(value: str) -> List[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices must be comma-separated GPU indices") from exc
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices must contain unique non-negative GPU indices")
    return devices


def _safe_clean_dir(path: str) -> None:
    target = Path(path).resolve()
    protected = {Path.cwd().resolve(), WORKSPACE_ROOT.resolve(), TRACE_ROOT.resolve(), Path(target.anchor)}
    if target in protected or len(target.parts) < 3:
        raise ValueError(f"Refusing to clean unsafe output directory: {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Expected an output directory, got a file: {target}")
        shutil.rmtree(target)


def _safe_clean_file(path: str) -> None:
    target = Path(path).resolve()
    if target.is_file():
        target.unlink()
    progress = Path(str(target) + ".progress")
    if progress.is_dir():
        shutil.rmtree(progress)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.is_file():
        temporary.unlink()


def _runtime_env(devices: Sequence[int]) -> dict:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
        "PYTHONPATH": str(TRACE_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "true",
        "WANDB_DISABLED": "true",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    })
    return env


def _torchrun(args, script: Path, script_args: Sequence[str]) -> None:
    if args.nproc_per_node != len(args.devices):
        raise ValueError(
            f"--nproc-per-node ({args.nproc_per_node}) must equal the number of --devices ({len(args.devices)}). "
            "For a one-GPU formal check, change both to 1 and use --devices 0."
        )
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={args.nproc_per_node}", str(script), *script_args,
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE_ROOT, env=_runtime_env(args.devices), check=True)


def _run(command: Sequence[str], devices: Sequence[int] | None = None) -> None:
    print("Launching:", " ".join(command), flush=True)
    env = _runtime_env(devices) if devices else os.environ.copy()
    subprocess.run(list(command), cwd=WORKSPACE_ROOT, env=env, check=True)


def _prepare_output(args, *, is_file: bool = False) -> None:
    if args.clean and args.resume != "none":
        raise ValueError("--clean and --resume cannot be used together")
    if args.clean:
        (_safe_clean_file if is_file else _safe_clean_dir)(args.output)
    output = Path(args.output)
    if args.resume == "none" and output.exists() and not args.clean:
        raise FileExistsError(f"{args.output} already exists; choose --clean or --resume auto/path")


def _resume_args(value: str) -> List[str]:
    return [] if value == "none" else ["--resume_from_checkpoint", value]


def _max_sample_args(value: int) -> List[str]:
    if value < 0:
        raise ValueError("--max-samples must be non-negative; 0 means full data")
    return [] if value == 0 else ["--max_samples", str(value)]


def _validate_train_architecture(args) -> None:
    if args.mm_projector_type == "ref_projector":
        expected_frames = 2 * args.bnd_frames + args.seg_frames
        if args.num_frames != expected_frames:
            raise ValueError(
                "ref_projector requires --num-frames = 2 * --bnd-frames + --seg-frames; "
                f"got {args.num_frames} != 2 * {args.bnd_frames} + {args.seg_frames}"
            )
    if args.closs == "True" and not Path(args.class_feature_path).is_file():
        raise FileNotFoundError(
            "--closs true requires the pre-computed class feature file: "
            f"{args.class_feature_path}"
        )


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_train_args(args, model_path: str, output: str) -> List[str]:
    _validate_train_architecture(args)
    save_args = ["--save_strategy", args.save_strategy]
    if args.save_strategy == "steps":
        if args.save_steps <= 0:
            raise ValueError("--save-steps must be positive when --save-strategy steps")
        save_args += ["--save_steps", str(args.save_steps)]
    return [
        "--deepspeed", args.deepspeed,
        "--version", args.version,
        "--vision_tower", args.vision_tower,
        "--mm_projector_type", args.mm_projector_type,
        "--closs", args.closs,
        "--class_feature_path", args.class_feature_path,
        "--freeze_mm_mlp_adapter", args.freeze_mm_mlp_adapter,
        "--tune_mm_mlp_adapter", args.tune_mm_mlp_adapter,
        "--tune_mm_embed_head", args.tune_mm_embed_head,
        "--tune_lm_embed_head", args.tune_lm_embed_head,
        "--model_name_or_path", model_path,
        "--data_path", args.annotation,
        "--data_folder", args.video_root,
        "--train_mode", "ref2",
        "--bnd_ratio", str(args.bnd_ratio),
        "--bnd_frames", str(args.bnd_frames),
        "--seg_frames", str(args.seg_frames),
        "--mm_vision_select_layer", str(args.mm_vision_select_layer),
        "--mm_use_im_start_end", args.mm_use_im_start_end,
        "--mm_use_im_patch_token", args.mm_use_im_patch_token,
        "--downsample_num", str(args.downsample_num),
        "--image_aspect_ratio", args.image_aspect_ratio,
        "--freeze_backbone", args.freeze_backbone,
        "--num_frames", str(args.num_frames),
        "--bf16", args.bf16,
        "--tf32", args.tf32,
        "--fp16", args.fp16,
        "--output_dir", output,
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", str(args.eval_batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--evaluation_strategy", "no",
        *save_args,
        "--save_total_limit", str(args.save_total_limit),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--logging_steps", str(args.logging_steps),
        "--disable_tqdm", "False",
        "--model_max_length", str(args.model_max_length),
        "--gradient_checkpointing", args.gradient_checkpointing,
        "--dataloader_num_workers", str(args.num_workers),
        "--report_to", "none",
        "--run_name", args.run_name,
        "--lazy_preprocess", "True",
        "--sample_scheme", args.sample_scheme,
        *_max_sample_args(args.max_samples),
        *_resume_args(args.resume),
    ]


def _rollout_audit_args(args) -> List[str]:
    if args.save_rollouts == "True" and not args.rollout_output:
        raise ValueError("--save-rollouts true requires --rollout-output")
    result = ["--save_rollouts", args.save_rollouts]
    if args.rollout_output:
        result += ["--rollout_audit_dir", args.rollout_output]
    return result


def _write_stage_manifest(args, stage: str, extra: dict) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "launcher": "Trace/run_opd_grpo.py", **extra}
    temporary = output / "stage_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "stage_manifest.json")


def _validate_distinct_teacher_base(student_checkpoint: str, teacher_checkpoint: str) -> None:
    student_path = Path(student_checkpoint)
    teacher_path = Path(teacher_checkpoint)
    if student_path.resolve() == teacher_path.resolve():
        raise ValueError(
            "Student and teacher must be separately SFT-trained checkpoints from the same base; "
            "they cannot be the same directory"
        )

    for role, checkpoint in (("student", student_path), ("teacher", teacher_path)):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"OPD {role} checkpoint directory does not exist: {checkpoint}")

    teacher_manifest_path = teacher_path / "stage_manifest.json"
    if not teacher_manifest_path.is_file():
        raise FileNotFoundError(
            "The separately trained paired teacher must have stage_manifest.json created by "
            f"this launcher: {teacher_manifest_path}"
        )
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("stage") != "paired_teacher_sft":
        raise ValueError("The distinct OPD teacher must be a paired_teacher_sft checkpoint")

    student_manifest_path = student_path / "stage_manifest.json"
    if not student_manifest_path.is_file():
        print(
            "WARNING: Student checkpoint has no stage_manifest.json, so its original base weights and "
            "SFT settings cannot be verified automatically. Treating it as a trusted pre-existing "
            f"candidate-only SFT checkpoint: {student_path}",
            flush=True,
        )
        return

    student_manifest = json.loads(student_manifest_path.read_text(encoding="utf-8"))
    if student_manifest.get("stage") != "candidate_sft":
        raise ValueError("The OPD student must be a candidate_sft checkpoint")
    student_base_value = student_manifest.get("base_checkpoint")
    teacher_base_value = teacher_manifest.get("base_checkpoint")
    if student_base_value is None or teacher_base_value is None:
        print(
            "WARNING: A legacy stage manifest does not record base_checkpoint; "
            "the shared original weights cannot be verified automatically.",
            flush=True,
        )
    else:
        student_base = Path(student_base_value).resolve()
        teacher_base = Path(teacher_base_value).resolve()
        if student_base != teacher_base:
            raise ValueError(
                f"Student and paired teacher do not share the same original base weights: {student_base} != {teacher_base}"
            )
    for key in ("mm_projector_type", "closs"):
        student_value = student_manifest.get(key)
        teacher_value = teacher_manifest.get(key)
        if student_value is None or teacher_value is None:
            print(
                f"WARNING: A legacy stage manifest does not record {key}; "
                "compatibility cannot be verified automatically.",
                flush=True,
            )
            continue
        if student_value != teacher_value:
            raise ValueError(
                f"Student and paired teacher must use the same {key}: "
                f"{student_value!r} != {teacher_value!r}"
            )
    if student_manifest.get("closs"):
        student_hash = student_manifest.get("class_feature_sha256")
        teacher_hash = teacher_manifest.get("class_feature_sha256")
        if not student_hash or student_hash != teacher_hash:
            raise ValueError(
                "Student and paired teacher must use the identical class_features_bge.pt file"
            )


def run_sft(args) -> None:
    _prepare_output(args)
    train_args = _common_train_args(args, args.base_checkpoint, args.output)
    train_args += ["--proposal_path", args.proposals, "--second_stage", "sft"]
    _torchrun(args, TRACE_ROOT / "trace" / "train_mt.py", train_args)
    _write_stage_manifest(args, "candidate_sft", {
        "base_checkpoint": args.base_checkpoint,
        "proposals": args.proposals,
        "annotation": args.annotation,
        "input_mode": "candidate_only",
        "mm_projector_type": args.mm_projector_type,
        "closs": args.closs == "True",
        "class_feature_path": args.class_feature_path,
        "class_feature_sha256": _file_sha256(args.class_feature_path) if args.closs == "True" else None,
        "freeze_backbone": args.freeze_backbone == "True",
    })


def run_paired_teacher_sft(args) -> None:
    _prepare_output(args)
    train_args = _common_train_args(args, args.base_checkpoint, args.output)
    train_args += ["--replay_path", args.replay, "--replay_balance", "none", "--second_stage", "opd_teacher_sft"]
    _torchrun(args, TRACE_ROOT / "trace" / "train_mt.py", train_args)
    _write_stage_manifest(args, "paired_teacher_sft", {
        "base_checkpoint": args.base_checkpoint,
        "replay": args.replay,
        "annotation": args.annotation,
        "input_mode": "upper_real_lower_candidate",
        "mm_projector_type": args.mm_projector_type,
        "closs": args.closs == "True",
        "class_feature_path": args.class_feature_path,
        "class_feature_sha256": _file_sha256(args.class_feature_path) if args.closs == "True" else None,
        "freeze_backbone": args.freeze_backbone == "True",
    })


def run_opd(args) -> None:
    _validate_distinct_teacher_base(args.student_checkpoint, args.teacher_checkpoint)
    _prepare_output(args)
    train_args = _common_train_args(args, args.student_checkpoint, args.output)
    train_args += [
        "--replay_path", args.replay, "--opd_teacher_cache_path", args.teacher_cache,
        "--replay_balance", "none", "--second_stage", "opd",
        "--opd_teacher_model_path", args.teacher_checkpoint,
        "--opd_weight", str(args.opd_weight), "--opd_temperature", str(args.opd_temperature),
        "--opd_disagreement_iou_gate", str(args.opd_disagreement_iou_gate),
        "--opd_false_refusal_weight", str(args.false_refusal_weight),
        "--opd_positive_error_weight", str(args.positive_error_weight),
        "--opd_negative_error_weight", str(args.negative_error_weight),
        "--opd_positive_anchor_weight", str(args.positive_anchor_weight),
        "--opd_negative_anchor_weight", str(args.negative_anchor_weight),
        "--opd_guided_positive_fraction", str(args.guided_positive_fraction),
        "--opd_guided_alpha", str(args.guided_alpha),
        "--opd_guided_max_tokens", str(args.guided_max_tokens),
        "--opd_guided_loss_coef", str(args.guided_loss_coef),
        "--grpo_max_new_tokens", str(args.rollout_max_new_tokens),
        *_rollout_audit_args(args),
    ]
    _torchrun(args, TRACE_ROOT / "trace" / "train_mt.py", train_args)
    _write_stage_manifest(args, "opd_student", {
        "student_checkpoint": args.student_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_cache": args.teacher_cache,
        "replay": args.replay,
        "student_input_mode": "candidate_only",
        "teacher_input_mode": "upper_real_lower_candidate",
        "mm_projector_type": args.mm_projector_type,
        "closs": args.closs == "True",
        "save_rollouts": args.save_rollouts == "True",
        "rollout_output": args.rollout_output,
    })


def run_grpo(args) -> None:
    _prepare_output(args)
    train_args = _common_train_args(args, args.opd_checkpoint, args.output)
    train_args += [
        "--replay_path", args.replay, "--replay_balance", "none", "--second_stage", "grpo",
        "--grpo_group_size", str(args.group_size), "--grpo_temperature", str(args.temperature),
        "--grpo_max_new_tokens", str(args.max_new_tokens), "--grpo_clip_range", str(args.clip_range),
        "--grpo_localization_weight", str(args.localization_weight),
        "--grpo_explanation_weight", str(args.explanation_weight),
        "--grpo_format_weight", str(args.format_weight),
        "--grpo_explanation_iou_gate", str(args.explanation_iou_gate),
        "--grpo_boundary_tolerance", str(args.boundary_tolerance),
        "--grpo_text_reward_mode", "entailment",
        "--grpo_text_require_candidate_observable", args.require_candidate_observable,
        "--grpo_structure_aware", args.structure_aware,
        "--grpo_kl_coef", str(args.kl_coef), "--grpo_sft_coef", str(args.sft_coef),
        *_rollout_audit_args(args),
    ]
    if args.explanation_weight > 0 and not args.entailment_model_path:
        raise ValueError("--entailment-model-path is required when --explanation-weight is positive")
    train_args += [
        "--grpo_text_nli_model_path", args.entailment_model_path,
        "--grpo_text_nli_device", args.entailment_device,
        "--grpo_text_nli_batch_size", str(args.entailment_batch_size),
    ]
    _torchrun(args, TRACE_ROOT / "trace" / "train_mt.py", train_args)
    _write_stage_manifest(args, "grpo", {
        "opd_checkpoint": args.opd_checkpoint, "replay": args.replay,
        "input_mode": "candidate_only", "text_reward_mode": "atomic_entailment_v3_aligned_contradiction",
        "entailment_model_path": args.entailment_model_path,
        "mm_projector_type": args.mm_projector_type,
        "closs": args.closs == "True",
        "save_rollouts": args.save_rollouts == "True",
        "rollout_output": args.rollout_output,
    })


def run_replay(args) -> None:
    if args.paired_only and not args.video_root:
        raise ValueError("--paired-only requires --video-root")
    command = [
        sys.executable, str(TRACE_ROOT / "scripts" / "build_opd_grpo_replay.py"),
        "--gt", args.annotation, "--proposals", args.proposals, "--output", args.output,
        "--near-negative-seconds", str(args.near_negative_seconds),
        "--max-records", str(args.max_records),
    ]
    if args.reference_map:
        command += ["--reference-map", args.reference_map]
    if args.evidence_audit:
        command += ["--evidence-audit", args.evidence_audit]
    if args.paired_only:
        command += ["--paired-only", "--video-root", args.video_root]
    if args.require_reference:
        command.append("--require-reference")
    if args.stratified_debug:
        command.append("--stratified-debug")
    if args.resume:
        command.append("--resume")
    if args.clean:
        command.append("--clean")
    _run(command)


def run_precheck(args) -> None:
    _validate_distinct_teacher_base(args.student_checkpoint, args.teacher_checkpoint)
    if args.clean and args.resume != "none":
        raise ValueError("--clean and --resume cannot be used together")
    script_args = [
        "--replay", args.replay, "--data-folder", args.video_root,
        "--student-model-path", args.student_checkpoint,
        "--model-path", args.teacher_checkpoint, "--vision-tower", args.vision_tower,
        "--output", args.output, "--selected-output", args.selected_output,
        "--prompt-file", args.prompt_file, "--version", args.version,
        "--bnd-ratio", str(args.bnd_ratio), "--bnd-frames", str(args.bnd_frames),
        "--seg-frames", str(args.seg_frames), "--max-new-tokens", str(args.max_new_tokens),
        "--teacher-iou-gate", str(args.teacher_iou_gate),
        "--max-samples", str(args.max_samples),
    ]
    if args.enforce_better == "True":
        script_args.append("--enforce-pair-benefit")
    if args.clean:
        script_args.append("--clean")
    if args.resume != "none":
        if args.resume != "auto":
            raise ValueError("Precheck accepts --resume auto (its progress location is fixed beside --output)")
        script_args.append("--resume")
    _torchrun(args, TRACE_ROOT / "scripts" / "precheck_opd_teacher.py", script_args)


def run_teacher_eval(args) -> None:
    if not Path(args.teacher_checkpoint).is_dir():
        raise FileNotFoundError(f"Teacher checkpoint directory does not exist: {args.teacher_checkpoint}")
    if args.clean and args.resume != "none":
        raise ValueError("--clean and --resume cannot be used together")
    output_dir = Path(args.output)
    if args.clean:
        _safe_clean_dir(args.output)
        _safe_clean_file(args.metrics_output)
    elif output_dir.exists() and args.resume == "none":
        raise FileExistsError(f"{args.output} already exists; choose --clean or --resume auto")
    elif not output_dir.exists() and args.resume != "none":
        raise FileNotFoundError(f"Cannot resume missing Teacher evaluation directory: {args.output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "proposal_report.json"
    prediction_path = output_dir / "paired_test_predictions.json"
    subset_gt_path = output_dir / "paired_test_gt.json"
    script_args = [
        "--teacher-only",
        "--replay", args.replay, "--data-folder", args.video_root,
        "--model-path", args.teacher_checkpoint, "--vision-tower", args.vision_tower,
        "--annotation", args.annotation,
        "--output", str(report_path),
        "--prediction-output", str(prediction_path),
        "--subset-gt-output", str(subset_gt_path),
        "--prompt-file", args.prompt_file, "--version", args.version,
        "--bnd-ratio", str(args.bnd_ratio), "--bnd-frames", str(args.bnd_frames),
        "--seg-frames", str(args.seg_frames), "--max-new-tokens", str(args.max_new_tokens),
        "--teacher-iou-gate", str(args.teacher_iou_gate),
        "--max-samples", str(args.max_samples),
    ]
    if args.resume != "none":
        if args.resume != "auto":
            raise ValueError("Teacher evaluation accepts --resume auto (progress is stored beside --output)")
        script_args.append("--resume")
    _torchrun(args, TRACE_ROOT / "scripts" / "precheck_opd_teacher.py", script_args)
    metrics_command = [
        sys.executable, str(WORKSPACE_ROOT / "evaluate_long.py"),
        "--gt_file", str(subset_gt_path),
        "--infer_file", str(prediction_path),
        "--output_file", args.metrics_output,
    ]
    if args.resume != "none":
        metrics_command.append("--reuse-existing")
    _run(metrics_command)


def _terminate_processes(processes: Iterable[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _merge_eval_results(args) -> Path:
    output_dir = Path(args.output)
    final = output_dir / f"fmt_aigc_test_f{args.num_frames}_result.json"
    if len(args.devices) == 1:
        if not final.is_file():
            raise FileNotFoundError(f"Single-GPU evaluation result is missing: {final}")
        return final
    chunks = [output_dir / f"fmt_aigc_test_f{args.num_frames}_result_chunk{i}.json" for i in range(len(args.devices))]
    missing = [str(path) for path in chunks if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation chunks are missing: {missing}")
    merged = []
    for path in chunks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected list result in {path}")
        merged.extend(payload)
    temporary = final.with_suffix(final.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(final)
    return final


def run_eval(args) -> None:
    if args.clean and args.resume:
        raise ValueError("--clean and --resume cannot be used together")
    if args.clean:
        _safe_clean_dir(args.output)
    output_dir = Path(args.output)
    if output_dir.exists() and not args.clean and not args.resume:
        raise FileExistsError(f"{args.output} already exists; choose --clean or --resume")
    if args.resume and not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing evaluation directory: {args.output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processes: List[subprocess.Popen] = []
    old_handlers = {}

    def stop_handler(signum, _frame):
        _terminate_processes(processes)
        raise KeyboardInterrupt(f"evaluation interrupted by signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop_handler)
        for chunk_idx, physical_device in enumerate(args.devices):
            command = [
                sys.executable, "-u", str(TRACE_ROOT / "trace" / "eval" / "evaluate_ref.py"),
                "--anno_path", str(TRACE_ROOT / "scripts" / "eval"),
                "--anno_file", args.proposals, "--video_path", args.video_root,
                "--gpu_id", "0", "--task", "dvc", "--dataset", "aigc",
                "--output_dir", args.output, "--split", "test",
                "--num_frames", str(args.num_frames), "--batch_size", str(args.batch_size),
                "--prompt_file", args.prompt_file, "--model_path", args.model_checkpoint,
                "--vision_tower", args.vision_tower, "--max_new_tokens", str(args.max_new_tokens),
                "--sample_num", str(args.sample_num), "--num_chunks", str(len(args.devices)),
                "--chunk_idx", str(chunk_idx), "--tqdm_position", str(chunk_idx),
                "--bnd_ratio", str(args.bnd_ratio), "--bnd_frames", str(args.bnd_frames),
                "--seg_frames", str(args.seg_frames),
            ]
            if chunk_idx != 0:
                command.append("--quiet_non_master")
            if args.resume:
                command.append("--resume")
            env = _runtime_env([physical_device])
            processes.append(subprocess.Popen(command, cwd=WORKSPACE_ROOT, env=env))
        failed = []
        for chunk_idx, process in enumerate(processes):
            code = process.wait()
            if code != 0:
                failed.append((chunk_idx, code))
        if failed:
            raise RuntimeError(f"Evaluation workers failed: {failed}; rerun with --resume")
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    final = _merge_eval_results(args)
    print(f"Merged {len(args.devices)} chunks: {final}")
    metrics_command = [
        sys.executable, str(WORKSPACE_ROOT / "evaluate_long.py"),
        "--gt_file", args.annotation,
        "--infer_file", str(final),
        "--output_file", args.metrics_output,
    ]
    if args.resume:
        metrics_command.append("--reuse-existing")
    _run(metrics_command)


def add_distributed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--devices", type=_devices, required=True, help="Visible GPUs, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, required=True)


def add_train_common(
        parser: argparse.ArgumentParser, *, default_lr: float,
        default_freeze_backbone: str = "True") -> None:
    add_distributed(parser)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="v1_mistral")
    parser.add_argument("--mm-projector-type", default="ref_projector")
    parser.add_argument("--closs", type=_bool, default="True")
    parser.add_argument(
        "--class-feature-path",
        default="../MSLoc_data/Trace/class_features_bge.pt",
        help="Pre-computed bge-large-en-v1.5 anomaly-class features used by --closs true",
    )
    parser.add_argument("--freeze-mm-mlp-adapter", type=_bool, default="False")
    parser.add_argument("--tune-mm-mlp-adapter", type=_bool, default="True")
    parser.add_argument("--tune-mm-embed-head", type=_bool, default="True")
    parser.add_argument("--tune-lm-embed-head", type=_bool, default="True")
    parser.add_argument("--freeze-backbone", type=_bool, default=default_freeze_backbone)
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--mm-vision-select-layer", type=int, default=-2)
    parser.add_argument("--mm-use-im-start-end", type=_bool, default="False")
    parser.add_argument("--mm-use-im-patch-token", type=_bool, default="False")
    parser.add_argument("--downsample-num", type=int, default=1)
    parser.add_argument("--image-aspect-ratio", default="pad")
    parser.add_argument("--bf16", type=_bool, default="True")
    parser.add_argument("--tf32", type=_bool, default="False")
    parser.add_argument("--fp16", type=_bool, default="False")
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-GPU batch size; never auto-scaled")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=default_lr)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", choices=("epoch", "steps"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=99)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--gradient-checkpointing", type=_bool, default="True")
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--sample-scheme", default="rand")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses the full formal dataset")
    parser.add_argument("--resume", default="none", help="none, auto, or an explicit checkpoint directory")
    parser.add_argument("--clean", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    sft = subparsers.add_parser(
        "student-sft", aliases=["sft"],
        help="Train the candidate-only student from the shared base checkpoint",
    )
    add_train_common(sft, default_lr=5e-6, default_freeze_backbone="False")
    sft.add_argument("--proposals", required=True)
    sft.add_argument("--base-model", "--base-checkpoint", dest="base_checkpoint", required=True)
    sft.set_defaults(handler=run_sft)

    replay = subparsers.add_parser(
        "build-samples", aliases=["build-replay"],
        help="Build OPD or GRPO training samples from real stage-1 proposals",
    )
    replay.add_argument("--annotation", required=True)
    replay.add_argument("--proposals", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--video-root")
    replay.add_argument("--reference-map")
    replay.add_argument("--evidence-audit")
    replay.add_argument("--near-negative-seconds", type=float, default=1.0)
    replay.add_argument("--max-records", type=int, default=0)
    replay.add_argument("--paired-only", action="store_true")
    replay.add_argument("--require-reference", action="store_true")
    replay.add_argument("--stratified-debug", action="store_true")
    replay.add_argument("--resume", action="store_true")
    replay.add_argument("--clean", action="store_true")
    replay.set_defaults(handler=run_replay)

    precheck = subparsers.add_parser(
        "check-teacher", aliases=["precheck"],
        help="Check whether the frozen teacher benefits from the real reference video",
    )
    add_distributed(precheck)
    precheck.add_argument("--training-samples", "--replay", dest="replay", required=True)
    precheck.add_argument("--video-root", required=True)
    precheck.add_argument("--student-model", "--student-checkpoint", dest="student_checkpoint", required=True)
    precheck.add_argument("--teacher-model", "--teacher-checkpoint", dest="teacher_checkpoint", required=True)
    precheck.add_argument("--vision-tower", required=True)
    precheck.add_argument("--output", required=True)
    precheck.add_argument("--selected-output", required=True)
    precheck.add_argument("--prompt-file", required=True)
    precheck.add_argument("--version", default="v1_mistral")
    precheck.add_argument("--bnd-ratio", type=float, default=0.2)
    precheck.add_argument("--bnd-frames", type=int, default=16)
    precheck.add_argument("--seg-frames", type=int, default=8)
    precheck.add_argument("--max-new-tokens", type=int, default=128)
    precheck.add_argument("--teacher-iou-gate", type=float, default=0.3)
    precheck.add_argument("--enforce-better", type=_bool, default="True")
    precheck.add_argument("--max-samples", type=int, default=0)
    precheck.add_argument("--resume", default="none", choices=("none", "auto"))
    precheck.add_argument("--clean", action="store_true")
    precheck.set_defaults(handler=run_precheck)

    teacher_eval = subparsers.add_parser(
        "test-teacher",
        help="Evaluate only the paired-input Teacher SFT checkpoint",
    )
    add_distributed(teacher_eval)
    teacher_eval.add_argument("--test-samples", "--replay", dest="replay", required=True)
    teacher_eval.add_argument("--annotation", required=True)
    teacher_eval.add_argument("--video-root", required=True)
    teacher_eval.add_argument("--teacher-model", "--teacher-checkpoint", dest="teacher_checkpoint", required=True)
    teacher_eval.add_argument("--vision-tower", required=True)
    teacher_eval.add_argument("--output", required=True)
    teacher_eval.add_argument("--metrics-output", required=True)
    teacher_eval.add_argument("--prompt-file", required=True)
    teacher_eval.add_argument("--version", default="v1_mistral")
    teacher_eval.add_argument("--bnd-ratio", type=float, default=0.2)
    teacher_eval.add_argument("--bnd-frames", type=int, default=16)
    teacher_eval.add_argument("--seg-frames", type=int, default=8)
    teacher_eval.add_argument("--max-new-tokens", type=int, default=128)
    teacher_eval.add_argument("--teacher-iou-gate", type=float, default=0.3)
    teacher_eval.add_argument("--max-samples", type=int, default=0)
    teacher_eval.add_argument("--resume", default="none", choices=("none", "auto"))
    teacher_eval.add_argument("--clean", action="store_true")
    teacher_eval.set_defaults(handler=run_teacher_eval)

    teacher = subparsers.add_parser(
        "train-paired-teacher", aliases=["paired-teacher-sft"],
        help="Train the fallback teacher on upper-real/lower-candidate inputs",
    )
    add_train_common(teacher, default_lr=5e-6, default_freeze_backbone="False")
    teacher.add_argument("--training-samples", "--replay", dest="replay", required=True)
    teacher.add_argument("--base-model", "--base-checkpoint", dest="base_checkpoint", required=True)
    teacher.set_defaults(handler=run_paired_teacher_sft)

    opd = subparsers.add_parser("opd", help="Candidate-only student with a frozen paired-input teacher")
    add_train_common(opd, default_lr=2e-6)
    opd.add_argument("--training-samples", "--replay", dest="replay", required=True)
    opd.add_argument("--student-model", "--student-checkpoint", dest="student_checkpoint", required=True)
    opd.add_argument("--teacher-model", "--teacher-checkpoint", dest="teacher_checkpoint", required=True)
    opd.add_argument("--teacher-check-result", "--teacher-cache", dest="teacher_cache", required=True)
    opd.add_argument("--opd-weight", type=float, default=1.0)
    opd.add_argument("--opd-temperature", type=float, default=1.0)
    opd.add_argument("--opd-disagreement-iou-gate", type=float, default=0.3)
    opd.add_argument("--false-refusal-weight", type=float, default=1.0)
    opd.add_argument("--positive-error-weight", type=float, default=0.8)
    opd.add_argument("--negative-error-weight", type=float, default=0.8)
    opd.add_argument("--positive-anchor-weight", type=float, default=0.2)
    opd.add_argument("--negative-anchor-weight", type=float, default=0.2)
    opd.add_argument("--guided-positive-fraction", type=float, default=1.0)
    opd.add_argument("--guided-alpha", type=float, default=0.5)
    opd.add_argument("--guided-max-tokens", type=int, default=16)
    opd.add_argument("--guided-loss-coef", type=float, default=0.25)
    opd.add_argument("--rollout-max-new-tokens", type=int, default=128)
    opd.add_argument("--save-rollouts", type=_bool, default="False")
    opd.add_argument("--rollout-output", default="")
    opd.set_defaults(handler=run_opd)

    grpo = subparsers.add_parser("grpo", help="Candidate-only structure-aware GRPO")
    add_train_common(grpo, default_lr=1e-6)
    grpo.add_argument("--training-samples", "--replay", dest="replay", required=True)
    grpo.add_argument("--starting-model", "--opd-checkpoint", dest="opd_checkpoint", required=True)
    grpo.add_argument("--group-size", type=int, default=4)
    grpo.add_argument("--temperature", type=float, default=0.7)
    grpo.add_argument("--max-new-tokens", type=int, default=128)
    grpo.add_argument("--clip-range", type=float, default=0.2)
    grpo.add_argument("--localization-weight", type=float, default=1.0)
    grpo.add_argument("--explanation-weight", type=float, default=0.3)
    grpo.add_argument("--format-weight", type=float, default=0.1)
    grpo.add_argument("--explanation-iou-gate", type=float, default=0.3)
    grpo.add_argument("--boundary-tolerance", type=float, default=1.0)
    grpo.add_argument("--require-candidate-observable", action="store_true", default=False)
    grpo.add_argument("--entailment-model-path", default="")
    grpo.add_argument("--entailment-device", default="cuda")
    grpo.add_argument("--entailment-batch-size", type=int, default=32)
    grpo.add_argument("--structure-aware", type=_bool, default="True")
    grpo.add_argument("--kl-coef", type=float, default=0.02)
    grpo.add_argument("--sft-coef", type=float, default=0.1)
    grpo.add_argument("--save-rollouts", type=_bool, default="False")
    grpo.add_argument("--rollout-output", default="")
    grpo.set_defaults(handler=run_grpo)

    evaluate = subparsers.add_parser(
        "test", aliases=["eval"],
        help="Test with one full model replica and one data shard per GPU",
    )
    evaluate.add_argument("--devices", type=_devices, required=True)
    evaluate.add_argument("--model", "--model-checkpoint", dest="model_checkpoint", required=True)
    evaluate.add_argument("--proposals", required=True)
    evaluate.add_argument("--annotation", required=True)
    evaluate.add_argument("--video-root", required=True)
    evaluate.add_argument("--vision-tower", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--metrics-output", required=True)
    evaluate.add_argument("--prompt-file", required=True)
    evaluate.add_argument("--num-frames", type=int, default=40)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--sample-num", type=int, default=-1)
    evaluate.add_argument("--bnd-ratio", type=float, default=0.2)
    evaluate.add_argument("--bnd-frames", type=int, default=16)
    evaluate.add_argument("--seg-frames", type=int, default=8)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--clean", action="store_true")
    evaluate.set_defaults(handler=run_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
