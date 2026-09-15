#!/usr/bin/env bash
# Validation only: this script never updates teacher or student parameters.
set -euo pipefail
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TRACE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_data" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
REPLAY_PATH=${REPLAY_PATH:?Set REPLAY_PATH to normalized replay with resolved real references}
STUDENT_CKPT=${STUDENT_CKPT:?Set STUDENT_CKPT to the candidate-only Student SFT checkpoint}
TEACHER_CKPT=${TEACHER_CKPT:?Set TEACHER_CKPT to the independently trained paired Teacher SFT checkpoint}
OUT_PATH=${OUT_PATH:-"$MSLOC_ASSETS/Trace/output/opd_teacher_precheck.json"}
SELECTED_OUT=${SELECTED_OUT:-"$MSLOC_ASSETS/Trace/output/opd_selected_samples.json"}

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-8} "$TRACE_DIR/scripts/precheck_opd_teacher.py" \
  --replay "$REPLAY_PATH" --data-folder "$DATA_ROOT/videos" \
  --student-model-path "$STUDENT_CKPT" --model-path "$TEACHER_CKPT" \
  --vision-tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
  --output "$OUT_PATH" --selected-output "$SELECTED_OUT" --version v1_mistral --bnd-ratio 0.2 --bnd-frames 16 --seg-frames 8 \
  --teacher-iou-gate ${TEACHER_IOU_GATE:-0.3} \
  --enforce-pair-benefit
