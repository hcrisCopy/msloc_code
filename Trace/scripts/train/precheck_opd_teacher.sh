#!/usr/bin/env bash
# Validation only: this script never updates teacher or student parameters.
set -euo pipefail

TRACE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_data" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
REPLAY_PATH=${REPLAY_PATH:?Set REPLAY_PATH to normalized replay with resolved real references}
SFT_CKPT=${SFT_CKPT:?Set SFT_CKPT to the candidate-only ref2-SFT checkpoint}
OUT_PATH=${OUT_PATH:-"$MSLOC_ASSETS/Trace/output/opd_teacher_precheck.json"}

python "$TRACE_DIR/scripts/precheck_opd_teacher.py" \
  --replay "$REPLAY_PATH" --data-folder "$DATA_ROOT/videos" \
  --model-path "$SFT_CKPT" --vision-tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
  --output "$OUT_PATH" --version v1_mistral --bnd-ratio 0.2 --bnd-frames 16 --seg-frames 8 \
  --teacher-iou-gate ${TEACHER_IOU_GATE:-0.3} \
  --minimum-recovery-improvement ${MIN_RECOVERY_IMPROVEMENT:-0.01} \
  --maximum-negative-noevent-drop ${MAX_NEGATIVE_NOEVENT_DROP:-0.02} \
  --enforce-pair-benefit
