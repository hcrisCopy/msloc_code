#!/usr/bin/env bash
# Candidate-only student OPD with a frozen, paired-input teacher.  The teacher
# is the SFT checkpoint itself viewed through a real-reference/candidate pair;
# it is never fine-tuned.  Run precheck_opd_teacher.py first.
set -euo pipefail
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TRACE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_data" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
REPLAY_PATH=${REPLAY_PATH:?Set REPLAY_PATH to normalized replay JSON with real references}
STUDENT_CKPT=${STUDENT_CKPT:?Set STUDENT_CKPT to the ref2-SFT checkpoint}
TEACHER_CACHE=${TEACHER_CACHE:?Set TEACHER_CACHE to the JSON produced by precheck_opd_teacher.py}
OUT_DIR=${OUT_DIR:-"$MSLOC_ASSETS/Trace/output/opd_student"}
REPORT_TO=${REPORT_TO:-none}
DEEPSPEED_CONFIG="$TRACE_DIR/scripts/zero2.json"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS=(--max_samples "$MAX_SAMPLES")
fi
SAVE_ARGS=(--save_strategy epoch)
if [[ -n "${SAVE_STEPS:-}" ]]; then
  SAVE_ARGS=(--save_strategy steps --save_steps "$SAVE_STEPS")
fi
if [[ "${CLEAN:-0}" == "1" ]]; then
  case "$OUT_DIR" in
    "$MSLOC_ASSETS"/*) rm -rf -- "$OUT_DIR" ;;
    *) echo "Refusing CLEAN outside MSLOC_ASSETS: $OUT_DIR" >&2; exit 2 ;;
  esac
fi

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-8} "$TRACE_DIR/trace/train_mt.py" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --version v1_mistral --vision_tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
  --mm_projector_type spatial_slot --tune_mm_mlp_adapter True --tune_mm_embed_head True --tune_lm_embed_head True \
  --model_name_or_path "$STUDENT_CKPT" --opd_teacher_model_path "$STUDENT_CKPT" \
  --data_path "$DATA_ROOT/annos/train_all_1209.json" --data_folder "$DATA_ROOT/videos" \
  --train_mode ref2 --replay_path "$REPLAY_PATH" --opd_teacher_cache_path "$TEACHER_CACHE" --replay_balance none --second_stage opd \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --opd_weight ${OPD_WEIGHT:-1.0} --opd_temperature ${OPD_TEMPERATURE:-1.0} \
  --opd_disagreement_iou_gate ${OPD_DISAGREEMENT_IOU_GATE:-0.30} \
  --opd_false_refusal_weight ${FALSE_REFUSAL_WEIGHT:-1.0} --opd_positive_error_weight ${POSITIVE_ERROR_WEIGHT:-0.8} --opd_negative_error_weight ${NEGATIVE_ERROR_WEIGHT:-0.8} \
  --opd_positive_anchor_weight ${POSITIVE_ANCHOR_WEIGHT:-0.2} --opd_negative_anchor_weight ${NEGATIVE_ANCHOR_WEIGHT:-0.2} \
  --opd_guided_positive_fraction ${GUIDED_POSITIVE_FRACTION:-1.0} --opd_guided_alpha ${GUIDED_ALPHA:-0.5} --opd_guided_max_tokens ${GUIDED_MAX_TOKENS:-16} \
  --bnd_ratio 0.2 --bnd_frames 16 --seg_frames 8 --bf16 True --output_dir "$OUT_DIR" \
  --num_train_epochs ${EPOCHS:-1} --per_device_train_batch_size ${BATCH_SIZE:-1} \
  --gradient_accumulation_steps ${GRAD_ACCUM:-4} --learning_rate ${LR:-2e-6} \
  "${SAVE_ARGS[@]}" --logging_steps 1 --disable_tqdm False --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers ${NUM_WORKERS:-0} \
  --report_to "$REPORT_TO" \
  --lazy_preprocess True --sample_scheme rand
