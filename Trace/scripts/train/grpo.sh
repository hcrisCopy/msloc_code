#!/usr/bin/env bash
# Deployment-distribution GRPO.  There is no reference video and no OPD teacher.
# Explanations are scored by frozen Qwen3-VL-235B through an OpenAI-compatible
# endpoint; it receives candidate-video frames, the student output, and an
# annotation-derived reference-evidence card, but never a paired reference
# video or GT timestamps.
set -euo pipefail

TRACE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_assets" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
REPLAY_PATH=${REPLAY_PATH:?Set REPLAY_PATH to normalized replay JSON}
OPD_CKPT=${OPD_CKPT:?Set OPD_CKPT to final candidate-only OPD student checkpoint}
QWEN_JUDGE_MODEL=${QWEN_JUDGE_MODEL:-Qwen/Qwen3-VL-235B-A22B-Instruct}
EXPLANATION_WEIGHT=${EXPLANATION_WEIGHT:-0.3}
JUDGE_ARGS=()
if [[ "$EXPLANATION_WEIGHT" != "0" && "$EXPLANATION_WEIGHT" != "0.0" && "$EXPLANATION_WEIGHT" != "0.00" ]]; then
  QWEN_JUDGE_ENDPOINT=${QWEN_JUDGE_ENDPOINT:?Set the frozen Qwen3-VL-235B OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1}
  EXPLANATION_JUDGE_COMMAND="${PYTHON_BIN:-python} $TRACE_DIR/scripts/qwen3_vl_explanation_judge.py --endpoint $QWEN_JUDGE_ENDPOINT --model $QWEN_JUDGE_MODEL"
  JUDGE_ARGS=(--grpo_explanation_judge_command "$EXPLANATION_JUDGE_COMMAND")
fi
OUT_DIR=${OUT_DIR:-"$MSLOC_ASSETS/Trace/output/grpo"}
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

torchrun --nproc_per_node=${NPROC_PER_NODE:-1} "$TRACE_DIR/trace/train_mt.py" \
  --deepspeed "$TRACE_DIR/scripts/zero3.json" \
  --version v1_mistral --vision_tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
  --mm_projector_type spatial_slot --tune_mm_mlp_adapter True --tune_mm_embed_head True --tune_lm_embed_head True \
  --model_name_or_path "$OPD_CKPT" --data_path "$DATA_ROOT/annos/train_all_1209.json" --data_folder "$DATA_ROOT/videos" \
  --train_mode ref2 --replay_path "$REPLAY_PATH" --replay_balance none --second_stage grpo \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --grpo_group_size ${GROUP_SIZE:-4} --grpo_temperature ${TEMPERATURE:-0.7} --grpo_max_new_tokens ${MAX_NEW_TOKENS:-128} \
  --grpo_localization_weight 1.0 --grpo_explanation_weight "$EXPLANATION_WEIGHT" --grpo_format_weight 0.1 \
  "${JUDGE_ARGS[@]}" --grpo_structure_aware True \
  --grpo_kl_coef ${KL_COEF:-0.02} --grpo_sft_coef ${SFT_COEF:-0.1} \
  --bnd_ratio 0.2 --bnd_frames 16 --seg_frames 8 --bf16 True --output_dir "$OUT_DIR" \
  --num_train_epochs ${EPOCHS:-1} --per_device_train_batch_size ${BATCH_SIZE:-1} \
  --gradient_accumulation_steps ${GRAD_ACCUM:-4} --learning_rate ${LR:-1e-6} \
  "${SAVE_ARGS[@]}" --logging_steps 1 --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers ${NUM_WORKERS:-0} \
  --lazy_preprocess True --sample_scheme rand
