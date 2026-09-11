#!/usr/bin/env bash
# Deployment-distribution GRPO. There is no reference video or OPD teacher.
# Explanation reward is computed locally from TASLE's reference evidence.
set -euo pipefail

TRACE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_data" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
REPLAY_PATH=${REPLAY_PATH:?Set REPLAY_PATH to the full candidate-only GRPO replay JSON}
OPD_CKPT=${OPD_CKPT:?Set OPD_CKPT to final candidate-only OPD student checkpoint}
EXPLANATION_WEIGHT=${EXPLANATION_WEIGHT:-0.3}
TEXT_REWARD_MODE=${TEXT_REWARD_MODE:-lexical}
TEXT_REWARD_ARGS=(--grpo_text_reward_mode "$TEXT_REWARD_MODE" --grpo_text_max_words "${TEXT_MAX_WORDS:-80}")
if [[ "$TEXT_REWARD_MODE" == "nli" ]]; then
  NLI_MODEL_PATH=${NLI_MODEL_PATH:?Set NLI_MODEL_PATH to a local frozen NLI model directory}
  TEXT_REWARD_ARGS+=(--grpo_text_nli_model_path "$NLI_MODEL_PATH" --grpo_text_nli_device "${NLI_DEVICE:-cpu}" --grpo_text_nli_batch_size "${NLI_BATCH_SIZE:-32}")
fi
OUT_DIR=${OUT_DIR:-"$MSLOC_ASSETS/Trace/output/grpo"}
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

torchrun --nproc_per_node=${NPROC_PER_NODE:-1} "$TRACE_DIR/trace/train_mt.py" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --version v1_mistral --vision_tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
  --mm_projector_type spatial_slot --tune_mm_mlp_adapter True --tune_mm_embed_head True --tune_lm_embed_head True \
  --model_name_or_path "$OPD_CKPT" --data_path "$DATA_ROOT/annos/train_all_1209.json" --data_folder "$DATA_ROOT/videos" \
  --train_mode ref2 --replay_path "$REPLAY_PATH" --replay_balance none --second_stage grpo \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --grpo_group_size ${GROUP_SIZE:-4} --grpo_temperature ${TEMPERATURE:-0.7} --grpo_max_new_tokens ${MAX_NEW_TOKENS:-128} \
  --grpo_localization_weight 1.0 --grpo_explanation_weight "$EXPLANATION_WEIGHT" --grpo_format_weight 0.1 \
  "${TEXT_REWARD_ARGS[@]}" --grpo_structure_aware True \
  --grpo_kl_coef ${KL_COEF:-0.02} --grpo_sft_coef ${SFT_COEF:-0.1} \
  --bnd_ratio 0.2 --bnd_frames 16 --seg_frames 8 --bf16 True --output_dir "$OUT_DIR" \
  --num_train_epochs ${EPOCHS:-1} --per_device_train_batch_size ${BATCH_SIZE:-1} \
  --gradient_accumulation_steps ${GRAD_ACCUM:-4} --learning_rate ${LR:-1e-6} \
  "${SAVE_ARGS[@]}" --logging_steps 1 --disable_tqdm False --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers ${NUM_WORKERS:-0} \
  --report_to "$REPORT_TO" \
  --lazy_preprocess True --sample_scheme rand
