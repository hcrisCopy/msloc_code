#!/bin/bash
# Train Trace in `ref2` mode using DeMamba 4-class proposals.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
MSLOC_ROOT=$(cd "$TRACE_DIR/.." && pwd)
MSLOC_ASSETS=${MSLOC_ASSETS:-"$(cd "$MSLOC_ROOT/../MSLoc_assets" && pwd)"}
DATA_ROOT=${DATA_ROOT:-"$MSLOC_ASSETS/data/Tasle-CoT-10K"}
PROPOSAL_PATH=${PROPOSAL_PATH:?Set PROPOSAL_PATH to DeMamba predictions.json generated on the training split}
BASE_CKPT=${BASE_CKPT:-"$MSLOC_ASSETS/Trace/ckpts/trace-uni"}

WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-16666}
RANK=${RANK:-0}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRAD_ACCUM:-2}
DENOMINATOR=$(($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS))
if [ "$GLOBAL_BATCH_SIZE" -lt "$DENOMINATOR" ] || [ $((GLOBAL_BATCH_SIZE % DENOMINATOR)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be a positive multiple of WORLD_SIZE*NPROC_PER_NODE*GRADIENT_ACCUMULATION_STEPS=$DENOMINATOR" >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE/$DENOMINATOR))
echo "LOCAL_BATCH_SIZE: $LOCAL_BATCH_SIZE"

export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=trace_vllava
export NCCL_P2P_LEVEL=NVL
export HCCL_BUFFSIZE=1024
RUN_NAME=trace_vllava
OUTP_DIR=${OUTP_DIR:-"$MSLOC_ASSETS/Trace/output/trace_vllava/ref2"}
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLE_ARGS=(--max_samples "$MAX_SAMPLES")
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
SAVE_ARGS=(--save_strategy epoch)
if [[ -n "${SAVE_STEPS:-}" ]]; then
    SAVE_ARGS=(--save_strategy steps --save_steps "$SAVE_STEPS")
fi
if [[ "${CLEAN:-0}" == "1" ]]; then
    case "$OUTP_DIR" in
        "$MSLOC_ASSETS"/*) rm -rf -- "$OUTP_DIR" ;;
        *) echo "Refusing CLEAN outside MSLOC_ASSETS: $OUTP_DIR" >&2; exit 2 ;;
    esac
fi

ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    "$TRACE_DIR/trace/train_mt.py" \
    --deepspeed "$TRACE_DIR/scripts/zero3.json" \
    --version v1_mistral \
    --vision_tower "$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336" \
    --mm_projector_type spatial_slot \
    --freeze_mm_mlp_adapter False \
    --tune_mm_mlp_adapter True \
    --tune_mm_embed_head True \
    --tune_lm_embed_head True \
    --model_name_or_path "$BASE_CKPT" \
    --data_path "$DATA_ROOT/annos/train_all_1209.json" \
    --data_folder "$DATA_ROOT/videos" \
    --train_mode ref2 \
    --proposal_path "$PROPOSAL_PATH" \
    "${MAX_SAMPLE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --bnd_ratio 0.2 \
    --bnd_frames 16 \
    --seg_frames 8 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --downsample_num 1 \
    --image_aspect_ratio pad \
    --freeze_backbone True \
    --num_frames 32 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --output_dir "$OUTP_DIR" \
    --num_train_epochs ${EPOCHS:-2} \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    "${SAVE_ARGS[@]}" \
    --save_total_limit 99 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${NUM_WORKERS:-4} \
    --run_name $RUN_NAME \
    --lazy_preprocess True \
    --sample_scheme "rand"
