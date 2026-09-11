#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
MSLOC_ASSETS="$(cd "$REPO_ROOT/../MSLoc_data" && pwd)" || return 1
DATA_ROOT="$MSLOC_ASSETS/data/Tasle-CoT-10K"
TRAIN_ANNO="$DATA_ROOT/annos/train_all_1209.json"
TEST_ANNO="$DATA_ROOT/annos/test_all_1209_0119.json"
VIDEO_ROOT="$DATA_ROOT/videos"
FRAME_ROOT="$MSLOC_ASSETS/DeMamba/video_frames"
TRACE_BASE="$MSLOC_ASSETS/Trace/ckpts/trace-uni"
VISION_TOWER="$MSLOC_ASSETS/Trace/ckpts/clip-vit-large-patch14-336"
STAGE1_CONFIG="$MSLOC_ASSETS/DeMamba/full/configs/xclip_neurons_full.yaml"
STAGE1_CKPT="$MSLOC_ASSETS/DeMamba/full/method/results/best_acc.pth"
EXP_ROOT="$MSLOC_ASSETS/Trace/experiments/opd_grpo"

export REPO_ROOT MSLOC_ASSETS DATA_ROOT TRAIN_ANNO TEST_ANNO VIDEO_ROOT
export FRAME_ROOT TRACE_BASE VISION_TOWER STAGE1_CONFIG STAGE1_CKPT EXP_ROOT

mkdir -p "$EXP_ROOT" || return 1
test -f "$TRAIN_ANNO" || return 1
test -f "$TEST_ANNO" || return 1
test -d "$VIDEO_ROOT" || return 1
test -d "$FRAME_ROOT" || return 1
test -d "$TRACE_BASE" || return 1
test -d "$VISION_TOWER" || return 1
test -f "$STAGE1_CONFIG" || return 1
test -f "$STAGE1_CKPT" || return 1

echo "OPD/GRPO paths configured."
