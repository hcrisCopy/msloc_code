#!/usr/bin/env bash
# Run from the MSLoc_code root after downloading the gated Hugging Face
# DINOv3 ViT-B/16 checkpoint to the path in DINOv3_Tasle_neurons.yaml.
set -euo pipefail

python DeMamba/build_probe_pairs.py \
  --annotations /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --fps 8 \
  --strict

python DeMamba/probe_dinov3_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --dinov3-backend huggingface \
  --dinov3-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov3 \
  --image-batch-size 32 \
  --crop-youku --amp --strict

python DeMamba/train.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42

python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval \
  --device-ids 0 \
  --val-batch-size 16

python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/predictions.json