#!/usr/bin/env bash
# Run from the MSLoc_code root after downloading the gated Hugging Face
# DINOv3 ViT-B/16 checkpoint to the path in DINOv3_Tasle_neurons.yaml.
set -euo pipefail

python DeMamba/build_probe_pairs.py \
  --annotations ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --fps 8 \
  --strict

python DeMamba/probe_dinov3_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --frame-root ../MSLoc_data/DeMamba/video_frames \
  --dinov3-backend huggingface \
  --dinov3-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov3 \
  --image-batch-size 32 \
  --crop-youku --amp --strict

python DeMamba/train.py --config DeMamba/configs/DINOv3_Tasle_neurons.yaml

python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval
