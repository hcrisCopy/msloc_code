# DINOv2 ViT-B/14 + DeMamba：下载与运行

所有命令均从项目根目录 `MSLoc_code` 执行

## 1. DINOv2

### 1.1 权重下载

```bash
hf download facebook/dinov2-base \
  --local-dir ../MSLoc_data/DeMamba/pretrained_weights/dinov2_hf
```

### 1.2 不使用神经元探测的基线训练与评测

该基线直接把 DINOv2 最后一层的全部 patch-token 隐藏特征输入 DeMamba，不读取神经元索引。

训练结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv2_Tasle.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

读取训练得到的 `best_acc.pth` 评测原论文测试集，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv2_Tasle.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov2_baseline_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval \
  --device-ids 0 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

读取上一步的预测并汇总指标，输出到 `../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov2_baseline_4/eval/metrics_long.json \
  --reuse-existing
```

### 1.3 使用神经元探测的训练与原论文测试集评测

#### 第一步：构造真假帧对

从训练标注和已抽取视频帧生成神经元探测所需的真假帧对，输出到 `../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl`；文件已存在时直接复用。

```bash
python DeMamba/build_probe_pairs.py \
  --annotations /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl \
  --fps 8 \
  --strict \
  --reuse-existing
```

#### 第二步：探测 DINOv2 神经元

读取上一步的真假帧对并计算敏感神经元，输出得分和索引到 `../MSLoc_data/DeMamba/neuron_probe_dinov2/`；中断后继续，完成后直接复用。

```bash
python DeMamba/probe_dinov2_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov2/train_pairs.jsonl \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --dinov2-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov2_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov2 \
  --image-batch-size 32 \
  --crop-youku \
  --amp \
  --strict \
  --resume
```

#### 第三步：训练 DINOv2 + DeMamba

读取上一步的神经元索引进行训练，checkpoint 输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

#### 第四步：评测原论文测试集

读取训练得到的 `best_acc.pth` 生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov2_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval \
  --device-ids 0 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

#### 第五步：汇总原论文指标

读取上一步的 `predictions.json` 计算最终指标，输出到 `../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval/metrics_long.json \
  --reuse-existing
```

### 1.4 评测新 benchmark：ActivityForensics

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov2_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics \
  --device-ids 0 \
  --batch-size 16 \
  --clean
```

DINOv2 使用配置中的 `image_size: 196` 和 `normalization: dinov2`；评测器会自动
按该配置处理原始视频。数据尚未下载完整时会只评测当前可用视频，并在输出中标记
`evaluation_scope: partial`；全量下载完成后同一命令会自动评测完整测试集。

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 2. DINOv3

### 2.1 权重下载

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir "../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf"
```

### 2.2 不使用神经元探测的基线训练与评测

该基线直接把 DINOv3 最后一层去除 CLS 和 register token 后的全部 patch-token 隐藏特征输入 DeMamba，不读取神经元索引。

训练结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv3_Tasle.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

读取训练得到的 `best_acc.pth` 评测原论文测试集，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_baseline_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval \
  --device-ids 0 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

读取上一步的预测并汇总指标，输出到 `../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov3_baseline_4/eval/metrics_long.json \
  --reuse-existing
```

### 2.3 使用神经元探测的训练与原论文测试集评测

#### 第一步：构造真假帧对

从训练标注和已抽取视频帧生成神经元探测所需的真假帧对，输出到 `../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl`；文件已存在时直接复用。

```bash
python DeMamba/build_probe_pairs.py \
  --annotations /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/annos/train_all_1209.json \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --output ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --fps 8 \
  --strict \
  --reuse-existing
```

#### 第二步：探测 DINOv3 神经元

读取上一步的真假帧对并计算敏感神经元，输出得分和索引到 `../MSLoc_data/DeMamba/neuron_probe_dinov3/`；中断后继续，完成后直接复用。

```bash
python DeMamba/probe_dinov3_neurons.py \
  --pairs ../MSLoc_data/DeMamba/neuron_probe_dinov3/train_pairs.jsonl \
  --frame-root /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/0826/dataset/Tasle-CoT-10K/video_frames \
  --dinov3-backend huggingface \
  --dinov3-hf-model-path ../MSLoc_data/DeMamba/pretrained_weights/dinov3_hf \
  --output-dir ../MSLoc_data/DeMamba/neuron_probe_dinov3 \
  --image-batch-size 32 \
  --crop-youku \
  --amp \
  --strict \
  --resume
```

#### 第三步：训练 DINOv3 + DeMamba

读取上一步的神经元索引进行训练，checkpoint 输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/`；中断后从最新 epoch 继续。

```bash
python DeMamba/train.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --device-ids 0,1,2,3,4,5,6,7 \
  --train-batch-size 32 \
  --val-batch-size 32 \
  --max-epoch 10 \
  --seed 42 \
  --resume-from-checkpoint auto
```

#### 第四步：评测原论文测试集

读取训练得到的 `best_acc.pth` 生成预测，结果输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/`；中断后继续，完成后直接复用。

```bash
python DeMamba/eval.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model_path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval \
  --device-ids 0 \
  --val-batch-size 16 \
  --cache-data \
  --resume \
  --reuse-existing
```

#### 第五步：汇总原论文指标

读取上一步的 `predictions.json` 计算最终指标，输出到 `../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/metrics_long.json`；文件已存在时直接复用。

```bash
python evaluate_long.py \
  --gt_file ../MSLoc_data/test_all_1209_0119_long.json \
  --infer_file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/predictions.json \
  --output_file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval/metrics_long.json \
  --reuse-existing
```

### 2.4 评测新 benchmark：ActivityForensics

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv3_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov3_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics \
  --device-ids 0 \
  --batch-size 16 \
  --clean
```

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov3_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 3. 原 XCLIP 评测新 benchmark：ActivityForensics

```bash
python DeMamba/eval_activityforensics.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --model-path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics \
  --device-ids 0 \
  --batch-size 16 \
  --clean
```

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 4. Baseline 评测新 benchmark：ActivityForensics

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/XCLIP_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/all_Class_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics \
  --device-ids 0 \
  --batch-size 16 \
  --clean
```

完成上述评测后，运行proposal质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

三个评测都会输出 TASLE 风格的 `Det_Acc`、`F1Det`、`F1Loc`，以及
ActivityForensics 的 AP/AR，并分别汇总 all、in-domain、out-of-domain 和各生成器结果。
`predictions.json` 同时保存真实时序标注和模型预测，因此可同时作为proposal质量评测的
`--gt-file` 与 `--infer-file`。
