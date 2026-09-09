# DINOv2 ViT-B/14 + DeMamba：下载与运行

所有命令均从项目根目录 `MSLoc_code` 执行

## 1. 下载 DINOv2 权重

```bash
hf download facebook/dinov2-base \
  --local-dir ../MSLoc_data/DeMamba/pretrained_weights/dinov2_hf
```


## 2. 正式神经元探测、训练与原论文测试集评测

```bash
bash DeMamba/run_dinov2_neuron_pipeline.sh
```

## 3. DINOv2 评测新 benchmark：ActivityForensics

下面的命令直接读取 ActivityForensics 原始视频，不需要预先抽帧。配置中的
`dinov2_hf_model_path`、神经元索引路径和 checkpoint 路径与
`DeMamba/run_dinov2_neuron_pipeline.sh` 保持一致。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/DINOv2_Tasle_neurons.yaml \
  --model-path ../MSLoc_data/DeMamba/results/dinov2_neurons_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics \
  --device-ids 0 \
  --batch-size 8 \
  --clean
```

DINOv2 使用配置中的 `image_size: 196` 和 `normalization: dinov2`；评测器会自动
按该配置处理原始视频。数据尚未下载完整时会只评测当前可用视频，并在输出中标记
`evaluation_scope: partial`；全量下载完成后同一命令会自动评测完整测试集。

完成上述评测后，运行时序提案质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/results/dinov2_neurons_4/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 4. 原 XCLIP 评测新 benchmark：ActivityForensics

```bash
python DeMamba/eval_activityforensics.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --model-path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics \
  --device-ids 0 \
  --batch-size 8 \
  --clean
```

完成上述评测后，运行时序提案质量评测：

```bash
python evaluate_proposal_quality.py \
  --gt-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --infer-file ../MSLoc_data/DeMamba/full/method/eval_activityforensics/predictions.json \
  --output-dir ../MSLoc_data/DeMamba/full/method/eval_activityforensics/proposal_quality \
  --iou-thresholds 0.1,0.3,0.5,0.7 \
  --domain-key tool_domain
```

## 5. Baseline 评测新 benchmark：ActivityForensics

Baseline 使用 `DeMamba/configs/XCLIP_Tasle.yaml`，checkpoint 使用
`../MSLoc_data/DeMamba/results/all_Class_4/best_acc.pth`。

```bash
python DeMamba/eval_activityforensics.py \
  --config DeMamba/configs/XCLIP_Tasle.yaml \
  --model-path ../MSLoc_data/DeMamba/results/all_Class_4/best_acc.pth \
  --annotation-dir ../MSLoc_data/ActivityForensics \
  --video-root ../MSLoc_data/ActivityForensics \
  --output-dir ../MSLoc_data/DeMamba/results/all_Class_4/eval_activityforensics \
  --device-ids 0 \
  --batch-size 8 \
  --clean
```

完成上述评测后，运行时序提案质量评测：

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
`predictions.json` 同时保存真实时序标注和模型预测，因此可同时作为提案质量评测的
`--gt-file` 与 `--infer-file`。提案质量评测最后首先打印 `Recall`（真实伪造时长被提案
覆盖的比例），然后打印 Union temporal IoU、Under-coverage 和 Over-coverage。
