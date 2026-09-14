# 第二阶段训练与测试运行说明

第一阶段先用 DeMamba 生成疑似伪造的 proposal。第二阶段用这些 proposal 训练 TRACE：先根据标注学习定位和解释（SFT）；再让能看到真实参考视频的教师纠正只看待检测视频的学生（OPD）；最后根据定位、解释和输出格式的得分继续优化模型（GRPO）。如果现有模型不能从真实视频对照中得到帮助，就先重新训练一个能理解上下拼接画面的教师。

第二阶段只使用第一阶段在训练集、测试集上实际预测的 proposal，以及 TASLE-CoT-10K 的标注和视频。

所有命令都在 `msloc_code` 根目录运行，路径均为相对路径。正式实验默认单机 8 卡，每张卡加载一份完整模型并处理约八分之一数据。

## 流程总览

```text
训练集 proposal + 训练标注
  ├─> 只看待检测视频的基础模型 ──────────────────> OPD 学生初始权重
  ├─> 带真实视频对照的训练样本 ──────────────────> 教师检查
  │                                                └─不合格：从 trace-uni 重新训练上下拼接教师，再检查
  │                                                                                  └─> OPD 冻结教师
  └─> 包含全部 proposal 的训练样本 ──────────────> GRPO

测试集 proposal + GRPO 模型 ───────────────────> 最终测试结果
```

第一阶段交给第二阶段的文件只有训练集、测试集 proposal JSON。第二阶段另外读取数据集标注和视频，不需要第一阶段的模型内部特征或训练状态。真实参考视频只供教师重训、教师检查和 OPD 教师使用，不会出现在 OPD 学生、GRPO 或测试输入中。

## 0. 运行前准备

DeMamba 使用 `msloc` 环境，TRACE 的训练和测试使用 `trace` 环境。

```bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
hf download cross-encoder/nli-deberta-v3-small \
  --local-dir ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small
```

- 第一次运行使用 `--clean` 清理这个步骤以前的输出。恢复中断任务时必须去掉 `--clean`。
- 恢复训练：把 `--resume none` 改为 `--resume auto`。
- 恢复教师检查：把 `--resume none` 改为 `--resume auto`。
- 恢复测试：加入 `--resume`。

## 1. 检查或生成训练集、测试集的 proposal

先检查下面两个文件是否已经生成：

```bash
ls -lh \
  ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  ../MSLoc_data/DeMamba/full/method/eval/predictions.json
```

如果文件存在且内容完整，还要确认：

- `eval_train/predictions.json` 来自当前 DeMamba 权重对 `train_all_1209.json` 的推理；
- `eval/predictions.json` 来自当前 DeMamba 权重对 `test_all_1209.json` 的推理。

两者都符合时，直接从第 2 节开始。缺少哪个文件，或者文件对应的权重、数据划分不对，就只运行下面相应的命令。

生成训练集 proposal：

```bash
conda activate msloc
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json \
  --dataset-base-path ../MSLoc_data/DeMamba/video_frames \
  --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/method/eval_train \
  --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --num-workers 4 \
  --save-progress \
  --cache-data \
  --clean
```

生成测试集 proposal：

```bash
python DeMamba/eval.py \
  --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml \
  --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json \
  --dataset-base-path ../MSLoc_data/DeMamba/video_frames \
  --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth \
  --output_dir ../MSLoc_data/DeMamba/full/method/eval \
  --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json \
  --device-ids 0,1,2,3,4,5,6,7 \
  --val-batch-size 16 \
  --num-workers 4 \
  --save-progress \
  --cache-data \
  --clean
```

中断恢复时去掉 `--clean`，加入 `--resume`。输出：

```text
../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
../MSLoc_data/DeMamba/full/method/eval/predictions.json
```

## 2. 训练只看待检测视频的基础模型

这一步把每个训练集 proposal 对应的视频片段交给模型，再根据数据集标注准备正确答案：proposal 与伪造区间有重叠时，答案包含片段内的伪造起止时间、类型和解释；没有重叠时，答案是“没有伪造”。SFT 就是让模型反复学习这些“视频片段—正确答案”样本，使它只看待检测视频也能按规定格式完成判断、定位和解释。训练结果既用于首次教师检查，也是后续 OPD 学生模型的初始权重。

```bash
conda activate trace
python Trace/run_opd_grpo.py sft \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft \
  --version v1_mistral \
  --mm-projector-type spatial_slot \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone true \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --num-frames 40 \
  --mm-vision-select-layer -2 \
  --mm-use-im-start-end false \
  --mm-use-im-patch-token false \
  --downsample-num 1 \
  --image-aspect-ratio pad \
  --bf16 true \
  --tf32 false \
  --fp16 false \
  --epochs 2 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --grad-accum 2 \
  --learning-rate 5e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --save-strategy epoch \
  --save-steps 0 \
  --save-total-limit 99 \
  --model-max-length 4096 \
  --gradient-checkpointing true \
  --num-workers 4 \
  --sample-scheme rand \
  --run-name ref2_sft \
  --max-samples 0 \
  --resume none \
  --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft/`。

## 3. 整理后续训练所需的样本文件

需要从训练集 proposal 和标注中整理出两个 JSON 文件：

- `opd_training_samples.json`：只保留能找到真实参考视频的样本，记录待检测视频、真实参考视频、proposal 和标注，用于教师检查、教师重训和 OPD。
- `grpo_training_samples.json`：保留全部有效 proposal，包括真实视频上的误报，用于 GRPO。

```bash
python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/opd_training_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 0 \
  --paired-only \
  --require-reference \
  --clean

python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/grpo_training_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 0 \
  --clean
```

输出：

```text
../MSLoc_data/Trace/experiments/opd_grpo/opd_training_samples.json
../MSLoc_data/Trace/experiments/opd_grpo/grpo_training_samples.json
```

## 4. 检查教师模型；不合格时重新训练

### 4.1 先检查第 2 步得到的模型

检查时不更新模型权重。同一个模型会分别处理两种输入：只看待检测视频，以及同时看“上方真实参考视频、下方待检测视频”的拼接画面。只有加入真实参考视频后定位效果确实改善，并且没有明显增加误报，检查才算通过。

```bash
python Trace/run_opd_grpo.py check-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/opd_training_samples.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/teacher_check_result.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --minimum-recovery-improvement 0.01 \
  --minimum-reliable-positive-rate 0.05 \
  --maximum-negative-noevent-drop 0.02 \
  --max-samples 0 \
  --resume none \
  --clean
```

如果检查通过，跳过 4.2。后续 OPD 的 `--teacher-model` 使用 `../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft`。

如果检查失败，程序仍会保存检查报告，但 OPD 会拒绝使用这份结果。此时继续执行 4.2。

### 4.2 从最初权重重新训练上下拼接教师

教师必须从最初的 `trace-uni` 权重重新训练，不能从第 2 步的模型继续训练。训练标签不变，但输入改成“上方真实参考视频、下方待检测视频”，文字提示也会明确说明上下画面的含义。

这个模型以后只作为冻结教师使用。OPD 学生仍从第 2 步的模型开始训练。

```bash
python Trace/run_opd_grpo.py train-paired-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/opd_training_samples.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_paired_sft \
  --version v1_mistral \
  --mm-projector-type spatial_slot \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone true \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --num-frames 40 \
  --mm-vision-select-layer -2 \
  --mm-use-im-start-end false \
  --mm-use-im-patch-token false \
  --downsample-num 1 \
  --image-aspect-ratio pad \
  --bf16 true \
  --tf32 false \
  --fp16 false \
  --epochs 2 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --grad-accum 2 \
  --learning-rate 5e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --save-strategy epoch \
  --save-steps 0 \
  --save-total-limit 99 \
  --model-max-length 4096 \
  --gradient-checkpointing true \
  --num-workers 4 \
  --sample-scheme rand \
  --run-name opd_teacher_paired_sft \
  --max-samples 0 \
  --resume none \
  --clean
```

训练完成后，重新运行 4.1 的检查命令，只改教师权重目录：

```text
--teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_paired_sft
```

保留 `--clean`，清掉上一次失败报告和旧检查进度。新的检查结果保存在 `../MSLoc_data/Trace/experiments/opd_grpo/teacher_check_result.json`。

## 5. 用合格教师训练 OPD 学生模型

下面的命令按“首次检查失败，重新训练的教师已经通过检查”填写。学生只看待检测视频，教师看上下拼接画面。

如果 4.1 首次检查已经通过，只把 `--teacher-model` 改为 `../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft`。`--student-model` 始终使用第 2 步的结果。

```bash
python Trace/run_opd_grpo.py opd \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/opd_training_samples.json \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_paired_sft \
  --teacher-check-result ../MSLoc_data/Trace/experiments/opd_grpo/teacher_check_result.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/opd \
  --version v1_mistral \
  --mm-projector-type spatial_slot \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone true \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --num-frames 40 \
  --mm-vision-select-layer -2 \
  --mm-use-im-start-end false \
  --mm-use-im-patch-token false \
  --downsample-num 1 \
  --image-aspect-ratio pad \
  --bf16 true \
  --tf32 false \
  --fp16 false \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 4 \
  --grad-accum 4 \
  --learning-rate 2e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --save-strategy epoch \
  --save-steps 0 \
  --save-total-limit 99 \
  --model-max-length 4096 \
  --gradient-checkpointing true \
  --num-workers 4 \
  --sample-scheme rand \
  --run-name opd \
  --max-samples 0 \
  --resume none \
  --opd-weight 1.0 \
  --opd-temperature 1.0 \
  --opd-disagreement-iou-gate 0.3 \
  --false-refusal-weight 1.0 \
  --positive-error-weight 0.8 \
  --negative-error-weight 0.8 \
  --positive-anchor-weight 0.2 \
  --negative-anchor-weight 0.2 \
  --guided-positive-fraction 1.0 \
  --guided-alpha 0.5 \
  --guided-max-tokens 16 \
  --rollout-max-new-tokens 128 \
  --guided-loss-coef 0.25 \
  --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/opd/`。

## 6. 用定位和解释得分继续训练

GRPO 从第 5 步的学生模型开始训练，只看待检测视频，不再使用真实参考视频或教师模型。训练样本中保留真实视频误报，用来约束模型不能一味把所有 proposal 都判断为伪造。

文字解释奖励的重点如下：

- 只给定位正确的伪造样本计算解释奖励。输出格式必须正确，并且预测时间段与标注的 IoU 至少达到 `0.3`；否则解释写得再好也不得分。
- 评分依据来自标注中的物体异常、异常开始和异常结束等信息，不要求生成文字逐字复述标注。
- NLI 模式使用冻结的 NLI 模型识别同义表达并惩罚矛盾；这个模型只负责评分，不参与训练。
- 空泛描述、重复内容和过长解释会被扣分。真实视频或不与伪造片段相交的 proposal 不要求生成解释，只检查是否正确输出“无伪造”。

```bash
python Trace/run_opd_grpo.py grpo \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/grpo_training_samples.json \
  --starting-model ../MSLoc_data/Trace/experiments/opd_grpo/opd \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli \
  --version v1_mistral \
  --mm-projector-type spatial_slot \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone true \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --num-frames 40 \
  --mm-vision-select-layer -2 \
  --mm-use-im-start-end false \
  --mm-use-im-patch-token false \
  --downsample-num 1 \
  --image-aspect-ratio pad \
  --bf16 true \
  --tf32 false \
  --fp16 false \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 4 \
  --grad-accum 4 \
  --learning-rate 1e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --save-strategy epoch \
  --save-steps 0 \
  --save-total-limit 99 \
  --model-max-length 4096 \
  --gradient-checkpointing true \
  --num-workers 4 \
  --sample-scheme rand \
  --run-name grpo_nli \
  --max-samples 0 \
  --resume none \
  --group-size 4 \
  --temperature 0.7 \
  --max-new-tokens 128 \
  --clip-range 0.2 \
  --localization-weight 1.0 \
  --explanation-weight 0.3 \
  --format-weight 0.1 \
  --explanation-iou-gate 0.3 \
  --boundary-tolerance 1.0 \
  --text-reward-mode nli \
  --text-max-words 80 \
  --require-candidate-observable false \
  --nli-model-path ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small \
  --nli-device cuda \
  --nli-batch-size 32 \
  --structure-aware true \
  --kl-coef 0.02 \
  --sft-coef 0.1 \
  --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli/`。

训练时主要查看定位、解释、格式、组内差异和文字矛盾这几项：`grpo_loc_reward`、`grpo_exp_reward`、`grpo_fmt_reward`、`grpo_group_std`、`grpo_text_contradiction`。

若要使用不加载 NLI 模型的词面评分，把 `--text-reward-mode nli` 改为 `--text-reward-mode lexical`，删除三个 `--nli-*` 参数，并换一个输出目录。两种方法都应从同一个 OPD 权重开始，不能接着彼此的结果继续训练。

## 7. 测试最终模型

测试时模型只看第一阶段在测试集上预测的 proposal。每张卡加载一份完整模型并负责一部分视频；所有显卡完成后，程序自动合并结果。

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/grpo_nli_test \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --num-frames 40 \
  --max-new-tokens 512 \
  --batch-size 1 \
  --sample-num -1 \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --clean
```

最终结果：`../MSLoc_data/Trace/inference_results/grpo_nli_test/fmt_aigc_test_f40_result.json`。
