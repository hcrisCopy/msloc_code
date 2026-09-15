# 第二阶段运行说明

所有命令都在 `msloc_code` 根目录运行，路径均为相对路径。第二阶段只接收第一阶段的训练集、测试集 proposal JSON；其余输入是数据集标注、视频和 TRACE 基础权重。

## 正确流程

```text
同一 TRACE 基础权重
  ├─ 训练 proposal + 待检视频 ─> Student SFT ─> 测试
  └─ 训练 proposal + 上真下待检视频 ─> Teacher SFT ─> paired 测试

训练 proposal 上逐条比较 Student 与 Teacher
  └─ 只保留 Teacher 二分类纠错成功或正样本定位 IoU 严格提升的样本
       └─ OPD ─> 测试 ─> GRPO ─> 测试
```

教师输入不是把两幅画面压成半高，也不是沿时间前后拼接。每个时间点构造“上方真实参考、下方待检”的逻辑画面；冻结 CLIP 分别编码两个原尺寸 336×336 视图，再按上下顺序拼 patch token。这样保留空间细节和逐帧对齐关系，也避免直接编码 672×336 长图带来的自注意力显存增长。学生、OPD 输出模型、GRPO 和正式测试始终只看待检视频。

现有 `ref_projector`、CLoss、OPD disagreement focusing、guided rollout、正负 anchor 和 structure-aware GRPO 全部保留。当前工作树和全部 Git 历史中没有名为 DMA、EMA、LAA 的实现或参数；本次没有删除它们，也没有根据缩写臆造代码。若它们来自另一分支，需要拿到那份代码后再合并。

## 0. 环境、八卡与恢复

```bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
hf download cross-encoder/nli-deberta-v3-small \
  --local-dir ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small
```

类别特征放在 `../MSLoc_data/Trace/class_features_bge.pt`。

- 首次运行用 `--clean`；恢复时删除 `--clean`。
- 训练恢复：把 `--resume none` 改为 `--resume auto`，也可传具体 checkpoint。
- 样本构建恢复：加入 `--resume`；预检恢复：改为 `--resume auto`；测试恢复：加入 `--resume`。
- 单卡正式检查只需把 `--devices` 改为 `0`、`--nproc-per-node` 改为 `1`，其余训练参数不变。测试只把 `--devices` 改为 `0`。
- DDP 每个 rank 在一张卡加载完整模型并处理约八分之一数据。代码不自动缩小 batch、帧数或模型。训练、预检和测试均有 tqdm；主进程保留指标打印；固定禁用 WandB。

第一阶段必须已经生成：

```text
../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
../MSLoc_data/DeMamba/full/method/eval/predictions.json
```

它们分别对应 `train_all_1209.json` 和 `test_all_1209.json`。

## 1. Student SFT

```bash
python Trace/run_opd_grpo.py student-sft \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --version v1_mistral \
  --mm-projector-type ref_projector \
  --closs true \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone false \
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
  --run-name student_sft \
  --max-samples 0 \
  --resume none \
  --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/student_sft/`。

## 2. 测试 Student SFT

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/student_sft_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/student_sft_test/metrics.json \
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

输出目录包含各卡进度、合并推理 JSON、严格解析状态和 `metrics.json`。

## 3. 构建三份样本

paired 样本只保留存在同时间轴真实参考的 fake 源视频，但同时保留其正、负 proposal；GRPO 样本保留全部 proposal，包括真实视频误报。

测试集是否带有可用的真实参考，不能只根据标注文件名推断。下面第二条命令会逐条检查标注中的参考路径或同目录 `*_real.mp4` 是否真实存在；只有检查通过，才能执行第 5 节的 Teacher paired 测试。若构建结果为空或报告缺失参考，应跳过 Teacher 测试，不能伪造参考视频，也不能拿训练集参考替代测试集参考。

```bash
python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/train_paired_samples.json \
  --near-negative-seconds 1.0 \
  --max-records 0 \
  --paired-only \
  --require-reference \
  --clean

python Trace/run_opd_grpo.py build-samples \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/test_paired_samples.json \
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

构建过程每 100 个 proposal 源记录原子写入一次 `.progress.json`，完成后自动删除。

## 4. Teacher SFT

教师必须从与学生相同的 `trace-uni` 基础权重独立训练，不能从 Student SFT 接着训练。

```bash
python Trace/run_opd_grpo.py train-paired-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/train_paired_samples.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --base-model ../MSLoc_data/Trace/ckpts/trace-uni \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft \
  --version v1_mistral \
  --mm-projector-type ref_projector \
  --closs true \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
  --freeze-mm-mlp-adapter false \
  --tune-mm-mlp-adapter true \
  --tune-mm-embed-head true \
  --tune-lm-embed-head true \
  --freeze-backbone false \
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
  --run-name teacher_sft \
  --max-samples 0 \
  --resume none \
  --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft/`。

## 5. 测试 Teacher SFT

仅当第 3 节确认测试 split 的真实参考文件存在时，才执行本节。教师输入与部署模型不同，所以用测试 paired proposal 单独报告 proposal 级真假/定位指标。`--enforce-better false` 只保存结果，不用测试集做训练筛选。

```bash
python Trace/run_opd_grpo.py check-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/test_paired_samples.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/teacher_sft_test/report.json \
  --selected-output ../MSLoc_data/Trace/inference_results/teacher_sft_test/teacher_better_samples.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --max-samples 0 \
  --enforce-better false \
  --resume none \
  --clean
```

`report.json` 分开给出两类指标：`binary_accuracy` 只判断 fake / no forgery；正样本的 `positive_mean_iou` 和 `positive_localized_rate` 再衡量片段定位。`joint_binary_localization_accuracy` 仅作辅助报告，其中正样本使用显式传入的 `--teacher-iou-gate 0.3`，这个阈值不参与 OPD 样本筛选。

## 6. 训练集预检与筛选

- 正 proposal：如果 Student 输出 `No forgery.` 或格式错误，而 Teacher 正确输出 fake，则按二分类纠错选中；如果两者都输出 fake，只要 Teacher IoU 严格大于 Student IoU 就选中，不再设置额外 IoU 增量门槛。
- 负 proposal：Teacher 正确输出 no-event，而 Student 输出事件或格式失败。
- Teacher 不可靠、只打平或更差的样本不会进入 OPD。

```bash
python Trace/run_opd_grpo.py check-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/train_paired_samples.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/teacher_precheck.json \
  --selected-output ../MSLoc_data/Trace/experiments/opd_grpo/opd_selected_samples.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --max-samples 0 \
  --enforce-better true \
  --resume none \
  --clean
```

如果没有任何 Teacher 严格优于 Student 的样本，报告仍保存，但命令退出且不能继续 OPD。这里不再设置整体提升门槛。OPD 只读 `opd_selected_samples.json`。

## 7. OPD

```bash
python Trace/run_opd_grpo.py opd \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --training-samples ../MSLoc_data/Trace/experiments/opd_grpo/opd_selected_samples.json \
  --student-model ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft \
  --teacher-check-result ../MSLoc_data/Trace/experiments/opd_grpo/teacher_precheck.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --deepspeed Trace/scripts/zero2.json \
  --output ../MSLoc_data/Trace/experiments/opd_grpo/opd \
  --version v1_mistral \
  --mm-projector-type ref_projector \
  --closs true \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
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

## 8. 测试 OPD

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/opd \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/opd_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/opd_test/metrics.json \
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

## 9. GRPO

三种奖励是定位、TRACE 格式和解释。解释奖励删除 lexical 路径，也不再依赖手写同义词表：将对象异常、开始、结束标注拆成最多三个带关系事实，用冻结 entailment cross-encoder 对生成的原子 claim 做一对一匹配，计算事实覆盖率、claim 精确率，并惩罚矛盾、重复和超长内容。只有正样本定位 IoU≥0.3 时才启用解释分。正式实验沿用 SFT 使用的人工解释标注作为监督，因此不额外要求 `candidate_observable` 审计；如果以后提供逐条可观察性审计，再把该开关设为 true。

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
  --output ../MSLoc_data/Trace/experiments/opd_grpo/grpo \
  --version v1_mistral \
  --mm-projector-type ref_projector \
  --closs true \
  --class-feature-path ../MSLoc_data/Trace/class_features_bge.pt \
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
  --run-name grpo \
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
  --text-max-words 80 \
  --require-candidate-observable false \
  --entailment-model-path ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small \
  --entailment-device cuda \
  --entailment-batch-size 32 \
  --structure-aware true \
  --kl-coef 0.02 \
  --sft-coef 0.1 \
  --clean
```

日志关注 `grpo_loc_reward`、`grpo_exp_reward`、`grpo_fmt_reward`、`grpo_group_std`、`grpo_text_graph_precision`、`grpo_text_graph_recall` 和 `grpo_text_contradiction`。

## 10. 测试 GRPO

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/grpo \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/grpo_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/grpo_test/metrics.json \
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

最终比较 Student SFT、Teacher SFT paired proposal 报告、OPD 和 GRPO。Student/OPD/GRPO 使用相同测试 proposal 和长视频指标；Teacher 输入不同，必须单列 paired proposal 指标，不能与部署模型的长视频指标混为一列。
