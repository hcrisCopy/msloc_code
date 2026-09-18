# 第二阶段运行说明

所有命令都在 `msloc_code` 根目录运行，路径均为相对路径。第二阶段只接收第一阶段的训练集、测试集 proposal JSON；其余输入是数据集标注、视频和 TRACE 基础权重。

## 流程

```text
同一 TRACE 基础权重
  ├─ 训练 proposal + 待检视频 ─> Student SFT ─> 测试
  └─ 训练 proposal + 上真下待检视频 ─> Teacher SFT ─> paired 测试

训练 proposal 上逐条比较 Student 与 Teacher
  └─ 只保留 Teacher 二分类纠错成功或正样本定位 IoU 严格提升的样本
       └─ OPD ─> 测试 ─> GRPO ─> 测试
```

## 0. 环境、八卡与恢复

```bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
hf download cross-encoder/nli-deberta-v3-small \
  --local-dir ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small
```

类别特征放在 `../MSLoc_data/Trace/class_features_bge.pt`。第一阶段需要先生成训练集和测试集 proposal：

```text
../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
../MSLoc_data/DeMamba/full/method/eval/predictions.json
```

两份文件分别对应 `train_all_1209.json` 和 `test_all_1209_0119.json`，也是第二阶段从第一阶段接收的产物。测试 proposal 是第一阶段按 `_0119` 版测试标注生成的，因此第二阶段构建测试样本和三次 Student 测试都必须继续使用同一版 GT。

- 首次运行用 `--clean`；恢复时删除 `--clean`。
- 训练恢复：把 `--resume none` 改为 `--resume auto`，也可传具体 checkpoint。
- 样本构建恢复：加入 `--resume`；预检恢复：改为 `--resume auto`；测试恢复：加入 `--resume`。
- 单卡正式检查只需把 `--devices` 改为 `0`、`--nproc-per-node` 改为 `1`，其余训练参数不变。测试只把 `--devices` 改为 `0`。
- 八卡时每张卡加载一份完整模型并分担数据；训练、预检和测试显示进度并在主进程打印指标，不使用 WandB。

训练输出目录会保留最终权重、按 epoch 保存的 `checkpoint-*` 和 `trainer_state.json`，其中包含 loss、OPD/GRPO 奖励等聚合日志。正式测试目录会保留逐视频的 refine 结果，包括片段边界、解释文本、严格解析状态和原始生成 token，以及汇总的 `metrics.json`；Teacher 预检还会在 `teacher_precheck.json` 中保留 Student/Teacher 的逐 proposal 输出。正式命令也会保存 OPD 和 GRPO 的每条在线 rollout，便于核对模型输出及分项奖励。

## 1. Student SFT

输入是 TRACE 基础权重、第一阶段训练 proposal、训练标注和待检视频。模型只看 proposal 对应的待检片段，学习判断 fake/no forgery、细化伪造边界并生成解释。

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

输出权重：`../MSLoc_data/Trace/experiments/opd_grpo/student_sft/`，供 Student 测试、训练集预检和 OPD 初始化使用。

## 2. 测试 Student SFT

使用第 1 步权重 refine 第一阶段测试 proposal，再与测试 GT 计算完整测试集指标。已有兼容的 Student SFT 权重时，可直接替换 `--model`。

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/student_sft \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
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

输出：`../MSLoc_data/Trace/inference_results/student_sft_test/`；汇总指标保存在其中的 `metrics.json`。

## 3. 构建三份样本

这一步把第一阶段 proposal 和数据集 GT 整理成后续训练需要的格式：

- `../MSLoc_data/Trace/experiments/opd_grpo/train_paired_samples.json`：训练 proposal 中有真实参考视频的样本，供 Teacher SFT 和训练集预检使用。
- `../MSLoc_data/Trace/experiments/opd_grpo/test_paired_samples.json`：测试 proposal 中有真实参考视频的样本，供 Teacher SFT 评测使用。
- `../MSLoc_data/Trace/experiments/opd_grpo/grpo_training_samples.json`：全部训练 proposal，供 GRPO 使用。

paired 样本根据标注映射或同路径 `*_real.mp4` 查找真实参考，并检查文件是否存在。正、负 proposal 都保留。

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
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
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

## 4. Teacher SFT

Teacher 用的是和 Student 同一份的 TRACE 基础权重训练。输入使用第 3 步的 `train_paired_samples.json`：每个时间点同时提供对齐的真实参考画面和待检画面，训练目标仍是对待检 proposal 做真假判断、边界 refine 和解释。

真实参考和待检视频先在相同时间戳各采样 40 帧，两个输入张量都是 `[40, 3, 336, 336]`。每个时间点在逻辑上组成“上方真实参考、下方待检”的 `[3, 672, 336]` 画面；进入冻结 CLIP 时沿高度拆回两个 `[3, 336, 336]` 视图分别编码，避免让 CLIP 直接处理双倍高度画面。CLIP-ViT-L/14-336 为每个视图产生 24×24=576 个 patch token。

在第 `t` 个时间点，代码先给 reference 的 576 个 token 加同一个可学习 reference 身份向量，再给 candidate 的 576 个 token 加另一个可学习 candidate 身份向量，然后按 `[reference tokens, candidate tokens]` 组成 1152 个 token。因此进入 `ref_projector` 的形状是 `[B, 40, 1152, D]`；两个身份向量的形状仅为 `[1, 1, 1, D]`，新增参数和显存很小。身份向量只在 paired 输入出现时使用，普通 Student、OPD 输出模型、GRPO 和正式测试的 576-token candidate-only 输入不受影响。

这里增加的是每帧内部的对照信息，时间维仍然是 40，而不是先输入 40 帧 reference、再输入 40 帧 candidate。`ref_projector` 仍按同一时间轴处理“左边界 16 帧 + proposal 内部 8 帧 + 右边界 16 帧”，所以 reference 第 `t` 帧和 candidate 第 `t` 帧始终直接对应；模型不需要在 80 帧中重新寻找配对关系。提示词进一步说明两个视图的含义，最终真假判断、边界和解释只针对 candidate。

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

输出权重：`../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft/`，供 Teacher 测试、训练集预检和 OPD 冻结教师使用。

## 5. 测试 Teacher SFT

输入是第 3 步的测试 paired proposal、第 4 步 Teacher 权重和测试 GT。Teacher 同时看时间对齐的真实参考与待检片段，对第一阶段测试 proposal 做真假判断和边界 refine，再用与 Student、OPD、GRPO 测试相同的评测程序计算指标。

```bash
python Trace/run_opd_grpo.py test-teacher \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --test-samples ../MSLoc_data/Trace/experiments/opd_grpo/test_paired_samples.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
  --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
  --teacher-model ../MSLoc_data/Trace/experiments/opd_grpo/teacher_sft \
  --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
  --output ../MSLoc_data/Trace/inference_results/teacher_sft_test \
  --metrics-output ../MSLoc_data/Trace/inference_results/teacher_sft_test/metrics.json \
  --prompt-file Trace/trace/prompts/dvc.txt \
  --version v1_mistral \
  --bnd-ratio 0.2 \
  --bnd-frames 16 \
  --seg-frames 8 \
  --max-new-tokens 128 \
  --teacher-iou-gate 0.3 \
  --max-samples 0 \
  --resume none \
  --clean
```

主评测结果保存在 `../MSLoc_data/Trace/inference_results/teacher_sft_test/metrics.json`，指标与其他测试完全相同，都是 `Det_Acc`、`Loc_F1` 和 `Loc_IoU`。区别只在数据范围：Teacher 必须有真实参考视频，所以在测试集的 paired 子集上评测；`proposal_report.json` 另外保存 proposal 级真假、IoU 和格式诊断，不替代统一指标。

## 6. 训练集预检与筛选

输入是第 3 步的训练 paired proposal，以及已经训练好的 Student 和 Teacher。Student 只输入 proposal 对应的待检片段；Teacher 输入同一时间段的真实参考片段和待检片段。两者都输出对该 proposal 的 fake/no forgery 判断、refine 后的边界和解释，再用训练 GT 判断谁的结果更好：

- 正 proposal：Student 输出 `No forgery.` 或格式错误而 Teacher 输出 fake，或者两者都输出 fake 且 Teacher IoU 更高。
- 负 proposal：Teacher 正确输出 no-event，而 Student 输出事件或格式失败。
- 其余 proposal 不进入 OPD。

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

输出包括完整对比报告 `../MSLoc_data/Trace/experiments/opd_grpo/teacher_precheck.json` 和筛选结果 `../MSLoc_data/Trace/experiments/opd_grpo/opd_selected_samples.json`。后者是 OPD 的训练集。

## 7. OPD

OPD 从第 1 步 Student 权重开始训练，第 4 步 Teacher 权重保持冻结。训练只读取第 6 步筛选出的 proposal：Student 看待检片段，Teacher 看同时间点的真实参考与待检片段，利用 Teacher 更好的真假判断和定位结果蒸馏 Student。

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
  --save-rollouts true \
  --rollout-output ../MSLoc_data/Trace/experiments/opd_grpo/opd/rollouts \
  --clean
```

输出权重：`../MSLoc_data/Trace/experiments/opd_grpo/opd/`，供下一步测试和 GRPO 初始化使用。逐条训练记录保存在 `opd/rollouts/opd_rank*.jsonl`：每条包含 proposal、Student 生成结果、解析状态、Teacher 预检结果、错误类型、蒸馏权重和 Student/Teacher reverse-KL。八卡各写一个文件，不会并发写同一文件；断点继续训练时会追加，并按 `audit_key` 跳过已经记录的 rollout。

## 8. 测试 OPD

这一步评测 OPD 训练后的学生模型。模型只看待检视频，对第一阶段生成的完整测试 proposal 做真假判断和边界 refine，再与测试 GT 计算 `Det_Acc`、`Loc_F1` 和 `Loc_IoU`。

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/opd \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
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

输出：`../MSLoc_data/Trace/inference_results/opd_test/`；汇总指标保存在其中的 `metrics.json`。

## 9. GRPO

GRPO 从第 7 步 OPD 权重继续训练，读取第 3 步的全部训练 proposal。Student 仍只看待检视频；同一个 proposal 生成 4 个回答，并按下面三项奖励比较：

- 定位奖励：负 proposal 正确回答 `No forgery.` 得正分；正 proposal 必须输出 fake 片段，再按片段 IoU 和起止边界准确度得分，多报片段会扣分。
- 格式奖励：回答能被严格解析为合法 TRACE 事件或 `No forgery.` 得 `+1`，格式错误得 `-1`。
- 解释奖励：从训练标注中读取与 SFT 相同的解释事实：`obj_cot[0]` 是主要异常，非 Round4 再加入 `bnd_cot_st[0]` 和 `bnd_cot_ed[0]` 作为开始、结束事实。生成解释先按句子拆成若干 claim，冻结的 entailment 模型计算每条 claim 对每条事实的“支持”和“矛盾”概率，再按支持概率做一对一匹配；一条生成句子最多解释一条事实，一条事实也只能被覆盖一次。事实覆盖越完整、生成句子中有标注依据的比例越高，奖励越高。矛盾只检查一对一匹配后的对应事实，例如结束句只与匹配到的结束事实判断，不再因为它和主要异常描述的是不同阶段而误扣分。为避免边界没找准却靠复述标注得分，只对正样本且定位 IoU 不低于 `0.3` 的回答计算解释奖励。

总奖励是三项的加权和，正式命令中的权重依次为 `1.0`、`0.1` 和 `0.3`（定位、格式、解释）。

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
  --entailment-model-path ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small \
  --entailment-device cuda \
  --entailment-batch-size 32 \
  --structure-aware true \
  --kl-coef 0.02 \
  --sft-coef 0.1 \
  --save-rollouts true \
  --rollout-output ../MSLoc_data/Trace/experiments/opd_grpo/grpo/rollouts \
  --clean
```

输出权重：`../MSLoc_data/Trace/experiments/opd_grpo/grpo/`。逐条训练记录保存在 `grpo/rollouts/grpo_rank*.jsonl`：每个回答都保存原始 token、解析后的真假/片段/解释、匹配到的解释事实、定位/格式/解释/总奖励及各自 advantage。八卡分别写文件，断点恢复时继续追加并去重。

输出权重：`../MSLoc_data/Trace/experiments/opd_grpo/grpo/`。训练日志分别打印定位、解释和格式奖励，以及解释覆盖率、精确率和矛盾分。

## 10. 测试 GRPO

这一步评测 GRPO 训练后的最终学生模型。模型仍然只看待检视频，对同一份第一阶段测试 proposal 做真假判断和边界 refine，再与测试 GT 计算最终的 `Det_Acc`、`Loc_F1` 和 `Loc_IoU`，用于和 Student SFT、OPD 的结果直接比较。

```bash
python Trace/run_opd_grpo.py test \
  --devices 0,1,2,3,4,5,6,7 \
  --model ../MSLoc_data/Trace/experiments/opd_grpo/grpo \
  --proposals ../MSLoc_data/DeMamba/full/method/eval/predictions.json \
  --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
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

输出：`../MSLoc_data/Trace/inference_results/grpo_test/`；汇总指标保存在其中的 `metrics.json`。
