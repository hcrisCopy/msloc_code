# 从 DeMamba proposal 到 TRACE SFT、OPD、GRPO 的完整运行说明

本说明在 Linux、NVIDIA GPU 和 Bash 环境执行。

流程：第一阶段在训练/测试集生成真实 proposal；candidate-only ref2 SFT；
paired-only replay；冻结教师预检；OPD；GRPO；candidate-only 测试。

训练集 proposal 必须来自第一阶段在训练集的真实推理，不能由 GT 时间段替代。

## 0. 一次性准备环境与路径

DeMamba 和已有 TRACE 推理可继续使用现有 `msloc` 环境。本流程的 TRACE 训练脚本
（ref2 SFT、OPD、GRPO）启用了 DeepSpeed；仓库固定依赖为 `torch<2.2`、
`torchvision<0.17` 和 `deepspeed==0.13.1`。不要将这套训练依赖安装到使用
PyTorch 2.6 的 `msloc` 环境中。首次运行：

~~~bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
python -c "import torch, torchvision, deepspeed; print(torch.__version__, torchvision.__version__, deepspeed.__version__)"
~~~

从仓库根目录执行。脚本自动使用仓库同级的 `../MSLoc_data`：

~~~bash
source Trace/scripts/setup_opd_grpo_env.sh
~~~

所有 TRACE 训练脚本默认保持原有的按 epoch 保存；调试时显式设置 `SAVE_STEPS`。使用 `CLEAN=1` 清理对应输出目录；中断后移除 `CLEAN=1` 并使用 `RESUME_FROM_CHECKPOINT=auto` 续跑。

## 1. 使用已有 DeMamba checkpoint 生成训练集 proposal

先执行 `conda activate msloc`。

正式运行：

~~~bash
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --device-ids 0 --val-batch-size 16 --save-progress --cache-data --clean

export STAGE1_TRAIN_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
export STAGE1_TEST_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval/predictions.json
~~~

小样本调试：

~~~bash
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --max-eval-videos 3 --save-progress --cache-data --num-workers 0 --device-ids 0 --val-batch-size 1 --clean

export STAGE1_TRAIN_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
export STAGE1_TEST_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval/predictions.json
~~~

中断后续跑时移除 `--clean` 并在原命令末尾使用 `--resume`；数据窗口缓存会自动复用。

## 2. candidate-only ref2 SFT

切换到 TRACE 环境：

~~~bash
conda activate trace
source Trace/scripts/setup_opd_grpo_env.sh
~~~

正式运行：

~~~bash
PROPOSAL_PATH=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json BASE_CKPT="$TRACE_BASE" OUTP_DIR="$EXP_ROOT/ref2_sft" CLEAN=1 bash Trace/scripts/train/ref2.sh

export SFT_CKPT="$EXP_ROOT/ref2_sft"
test -f "$SFT_CKPT/config.json"
~~~

小样本调试：

~~~bash
PROPOSAL_PATH=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json BASE_CKPT="$TRACE_BASE" OUTP_DIR="$EXP_ROOT/ref2_sft" MAX_SAMPLES=3 EPOCHS=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 bash Trace/scripts/train/ref2.sh

export SFT_CKPT="$EXP_ROOT/ref2_sft"
~~~

中断后在原命令中移除 `CLEAN=1` 并加入 `RESUME_FROM_CHECKPOINT=auto`。


## 3. 构建 paired-only replay

只保留 fake 源视频并且同时间轴真实 counterpart 存在的记录。每条第一阶段
proposal 的真实视频与 candidate 均截取相同的 start/end 区间。假视频中落在真实
区间的 proposal 会保留为 paired no-forgery 校准样本；独立 real 视频与没有
_real.mp4 的 fake 视频不进入 OPD/GRPO。

正式运行：

~~~bash
export REPLAY_PATH="$EXP_ROOT/opd_grpo_replay.json"

python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --paired-only --video-root "$VIDEO_ROOT" --require-reference --output "$REPLAY_PATH"

test -f "$REPLAY_PATH"
~~~

小样本调试：

~~~bash
export REPLAY_PATH="$EXP_ROOT/opd_grpo_replay.json"

python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --paired-only --video-root "$VIDEO_ROOT" --require-reference --max-records 3 --output "$REPLAY_PATH"
~~~

仅在真实 counterpart 不遵循 clip.mp4 到 clip_real.mp4 命名规则时，加入
--reference-map 指向真实映射 JSON。不要为没有真实源视频的样本编造映射。

## 4. 冻结教师预检：必须先执行

学生和教师都加载同一个冻结 SFT checkpoint。学生只看 candidate；教师看原尺寸
上下对照视频，帧尺寸为 672x336。此步骤只推理，不更新参数。

小样本调试，验证非方形视觉输入、显存和输出解析：

~~~bash
export PRECHECK_SMOKE="$EXP_ROOT/precheck_smoke.json"

PYTHONPATH="$REPO_ROOT/Trace:$PYTHONPATH" python Trace/scripts/precheck_opd_teacher.py --replay "$REPLAY_PATH" --data-folder "$VIDEO_ROOT" --model-path "$SFT_CKPT" --vision-tower "$VISION_TOWER" --output "$PRECHECK_SMOKE" --version v1_mistral --max-samples 3

export TEACHER_CACHE="$PRECHECK_SMOKE"
~~~

确认没有 shape error、OOM 或解析错误后运行完整预检。完整预检要求 paired 输入使
正 proposal 恢复率至少提高 1 个百分点，同时负 proposal 的 no-forgery 正确率下降
不超过 2 个百分点。

~~~bash
export TEACHER_CACHE="$EXP_ROOT/opd_teacher_precheck.json"

REPLAY_PATH="$REPLAY_PATH" SFT_CKPT="$SFT_CKPT" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_PATH="$TEACHER_CACHE" MIN_RECOVERY_IMPROVEMENT=0.01 MAX_NEGATIVE_NOEVENT_DROP=0.02 bash Trace/scripts/train/precheck_opd_teacher.sh

test -f "$TEACHER_CACHE"
~~~

若完整预检返回非零，停止 OPD，检查预检 JSON 的 candidate 与 paired 指标；不可
通过降低阈值强行训练。

## 5. OPD

学生只看 candidate，冻结教师仅在 KL 打分时看上下对照。每个 batch 保留 SFT loss；
OPD 的 reverse KL 仅计算定位结构 token。

默认动态权重：错误拒答 1.0；正例其他错误和负例乱报 0.8；正确正例和正确负例
anchor 0.2。可靠正例中 25% 在前 16 个结构 token 使用 guided rollout。

正式运行：

~~~bash
export OPD_OUT="$EXP_ROOT/opd"

REPLAY_PATH="$REPLAY_PATH" STUDENT_CKPT="$SFT_CKPT" TEACHER_CACHE="$TEACHER_CACHE" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_DIR="$OPD_OUT" NPROC_PER_NODE=1 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 bash Trace/scripts/train/opd.sh

export OPD_CKPT="$OPD_OUT"
test -f "$OPD_CKPT/config.json"
~~~

小样本调试：

~~~bash
export OPD_OUT="$EXP_ROOT/opd"

REPLAY_PATH="$REPLAY_PATH" STUDENT_CKPT="$SFT_CKPT" TEACHER_CACHE="$PRECHECK_SMOKE" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_DIR="$OPD_OUT" MAX_SAMPLES=3 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/opd.sh

export OPD_CKPT="$OPD_OUT"
~~~

中断后在原命令中移除 `CLEAN=1` 并加入 `RESUME_FROM_CHECKPOINT=auto`。

日志中应出现 opd_reverse_kl、opd_false_refusal_rollouts 与
opd_reliable_teacher_rate。可靠教师比例接近零时，不应进入 GRPO。

## 6. GRPO 冒烟测试：不调用 Qwen

先验证 candidate-only GRPO rollout、定位奖励与格式奖励。EXPLANATION_WEIGHT=0
时不要求 Qwen endpoint，且不会调用评审器。

~~~bash
export GRPO_SMOKE_OUT="$EXP_ROOT/grpo_smoke"

REPLAY_PATH="$REPLAY_PATH" OPD_CKPT="$OPD_CKPT" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_DIR="$GRPO_SMOKE_OUT" MAX_SAMPLES=3 EXPLANATION_WEIGHT=0 GROUP_SIZE=2 MAX_NEW_TOKENS=64 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh
~~~

中断后在原命令中移除 `CLEAN=1` 并加入 `RESUME_FROM_CHECKPOINT=auto`。

## 7. 启动 Qwen3-VL-235B 解释评审服务

在单独 GPU 进程或单独节点启动 OpenAI-compatible Qwen3-VL 服务。以下为 vLLM
示例；TP_SIZE 由模型显存需求和 GPU 数量决定。

~~~bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3-VL-235B-A22B-Instruct
export QWEN_JUDGE_MODEL=Qwen3-VL-235B-A22B-Instruct
export TP_SIZE=8

vllm serve "$QWEN_MODEL_PATH" --served-model-name "$QWEN_JUDGE_MODEL" --tensor-parallel-size "$TP_SIZE" --trust-remote-code --host 0.0.0.0 --port 8000
~~~

在训练节点验证：

~~~bash
export QWEN_JUDGE_ENDPOINT=http://QWEN_HOST:8000/v1
curl "$QWEN_JUDGE_ENDPOINT/models"
~~~

评审器只收到 candidate 视频帧、模型预测时间段、模型解释和标注证据卡；不会收到
真实参考视频或 GT 时间戳。

## 8. 完整 GRPO

正式运行：

~~~bash
export GRPO_OUT="$EXP_ROOT/grpo"

REPLAY_PATH="$REPLAY_PATH" OPD_CKPT="$OPD_CKPT" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_DIR="$GRPO_OUT" QWEN_JUDGE_ENDPOINT="$QWEN_JUDGE_ENDPOINT" QWEN_JUDGE_MODEL="$QWEN_JUDGE_MODEL" EXPLANATION_WEIGHT=0.3 GROUP_SIZE=4 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh

export FINAL_CKPT="$GRPO_OUT"
test -f "$FINAL_CKPT/config.json"
~~~

小样本调试：

~~~bash
export GRPO_OUT="$EXP_ROOT/grpo"

REPLAY_PATH="$REPLAY_PATH" OPD_CKPT="$OPD_CKPT" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" OUT_DIR="$GRPO_OUT" QWEN_JUDGE_ENDPOINT="$QWEN_JUDGE_ENDPOINT" QWEN_JUDGE_MODEL="$QWEN_JUDGE_MODEL" MAX_SAMPLES=3 EXPLANATION_WEIGHT=0.3 GROUP_SIZE=2 MAX_NEW_TOKENS=64 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh

export FINAL_CKPT="$GRPO_OUT"
~~~

中断后在原命令中移除 `CLEAN=1` 并加入 `RESUME_FROM_CHECKPOINT=auto`。

Qwen 服务和 TRACE 训练不能使用同一组 GPU。Qwen 不稳定时停止完整 GRPO，不要将
解释奖励替换为纯文本相似度。

## 9. 最终 candidate-only 测试

测试时绝不构造真实参考对照，输入是第一阶段在测试集真实生成的 proposal。

正式运行：

~~~bash
MODEL_DIR="$FINAL_CKPT" TEST_ANNO_FILE="$STAGE1_TEST_PROPOSALS" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" VIDEO_DIR="$VIDEO_ROOT" bash Trace/scripts/eval/ref2_eval.sh
~~~

小样本调试：

~~~bash
MODEL_DIR="$FINAL_CKPT" TEST_ANNO_FILE="$STAGE1_TEST_PROPOSALS" MSLOC_ASSETS="$MSLOC_ASSETS" DATA_ROOT="$DATA_ROOT" VIDEO_DIR="$VIDEO_ROOT" SAMPLE_NUM=3 NUM_GPUS=1 CLEAN=1 bash Trace/scripts/eval/ref2_eval.sh
~~~

结果默认写入：

~~~text
$MSLOC_ASSETS/Trace/inference_results/ref2_aigc_test/fmt_aigc_test_f40_result.json
~~~

## 对比与报告

在同一份 STAGE1_TEST_PROPOSALS 上分别测试 SFT_CKPT、OPD_CKPT、FINAL_CKPT。
报告第一阶段 proposal 数量、paired/candidate 教师预检指标、fake 错误拒答率、
no-event 误报率、定位 IoU/边界误差、格式失败率，以及仅在定位正确条件下计算的
解释质量。
