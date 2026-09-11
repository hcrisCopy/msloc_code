# DeMamba → TRACE SFT → OPD → GRPO

在仓库根目录执行。数据目录固定为同级 `../MSLoc_data`。

## 0. 环境

```bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
source Trace/scripts/setup_opd_grpo_env.sh
```

训练命令用 `CLEAN=1` 清理对应输出目录。中断恢复时移除 `CLEAN=1`，加入
`RESUME_FROM_CHECKPOINT=auto`。训练与评测均显示 tqdm 进度。

## 1. 生成训练集 proposal

先切换到 DeMamba 使用的环境 conda activate msloc。

正式：

```bash
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --device-ids 0 --val-batch-size 16 --save-progress --cache-data --clean

export STAGE1_TRAIN_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json
export STAGE1_TEST_PROPOSALS=../MSLoc_data/DeMamba/full/method/eval/predictions.json
```

小样本：

```bash
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --max-eval-videos 3 --num-workers 0 --device-ids 0 --val-batch-size 1 --save-progress --cache-data --clean
```

DeMamba 评测续跑：移除 `--clean`，加入 `--resume`。

## 2. candidate-only ref2 SFT

```bash
conda activate trace
source Trace/scripts/setup_opd_grpo_env.sh

PROPOSAL_PATH="$STAGE1_TRAIN_PROPOSALS" BASE_CKPT="$TRACE_BASE" OUTP_DIR="$EXP_ROOT/ref2_sft" CLEAN=1 bash Trace/scripts/train/ref2.sh
export SFT_CKPT="$EXP_ROOT/ref2_sft"
```

小样本：

```bash
export SFT_SMOKE="$EXP_ROOT/ref2_sft_smoke"
PROPOSAL_PATH="$STAGE1_TRAIN_PROPOSALS" BASE_CKPT="$TRACE_BASE" OUTP_DIR="$SFT_SMOKE" MAX_SAMPLES=3 GLOBAL_BATCH_SIZE=1 GRAD_ACCUM=1 EPOCHS=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 bash Trace/scripts/train/ref2.sh
```

## 3. 构建 OPD/GRPO replay

OPD 需要真实配对；GRPO 使用完整 candidate proposal 分布，包含真实视频误报。

```bash
export OPD_REPLAY="$EXP_ROOT/opd_paired_replay.json"
export GRPO_REPLAY="$EXP_ROOT/grpo_candidate_replay.json"
python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --paired-only --video-root "$VIDEO_ROOT" --require-reference --output "$OPD_REPLAY" --clean
python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --output "$GRPO_REPLAY" --clean
```

小样本：

```bash
export OPD_REPLAY_SMOKE="$EXP_ROOT/opd_paired_replay_smoke.json"
export GRPO_REPLAY_SMOKE="$EXP_ROOT/grpo_candidate_replay_smoke.json"
python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --paired-only --video-root "$VIDEO_ROOT" --require-reference --max-records 3 --output "$OPD_REPLAY_SMOKE" --clean
python Trace/scripts/build_opd_grpo_replay.py --gt "$TRAIN_ANNO" --proposals "$STAGE1_TRAIN_PROPOSALS" --max-records 3 --output "$GRPO_REPLAY_SMOKE" --clean
```

## 4. 冻结 paired teacher 预检

正式：

```bash
export TEACHER_CACHE="$EXP_ROOT/opd_teacher_precheck.json"
REPLAY_PATH="$OPD_REPLAY" SFT_CKPT="$SFT_CKPT" OUT_PATH="$TEACHER_CACHE" MIN_RECOVERY_IMPROVEMENT=0.01 MIN_RELIABLE_POSITIVE_RATE=0.05 MAX_NEGATIVE_NOEVENT_DROP=0.02 bash Trace/scripts/train/precheck_opd_teacher.sh
```

小样本只检查链路：

```bash
export PRECHECK_SMOKE="$EXP_ROOT/precheck_smoke.json"
PYTHONPATH="$REPO_ROOT/Trace:$PYTHONPATH" python Trace/scripts/precheck_opd_teacher.py --replay "$OPD_REPLAY_SMOKE" --data-folder "$VIDEO_ROOT" --model-path "$SFT_CKPT" --vision-tower "$VISION_TOWER" --output "$PRECHECK_SMOKE" --version v1_mistral --max-samples 3
```

正式预检失败时不要进入 OPD。

## 5. OPD

正式：

```bash
export OPD_OUT="$EXP_ROOT/opd"
REPLAY_PATH="$OPD_REPLAY" STUDENT_CKPT="$SFT_CKPT" TEACHER_CACHE="$TEACHER_CACHE" OUT_DIR="$OPD_OUT" EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/opd.sh
export OPD_CKPT="$OPD_OUT"
```

小样本：

```bash
export OPD_SMOKE="$EXP_ROOT/opd_smoke"
REPLAY_PATH="$OPD_REPLAY_SMOKE" STUDENT_CKPT="$SFT_CKPT" TEACHER_CACHE="$PRECHECK_SMOKE" OUT_DIR="$OPD_SMOKE" MAX_SAMPLES=3 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/opd.sh
```

## 6. Lexical 文字奖励小样本

GRPO 保留定位、解释、格式三类奖励。解释奖励使用标注中的
`object/start/end class + caption`，不再调用 235B 或读取视频。
它衡量与参考答案的一致性，不等同于视觉真实性验证。

`lexical`：关键词规范化、领域同义词和 token F1；无需额外模型，速度最快。

```bash
export GRPO_LEXICAL_SMOKE="$EXP_ROOT/grpo_lexical_smoke"
REPLAY_PATH="$GRPO_REPLAY_SMOKE" OPD_CKPT="$OPD_CKPT" OUT_DIR="$GRPO_LEXICAL_SMOKE" TEXT_REWARD_MODE=lexical EXPLANATION_WEIGHT=0.3 MAX_SAMPLES=3 GROUP_SIZE=2 MAX_NEW_TOKENS=64 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh
```

## 7. NLI 文字奖励小样本

`nli` 是 lexical 加强版：保留 lexical 匹配，再增加语义同义改写和矛盾检测。
NLI 模型冻结，只推理、不训练。

```bash
hf download cross-encoder/nli-deberta-v3-small --local-dir ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small
export NLI_MODEL_PATH=../MSLoc_data/Trace/ckpts/nli-deberta-v3-small

export GRPO_NLI_SMOKE="$EXP_ROOT/grpo_nli_smoke"
REPLAY_PATH="$GRPO_REPLAY_SMOKE" OPD_CKPT="$OPD_CKPT" OUT_DIR="$GRPO_NLI_SMOKE" TEXT_REWARD_MODE=nli NLI_MODEL_PATH="$NLI_MODEL_PATH" NLI_DEVICE=cpu EXPLANATION_WEIGHT=0.3 MAX_SAMPLES=3 GROUP_SIZE=2 MAX_NEW_TOKENS=64 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh
```

## 8. 正式 GRPO

正式训练只选一种模式，不要先后训练两轮。优先使用 `nli`；速度不足时改成
`TEXT_REWARD_MODE=lexical`，并删除命令中的 NLI 参数。

```bash
export GRPO_OUT="$EXP_ROOT/grpo"
REPLAY_PATH="$GRPO_REPLAY" OPD_CKPT="$OPD_CKPT" OUT_DIR="$GRPO_OUT" TEXT_REWARD_MODE=nli NLI_MODEL_PATH="$NLI_MODEL_PATH" NLI_DEVICE=cpu EXPLANATION_WEIGHT=0.3 GROUP_SIZE=4 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 CLEAN=1 NPROC_PER_NODE=1 bash Trace/scripts/train/grpo.sh
export FINAL_CKPT="$GRPO_OUT"
```

关注日志：`grpo_exp_reward`、`grpo_text_graph_precision`、
`grpo_text_graph_recall`、`grpo_text_contradiction`。

当前为 `num_iterations=1` 的单次更新 masked GRPO；组内优势和 reference KL
有效，PPO clipping 在首次更新时不提供额外约束。

## 9. candidate-only 测试

正式：

```bash
MODEL_DIR="$FINAL_CKPT" TEST_ANNO_FILE="$STAGE1_TEST_PROPOSALS" bash Trace/scripts/eval/ref2_eval.sh
```

小样本：

```bash
MODEL_DIR="$FINAL_CKPT" TEST_ANNO_FILE="$STAGE1_TEST_PROPOSALS" SAMPLE_NUM=3 NUM_GPUS=1 CLEAN=1 bash Trace/scripts/eval/ref2_eval.sh
```

在同一份测试 proposals 上分别评测 SFT、OPD 和 GRPO checkpoint。
