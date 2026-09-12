# DeMamba → SFT → OPD → GRPO

全部命令在 `msloc_code` 根目录执行。代码在 `Trace/`，数据和模型在同级 `../MSLoc_data/`。

正式训练默认单机 8 卡。单卡时，将 `NPROC_PER_NODE=8`、`NUM_GPUS=8` 改为 `1`，DeMamba 的 `--device-ids` 改为 `0`。

## 0. 准备

`msloc` 环境运行 DeMamba，`trace` 环境运行其余阶段。NLI 模型只下载一次。

```bash
conda create -n trace python=3.10 -y
conda activate trace
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Trace/requirements.txt
hf download cross-encoder/nli-deberta-v3-small --local-dir ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small
```

`CLEAN=1` 会删除该阶段旧输出。中断恢复时删除 `CLEAN=1`，改为 `RESUME_FROM_CHECKPOINT=auto`。

## 1. DeMamba 生成 proposal

作用：找出可能被伪造的时间窗口。依赖 DeMamba 权重、帧数据和训练标注。

正式：

```bash
conda activate msloc
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --device-ids 0,1,2,3,4,5,6,7 --val-batch-size 16 --num-workers 4 --save-progress --cache-data --clean
```

输出：`../MSLoc_data/DeMamba/full/method/eval_train/predictions.json`。

小样本：

```bash
python Trace/scripts/build_smoke_annotations.py --input ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --output ../MSLoc_data/data/Tasle-CoT-10K/annos/train_smoke.json --fake 2 --real 5 --clean
python DeMamba/eval.py --config ../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml --neuron-indices-path ../MSLoc_data/DeMamba/full/method/evidence_probe/xclip_neuron_indices_checkpoint.json --dataset-base-path ../MSLoc_data/DeMamba/video_frames --model_path ../MSLoc_data/DeMamba/full/method/results/best_acc.pth --output_dir ../MSLoc_data/DeMamba/full/method/eval_train_smoke --anno-file ../MSLoc_data/data/Tasle-CoT-10K/annos/train_smoke.json --device-ids 0,1,2,3,4,5,6,7 --val-batch-size 1 --num-workers 0 --save-progress --cache-data --clean
```

输出：`../MSLoc_data/DeMamba/full/method/eval_train_smoke/predictions.json`。中断恢复时去掉 `--clean`，加入 `--resume`。

## 2. SFT

作用：让 TRACE 根据 candidate proposal 输出时间和文字解释。依赖阶段 1 proposal、TRACE 基座模型和 CLIP。

正式：

```bash
conda activate trace
PROPOSAL_PATH=../MSLoc_data/DeMamba/full/method/eval_train/predictions.json BASE_CKPT=../MSLoc_data/Trace/ckpts/trace-uni OUTP_DIR=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft EPOCHS=2 BATCH_SIZE=2 GRAD_ACCUM=2 NUM_WORKERS=4 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/ref2.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft/`。

小样本：

```bash
PROPOSAL_PATH=../MSLoc_data/DeMamba/full/method/eval_train_smoke/predictions.json BASE_CKPT=../MSLoc_data/Trace/ckpts/trace-uni OUTP_DIR=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft_smoke MAX_SAMPLES=3 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/ref2.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft_smoke/`。

## 3. 构建 replay

作用：把 proposal、GT 和解释标注整理成训练 JSON。OPD 需要真实参考视频；GRPO 只使用 candidate，并保留真实视频误报。

正式：

```bash
python Trace/scripts/build_opd_grpo_replay.py --gt ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json --paired-only --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos --require-reference --output ../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay.json --clean
python Trace/scripts/build_opd_grpo_replay.py --gt ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --proposals ../MSLoc_data/DeMamba/full/method/eval_train/predictions.json --output ../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay.json --clean
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay.json` 和 `grpo_candidate_replay.json`。

小样本：

```bash
python Trace/scripts/build_opd_grpo_replay.py --gt ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --proposals ../MSLoc_data/DeMamba/full/method/eval_train_smoke/predictions.json --paired-only --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos --require-reference --max-records 3 --stratified-debug --output ../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay_smoke.json --clean
python Trace/scripts/build_opd_grpo_replay.py --gt ../MSLoc_data/data/Tasle-CoT-10K/annos/train_all_1209.json --proposals ../MSLoc_data/DeMamba/full/method/eval_train_smoke/predictions.json --max-records 3 --stratified-debug --output ../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay_smoke.json --clean
```

输出：同目录下两个带 `_smoke` 的 replay 文件。

## 4. OPD teacher 预检

作用：冻结 SFT teacher，让它看真实参考与 candidate；记录哪些回答可靠。

正式：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay.json SFT_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft OUT_PATH=../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_precheck.json TEACHER_IOU_GATE=0.3 MIN_RECOVERY_IMPROVEMENT=0.01 MIN_RELIABLE_POSITIVE_RATE=0.05 MAX_NEGATIVE_NOEVENT_DROP=0.02 NPROC_PER_NODE=8 bash Trace/scripts/train/precheck_opd_teacher.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_precheck.json`。正式预检失败时不能继续 OPD。

小样本只检查链路，允许质量门失败：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay_smoke.json SFT_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft_smoke OUT_PATH=../MSLoc_data/Trace/experiments/opd_grpo/precheck_smoke.json TEACHER_IOU_GATE=0.3 MIN_RECOVERY_IMPROVEMENT=0 MIN_RELIABLE_POSITIVE_RATE=0.05 MAX_NEGATIVE_NOEVENT_DROP=0.02 SMOKE_ALLOW_FAILED_PRECHECK=1 NPROC_PER_NODE=8 bash Trace/scripts/train/precheck_opd_teacher.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/precheck_smoke.json`，仅供小样本使用。

## 5. OPD

作用：student 只看 candidate，teacher 看真实参考与 candidate，通过结构 token 的 reverse KL 修正 student。原 student rollout 决定错误权重；guided rollout 只增加 `0.25` 倍辅助 KL。

正式：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay.json STUDENT_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft TEACHER_CACHE=../MSLoc_data/Trace/experiments/opd_grpo/opd_teacher_precheck.json OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/opd OPD_WEIGHT=1.0 OPD_TEMPERATURE=1.0 OPD_DISAGREEMENT_IOU_GATE=0.3 FALSE_REFUSAL_WEIGHT=1.0 POSITIVE_ERROR_WEIGHT=0.8 NEGATIVE_ERROR_WEIGHT=0.8 POSITIVE_ANCHOR_WEIGHT=0.2 NEGATIVE_ANCHOR_WEIGHT=0.2 GUIDED_POSITIVE_FRACTION=1.0 GUIDED_ALPHA=0.5 GUIDED_MAX_TOKENS=16 GUIDED_LOSS_COEF=0.25 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 NUM_WORKERS=4 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/opd.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/opd/`。

小样本：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/opd_paired_replay_smoke.json STUDENT_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/ref2_sft_smoke TEACHER_CACHE=../MSLoc_data/Trace/experiments/opd_grpo/precheck_smoke.json OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/opd_smoke SMOKE_ALLOW_UNVALIDATED_TEACHER=1 MAX_SAMPLES=3 OPD_WEIGHT=1.0 OPD_TEMPERATURE=1.0 OPD_DISAGREEMENT_IOU_GATE=0.3 GUIDED_POSITIVE_FRACTION=1.0 GUIDED_ALPHA=0.5 GUIDED_MAX_TOKENS=16 GUIDED_LOSS_COEF=0.25 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/opd.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/opd_smoke/`。teacher 不可靠时 OPD KL 可以为 0，此命令只检查程序能否运行。

## 6. GRPO

作用：在 candidate-only 分布上优化定位、格式、文字解释三类奖励。文字奖励只有定位 IoU 达标后才发放。先运行所选方法的小样本命令；通过后，再从正式 OPD 模型开始正式训练。Lexical 和 NLI 二选一，不连续训练。

### 6.1 Lexical

关键词、同义词和 token F1，不加载额外模型。

正式：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay.json OPD_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/opd OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/grpo_lexical TEXT_REWARD_MODE=lexical TEXT_MAX_WORDS=80 EXPLANATION_WEIGHT=0.3 GROUP_SIZE=4 MAX_NEW_TOKENS=128 TEMPERATURE=0.7 KL_COEF=0.02 SFT_COEF=0.1 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 NUM_WORKERS=4 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/grpo.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/grpo_lexical/`。

小样本：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay_smoke.json OPD_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/opd_smoke OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/grpo_lexical_smoke TEXT_REWARD_MODE=lexical TEXT_MAX_WORDS=80 EXPLANATION_WEIGHT=0.3 GROUP_SIZE=2 MAX_NEW_TOKENS=64 TEMPERATURE=0.7 KL_COEF=0.02 SFT_COEF=0.1 MAX_SAMPLES=3 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/grpo.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/grpo_lexical_smoke/`。

### 6.2 NLI

保留 Lexical 分数，再用冻结的小型 NLI 模型检查语义一致和矛盾；NLI 不训练。

正式：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay.json OPD_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/opd OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli TEXT_REWARD_MODE=nli NLI_MODEL_PATH=../MSLoc_data/Trace/ckpts/nli-deberta-v3-small NLI_DEVICE=cuda NLI_BATCH_SIZE=32 TEXT_MAX_WORDS=80 EXPLANATION_WEIGHT=0.3 GROUP_SIZE=4 MAX_NEW_TOKENS=128 TEMPERATURE=0.7 KL_COEF=0.02 SFT_COEF=0.1 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=4 NUM_WORKERS=4 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/grpo.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli/`。关注 `grpo_loc_reward`、`grpo_exp_reward`、`grpo_fmt_reward` 和 `grpo_text_contradiction`。

小样本：

```bash
REPLAY_PATH=../MSLoc_data/Trace/experiments/opd_grpo/grpo_candidate_replay_smoke.json OPD_CKPT=../MSLoc_data/Trace/experiments/opd_grpo/opd_smoke OUT_DIR=../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli_smoke TEXT_REWARD_MODE=nli NLI_MODEL_PATH=../MSLoc_data/Trace/ckpts/nli-deberta-v3-small NLI_DEVICE=cuda NLI_BATCH_SIZE=32 TEXT_MAX_WORDS=80 EXPLANATION_WEIGHT=0.3 GROUP_SIZE=2 MAX_NEW_TOKENS=64 TEMPERATURE=0.7 KL_COEF=0.02 SFT_COEF=0.1 MAX_SAMPLES=3 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 NUM_WORKERS=0 SAVE_STEPS=1 NPROC_PER_NODE=8 CLEAN=1 bash Trace/scripts/train/grpo.sh
```

输出：`../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli_smoke/`。

## 7. candidate-only 测试

作用：最终模型只看 candidate proposal，生成定位和解释。测试 proposal 默认位于 `../MSLoc_data/DeMamba/full/method/eval/predictions.json`。

```bash
MODEL_DIR=../MSLoc_data/Trace/experiments/opd_grpo/grpo_nli TEST_ANNO_FILE=../MSLoc_data/DeMamba/full/method/eval/predictions.json RAW_ANNO_FILE=../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209.json VIDEO_DIR=../MSLoc_data/data/Tasle-CoT-10K/videos OUTPUT_DIR=../MSLoc_data/Trace/inference_results/grpo_nli_test NUM_FRAME=40 MAX_NEW_TOKENS=512 SAMPLE_NUM=-1 NUM_GPUS=8 CLEAN=1 bash Trace/scripts/eval/ref2_eval.sh
```

输出：`../MSLoc_data/Trace/inference_results/grpo_nli_test/fmt_aigc_test_f40_result.json`。若选择 Lexical，将 `MODEL_DIR` 改为 `grpo_lexical`，同时换一个 `OUTPUT_DIR`。小样本测试把 `SAMPLE_NUM=-1` 改为 `3`。
