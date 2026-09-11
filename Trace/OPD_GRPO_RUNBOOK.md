# TRACE second-stage pipeline: SFT -> OPD -> GRPO

This implementation preserves TRACE's original output contract. A forged proposal emits a text `<sync>`, a non-empty time stream, `<score>`, and an explanation. A real proposal emits an empty time stream followed by the exact caption `No forgery.`. There is no extra fake/real classifier at inference.

## Environment

Use the TRACE environment before invoking any stage. Install a CUDA-matched
PyTorch/torchvision wheel first when necessary, then install the repository's
remaining pinned dependencies:

```bash
cd Trace
pip install -r requirements.txt
```

The paired-teacher precheck imports einops, decord, and torchvision as well as
the model stack. A missing package is an environment setup failure, not a valid
precheck result.

## 1. Candidate-only ref2 SFT from actual stage-1 training proposals

First run DeMamba on the **training split** and save its raw `predictions.json`.
Then train the candidate-only TRACE student; this is the checkpoint used by
both the frozen teacher precheck and OPD.

```bash
cd Trace
PROPOSAL_PATH=/absolute/path/stage1_train_predictions.json \
BASE_CKPT=/absolute/path/trace-uni \
OUTP_DIR=/absolute/path/ref2_sft \
bash scripts/train/ref2.sh
```

Set `SFT_CKPT` in the next stages to `OUTP_DIR` itself: `train_mt.py` writes
the final loadable model into that output directory after training. The
intermediate `checkpoint-*` directories are optional recovery snapshots, not a
required `checkpoint-final` directory. The ref2 script refuses an unset
proposal path rather than silently using a stale prediction file.

## 2. Build paired replay from actual stage-1 training proposals

The replay builder never synthesizes proposal windows. It reads every proposal actually emitted by stage 1 and labels it as `positive`, `hard_positive`, `near_hard_negative`, or `real_false_positive` for analysis. All four remain in the replay; use `--replay_balance none` in OPD and GRPO.

```bash
python Trace/scripts/build_opd_grpo_replay.py \
  --gt "$DATA_ROOT/annos/train_all_1209.json" \
  --proposals "$STAGE1_TRAIN_PREDICTIONS" \
  --paired-only --video-root "$DATA_ROOT/videos" \
  --require-reference \
  --output "$REPLAY_PATH"
```

This follows the pairing already used in `DeMamba/build_probe_pairs.py`: for a fake `clip.mp4`, the builder first uses an annotation's `normal_video` / `original_video` field and otherwise infers its same-timeline real counterpart `clip_real.mp4`. `--paired-only` verifies that counterpart exists and discards every video without one. Therefore every retained proposal `[s,e]` is cropped as fake `[s,e]` and real `[s,e]`. No hand-written proposal-level map is required under the normal TASLE naming convention.

Only use `--reference-map "$REAL_REFERENCE_MAP"` when that convention is not valid, for example a different counterpart filename or a known time offset. An override entry is:

```json
{
  "candidate/video.mp4": {
    "reference_video": "real/source.mp4",
    "same_timeline": true
  }
}
```

For a known temporal offset, use `"time_offset_seconds": -1.25`. For videos with a linear retiming, use both `candidate_interval` and `reference_interval`. The builder resolves and writes a distinct `reference_segment` into every retained proposal record. A proposal that falls outside the forged interval is still retained if it comes from a paired fake video: its upper real and lower candidate clips naturally look the same, giving a paired no-forgery example. Independently collected real videos and fake videos without `_real.mp4` are excluded from OPD.

## 3. Validate the frozen paired-video teacher before OPD

No teacher is trained. On every retained proposal, the candidate-only ref2-SFT checkpoint is frozen and run twice: once on the candidate, and once on a video whose each sampled frame is `[upper real reference; lower candidate]`. **Neither half is resized**: a `336x336` reference plus a `336x336` candidate becomes a `672x336` teacher-only canvas. TRACE's CLIP wrapper locally interpolates its learned 2-D *positional embeddings* to the `48x24` patch grid; it never interpolates pixels. This is required because the pinned `transformers==4.40.1` CLIP API does not expose the newer interpolation flag. The teacher receives the explicit statement that only the lower half is to be judged. For a proposal outside the forged interval, upper and lower content may naturally be identical; it remains a same-source paired no-forgery sample. The script reports candidate-vs-pair positive false-rejection rate, localized rate, negative no-event rate, format failures, and saves one immutable teacher decision per proposal.

This pair has twice as many visual patches as the normal student frame (1152 rather than 576), so CLIP self-attention is roughly four times more expensive on the teacher branch. Before a full precheck, run the four-sample smoke test below on the target GPU and confirm memory headroom. Do not replace this with half-height resampling: that would discard the facial, texture, and boundary evidence this experiment is supposed to recover.

```bash
cd Trace
REPLAY_PATH=/absolute/path/opd_grpo_replay.json \
SFT_CKPT=/absolute/path/ref2_sft_checkpoint \
OUT_PATH=/absolute/path/opd_teacher_precheck.json \
bash scripts/train/precheck_opd_teacher.sh
```

The wrapper exits nonzero if the paired input fails to improve positive event recovery or damages negative no-event accuracy by more than 2 percentage points. The report is still written for inspection. Do not start OPD unless this validation passes. For a smoke test only, call `precheck_opd_teacher.py` directly with `--max-samples 4` and omit `--enforce-pair-benefit`.

## 4. OPD

OPD trains a candidate-only student. The frozen teacher is loaded from the same SFT checkpoint but sees the paired video and the paired-video prompt. The precheck cache gates a token-level **reverse KL** term, `KL(p_student || p_teacher)`: a positive needs a valid teacher event with IoU >= 0.3, while a negative needs a valid teacher no-event output. The KL is computed only at TRACE's structural text/time positions, never on free explanation text. SFT loss still covers every actual proposal; unreliable teacher samples simply receive no OPD term.

All reliable replay proposals remain in OPD. The coefficient is dynamic on the student's *current sampled answer*: a positive `No forgery` stream gets the largest false-refusal weight, malformed or low-IoU positives get a smaller error weight, and negative false events are also weighted strongly. Correct positive and negative answers retain a nonzero `0.2` anchor weight. This concentrates correction on the observed false-refusal failure without teaching every real/near-boundary proposal to emit a timestamp.

By default, 25% of reliable positive samples use guided rollout for the first 16 structural tokens. The remainder use the student's natural rollout, and negative proposals are never guided. Thus guided rollout supplies some full time/boundary trajectories without replacing the student-error distribution.

```bash
cd Trace
REPLAY_PATH=/absolute/path/opd_grpo_replay.json \
STUDENT_CKPT=/absolute/path/ref2_sft_checkpoint \
TEACHER_CACHE=/absolute/path/opd_teacher_precheck.json \
OUT_DIR=/absolute/path/opd_student \
bash scripts/train/opd.sh
```

The script deliberately sets `--opd_teacher_model_path` equal to `STUDENT_CKPT`; it does not accept a teacher-SFT checkpoint. Its default dynamic weights are `1.0` (false refusal), `0.8` (other positive or negative errors), and `0.2` (correct anchors). They are environment-overridable in `scripts/train/opd.sh`.

## 5. GRPO

GRPO begins from the final OPD student and sees only the candidate proposal. For each proposal it samples a group of candidate-only outputs. Its rewards are: localization (event/no-event decision, IoU, boundaries and extra-event penalty), explanation, and output format. The frozen Qwen3-VL-235B judge sees candidate frames, the model's predicted time segment and explanation, plus an unseen annotation-derived reference-evidence card. It does not see the paired reference video or GT timestamps. It checks both coverage of the reference explanation and whether the candidate frames visibly support the claim.

```bash
cd Trace
REPLAY_PATH=/absolute/path/opd_grpo_replay.json \
OPD_CKPT=/absolute/path/opd_student \
QWEN_JUDGE_ENDPOINT=http://JUDGE_HOST:8000/v1 \
QWEN_JUDGE_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct \
OUT_DIR=/absolute/path/grpo_student \
bash scripts/train/grpo.sh
```

The judge endpoint must implement the OpenAI-compatible multimodal chat API. Run a small candidate-only GRPO smoke test with `EXPLANATION_WEIGHT=0 EPOCHS=1 GROUP_SIZE=2` first; it verifies model and training plumbing without calling the external judge. Then enable the Qwen explanation reward for the full run.

## 6. Candidate-only test evaluation

Evaluation never constructs a reference pair. Point it to the checkpoint from the stage being compared (normally the final GRPO checkpoint) and to DeMamba's **test-split** `predictions.json`:

```bash
cd Trace
MODEL_DIR=/absolute/path/grpo_student \
TEST_ANNO_FILE=/absolute/path/stage1_test_predictions.json \
bash scripts/eval/ref2_eval.sh
```

## Required reporting

Report stage-1 proposal counts and all four replay buckets; paired-teacher precheck metrics; positive false-rejection rate; negative false-event rate; localized acceptance / IoU; boundary error; format-failure rate; explanation quality conditioned on correct localization; and joint localization-plus-explanation success. If the paired frozen teacher does not pass precheck, the OPD hypothesis is unsupported and the experiment must stop before GRPO.
