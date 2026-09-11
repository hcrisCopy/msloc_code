# OPD paired-teacher implementation audit

Date: 2026-09-10  
Scope: TRACE ref2 SFT -> paired-teacher OPD -> candidate-only GRPO for
proposal-level video-forgery localization and explanation.

## Decision

The old approach that resized the upper reference and lower candidate to half
height is rejected. It destroys the small spatial evidence (face boundaries,
blending texture, and local temporal inconsistency) that the paired input is
intended to reveal.

The implementation now makes a teacher-only 672 x 336 canvas by vertically
concatenating two unchanged 336 x 336 views. The student still receives only
the normal candidate 336 x 336 video. The frozen CLIP visual tower interpolates
its learned 2-D **position embeddings**, not the pixels, from a 24 x 24 grid to
a 48 x 24 grid. This is the least invasive representation compatible with the
present TRACE architecture.

It is an experimental teacher input, not an assumption that a prompt alone
makes a VLM reliably compare two videos. The precheck remains a mandatory
gate: if paired input does not lower positive false refusal without materially
hurting paired no-event accuracy, OPD must not be run.

## What comparable open implementations actually do

| Work / code | Relevant mechanism | Consequence for this repository |
|---|---|---|
| [Video-OPD](https://github.com/SeerRay-Lab/Video-OPD) | Its training script uses on-policy distillation (--use_on_policy_distillation true) with a frozen teacher model; the project describes dense token-level **reverse KL** and Teacher-Validated Disagreement Focusing. | Replace our symmetric JSD with KL(p_student || p_teacher) on the student's sampled structural prefix. Retain a teacher-reliability gate and give current false refusals the largest weight. |
| [Vision-OPD](https://github.com/Mr-Neko/Vision-OPD) | prepare_data.py writes separate images (student full image) and bbox_images (teacher privileged crop). run_vision_opd.sh passes separate student and teacher image keys; it does not stitch a compressed teacher image into a student image. | The principle of privileged teacher perception is valid, but copying this interface directly is impossible in unmodified TRACE. |
| [VLM2-Bench](https://github.com/VLM2-Bench/VLM2-Bench) | The benchmark is specifically about multi-image/multi-video visual linking and reports that it remains difficult for current MLLMs. | The upper/lower prompt is insufficient evidence by itself. The paired-teacher precheck is a scientific requirement, not merely an engineering smoke test. |
| [IDForge](https://github.com/junly123/IDForge) | A reference-assisted forgery detector shows that authentic reference material can improve forgery detection, but uses a purpose-trained reference-assisted detector rather than a frozen prompted MLLM. | It supports testing authentic counterparts, but does not justify claiming that a frozen TRACE teacher will necessarily succeed. |

## Repository constraints found in the audit

1. trace/model/trace_arch.py asserts len(video_position) == 1; TRACE cannot
   accept separate reference and candidate videos in one forward pass.
   Therefore the Vision-OPD two-media interface cannot be pasted into this
   model without redesigning its multimodal tokenizer and connector.
2. requirements.txt pins transformers==4.40.1. That version's
   CLIPVisionModel.forward has no public interpolate_pos_encoding argument
   (compare the [v4.40.1 source](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/clip/modeling_clip.py)).
   Passing a non-square canvas into the old forward path would fail on the
   fixed position table.
3. The configured spatial_slot projector consumes a variable number of visual
   tokens, so the 48 x 24 = 1152 teacher patch sequence is not rejected merely
   because it is non-square. Its computation is nevertheless more expensive.

## Changes made from those findings

| Module | Change | Why it is needed |
|---|---|---|
| trace/mm_utils.py | make_vertical_reference_pair now directly concatenates full-resolution tensors. | No loss of reference or candidate pixels. |
| trace/model/multimodal_encoder/clip_encoder.py | Added a local, frozen CLIP 2-D position-interpolation path for non-native grids; native student frames retain the exact original code path. | Makes the full-height teacher canvas executable without upgrading a dependency shared by TRACE. |
| trace/trace_trainer.py | OPD uses temperature-scaled reverse KL only at TRACE localization tokens. The sampled answer dynamically gets high weight for a positive no-event stream, positive format/IoU failure, or negative false event; correct outputs remain nonzero anchors. | Matches Video-OPD's on-policy/teacher-validated disagreement principle while directly addressing this project's measured 67.68% positive false-refusal error. |
| trace/train_mt.py and scripts/train/opd.sh | Added validated, exposed disagreement weights and an IoU gate. | Makes the task-specific focusing reproducible rather than a hidden hard-coded filter. |
| OPD_GRPO_RUNBOOK.md | Documents the native pair, compute cost, mandatory precheck, and reverse-KL objective. | Prevents a future run from silently reverting to the invalid half-height design. |

## Exact training contract after the audit

1. Build replay only from fake source videos with an existing aligned
   _real.mp4 counterpart. Keep all Stage-1 proposals of these source videos:
   event-containing windows are positives; empty windows are paired no-event
   calibration samples.
2. SFT and deployment are candidate-only.
3. Freeze the SFT checkpoint. For each replay proposal [s,e], crop aligned
   real[s:e] and candidate[s:e], form the full-resolution vertical pair only
   for the teacher, and precheck candidate-only versus paired answers.
4. If the precheck succeeds, OPD keeps the ordinary SFT loss on all replay
   rows and adds reverse KL only for prechecked-reliable teachers. The student
   rollout decides the disagreement weight; it is not fixed using stale
   initial-SFT predictions. A small fraction of reliable positives retains
   guided rollout for early structural timestamp tokens.
5. GRPO starts from OPD, returns to candidate-only input, and rewards
   localization, a frozen Qwen3-VL-235B explanation judgment against candidate
   frames plus reference evidence text, and the TRACE output format. The judge
   receives neither the reference pair nor GT timestamps.

## Required empirical checks before a paper claim

The code is statically checked, but this workstation has no downloaded
checkpoint/data/GPU environment. Run, in order:

1. four-sample paired precheck (--max-samples 4, no benefit enforcement) to
   verify the 672 x 336 CLIP path, shape, and memory;
2. full paired precheck with benefit enforcement;
3. one-epoch OPD smoke run and inspect opd_false_refusal_rollouts,
   opd_reverse_kl, and nonzero negative-anchor counts;
4. candidate-only GRPO smoke run with explanation reward off, then a small
   Qwen-judge run; and
5. ablate candidate-only SFT, OPD without pair, OPD pair without
   disagreement focusing, and full OPD+GRPO.

Do not report the paired teacher as beneficial solely because it can be
constructed. The full precheck and these ablations are the evidence needed for
the method claim.
