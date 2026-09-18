import importlib.util
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opd = load_module("opd_grpo_test", "trace/opd_grpo.py")
replay = load_module("replay_test", "scripts/build_opd_grpo_replay.py")
text_reward = load_module("text_reward_test", "trace/text_explanation_reward.py")
rollout_audit = load_module("rollout_audit_test", "trace/rollout_audit.py")
launcher = load_module("run_opd_grpo_test", "run_opd_grpo.py")


class OpdGrpoTests(unittest.TestCase):
    def setUp(self):
        self.spec = opd.TraceTokenSpec(
            text_vocab_size=100,
            time_vocab={"<sync>": 0, "<sep>": 1, "0": 2, "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8, "7": 9, "8": 10, "9": 11, ".": 12},
            score_vocab_size=13,
        )

    def time_ids(self, text):
        return [self.spec.time_start_id + self.spec.time_vocab[ch] for ch in text]

    def test_rollout_audit_resumes_without_duplicates_and_repairs_partial_tail(self):
        with tempfile.TemporaryDirectory() as root:
            writer = rollout_audit.RolloutAuditWriter(root, "grpo", rank=3, enabled=True)
            first = {"audit_key": "step0-sample-a-rollout0", "reward": {"total": 1.0}}
            self.assertEqual(writer.write([first]), 1)
            self.assertEqual(writer.write([first]), 0)
            with writer.path.open("a", encoding="utf-8") as handle:
                handle.write('{"audit_key":"incomplete"')

            resumed = rollout_audit.RolloutAuditWriter(root, "grpo", rank=3, enabled=True)
            second = {"audit_key": "step1-sample-b-rollout0", "reward": {"total": 0.5}}
            self.assertEqual(resumed.write([first, second]), 1)
            records = [json.loads(line) for line in resumed.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["audit_key"] for record in records], [
                "step0-sample-a-rollout0", "step1-sample-b-rollout0",
            ])

    def test_valid_event_and_rewards(self):
        # <sync> 0000.2 <sep> 0003.4 <time-sync> <score-sync> caption
        tokens = [self.spec.text_sync_id]
        tokens += self.time_ids("0000.2")
        tokens += [self.spec.time_sep_id]
        tokens += self.time_ids("0003.4")
        tokens += [self.spec.time_sync_id, self.spec.score_start_id]
        tokens += [7, 8]
        parsed = opd.parse_trace_tokens(tokens, self.spec, lambda _: "mouth boundary flickers", window_duration=5.0)
        self.assertEqual(parsed.status, opd.VALID_EVENT)
        self.assertEqual(parsed.segments, [(0.2, 3.4)])
        reward = opd.score_trace_output(
            parsed,
            target_segments=[(0.2, 3.4)],
            candidate_video="candidate.mp4",
            proposal=(10.0, 15.0),
            evidence=None,
            sample_id="unit",
            config=opd.RewardConfig(),
        )
        self.assertGreater(reward.localization, 0.9)
        self.assertEqual(reward.format, 1.0)

    def test_zero_iou_segments_are_not_matched_or_rewarded(self):
        self.assertEqual(opd.match_segments([(0.0, 1.0)], [(2.0, 3.0)]), [])
        parsed = opd.ParsedTraceOutput(
            opd.VALID_EVENT, segments=[(0.0, 1.0)], caption="unrelated event"
        )
        reward = opd.score_trace_output(
            parsed,
            target_segments=[(2.0, 3.0)],
            candidate_video="candidate.mp4",
            proposal=(0.0, 4.0),
            evidence=None,
            sample_id="zero-iou",
            config=opd.RewardConfig(),
        )
        self.assertEqual(reward.matched_iou, 0.0)
        self.assertEqual(reward.localization, 0.0)

    def test_invalid_is_not_a_real_prediction(self):
        parsed = opd.parse_trace_tokens([self.spec.text_sync_id, 3], self.spec, lambda _: "No forgery.", window_duration=2.0)
        self.assertEqual(parsed.status, opd.FORMAT_FAILURE)
        negative = opd.score_trace_output(
            parsed, target_segments=[], candidate_video="x", proposal=(0.0, 2.0), evidence=None,
            sample_id="unit", config=opd.RewardConfig(),
        )
        self.assertEqual(negative.localization, -0.5)

    def test_explanation_reward_requires_candidate_only_judge(self):
        parsed = opd.ParsedTraceOutput(opd.VALID_EVENT, segments=[(0.2, 1.0)], caption="face flickers")
        with self.assertRaises(RuntimeError):
            opd.score_trace_output(
                parsed, target_segments=[(0.2, 1.0)], candidate_video="x", proposal=(0.0, 2.0),
                evidence=opd.EvidenceCard("face flickers", candidate_observable=True), sample_id="unit",
                config=opd.RewardConfig(explanation_weight=0.5), explanation_judge=None,
            )

    def test_component_masks_cover_total_without_changing_component_masks(self):
        tokens = [self.spec.text_sync_id, *self.time_ids("1"), 7]
        masks = opd.action_component_masks(tokens, self.spec)
        self.assertEqual(masks["total"], [1.0, 1.0, 1.0])
        self.assertEqual(masks["localization"], [1.0, 1.0, 0.0])
        self.assertEqual(masks["explanation"], [0.0, 0.0, 1.0])

    def test_replay_keeps_boundary_and_object_annotations(self):
        gt = [{"video_path": "candidate.mp4", "annotations": [{
            "segment": [2.0, 5.0], "combine_dir": "Round3",
            "bnd_cot_st": [{"bnd_caption": "transition begins", "bnd_class": "start"}],
            "obj_cot": [{"obj_caption": "mouth flickers", "bnd_sub_class": "mouth"}],
            "bnd_cot_ed": [{"bnd_caption": "transition ends", "bnd_class": "end"}],
        }]}]
        proposals = [{"video_path": "candidate.mp4", "model_inference": {"segment": [[1.0, 4.0]], "response": ["fake"]}}]
        records = replay.build_records(
            gt, proposals, {"candidate.mp4": {"reference_video": "real.mp4", "same_timeline": True}},
            evidence_audit={"candidate.mp4::2.000-5.000": {"candidate_observable": True}},
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_positive"])
        self.assertEqual(records[0]["targets"][0]["relative_segment"], [1.0, 3.0])
        self.assertEqual(records[0]["targets"][0]["evidence"]["object_caption"], "mouth flickers")
        self.assertTrue(records[0]["targets"][0]["evidence"]["candidate_observable"])
        self.assertEqual(records[0]["replay_bucket"], "positive")
        self.assertEqual(records[0]["reference"]["reference_video"], "real.mp4")
        self.assertEqual(records[0]["reference"]["reference_segment"], [1.0, 4.0])

    def test_paired_only_uses_demamba_real_counterpart_and_keeps_negative_proposals(self):
        gt = [{"video_path": "candidate.mp4", "type": "fake", "annotations": [
            {"segment": [2.0, 5.0], "combine_dir": "Round3"}
        ]}]
        proposals = [{"video_path": "candidate.mp4", "model_inference": {
            "segment": [[1.0, 4.0], [6.0, 7.0]]
        }}]
        records = replay.build_records(gt, proposals, {}, paired_only=True)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["reference"]["reference_video"], "candidate_real.mp4")
        self.assertEqual(records[0]["reference"]["reference_segment"], [1.0, 4.0])
        self.assertFalse(records[1]["is_positive"])
        self.assertEqual(records[1]["reference"]["reference_segment"], [6.0, 7.0])

    def test_candidate_replay_keeps_real_video_false_positives(self):
        gt = [{"video_path": "real.mp4", "type": "real", "annotations": []}]
        proposals = [{"video_path": "real.mp4", "type": "real", "model_inference": {
            "segment": [[1.0, 3.0]]
        }}]
        records = replay.build_records(gt, proposals, {}, paired_only=False)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["is_positive"])
        self.assertEqual(records[0]["replay_bucket"], "real_false_positive")
        self.assertIsNone(records[0]["reference"])

    def test_stratified_debug_covers_reward_branches(self):
        records = [
            {"id": "p", "is_positive": True, "replay_bucket": "positive"},
            {"id": "p2", "is_positive": True, "replay_bucket": "positive"},
            {"id": "n", "is_positive": False, "replay_bucket": "near_hard_negative"},
            {"id": "r", "is_positive": False, "replay_bucket": "real_false_positive"},
        ]
        selected = replay.stratified_debug_records(records, 3)
        self.assertEqual([row["id"] for row in selected], ["p", "n", "r"])

    def test_reference_text_reward_matches_evidence_and_penalises_generic_text(self):
        evidence = opd.EvidenceCard(
            object_caption="The mouth flickers and changes shape unnaturally.",
            object_class="Object Flickering/Instantaneous Changes",
        )
        class FakeNLI:
            def probabilities(self, pairs):
                results = []
                for fact, claim in pairs:
                    aligned = "mouth" in fact.lower() and "mouth" in claim.lower() and "flicker" in claim.lower()
                    results.append((0.95, 0.01) if aligned else (0.02, 0.05))
                return results

        judge = text_reward.EntailmentExplanationJudge.__new__(text_reward.EntailmentExplanationJudge)
        judge.require_candidate_observable = False
        judge.nli = FakeNLI()
        matched = judge.score(caption="The mouth flickers with unnatural shape changes.", evidence=evidence)
        generic = judge.score(caption="The video is fake.", evidence=evidence)
        self.assertGreater(matched.graph_f1, generic.graph_f1)
        self.assertGreater(matched.reward, generic.reward)
        self.assertLess(generic.graph_precision, matched.graph_precision)

    def test_text_reward_uses_object_and_boundary_facts(self):
        evidence = opd.EvidenceCard(
            object_caption="The hand is deformed.",
            start_caption="The deformation appears abruptly.",
            end_caption="The hand returns to normal.",
            object_class="Object Deformation",
            start_class="abrupt onset",
            end_class="abrupt offset",
        )
        facts = text_reward.evidence_facts(evidence)
        self.assertEqual([fact.relation for fact in facts], ["object_anomaly", "onset", "offset"])
        self.assertEqual([fact.weight for fact in facts], [2.0, 1.0, 1.0])

    def test_text_reward_does_not_cross_penalize_different_phases(self):
        evidence = opd.EvidenceCard(
            object_caption="The hand is deformed during the forged interval.",
            end_caption="The hand returns to normal at the end.",
        )

        class PhaseAwareNLI:
            def probabilities(self, pairs):
                results = []
                for fact, claim in pairs:
                    same_phase = (
                        ("deformed" in fact.lower() and "deformed" in claim.lower())
                        or ("returns to normal" in fact.lower() and "returns to normal" in claim.lower())
                    )
                    results.append((0.95, 0.01) if same_phase else (0.01, 0.99))
                return results

        judge = text_reward.EntailmentExplanationJudge.__new__(text_reward.EntailmentExplanationJudge)
        judge.require_candidate_observable = False
        judge.nli = PhaseAwareNLI()
        verdict = judge.score(
            caption="The hand is deformed. The hand returns to normal.",
            evidence=evidence,
        )
        self.assertAlmostEqual(verdict.contradiction, 0.01)
        self.assertEqual(verdict.judge_id, "atomic-entailment-v3-aligned-contradiction")

    def test_candidate_sft_aliases_use_paper_architecture_without_pairing(self):
        common = [
            "--devices", "0", "--nproc-per-node", "1",
            "--annotation", "train.json", "--video-root", "videos",
            "--vision-tower", "clip", "--deepspeed", "zero2.json",
            "--output", "output", "--epochs", "2", "--batch-size", "2",
            "--grad-accum", "2", "--num-workers", "4", "--run-name", "sft",
            "--proposals", "proposals.json", "--base-model", "trace-uni",
        ]
        parser = launcher.build_parser()
        for command in ("sft", "student-sft"):
            args = parser.parse_args([command, *common])
            self.assertIs(args.handler, launcher.run_sft)
            self.assertEqual(args.mm_projector_type, "ref_projector")
            self.assertEqual(args.closs, "True")
            self.assertEqual(args.num_frames, 40)
            self.assertEqual(args.bnd_frames, 16)
            self.assertEqual(args.seg_frames, 8)
            self.assertEqual(args.freeze_backbone, "False")

        paired = parser.parse_args([
            "paired-teacher-sft",
            "--devices", "0", "--nproc-per-node", "1",
            "--annotation", "train.json", "--video-root", "videos",
            "--vision-tower", "clip", "--deepspeed", "zero2.json",
            "--output", "output", "--epochs", "2", "--batch-size", "2",
            "--grad-accum", "2", "--num-workers", "4", "--run-name", "paired",
            "--training-samples", "samples.json", "--base-model", "trace-uni",
        ])
        self.assertIs(paired.handler, launcher.run_paired_teacher_sft)

        teacher_eval = parser.parse_args([
            "test-teacher",
            "--devices", "0", "--nproc-per-node", "1",
            "--test-samples", "test_paired.json", "--video-root", "videos",
            "--teacher-model", "teacher_sft", "--vision-tower", "clip",
            "--annotation", "test.json", "--output", "teacher_eval",
            "--metrics-output", "teacher_eval/metrics.json", "--prompt-file", "dvc.txt",
        ])
        self.assertIs(teacher_eval.handler, launcher.run_teacher_eval)
        self.assertFalse(hasattr(teacher_eval, "student_checkpoint"))

    def test_external_student_without_manifest_is_allowed_with_warning(self):
        with tempfile.TemporaryDirectory() as root:
            student = pathlib.Path(root) / "external_student"
            teacher = pathlib.Path(root) / "paired_teacher"
            student.mkdir()
            teacher.mkdir()
            (teacher / "stage_manifest.json").write_text(json.dumps({
                "stage": "paired_teacher_sft",
                "base_checkpoint": "trace-uni",
                "mm_projector_type": "ref_projector",
                "closs": True,
            }), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                launcher._validate_distinct_teacher_base(str(student), str(teacher))
            self.assertIn("trusted pre-existing candidate-only SFT checkpoint", output.getvalue())

    def test_paired_teacher_still_requires_launcher_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            student = pathlib.Path(root) / "external_student"
            teacher = pathlib.Path(root) / "paired_teacher"
            student.mkdir()
            teacher.mkdir()
            with self.assertRaises(FileNotFoundError):
                launcher._validate_distinct_teacher_base(str(student), str(teacher))

    def test_legacy_student_manifest_with_missing_architecture_is_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            student = pathlib.Path(root) / "legacy_student"
            teacher = pathlib.Path(root) / "paired_teacher"
            student.mkdir()
            teacher.mkdir()
            (student / "stage_manifest.json").write_text(json.dumps({
                "stage": "candidate_sft",
                "base_checkpoint": "trace-uni",
            }), encoding="utf-8")
            (teacher / "stage_manifest.json").write_text(json.dumps({
                "stage": "paired_teacher_sft",
                "base_checkpoint": "trace-uni",
                "mm_projector_type": "ref_projector",
                "closs": True,
            }), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                launcher._validate_distinct_teacher_base(str(student), str(teacher))
            self.assertIn("does not record mm_projector_type", output.getvalue())
            self.assertIn("does not record closs", output.getvalue())

    def test_explicit_student_architecture_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            student = pathlib.Path(root) / "student"
            teacher = pathlib.Path(root) / "paired_teacher"
            student.mkdir()
            teacher.mkdir()
            common = {"base_checkpoint": "trace-uni", "closs": True}
            (student / "stage_manifest.json").write_text(json.dumps({
                **common, "stage": "candidate_sft", "mm_projector_type": "spatial_slot",
            }), encoding="utf-8")
            (teacher / "stage_manifest.json").write_text(json.dumps({
                **common, "stage": "paired_teacher_sft", "mm_projector_type": "ref_projector",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same mm_projector_type"):
                launcher._validate_distinct_teacher_base(str(student), str(teacher))


if __name__ == "__main__":
    unittest.main()
