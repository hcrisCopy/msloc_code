import importlib.util
import pathlib
import sys
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
qwen_judge = load_module("qwen_judge_test", "scripts/qwen3_vl_explanation_judge.py")


class OpdGrpoTests(unittest.TestCase):
    def setUp(self):
        self.spec = opd.TraceTokenSpec(
            text_vocab_size=100,
            time_vocab={"<sync>": 0, "<sep>": 1, "0": 2, "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8, "7": 9, "8": 10, "9": 11, ".": 12},
            score_vocab_size=13,
        )

    def time_ids(self, text):
        return [self.spec.time_start_id + self.spec.time_vocab[ch] for ch in text]

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

    def test_qwen_judge_is_candidate_only_and_samples_predicted_boundaries(self):
        payload = {
            "sample_id": "unit",
            "candidate_video": "candidate.mp4",
            "proposal": [10.0, 20.0],
            "predicted_segments": [[2.0, 4.0]],
            "caption": "the mouth flickers",
            "protocol": "candidate_only_v1",
            "rubric": {},
            "reference_evidence": {"object_caption": "mouth flickers"},
        }
        qwen_judge.validate_candidate_only_payload(payload)
        times = qwen_judge.requested_frame_times(payload, max_frames=8)
        self.assertIn(12.0, times)
        self.assertIn(14.0, times)
        with self.assertRaises(ValueError):
            qwen_judge.validate_candidate_only_payload({**payload, "evidence": {"object_caption": "GT text"}})


if __name__ == "__main__":
    unittest.main()
