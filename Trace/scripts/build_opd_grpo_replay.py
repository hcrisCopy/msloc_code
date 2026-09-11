#!/usr/bin/env python3
"""Build an auditable proposal replay buffer for ref2 SFT, OPD and GRPO.

The input is the first-stage proposal JSON already consumed by ``ref2`` plus
the TASLE GT JSON.  Unlike the old ref2 path, the result preserves target
segments, evidence annotations, proposal provenance and (optionally) a real
reference-video mapping.  It never creates an OPD reference by duplicating the
candidate video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def video_id(item: Mapping[str, Any]) -> Optional[str]:
    return item.get("video_path") or item.get("video") or item.get("image_id")


def first(container: Any) -> Dict[str, Any]:
    if isinstance(container, dict):
        return container
    if isinstance(container, list) and container and isinstance(container[0], dict):
        return container[0]
    return {}


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    left, right = max(a[0], b[0]), min(a[1], b[1])
    return (left, right) if right > left else None


def temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = overlap(a, b)
    if inter is None:
        return 0.0
    return (inter[1] - inter[0]) / (max(a[1], b[1]) - min(a[0], b[0]))


def temporal_gap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    if overlap(a, b) is not None:
        return 0.0
    return max(b[0] - a[1], a[0] - b[1], 0.0)


def normalise_reference_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Accept either {candidate: descriptor} or a list of descriptors."""
    if isinstance(raw, dict) and "records" in raw:
        raw = raw["records"]
    if isinstance(raw, dict):
        result = {}
        for key, value in raw.items():
            if isinstance(value, str):
                value = {"reference_video": value}
            if not isinstance(value, dict):
                raise ValueError(f"reference map entry for {key!r} must be an object")
            result[key] = dict(value)
        return result
    if not isinstance(raw, list):
        raise ValueError("reference map must be an object or list")
    result = {}
    for row in raw:
        candidate = row.get("candidate_video") or row.get("video_path") or row.get("video")
        if not candidate:
            raise ValueError("reference map row lacks candidate_video")
        result[candidate] = dict(row)
    return result


def _finite_interval(value: Any, field_name: str) -> Tuple[float, float]:
    """Validate an interval supplied by the reference-alignment manifest."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be [start_seconds, end_seconds]")
    start, end = float(value[0]), float(value[1])
    if not start < end:
        raise ValueError(f"{field_name} must have start < end, got {value!r}")
    return start, end


def resolve_reference_segment(descriptor: Optional[Mapping[str, Any]], proposal: Tuple[float, float]) -> Optional[Dict[str, Any]]:
    """Resolve one candidate proposal to its corresponding real-video interval.

    A reference is useful only when it is aligned *per proposal*.  The manifest
    therefore has to provide one of the explicit mappings below:

    - ``same_timeline: true``: candidate and real reference share a time axis;
      the proposal interval is used unchanged.
    - ``time_offset_seconds``: reference_time = candidate_time + offset.
    - ``candidate_interval`` + ``reference_interval``: a linear alignment
      between two known corresponding intervals.
    - legacy ``reference_segment``: accepted only when it has exactly the same
      duration as this proposal.  This prevents silently reusing one unrelated
      reference window for every proposal of a video.
    """
    if not descriptor or not descriptor.get("reference_video"):
        return None
    start, end = proposal
    resolved = dict(descriptor)
    alignment_type: Optional[str] = None
    if bool(descriptor.get("same_timeline", False)):
        ref_start, ref_end = start, end
        alignment_type = "same_timeline"
    elif descriptor.get("time_offset_seconds") is not None:
        offset = float(descriptor["time_offset_seconds"])
        ref_start, ref_end = start + offset, end + offset
        alignment_type = "time_offset_seconds"
    elif descriptor.get("candidate_interval") is not None or descriptor.get("reference_interval") is not None:
        candidate_interval = _finite_interval(descriptor.get("candidate_interval"), "candidate_interval")
        reference_interval = _finite_interval(descriptor.get("reference_interval"), "reference_interval")
        scale = (reference_interval[1] - reference_interval[0]) / (candidate_interval[1] - candidate_interval[0])
        ref_start = reference_interval[0] + (start - candidate_interval[0]) * scale
        ref_end = reference_interval[0] + (end - candidate_interval[0]) * scale
        alignment_type = "linear_interval"
    elif descriptor.get("reference_segment") is not None or descriptor.get("reference_window") is not None:
        # This was the old schema.  It is valid only for a single proposal (or
        # a manifest writer that has already made it proposal-specific).
        ref_start, ref_end = _finite_interval(
            descriptor.get("reference_segment") or descriptor.get("reference_window"),
            "reference_segment",
        )
        if abs((ref_end - ref_start) - (end - start)) > 1e-3:
            raise ValueError(
                "legacy reference_segment has a different duration from proposal "
                f"{proposal}; provide same_timeline, time_offset_seconds, or "
                "candidate_interval/reference_interval for proposal-level alignment"
            )
        alignment_type = "legacy_equal_duration"
    else:
        raise ValueError(
            "reference entry needs same_timeline, time_offset_seconds, "
            "candidate_interval/reference_interval, or a duration-matched reference_segment"
        )
    if ref_start < 0 or not ref_start < ref_end:
        raise ValueError(f"resolved reference segment is invalid: {[ref_start, ref_end]}")
    resolved["reference_segment"] = [ref_start, ref_end]
    resolved["alignment_type"] = alignment_type
    return resolved


def infer_real_counterpart(video: str) -> str:
    """Match the TASLE convention already used by DeMamba's neuron probe.

    ``foo.mp4`` and ``foo_real.mp4`` share a time axis.  This is the normal
    case for fake proposals, so their reference interval is exactly the same
    proposal interval rather than a hand-written per-proposal map.
    """
    directory, filename = str(video).rsplit("/", 1) if "/" in str(video) else ("", str(video))
    stem, suffix = filename.rsplit(".", 1) if "." in filename else (filename, "")
    if stem.endswith("_real"):
        return video
    reference_name = f"{stem}_real.{suffix}" if suffix else f"{stem}_real"
    return f"{directory}/{reference_name}" if directory else reference_name


def inferred_reference_descriptor(item: Mapping[str, Any], candidate: str) -> Optional[Dict[str, Any]]:
    """Obtain the aligned real counterpart using DeMamba's pairing policy.

    DeMamba uses an explicit normal/original path in the annotation when it is
    available, otherwise the TASLE ``_real`` filename convention.  It creates
    pairs only for fake source videos.  A genuine candidate has no independent
    real counterpart and must remain a candidate-only no-event anchor rather
    than being duplicated into a misleading "pair".
    """
    if item.get("type") != "fake":
        return None
    reference_video = (
        item.get("normal_video") or item.get("normal_video_path") or
        item.get("original_video") or item.get("original_video_path") or
        infer_real_counterpart(candidate)
    )
    return {"reference_video": str(reference_video), "same_timeline": True, "mapping_source": "demamba_counterpart"}


def evidence(annotation: Mapping[str, Any], audit: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    obj, start, end = first(annotation.get("obj_cot")), first(annotation.get("bnd_cot_st")), first(annotation.get("bnd_cot_ed"))
    round4 = "Round4" in str(annotation.get("combine_dir", ""))
    result = {
        "manipulation_type": "spatio-temporal" if round4 else "temporal",
        "object_caption": obj.get("obj_caption", ""),
        "object_class": obj.get("bnd_sub_class", ""),
        "start_caption": "" if round4 else start.get("bnd_caption", ""),
        "start_class": "" if round4 else start.get("bnd_class", ""),
        "end_caption": "" if round4 else end.get("bnd_caption", ""),
        "end_class": "" if round4 else end.get("bnd_class", ""),
        # TASLE rationales were written with a reference/candidate comparison.
        # They are NOT assumed candidate-observable until an independent audit
        # explicitly marks them so.
        "candidate_observable": bool(annotation.get("candidate_observable", False)),
    }
    if audit:
        result.update({key: value for key, value in audit.items() if key in result})
    return result


def _exists_under_root(video_root: Optional[Path], video: str) -> bool:
    if video_root is None:
        return True
    path = Path(video)
    return path.is_file() if path.is_absolute() else (video_root / path).is_file()


def build_records(
    gt: List[Mapping[str, Any]],
    proposals: List[Mapping[str, Any]],
    reference_map: Dict[str, Dict[str, Any]],
    near_negative_seconds: float = 1.0,
    evidence_audit: Optional[Mapping[str, Mapping[str, Any]]] = None,
    paired_only: bool = False,
    video_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    gt_by_video = {video_id(item): item for item in gt if video_id(item)}
    records: List[Dict[str, Any]] = []
    for proposal_source in proposals:
        candidate = video_id(proposal_source)
        gt_item = gt_by_video.get(candidate)
        if not candidate or not gt_item:
            continue
        # The paired OPD setting is deliberately restricted to fake source
        # videos with an authentic counterpart.  It includes positive and
        # negative *proposals* from those videos, but excludes unrelated real
        # videos rather than fabricating a reference by copying their input.
        if paired_only and gt_item.get("type") != "fake":
            continue
        base_reference = reference_map.get(candidate) or inferred_reference_descriptor(gt_item, candidate)
        if paired_only and (not base_reference or not base_reference.get("reference_video")
                            or not _exists_under_root(video_root, str(base_reference["reference_video"]))):
            continue
        inferred = proposal_source.get("model_inference", {})
        for proposal_index, raw_segment in enumerate(inferred.get("segment", [])):
            if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
                continue
            start, end = float(raw_segment[0]), float(raw_segment[1])
            if not start < end:
                continue
            matched = []
            all_gt_segments = []
            for ann in gt_item.get("annotations", []):
                segment = ann.get("segment", [])
                if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                    continue
                actual = (float(segment[0]), float(segment[1]))
                all_gt_segments.append(actual)
                inter = overlap((start, end), actual)
                if inter is None:
                    continue
                audit_key = f"{candidate}::{actual[0]:.3f}-{actual[1]:.3f}"
                matched.append({
                    "relative_segment": [inter[0] - start, inter[1] - start],
                    "absolute_segment": [inter[0], inter[1]],
                    "annotation": ann,
                    "evidence": evidence(ann, (evidence_audit or {}).get(audit_key)),
                })
            descriptor = resolve_reference_segment(base_reference, (start, end))
            best_iou = max((temporal_iou((start, end), gt_segment) for gt_segment in all_gt_segments), default=0.0)
            nearest_gap = min((temporal_gap((start, end), gt_segment) for gt_segment in all_gt_segments), default=float("inf"))
            if matched:
                bucket = "positive" if best_iou >= 0.5 else "hard_positive"
            elif nearest_gap <= near_negative_seconds:
                bucket = "near_hard_negative"
            else:
                bucket = "real_false_positive"
            record = {
                "id": f"{candidate}::{proposal_index}::{start:.3f}-{end:.3f}",
                "candidate_video": candidate,
                "proposal": [start, end],
                "proposal_duration": end - start,
                "source": {
                    "stage1_type": proposal_source.get("type"),
                    "proposal_index": proposal_index,
                    "stage1_response": (inferred.get("response") or [""] * (proposal_index + 1))[proposal_index] if proposal_index < len(inferred.get("response") or []) else "",
                },
                "is_positive": bool(matched),
                "replay_bucket": bucket,
                "max_gt_iou": best_iou,
                "nearest_gt_gap_seconds": None if nearest_gap == float("inf") else nearest_gap,
                "targets": matched,
                "reference": descriptor,
            }
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="TASLE training annotation JSON")
    parser.add_argument("--proposals", required=True, help="Stage-1 proposal JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-map", help="Optional overrides for fake->real counterpart paths or time alignment")
    parser.add_argument("--paired-only", action="store_true", help="Keep only proposals from fake videos with an existing authentic counterpart. This is the OPD paired-video setting.")
    parser.add_argument("--video-root", help="Video root used to verify inferred/overridden reference files; required by --paired-only")
    parser.add_argument("--require-reference", action="store_true", help="Fail if a retained positive proposal lacks a real counterpart")
    parser.add_argument("--near-negative-seconds", type=float, default=1.0, help="Gap threshold for near-hard-negative replay bucket")
    parser.add_argument("--evidence-audit", help="Optional candidate-observability diagnostic keyed as video::start-end; it does not gate GRPO reward")
    args = parser.parse_args()
    if args.paired_only and not args.video_root:
        parser.error("--paired-only requires --video-root so missing _real.mp4 counterparts can be excluded")

    with open(args.gt, "r", encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(args.proposals, "r", encoding="utf-8") as handle:
        proposals = json.load(handle)
    reference_map: Dict[str, Dict[str, Any]] = {}
    if args.reference_map:
        with open(args.reference_map, "r", encoding="utf-8") as handle:
            reference_map = normalise_reference_map(json.load(handle))
    evidence_audit = None
    if args.evidence_audit:
        with open(args.evidence_audit, "r", encoding="utf-8") as handle:
            evidence_audit = json.load(handle)
        if not isinstance(evidence_audit, dict):
            raise ValueError("evidence_audit must be a JSON object keyed as video::start-end")
    records = build_records(
        gt, proposals, reference_map, args.near_negative_seconds, evidence_audit,
        paired_only=args.paired_only,
        video_root=Path(args.video_root).resolve() if args.video_root else None,
    )
    if args.paired_only and not records:
        raise SystemExit(
            "Paired-only replay is empty: no stage-1 proposals came from a fake video "
            "with a verified same-timeline real counterpart. Check --video-root, the "
            "proposal paths, and the *_real.mp4 naming/annotation mapping."
        )
    missing = [r["id"] for r in records if r["is_positive"] and (not r["reference"] or not r["reference"].get("reference_video"))]
    if args.require_reference and missing:
        raise SystemExit(
            f"{len(missing)} positive replay records lack a real reference_video mapping; refusing to fabricate OPD pairs. "
            f"First missing id: {missing[0]}"
        )
    output = {
        "schema_version": 2,
        "paired_only": bool(args.paired_only),
        "records": records,
        "statistics": {
            "records": len(records),
            "positive": sum(r["is_positive"] for r in records),
            "negative": sum(not r["is_positive"] for r in records),
            "with_reference": sum(bool(r["reference"] and r["reference"].get("reference_video")) for r in records),
            "positive_without_reference": len(missing),
            "buckets": {bucket: sum(record["replay_bucket"] == bucket for record in records) for bucket in ("positive", "hard_positive", "near_hard_negative", "real_false_positive")},
            "candidate_observable_evidence": sum(any(target["evidence"].get("candidate_observable", False) for target in record["targets"]) for record in records),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output["statistics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
