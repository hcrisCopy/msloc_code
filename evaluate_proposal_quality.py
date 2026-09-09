"""Evaluate whether first-stage temporal proposals are usable by Trace.

This is intentionally independent from ``evaluate_long.py``.  It consumes the
same annotation/prediction JSON schema, but answers a different question:
whether the proposal passed from Stage 1 to Stage 2 covers the forged interval
without making the second stage inspect too much irrelevant video.

The report contains, for each source/tool domain and overall:

* event recall and one-to-one F1 at several temporal-IoU thresholds;
* temporal recall (fraction of forged time covered by proposals);
* union temporal IoU between all predicted and GT intervals of a fake video;
* start/end boundary MAE on overlapping matched events;
* under-coverage (GT time missed by the proposal);
* over-coverage (proposal time that is not forged content).

Run from the repository root, for example::

    python evaluate_proposal_quality.py --gt-file path/to/test.json \
        --infer-file path/to/predictions.json --output-dir proposal_quality
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


def _as_segment_list(value: Any) -> list[tuple[float, float]]:
    """Normalise ``[s,e]`` and ``[[s,e], ...]`` to valid half-open intervals."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(item, (int, float)) for item in value
    ):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    segments = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            segments.append((start, end))
    return segments


def merge_segments(segments: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals, keeping touching intervals together."""
    ordered = sorted(segments)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def total_length(segments: Iterable[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in merge_segments(segments)))


def intersection_length(
    first: Iterable[tuple[float, float]], second: Iterable[tuple[float, float]]
) -> float:
    left, right = merge_segments(first), merge_segments(second)
    i = j = 0
    overlap = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        overlap += max(0.0, end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return float(overlap)


def temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - overlap
    return overlap / union if union > EPS else 0.0


def greedy_matches(
    predictions: list[tuple[float, float]], ground_truths: list[tuple[float, float]], threshold: float
) -> list[tuple[int, int, float]]:
    """One-to-one event matches, sorted by IoU as in standard detection evaluation."""
    candidates = [
        (temporal_iou(prediction, ground_truth), pred_index, gt_index)
        for pred_index, prediction in enumerate(predictions)
        for gt_index, ground_truth in enumerate(ground_truths)
    ]
    candidates.sort(reverse=True)
    matched_predictions, matched_ground_truths, matches = set(), set(), []
    for iou, pred_index, gt_index in candidates:
        if iou < threshold:
            break
        if pred_index in matched_predictions or gt_index in matched_ground_truths:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truths.add(gt_index)
        matches.append((pred_index, gt_index, float(iou)))
    return matches


def _domain(entry: dict[str, Any], fallback_key: str) -> str:
    value = entry.get(fallback_key) or entry.get("tool_domain") or entry.get("source")
    if not value and entry.get("annotations"):
        annotation = entry["annotations"][0]
        if isinstance(annotation, dict):
            value = annotation.get(fallback_key) or annotation.get("tool_domain") or annotation.get("model")
    return str(value) if value not in (None, "") else "unknown"


def _gt_segments(entry: dict[str, Any]) -> list[tuple[float, float]]:
    if entry.get("type") != "fake":
        return []
    segments = []
    for annotation in entry.get("annotations", []):
        if isinstance(annotation, dict):
            segments.extend(_as_segment_list(annotation.get("segment")))
    if not segments:
        duration = entry.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            # This follows the existing dataloader's convention for a fake
            # video with no explicit temporal annotation.
            segments = [(0.0, float(duration))]
    return merge_segments(segments)


def _prediction_segments(entry: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not entry:
        return []
    inference = entry.get("model_inference", entry)
    if not isinstance(inference, dict):
        return []
    return merge_segments(_as_segment_list(inference.get("segment")))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > EPS else None


def evaluate(
    gt_entries: list[dict[str, Any]], prediction_entries: list[dict[str, Any]], thresholds: list[float], domain_key: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_map = {str(entry.get("video_path")): entry for entry in prediction_entries if entry.get("video_path")}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in gt_entries:
        groups[_domain(entry, domain_key)].append(entry)
    groups["Total"] = gt_entries

    report: dict[str, Any] = {"schema_version": 2, "iou_thresholds": thresholds, "domains": {}}
    rows: list[dict[str, Any]] = []

    for domain, entries in groups.items():
        fake_video_rows = []
        event_counts = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
        start_errors, end_errors, match_ious = [], [], []
        real_predicted_seconds, real_durations = [], []

        for gt_entry in entries:
            video_path = str(gt_entry.get("video_path", ""))
            prediction = prediction_map.get(video_path)
            predicted = _prediction_segments(prediction)
            gt = _gt_segments(gt_entry)
            video_type = str(gt_entry.get("type", "unknown"))
            predicted_duration = total_length(predicted)
            duration = float(gt_entry.get("duration", 0.0) or 0.0)

            row: dict[str, Any] = {
                "domain": domain,
                "video_path": video_path,
                "video_type": video_type,
                "num_gt_events": len(gt),
                "num_pred_events": len(predicted),
                "gt_seconds": total_length(gt),
                "proposal_seconds": predicted_duration,
                "prediction_found": prediction is not None,
            }
            if video_type != "fake":
                row.update({
                    "temporal_recall": None,
                    "union_temporal_iou": None,
                    "under_coverage_ratio": None,
                    "over_coverage_ratio": 1.0 if predicted_duration > EPS else 0.0,
                    "start_boundary_mae_seconds": None,
                    "end_boundary_mae_seconds": None,
                })
                if duration > EPS:
                    real_predicted_seconds.append(predicted_duration)
                    real_durations.append(duration)
                rows.append(row)
                continue

            overlap = intersection_length(predicted, gt)
            gt_duration = total_length(gt)
            union = predicted_duration + gt_duration - overlap
            under_coverage = 1.0 - (overlap / gt_duration if gt_duration > EPS else 1.0)
            over_coverage = 1.0 - (overlap / predicted_duration if predicted_duration > EPS else 1.0)
            row.update({
                "temporal_recall": overlap / gt_duration if gt_duration > EPS else 1.0,
                "union_temporal_iou": overlap / union if union > EPS else 1.0,
                "under_coverage_ratio": under_coverage,
                "over_coverage_ratio": over_coverage,
                "missed_seconds": gt_duration - overlap,
                "redundant_seconds": predicted_duration - overlap,
            })

            # Boundary errors use the least restrictive positive-overlap
            # matching.  Unmatched events are reported through recall and
            # under-coverage instead of assigning an arbitrary boundary value.
            positive_matches = greedy_matches(predicted, gt, threshold=EPS)
            if positive_matches:
                row["start_boundary_mae_seconds"] = float(np.mean([
                    abs(predicted[pred_index][0] - gt[gt_index][0])
                    for pred_index, gt_index, _ in positive_matches
                ]))
                row["end_boundary_mae_seconds"] = float(np.mean([
                    abs(predicted[pred_index][1] - gt[gt_index][1])
                    for pred_index, gt_index, _ in positive_matches
                ]))
                start_errors.extend(abs(predicted[p][0] - gt[g][0]) for p, g, _ in positive_matches)
                end_errors.extend(abs(predicted[p][1] - gt[g][1]) for p, g, _ in positive_matches)
                match_ious.extend(iou for _, _, iou in positive_matches)
            else:
                row["start_boundary_mae_seconds"] = None
                row["end_boundary_mae_seconds"] = None

            for threshold in thresholds:
                matches = greedy_matches(predicted, gt, threshold)
                event_counts[threshold]["tp"] += len(matches)
                event_counts[threshold]["fp"] += len(predicted) - len(matches)
                event_counts[threshold]["fn"] += len(gt) - len(matches)
                row[f"event_recall_iou_{threshold:g}"] = _safe_divide(len(matches), len(gt))
            fake_video_rows.append(row)
            rows.append(row)

        fake_count = len(fake_video_rows)
        summary: dict[str, Any] = {
            "num_videos": len(entries),
            "num_fake_videos": fake_count,
            "num_real_videos": len(entries) - fake_count,
            "proposal_found_rate_fake": _mean([float(row["proposal_seconds"] > EPS) for row in fake_video_rows]),
            "mean_temporal_recall": _mean([row["temporal_recall"] for row in fake_video_rows]),
            "mean_union_temporal_iou": _mean([row["union_temporal_iou"] for row in fake_video_rows]),
            "mean_under_coverage_ratio": _mean([row["under_coverage_ratio"] for row in fake_video_rows]),
            "mean_over_coverage_ratio": _mean([row["over_coverage_ratio"] for row in fake_video_rows]),
            "mean_missed_seconds": _mean([row["missed_seconds"] for row in fake_video_rows]),
            "mean_redundant_seconds": _mean([row["redundant_seconds"] for row in fake_video_rows]),
            "start_boundary_mae_seconds": _mean(start_errors),
            "end_boundary_mae_seconds": _mean(end_errors),
            "mean_matched_event_iou": _mean(match_ious),
            "real_video_proposal_duration_ratio": _safe_divide(sum(real_predicted_seconds), sum(real_durations)),
            "event_metrics": {},
        }
        for threshold, counts in event_counts.items():
            precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"]) or 0.0
            recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"]) or 0.0
            summary["event_metrics"][f"IoU@{threshold:g}"] = {
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        report["domains"][domain] = summary
    return report, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-file", required=True, type=Path, help="Ground-truth annotation JSON")
    parser.add_argument("--infer-file", required=True, type=Path, help="Stage-1 inference/predictions JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("proposal_quality"))
    parser.add_argument("--iou-thresholds", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--domain-key", default="tool_domain", help="Root/annotation field used for per-domain reports")
    args = parser.parse_args()
    thresholds = [float(value.strip()) for value in args.iou_thresholds.split(",") if value.strip()]
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        parser.error("--iou-thresholds must be comma-separated values in [0, 1]")
    if not args.gt_file.is_file() or not args.infer_file.is_file():
        raise FileNotFoundError("--gt-file and --infer-file must exist")

    gt_entries = json.loads(args.gt_file.read_text(encoding="utf-8"))
    prediction_entries = json.loads(args.infer_file.read_text(encoding="utf-8"))
    if not isinstance(gt_entries, list) or not isinstance(prediction_entries, list):
        raise ValueError("Both JSON files must contain a list of video entries")
    report, rows = evaluate(gt_entries, prediction_entries, thresholds, args.domain_key)
    report["inputs"] = {"gt_file": str(args.gt_file), "infer_file": str(args.infer_file)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "proposal_quality_summary.json"
    csv_path = args.output_dir / "proposal_quality_per_video.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    total = report["domains"]["Total"]
    print("Proposal quality (fake videos):")
    print(f"  Recall:             {total['mean_temporal_recall']!s}")
    print(f"  Union temporal IoU: {total['mean_union_temporal_iou']!s}")
    print(f"  Under-coverage:     {total['mean_under_coverage_ratio']!s}")
    print(f"  Over-coverage:      {total['mean_over_coverage_ratio']!s}")
    print(f"Saved summary: {json_path}")
    print(f"Saved per-video rows: {csv_path}")


if __name__ == "__main__":
    main()
