"""Zero-shot evaluation of a TASLE-trained DeMamba model on ActivityForensics.

The Hugging Face snapshot is already ``metadata/test.csv`` plus raw MP4 files;
it needs no extraction. All defaults are relative to the MSLoc_code root.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataloader import normalization_params, resize_interpolation
from util import build_model, uses_marginal_fake_score


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_GENERATORS = ("wan", "scifi", "fcvg", "vace", "ltx", "vidu")
IN_DOMAIN = frozenset({"wan", "vace", "ltx", "vidu"})
OUT_OF_DOMAIN = frozenset({"scifi", "fcvg"})
TASLE_TIOUS = (0.1, 0.3, 0.5, 0.7)
KNOWN_OUTPUTS = (
    "dataset_manifest.json", "window_predictions.json", "predictions.json",
    "metrics.json", "metrics.csv", "detailed_results.csv",
    "summary_results.json", "skipped_videos.json", "failures.json",
)


@dataclass(frozen=True)
class VideoAnnotation:
    video_id: str
    duration: float | None
    segments: tuple[tuple[float, float], ...]
    split: str
    generator: str


@dataclass(frozen=True)
class EvaluationItem:
    record: VideoAnnotation
    path: Path
    decoded_duration: float
    duration: float
    window_count: int


def normalise_generator(value: str) -> str:
    text = value.strip().lower().replace("_", "-")
    text = re.sub(r"^\d+-", "", text)
    return {
        "wan2.1": "wan", "wan-2.1": "wan", "vace-1.3b": "vace",
        "vace1.3b": "vace", "ltx-video": "ltx", "ltxvideo": "ltx",
    }.get(text, text or "unknown")


def parse_device_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--device-ids must be comma-separated integers") from exc
    if not ids or min(ids) < 0 or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--device-ids must be unique non-negative integers")
    return ids


def parse_generators(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(normalise_generator(x) for x in value.split(",") if x.strip()))
    if not values:
        raise argparse.ArgumentTypeError("--generators cannot be empty")
    return values


def parse_official_txt(path: Path) -> list[VideoAnnotation]:
    generator = normalise_generator(path.stem.split("@", 1)[-1])
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) not in (2, 3):
            raise ValueError(f"{path}:{line_number}: malformed annotation: {raw!r}")
        video_id = fields[0]
        try:
            duration = float(fields[1])
            segments = [] if len(fields) == 2 else [
                tuple(map(float, pair.split("=", 1))) for pair in fields[2].split("+")
            ]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid duration/segment") from exc
        clipped = tuple((max(0.0, a), min(duration, b)) for a, b in segments if min(duration, b) > max(0.0, a))
        records.append(VideoAnnotation(video_id, duration, clipped, path.stem, generator))
    if not records:
        raise ValueError(f"No valid records in {path}")
    return records


def parse_hf_csv(path: Path) -> list[VideoAnnotation]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file_name" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a file_name column")
        for line_number, row in enumerate(reader, 2):
            video_id = (row.get("file_name") or "").strip()
            if not video_id:
                raise ValueError(f"{path}:{line_number}: empty file_name")
            segments = tuple(
                (float(a), float(b)) for a, b in re.findall(
                    r"(?:^|\+)(\d+(?:\.\d+)?)=(\d+(?:\.\d+)?)=", Path(video_id).stem
                )
            )
            if any(b <= a for a, b in segments):
                raise ValueError(f"{path}:{line_number}: non-positive segment")
            if segments:
                parent = Path(video_id.replace("\\", "/")).parent.name
                generator = normalise_generator(parent)
            else:
                generator = "real"
            records.append(VideoAnnotation(video_id, None, segments, f"{path.stem}@{generator}", generator))
    if not records:
        raise ValueError(f"No valid rows in {path}")
    return records


def load_annotations(root: Path, pattern: str, source: str) -> list[VideoAnnotation]:
    if not root.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {root}")
    if source == "auto":
        source = "hf-csv" if any(root.rglob("test*.csv")) else "official-txt"
    if source == "hf-csv" and pattern == "test@*.txt":
        pattern = "test*.csv"
    suffix = ".csv" if source == "hf-csv" else ".txt"
    parser = parse_hf_csv if source == "hf-csv" else parse_official_txt
    files = sorted(p for p in root.rglob(f"*{suffix}") if fnmatch.fnmatch(p.name, pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern!r} annotation files below {root}")
    records, seen = [], set()
    for path in tqdm(files, desc="Loading annotations", unit="file"):
        parsed = parser(path)
        tqdm.write(f"[ok] {path.name}: {len(parsed)} rows")
        for record in parsed:
            key = f"{record.split}:{record.video_id}"
            if key in seen:
                raise ValueError(f"Duplicate annotation: {key}")
            seen.add(key)
            records.append(record)
    return records


def select_generators(records: Sequence[VideoAnnotation], generators: Sequence[str]) -> list[VideoAnnotation]:
    selected = [r for r in records if not r.segments or r.generator in generators]
    excluded: dict[str, int] = defaultdict(int)
    for record in records:
        if record.segments and record.generator not in generators:
            excluded[record.generator] += 1
    if excluded:
        print("[info] Excluded fake generators: " + ", ".join(f"{k}={v}" for k, v in sorted(excluded.items())))
    if not selected:
        raise ValueError("No records remain after generator filtering")
    return selected


def video_key(value: str) -> str:
    return value.replace("\\", "/").lower().removesuffix(".mp4")


def build_video_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Video root not found: {root}")
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not candidates:
        archives = [p for p in root.rglob("*") if p.is_file() and (
            p.suffix.lower() in {".zip", ".tar", ".tgz"}
            or p.name.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
        )]
        note = f" Found {len(archives)} archive(s); extract them first." if archives else ""
        raise FileNotFoundError(
            f"No raw videos below {root}.{note} The official HF snapshot contains direct MP4 files and needs no extraction."
        )
    relative, basename = {}, defaultdict(list)
    for path in tqdm(candidates, desc="Indexing raw videos", unit="video"):
        relative[video_key(str(path.relative_to(root)))] = path
        basename[video_key(path.name)].append(path)
    return relative, basename


def resolve_video(record: VideoAnnotation, root: Path, relative: dict, basename: dict) -> Path:
    direct = root / record.video_id
    requested = video_key(record.video_id)
    if direct.is_file():
        return direct
    if requested in relative:
        return relative[requested]
    matches = basename.get(Path(requested).name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No raw video for {record.video_id!r}")
    raise RuntimeError(f"Ambiguous basename for {record.video_id!r}: {matches[:5]}")


def video_metadata(path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV cannot open it")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise RuntimeError(f"invalid fps/frame count ({fps}, {frames})")
        return fps, frames, frames / fps
    finally:
        cap.release()


def intervals(duration: float, window: float, stride: float) -> list[tuple[float, float]]:
    return [(float(s), float(min(duration, s + window))) for s in np.arange(0, duration, stride)]


def build_manifest(records: Sequence[VideoAnnotation], root: Path, relative: dict, basename: dict,
                   window: float, stride: float) -> tuple[list[EvaluationItem], list[dict]]:
    items, skipped = [], []
    for record in tqdm(records, desc="Validating test videos", unit="video"):
        try:
            path = resolve_video(record, root, relative, basename)
            _, _, decoded = video_metadata(path)
            duration = record.duration if record.duration is not None else decoded
            items.append(EvaluationItem(record, path, decoded, duration, len(intervals(duration, window, stride))))
        except Exception as exc:
            skipped.append({
                "video_id": record.video_id,
                "generator": record.generator,
                "reason": str(exc),
            })
    return items, skipped


def domain(generator: str) -> str:
    if generator in IN_DOMAIN:
        return "in_domain"
    if generator in OUT_OF_DOMAIN:
        return "out_of_domain"
    return "real" if generator == "real" else "unassigned"


def print_manifest(items: Sequence[EvaluationItem], skipped: Sequence[dict],
                   annotation_count: int, generators: Sequence[str]) -> None:
    print("\n" + "=" * 72)
    print("ActivityForensics evaluation-set confirmation (before model loading)")
    print("=" * 72)
    print(f"Selected annotation rows : {annotation_count}")
    print(f"Available test videos    : {len(items)}")
    print(f"Missing/unreadable       : {len(skipped)}")
    print(f"Available real videos    : {sum(not x.record.segments for x in items)}")
    for generator in generators:
        count = sum(bool(x.record.segments) and x.record.generator == generator for x in items)
        print(f"Available fake {generator:<9}: {count:>6}  [{'OOD' if generator in OUT_OF_DOMAIN else 'ID'}]")
    print(f"Available inference windows: {sum(x.window_count for x in items)}")
    print("=" * 72 + "\n")


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device, device_ids: Sequence[int]) -> torch.nn.Module:
    model = build_model(
        cfg["model"], neuron_indices_path=cfg.get("neuron_indices_path"),
        xclip_model_path=cfg.get("xclip_model_path"), dinov3_repo_path=cfg.get("dinov3_repo_path"),
        dinov3_weights_path=cfg.get("dinov3_weights_path"), dinov3_model_name=cfg.get("dinov3_model_name"),
        dinov3_backend=cfg.get("dinov3_backend"), dinov3_hf_model_path=cfg.get("dinov3_hf_model_path"),
        dinov2_hf_model_path=cfg.get("dinov2_hf_model_path"),
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except (OSError, RuntimeError, EOFError) as exc:
        file_size = checkpoint_path.stat().st_size
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        raise RuntimeError(
            "Checkpoint could not be deserialized. It is probably incomplete or corrupted during upload. "
            f"File={checkpoint_path}, bytes={file_size}, sha256={digest.hexdigest()}. "
            "Compare these values with the source file and re-upload the checkpoint if they differ. "
            f"Original torch.load error: {exc}"
        ) from exc
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/config mismatch. Missing={missing[:8]}, unexpected={unexpected[:8]}")
    model.eval()
    print(f"[ok] Loaded checkpoint: {checkpoint_path}")
    if len(device_ids) > 1:
        print("[note] This evaluator uses only the first requested GPU")
    return model


def decode_window(cap: cv2.VideoCapture, start: float, end: float, count: int,
                  image_size: int, normalisation: tuple, interpolation: int) -> np.ndarray:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise RuntimeError("invalid FPS/frame count")
    first = min(max(int(start * fps), 0), frame_count - 1)
    last = min(max(int(end * fps), first), frame_count - 1)
    available = last - first + 1
    indices = list(range(first, last + 1, max(1, (last - first) // count)))[:count] if available >= count else list(range(first, last + 1))
    indices.extend([indices[-1]] * (count - len(indices)))
    mean, std = (np.asarray(x, dtype=np.float32) for x in normalisation)
    output, previous = [], None
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            if previous is None:
                raise RuntimeError(f"cannot decode frame {index}")
            frame = previous
        else:
            previous = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=interpolation).astype(np.float32) / 255.0
        output.append(np.transpose((image - mean) / std, (2, 0, 1)))
    return np.stack(output)


def merge_windows(windows: Sequence[dict]) -> list[dict]:
    proposals, active = [], None
    for item in windows:
        if not item["foreground"]:
            if active is not None:
                proposals.append(active)
                active = None
        elif active is None:
            active = {"segment": [item["start"], item["end"]], "score": item["score"], "window_count": 1}
        else:
            active["segment"][1] = item["end"]
            active["score"] = max(active["score"], item["score"])
            active["window_count"] += 1
    if active is not None:
        proposals.append(active)
    return proposals


def infer_video(model: torch.nn.Module, item: EvaluationItem, device: torch.device, frame_count: int,
                image_size: int, window: float, stride: float, batch_size: int, normalisation: tuple,
                interpolation: int, marginal_fake_score: bool, progress: Callable[[int], None]) -> dict:
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {item.path}")
    windows = []
    all_intervals = intervals(item.duration, window, stride)
    try:
        for offset in range(0, len(all_intervals), batch_size):
            batch_intervals = all_intervals[offset:offset + batch_size]
            batch = np.stack([
                decode_window(cap, a, b, frame_count, image_size, normalisation, interpolation)
                for a, b in batch_intervals
            ])
            tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=1).detach().cpu().numpy()
            for (start, end), probability in zip(batch_intervals, probabilities):
                if marginal_fake_score:
                    score = float(np.sum(probability[1:]))
                    foreground = score > 0.5
                else:
                    # Preserve the original XCLIP proposal score and decision rule.
                    score = float(np.max(probability[1:]))
                    foreground = score > float(probability[0])
                windows.append({
                    "start": start, "end": end, "foreground": foreground,
                    "score": score, "predicted_class": int(np.argmax(probability)),
                    "probabilities": probability.tolist(),
                })
            progress(len(batch_intervals))
    finally:
        cap.release()
    gt = [[max(0.0, a), min(item.duration, b)] for a, b in item.record.segments if min(item.duration, b) > max(0.0, a)]
    return {
        "video_id": item.record.video_id, "split": item.record.split,
        "generator": item.record.generator, "domain": domain(item.record.generator),
        "path": str(item.path), "duration": item.duration, "ground_truth": gt,
        "windows": windows, "proposals": merge_windows(windows),
    }


def temporal_iou(segment: Sequence[float], targets: np.ndarray) -> np.ndarray:
    if len(targets) == 0:
        return np.empty(0)
    intersection = np.maximum(0.0, np.minimum(segment[1], targets[:, 1]) - np.maximum(segment[0], targets[:, 0]))
    union = segment[1] - segment[0] + targets[:, 1] - targets[:, 0] - intersection
    return intersection / np.maximum(union, 1e-12)


def average_precision(predictions: Sequence[dict], ground_truth: dict[str, np.ndarray], threshold: float) -> float:
    total = sum(len(x) for x in ground_truth.values())
    if not total:
        return 0.0
    ordered = sorted(predictions, key=lambda x: x["score"], reverse=True)
    used = {key: np.zeros(len(value), dtype=bool) for key, value in ground_truth.items()}
    tp, fp = np.zeros(len(ordered)), np.zeros(len(ordered))
    for index, prediction in enumerate(ordered):
        overlaps = temporal_iou(prediction["segment"], ground_truth.get(prediction["video_id"], np.empty((0, 2))))
        if len(overlaps):
            best = int(np.argmax(overlaps))
            if overlaps[best] >= threshold and not used[prediction["video_id"]][best]:
                tp[index], used[prediction["video_id"]][best] = 1, True
                continue
        fp[index] = 1
    recall = np.cumsum(tp) / total
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    return float(np.mean([np.max(precision[recall >= point], initial=0.0) for point in np.linspace(0, 1, 101)]))


def average_recall(predictions: dict[str, list[dict]], ground_truth: dict[str, np.ndarray], k: int,
                   thresholds: Sequence[float]) -> float:
    total = sum(len(x) for x in ground_truth.values())
    if not total:
        return 0.0
    values = []
    for threshold in thresholds:
        hits = 0
        for video_id, targets in ground_truth.items():
            ranked = sorted(predictions.get(video_id, []), key=lambda x: x["score"], reverse=True)[:k]
            proposals = np.asarray([x["segment"] for x in ranked], dtype=float).reshape(-1, 2)
            for target in targets:
                if len(proposals) and np.max(temporal_iou(target, proposals), initial=0.0) >= threshold:
                    hits += 1
        values.append(hits / total)
    return float(np.mean(values))


def activity_metrics(results: Sequence[dict], ap_tious: Sequence[float], ar_tious: Sequence[float], ar_ks: Sequence[int]) -> dict:
    ground_truth, per_video, predictions = {}, {}, []
    for result in results:
        video_id = result["video_id"]
        ground_truth[video_id] = np.asarray(result["ground_truth"], dtype=float).reshape(-1, 2)
        current = [{"video_id": video_id, **p} for p in result["proposals"]]
        per_video[video_id] = current
        predictions.extend(current)
    real = [r for r in results if not r["ground_truth"]]
    metrics = {
        "videos": len(results), "positive_videos": len(results) - len(real), "real_videos": len(real),
        "proposals": len(predictions),
    }
    aps = []
    for threshold in ap_tious:
        value = average_precision(predictions, ground_truth, threshold)
        metrics[f"AP@{threshold:.2f}"] = value
        aps.append(value)
    metrics["mAP"] = float(np.mean(aps)) if aps else 0.0
    for k in ar_ks:
        metrics[f"AR@{k}"] = average_recall(per_video, ground_truth, k, ar_tious)
    metrics["real_video_FPR"] = sum(bool(r["proposals"]) for r in real) / len(real) if real else 0.0
    return metrics


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grid_counts(result: dict, resolution: float = 0.01) -> dict:
    count = int(math.ceil(result["duration"] / resolution))
    pred, gt = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    def mark(mask: np.ndarray, segments: Iterable[Sequence[float]]) -> None:
        for start, end in segments:
            first = max(0, int(math.floor(max(0.0, start) / resolution)))
            last = min(count, int(math.ceil(min(result["duration"], end) / resolution)))
            mask[first:last] = True
    mark(pred, (x["segment"] for x in result["proposals"]))
    mark(gt, result["ground_truth"])
    return {"tp": int(np.sum(pred & gt)), "fp": int(np.sum(pred & ~gt)),
            "fn": int(np.sum(~pred & gt)), "tn": int(np.sum(~pred & ~gt))}


def match_counts(result: dict, threshold: float) -> dict:
    predictions = [x["segment"] for x in result["proposals"]]
    targets = np.asarray(result["ground_truth"], dtype=float).reshape(-1, 2)
    pairs = []
    for pi, prediction in enumerate(predictions):
        for gi, overlap in enumerate(temporal_iou(prediction, targets)):
            if overlap >= threshold:
                pairs.append((float(overlap), pi, gi))
    matched_p, matched_g = set(), set()
    for _, pi, gi in sorted(pairs, reverse=True):
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi); matched_g.add(gi)
    return {"tp": len(matched_p), "fp": len(predictions) - len(matched_p), "fn": len(targets) - len(matched_g)}


def tasle_metrics(results: Sequence[dict]) -> dict:
    det_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for result in results:
        for key, value in grid_counts(result).items():
            det_counts[key] += value
    detection = prf(det_counts["tp"], det_counts["fp"], det_counts["fn"])
    segment, loc_values = {}, []
    for threshold in TASLE_TIOUS:
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for result in results:
            for key, value in match_counts(result, threshold).items():
                counts[key] += value
        values = {**prf(counts["tp"], counts["fp"], counts["fn"]), **counts}
        segment[f"IoU@{threshold:.1f}"] = values
        loc_values.append(values["f1"])
    accuracy = sum(bool(x["proposals"]) == bool(x["ground_truth"]) for x in results) / len(results)
    f1loc = float(np.mean(loc_values))
    return {
        "Det_Acc": accuracy, "Loc_F1": detection["f1"], "Loc_IoU": f1loc,
        "F1Det": detection["f1"], "F1Loc": f1loc,
        "paper": {"F1Det": detection["f1"], "F1Det_precision": detection["precision"],
                  "F1Det_recall": detection["recall"], **{f"F1Det_{k}": v for k, v in det_counts.items()},
                  "F1Loc": f1loc},
        "video_level": {"accuracy": accuracy}, "frame_level": detection,
        "segment_level": segment,
        "counts": {"total_videos": len(results), "fake_videos": sum(bool(x["ground_truth"]) for x in results),
                   "real_videos": sum(not x["ground_truth"] for x in results)},
    }


def report_groups(results: Sequence[dict], generators: Sequence[str]) -> dict[str, list[dict]]:
    real = [x for x in results if not x["ground_truth"]]
    fake = [x for x in results if x["ground_truth"]]
    id_fake = [x for x in fake if x["generator"] in IN_DOMAIN]
    ood_fake = [x for x in fake if x["generator"] in OUT_OF_DOMAIN]
    groups = {"all": list(results)}
    if id_fake:
        groups["in_domain"] = real + id_fake
    if ood_fake:
        groups["out_of_domain"] = real + ood_fake
    for generator in generators:
        generator_fake = [x for x in fake if x["generator"] == generator]
        if generator_fake:
            groups[generator] = real + generator_fake
    return {key: value for key, value in groups.items() if value}


def evaluate_groups(results: Sequence[dict], generators: Sequence[str], ap_tious: Sequence[float],
                    ar_tious: Sequence[float], ar_ks: Sequence[int]) -> dict:
    return {name: {"tasle": tasle_metrics(group), "activityforensics": activity_metrics(group, ap_tious, ar_tious, ar_ks)}
            for name, group in report_groups(results, generators).items()}


def print_metrics(metrics: dict) -> None:
    for name, report in metrics.items():
        tasle, activity = report["tasle"], report["activityforensics"]
        print(f"\n--- Domain: {name} ---")
        print(f"TASLE Det_Acc={tasle['Det_Acc']:.4f} Loc_F1/F1Det={tasle['F1Det']:.4f} Loc_IoU/F1Loc={tasle['F1Loc']:.4f}")
        for threshold, values in tasle["segment_level"].items():
            print(f"  {threshold}: Precision={values['precision']:.4f}, Recall={values['recall']:.4f}, F1={values['f1']:.4f}")
        aps = "  ".join(f"{k}={v:.4f}" for k, v in activity.items() if k.startswith("AP@"))
        ars = "  ".join(f"{k}={v:.4f}" for k, v in activity.items() if k.startswith("AR@"))
        print(f"ActivityForensics mAP={activity['mAP']:.4f}  {aps}")
        print(f"ActivityForensics {ars}  real_video_FPR={activity['real_video_FPR']:.4f}")


def compatible_predictions(results: Sequence[dict]) -> list[dict]:
    output = []
    for result in results:
        predicted = [x["segment"] for x in result["proposals"]]
        output.append({
            "video_path": result["video_id"], "duration": result["duration"],
            "type": "fake" if result["ground_truth"] else "real", "tool_domain": result["generator"],
            "annotations": [{"segment": x, "segment_label": "fake", "tool_domain": result["generator"]}
                            for x in result["ground_truth"]],
            "model_inference": {"type": "fake" if predicted else "real", "segment": predicted},
        })
    return output


def dump_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def flatten(value: dict, prefix: str = "") -> dict:
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        output.update(flatten(item, name) if isinstance(item, dict) else {name: item})
    return output


def clean_outputs(root: Path) -> None:
    removed = []
    for name in KNOWN_OUTPUTS:
        path = root / name
        if path.is_file():
            path.unlink(); removed.append(name)
    print(f"[clean] Removed {len(removed)} known old artifact(s): {', '.join(removed) if removed else 'none'}")


def write_manifest(path: Path, items: Sequence[EvaluationItem], skipped: Sequence[dict],
                   annotation_count: int, generators: Sequence[str]) -> None:
    dump_json(path, {
        "selected_generators": list(generators), "in_domain_generators": sorted(IN_DOMAIN & set(generators)),
        "out_of_domain_generators": sorted(OUT_OF_DOMAIN & set(generators)),
        "evaluation_scope": "partial" if skipped else "complete",
        "selected_annotation_count": annotation_count,
        "available_video_count": len(items), "skipped_video_count": len(skipped),
        "window_count": sum(x.window_count for x in items),
        "videos": [{"video_id": x.record.video_id, "path": str(x.path), "generator": x.record.generator,
                    "domain": domain(x.record.generator), "is_fake": bool(x.record.segments),
                    "duration": x.duration, "decoded_duration": x.decoded_duration,
                    "window_count": x.window_count} for x in items],
    })


def write_detailed_csv(path: Path, results: Sequence[dict]) -> None:
    fields = ["video_id", "generator", "domain", "duration", "true_type", "predicted_type", "correct",
              "ground_truth_segments", "predicted_segments", "windows"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results:
            true_fake, pred_fake = bool(result["ground_truth"]), bool(result["proposals"])
            writer.writerow({"video_id": result["video_id"], "generator": result["generator"],
                             "domain": result["domain"], "duration": result["duration"],
                             "true_type": "fake" if true_fake else "real",
                             "predicted_type": "fake" if pred_fake else "real", "correct": true_fake == pred_fake,
                             "ground_truth_segments": json.dumps(result["ground_truth"]),
                             "predicted_segments": json.dumps([x["segment"] for x in result["proposals"]]),
                             "windows": len(result["windows"])})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../MSLoc_data/DeMamba/full/configs/xclip_neurons_full.yaml"))
    parser.add_argument("--model-path", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/results/best_acc.pth"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("../MSLoc_data/ActivityForensics"))
    parser.add_argument("--video-root", type=Path, default=Path("../MSLoc_data/ActivityForensics"))
    parser.add_argument("--output-dir", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/eval_activityforensics"))
    parser.add_argument("--annotation-source", choices=["auto", "hf-csv", "official-txt"], default="auto")
    parser.add_argument("--annotation-pattern", default="test@*.txt")
    parser.add_argument("--generators", type=parse_generators, default=DEFAULT_GENERATORS,
                        help="Default: wan,scifi,fcvg,vace,ltx,vidu; Vidu is in-domain")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--frames-per-window", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device-ids", type=parse_device_ids, default=[0])
    parser.add_argument("--ap-tious", type=float, nargs="+", default=[0.75, 0.85, 0.95])
    parser.add_argument("--ar-tious", type=float, nargs="+", default=[0.75, 0.85, 0.95])
    parser.add_argument("--ar-ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--clean", action="store_true", help="Remove only this evaluator's known old artifacts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if max(args.device_ids) >= torch.cuda.device_count():
        raise ValueError(f"Requested GPUs {args.device_ids}, only {torch.cuda.device_count()} visible")
    if not args.config.is_file() or not args.model_path.is_file():
        raise FileNotFoundError("Config/checkpoint not found")
    if args.batch_size < 1 or any(x <= 0 for x in args.ar_ks):
        raise ValueError("Batch size and AR K values must be positive")
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    window = args.window_seconds or float(cfg.get("window_length", 2.0))
    stride = args.stride_seconds or window
    frame_count = args.frames_per_window or int(cfg.get("frames_per_window", 8))
    transform_config = cfg.get("transform_config", {})
    image_size = int(transform_config.get("image_size", 224))
    if min(window, stride, frame_count, image_size) <= 0:
        raise ValueError("Window, stride, frame count and image size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_outputs(args.output_dir)
    print("[1/5] Loading annotations")
    records = select_generators(load_annotations(args.annotation_dir, args.annotation_pattern, args.annotation_source), args.generators)
    print("[2/5] Indexing raw videos")
    relative, basename = build_video_index(args.video_root)
    print("[3/5] Validating and counting the evaluation set")
    manifest, skipped = build_manifest(records, args.video_root, relative, basename, window, stride)
    print_manifest(manifest, skipped, len(records), args.generators)
    write_manifest(args.output_dir / "dataset_manifest.json", manifest, skipped, len(records), args.generators)
    if skipped:
        dump_json(args.output_dir / "skipped_videos.json", skipped)
        print(f"[partial] Continuing with {len(manifest)} available videos; {len(skipped)} were skipped")
    elif (args.output_dir / "skipped_videos.json").is_file():
        (args.output_dir / "skipped_videos.json").unlink()
    if not manifest:
        raise RuntimeError("None of the selected test videos is currently available and readable")

    print("[4/5] Loading checkpoint")
    torch.cuda.set_device(args.device_ids[0]); device = torch.device(f"cuda:{args.device_ids[0]}")
    model = load_model(cfg, args.model_path, device, args.device_ids)
    normalisation = normalization_params(transform_config)
    interpolation = resize_interpolation(transform_config)
    marginal_fake_score = uses_marginal_fake_score(cfg.get("model"))
    print("[5/5] Running inference")
    results, failures = [], []
    with tqdm(total=sum(x.window_count for x in manifest), desc="Inference windows", unit="window", position=0) as window_bar:
        with tqdm(total=len(manifest), desc="Evaluating videos", unit="video", position=1) as video_bar:
            for item in manifest:
                windows_before = window_bar.n
                try:
                    results.append(infer_video(model, item, device, frame_count, image_size, window, stride,
                                               args.batch_size, normalisation, interpolation, marginal_fake_score,
                                               window_bar.update))
                except Exception as exc:
                    failures.append({"video_id": item.record.video_id, "generator": item.record.generator, "error": str(exc)})
                    tqdm.write(f"[warning] {item.record.video_id}: {exc}")
                    completed = window_bar.n - windows_before
                    window_bar.update(max(0, item.window_count - completed))
                finally:
                    video_bar.update(1)
    if failures:
        dump_json(args.output_dir / "failures.json", failures)
        print(f"[partial] {len(failures)} available videos failed during inference; see failures.json")
    elif (args.output_dir / "failures.json").is_file():
        (args.output_dir / "failures.json").unlink()
    if not results:
        raise RuntimeError("No video completed inference; metrics were not written")
    print(f"[result] Computing metrics from {len(results)}/{len(records)} selected annotation rows")

    dump_json(args.output_dir / "window_predictions.json", results)
    dump_json(args.output_dir / "predictions.json", compatible_predictions(results))
    metrics = evaluate_groups(results, args.generators, args.ap_tious, args.ar_tious, args.ar_ks)
    dump_json(args.output_dir / "metrics.json", metrics)
    dump_json(args.output_dir / "summary_results.json", {
        "config": str(args.config), "model_path": str(args.model_path),
        "evaluation_scope": "partial" if skipped or failures else "complete",
        "selected_annotation_count": len(records), "available_video_count": len(manifest),
        "evaluated_video_count": len(results), "preflight_skipped_count": len(skipped),
        "inference_failure_count": len(failures),
        "selected_generators": list(args.generators), "metrics": metrics,
    })
    rows = [{"group": group, **flatten(values)} for group, values in metrics.items()]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group", *sorted({key for row in rows for key in row if key != "group"})]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_detailed_csv(args.output_dir / "detailed_results.csv", results)
    print_metrics(metrics)
    print("\n[ok] Output files:")
    for name in KNOWN_OUTPUTS:
        path = args.output_dir / name
        if path.is_file():
            print(f"  - {path}")


if __name__ == "__main__":
    main()
