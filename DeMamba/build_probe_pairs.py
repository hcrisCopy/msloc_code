"""Build frame-level fake/real pairs for frozen-XCLIP neuron probing.

Each output record is one *annotated fake frame* and the frame with the same
number from its temporally aligned real counterpart.  Frame numbers are read
from the extracted files themselves.  With the project's fixed 8-fps
extraction, ``frame_1.jpg`` has timestamp 0 and ``frame_n.jpg`` has timestamp
``(n - 1) / fps``.  This avoids treating a position in a directory listing as
the video frame number.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


def load_records(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value.get("pairs", [])


def load_pair_map(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    records = (json.loads(text) if text.startswith("[") or text.startswith("{")
               else [json.loads(line) for line in text.splitlines() if line.strip()])
    if isinstance(records, dict):
        if "pairs" in records or "records" in records:
            records = records.get("pairs", records.get("records", []))
        else:
            records = [{"fake_video": fake, "normal_video": normal}
                       for fake, normal in records.items()]
    mapping = {}
    for record in records:
        fake = record.get("fake_video") or record.get("fake_video_path") or record.get("video_path")
        normal = (record.get("normal_video") or record.get("normal_video_path") or
                  record.get("original_video_path"))
        if fake and normal:
            mapping[str(fake)] = str(normal)
    if not mapping:
        raise ValueError(f"No fake/normal mapping records found in {path}")
    return mapping


def normal_path_from_record(record):
    return (record.get("normal_video") or record.get("normal_video_path") or
            record.get("original_video") or record.get("original_video_path"))


def infer_real_counterpart(fake_video):
    """Tasle-CoT-10K convention: ``clip.mp4`` -> ``clip_real.mp4``."""
    directory, filename = str(fake_video).rsplit("/", 1) if "/" in str(fake_video) else ("", str(fake_video))
    stem, suffix = filename.rsplit(".", 1) if "." in filename else (filename, "")
    if stem.endswith("_real"):
        return fake_video
    real_name = f"{stem}_real.{suffix}" if suffix else f"{stem}_real"
    return f"{directory}/{real_name}" if directory else real_name


def frame_directory(frame_root: Path, video_path: str) -> Path:
    return frame_root / Path(video_path).with_suffix("")


def numbered_frame_paths(directory: Path) -> dict[int, Path]:
    """Return ``frame number -> exact filename``, never implicit offsets."""
    paths = {}
    for path in directory.glob("frame_*.jpg"):
        match = FRAME_PATTERN.match(path.name)
        if match:
            number = int(match.group(1))
            if number in paths:
                raise ValueError(f"Duplicate extracted frame number {number} in {directory}")
            paths[number] = path
    return paths


def fake_segments(record) -> list[tuple[float, float]]:
    segments = []
    for annotation in record.get("annotations", []):
        if annotation.get("segment_label", "fake") != "fake" or "segment" not in annotation:
            continue
        start, end = map(float, annotation["segment"])
        if end > start:
            segments.append((start, end))
    return segments


def choose_evenly(items: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(items) <= limit:
        return items
    # Deterministic, source-order-preserving subsampling.  It prevents long
    # fake segments from dominating the neuron statistic.
    return [items[(position * len(items)) // limit] for position in range(limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Train annotation JSON")
    parser.add_argument("--frame-root", type=Path, required=True,
                        help="Root containing 8-fps frame_<number>.jpg directories")
    parser.add_argument("--pair-map", type=Path,
                        help="Optional JSON/JSONL fake-to-normal mapping")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Extraction FPS used to map frame numbers to timestamps")
    parser.add_argument("--boundary-margin", type=float, default=0.0,
                        help="Seconds excluded at each annotated fake-segment boundary")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every nth eligible extracted fake frame")
    parser.add_argument("--max-frames-per-video", type=int, default=0,
                        help="Optional deterministic cap; 0 keeps every eligible frame")
    parser.add_argument("--max-fake-videos", type=int, default=0,
                        help="Optional deterministic cap on fake source videos; 0 keeps every usable video")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if an aligned real frame is missing instead of skipping it")
    args = parser.parse_args()
    if (args.fps <= 0 or args.boundary_margin < 0 or args.frame_stride <= 0 or
            args.max_frames_per_video < 0 or args.max_fake_videos < 0):
        parser.error("fps/stride must be positive; margins and maximums must be non-negative")

    frame_root = args.frame_root.resolve()
    mapping = load_pair_map(args.pair_map) if args.pair_map else {}
    pairs, skipped, unresolved, selected_fake_videos = [], [], [], set()
    for item in load_records(args.annotations):
        if item.get("type") != "fake":
            continue
        if args.max_fake_videos and len(selected_fake_videos) >= args.max_fake_videos:
            break
        fake_video = str(item["video_path"])
        normal_video = (mapping.get(fake_video) or normal_path_from_record(item) or
                        infer_real_counterpart(fake_video))
        segments = fake_segments(item)
        if not normal_video or not segments:
            unresolved.append(fake_video)
            continue

        fake_dir = frame_directory(frame_root, fake_video)
        normal_dir = frame_directory(frame_root, normal_video)
        fake_frames = numbered_frame_paths(fake_dir)
        normal_frames = numbered_frame_paths(normal_dir)
        fake_numbers = sorted(fake_frames)
        if not fake_numbers or not normal_frames:
            skipped.append(f"{fake_video}: missing extracted fake or real frames")
            continue

        # A frame belongs to [start, end) exactly when the timestamp assigned
        # by its *filename* belongs to that half-open annotation interval.
        # frame_1 is t=0, not t=1/fps.
        eligible = []
        for number in fake_numbers:
            timestamp = (number - 1) / args.fps
            in_fake = any(start + args.boundary_margin <= timestamp < end - args.boundary_margin
                          for start, end in segments)
            if in_fake:
                eligible.append(number)
        eligible = eligible[::args.frame_stride]
        eligible = choose_evenly(eligible, args.max_frames_per_video)

        candidate_pairs = []
        for number in eligible:
            if number not in normal_frames:
                message = (f"{fake_video}: frame_{number}.jpg is fake-labelled at "
                           f"t={(number - 1) / args.fps:.6f}, but its aligned real frame is missing")
                if args.strict:
                    raise FileNotFoundError(message)
                skipped.append(message)
                continue
            timestamp = (number - 1) / args.fps
            candidate_pairs.append({
                "pair_id": f"{Path(fake_video).stem}_frame_{number:08d}",
                "fake_video": fake_video,
                "normal_video": str(normal_video),
                "frame_number": number,
                "normal_frame_number": number,
                # Store the real filename too: old extractions sometimes use
                # zero-padded names (frame_000001.jpg).  The probe must open
                # this exact file, not reconstruct a possibly different name.
                "fake_frame_file": fake_frames[number].name,
                "normal_frame_file": normal_frames[number].name,
                "timestamp": round(timestamp, 6),
                "fps": args.fps,
            })
        # Apply the source-video cap only after the video has yielded valid,
        # aligned pairs.  A missing/broken early video therefore cannot consume
        # a smoke-test slot.
        if not candidate_pairs:
            continue
        selected_fake_videos.add(fake_video)
        pairs.extend(candidate_pairs)

    if not pairs:
        raise RuntimeError("No valid annotated fake-frame/real-frame pairs were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} frame pairs from {len(selected_fake_videos)} fake videos to {args.output}")
    if skipped:
        path = args.output.with_suffix(".skipped.txt")
        path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"Skipped {len(skipped)} frame pair(s); see {path}")
    if unresolved:
        path = args.output.with_suffix(".unmapped_fake_videos.txt")
        path.write_text("\n".join(sorted(set(unresolved))) + "\n", encoding="utf-8")
        print(f"Skipped {len(set(unresolved))} videos without usable fake segments/counterparts; see {path}")


if __name__ == "__main__":
    main()
