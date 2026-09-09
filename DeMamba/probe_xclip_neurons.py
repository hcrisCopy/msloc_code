"""Probe one set of frame-level fake-sensitive XCLIP neurons.

The input manifest is produced by :mod:`build_probe_pairs`.  Each record is an
annotated fake frame and the temporally aligned frame from its real
counterpart.  Statistics are averaged within each source video first, then
across videos, so a long video cannot dominate the selected neurons.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import XCLIPVisionModel


CLIP_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
CLIP_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


class RunningVectorStats:
    """Online mean/variance for one vector per source video."""

    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, value):
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def signed_effect(self, eps=1e-4):
        if self.count < 2:
            return np.zeros_like(self.mean)
        std = np.sqrt(self.m2 / (self.count - 1))
        nonzero = std[std > eps]
        # A robust floor prevents nearly constant channels from receiving an
        # artificial infinite score.
        floor = max(eps, float(np.median(nonzero)) * 0.05) if nonzero.size else 1.0
        return self.mean / np.sqrt(std * std + floor * floor)


def frame_directory(frame_root, video_path):
    return Path(frame_root) / Path(str(video_path)).with_suffix("")


def read_pairs(path: Path):
    """Read a JSONL manifest (kept public for the legacy visualization tool)."""
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pairs:
        raise ValueError(f"No frame pairs in {path}")
    return pairs


def read_frame_pairs(path: Path):
    """Validate the frame-level manifest and its timestamp/frame invariant."""
    pairs = read_pairs(path)
    for pair in pairs:
        required = {"fake_video", "normal_video", "frame_number", "normal_frame_number",
                    "fake_frame_file", "normal_frame_file", "timestamp", "fps"}
        missing = required.difference(pair)
        if missing:
            raise ValueError(f"Invalid frame-pair record missing {sorted(missing)}: {pair}")
        frame_number = int(pair["frame_number"])
        normal_frame_number = int(pair["normal_frame_number"])
        fps = float(pair["fps"])
        if frame_number < 1 or normal_frame_number < 1 or fps <= 0:
            raise ValueError(f"Invalid frame number/fps in pair {pair.get('pair_id', pair)}")
        for filename_key, number in (("fake_frame_file", frame_number),
                                     ("normal_frame_file", normal_frame_number)):
            filename = Path(str(pair[filename_key]))
            match = FRAME_PATTERN.match(filename.name)
            if filename.name != str(pair[filename_key]) or not match or int(match.group(1)) != number:
                raise ValueError(
                    f"{filename_key} in {pair.get('pair_id', pair)} does not name its declared frame number"
                )
        expected_timestamp = (frame_number - 1) / fps
        if not math.isclose(float(pair["timestamp"]), expected_timestamp, abs_tol=1e-6):
            raise ValueError(
                f"Timestamp mismatch for {pair.get('pair_id', pair)}: "
                f"got {pair['timestamp']}, expected (frame_number - 1) / fps = {expected_timestamp}"
            )
    return pairs


def load_frame(frame_root, video_path, frame_number, frame_file, image_size, crop_youku):
    """Load the manifest's exact filename; never use a list offset."""
    path = frame_directory(frame_root, video_path) / str(frame_file)
    if not path.is_file():
        raise FileNotFoundError(f"Required aligned frame does not exist: {path}")
    match = FRAME_PATTERN.match(path.name)
    if not match or int(match.group(1)) != int(frame_number):
        raise ValueError(f"Frame filename does not match requested frame number: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable frame: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_youku and "youku" in str(path).lower():
        height, width = image.shape[:2]
        if width > height:
            image = image[:, int(width * .15):int(width * .85)]
        else:
            image = image[int(height * .15):int(height * .85), :]
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return torch.from_numpy(((image - CLIP_MEAN) / CLIP_STD)).permute(2, 0, 1).contiguous()


def load_window(frame_root, video_path, start, end, frames_per_window, image_size, crop_youku,
                reference_times=None):
    """Compatibility helper for the legacy three-target visualization script.

    It is deliberately not used for frame-level neuron selection.
    """
    directory = frame_directory(frame_root, video_path)
    files = sorted(directory.glob("frame_*.jpg"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No extracted frames found in {directory}")
    fps = 8.0
    first, last = max(0, int(start * fps)), min(len(files) - 1, max(0, int(end * fps)))
    if reference_times is not None:
        times = np.asarray(reference_times, dtype=np.float32)
        indices = np.clip(np.rint(times * fps).astype(np.int64), 0, len(files) - 1).tolist()
    elif last - first + 1 >= frames_per_window:
        step = max(1, (last - first) // frames_per_window)
        indices = list(range(first, last + 1, step))[:frames_per_window]
        times = np.asarray(indices, dtype=np.float32) / fps
    else:
        indices = list(range(first, last + 1))
        indices.extend([indices[-1]] * (frames_per_window - len(indices)))
        times = np.asarray(indices, dtype=np.float32) / fps
    images = []
    for index in indices:
        image = cv2.imread(str(files[int(index)]))
        if image is None:
            raise ValueError(f"Unreadable frame: {files[int(index)]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_youku and "youku" in str(files[int(index)]).lower():
            height, width = image.shape[:2]
            image = image[:, int(width * .15):int(width * .85)] if width > height else image[int(height * .15):int(height * .85), :]
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        images.append((image - CLIP_MEAN) / CLIP_STD)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous(), times


def grouped_by_video(pairs):
    groups = {}
    for pair in pairs:
        # Keep one fake/real counterpart pair together even if a custom mapping
        # maps one fake path to a different real path.
        key = (str(pair["fake_video"]), str(pair["normal_video"]))
        groups.setdefault(key, []).append(pair)
    # XCLIP processes each chunk as an ordered temporal sequence.
    return [sorted(group, key=lambda pair: int(pair["frame_number"])) for group in groups.values()]


def make_stats(config):
    return {layer: RunningVectorStats(int(config.hidden_size))
            for layer in range(1, int(config.num_hidden_layers) + 1)}


def process_groups(model, config, device, args, groups, description):
    """Accumulate one mean fake-minus-real vector for every valid video."""
    stats = make_stats(config)
    failures, processed_frames, processed_videos = [], 0, 0
    temporal_width = int(config.num_frames)
    if temporal_width < 1:
        raise ValueError(f"XCLIP config has invalid num_frames={temporal_width}")
    pbar = tqdm(groups, desc=description, unit="video", dynamic_ncols=True)
    for group in pbar:
        sums = {layer: np.zeros(int(config.hidden_size), dtype=np.float64) for layer in stats}
        valid = 0
        loaded = []
        for pair in group:
            try:
                fake = load_frame(args.frame_root, pair["fake_video"], pair["frame_number"], pair["fake_frame_file"],
                                  int(config.image_size), args.crop_youku)
                normal = load_frame(args.frame_root, pair["normal_video"], pair["normal_frame_number"], pair["normal_frame_file"],
                                    int(config.image_size), args.crop_youku)
                loaded.append((pair, fake, normal))
            except Exception as error:
                message = f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}"
                if args.strict:
                    raise RuntimeError(message) from error
                failures.append(message)

        # XCLIPVisionModel groups a flat image batch into sequences of exactly
        # config.num_frames images.  Keep fake and real as two separate 8-frame
        # sequences.  The last partial group is padded for XCLIP only; padded
        # positions are excluded from all activation statistics.
        for first in range(0, len(loaded), temporal_width):
            chunk = loaded[first:first + temporal_width]
            count = len(chunk)
            fake_frames = [item[1] for item in chunk]
            normal_frames = [item[2] for item in chunk]
            if count < temporal_width:
                fake_frames.extend([fake_frames[-1]] * (temporal_width - count))
                normal_frames.extend([normal_frames[-1]] * (temporal_width - count))
            batch = torch.stack(fake_frames + normal_frames, dim=0).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model(batch, output_hidden_states=True)
            # hidden: [fake_0 ... fake_7, real_0 ... real_7, tokens, channels].
            # The paired delta is retained only for the count genuine entries;
            # pad copies are deliberately ignored.
            for layer, hidden in enumerate(outputs.hidden_states[1:], 1):
                fake_activation = hidden[:count, 1:, :].float().mean(dim=1)
                normal_activation = hidden[temporal_width:temporal_width + count, 1:, :].float().mean(dim=1)
                sums[layer] += (fake_activation - normal_activation).sum(dim=0).cpu().numpy()
            valid += count
            processed_frames += count
        if valid:
            for layer in stats:
                stats[layer].update(sums[layer] / valid)
            processed_videos += 1
        pbar.set_postfix(frames=processed_frames, videos=processed_videos, fail=len(failures))
    return stats, failures, processed_frames, processed_videos


def _merge_stats(base: RunningVectorStats, other: RunningVectorStats):
    if other.count == 0:
        return base
    if base.count == 0:
        base.count, base.mean, base.m2 = other.count, other.mean.copy(), other.m2.copy()
        return base
    total = base.count + other.count
    delta = other.mean - base.mean
    base.m2 = base.m2 + other.m2 + delta ** 2 * base.count * other.count / total
    base.mean = (base.mean * base.count + other.mean * other.count) / total
    base.count = total
    return base


def _worker(rank, args, groups, result_queue):
    device = torch.device(f"cuda:{rank}")
    model = XCLIPVisionModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    stats, failures, frames, videos = process_groups(model, model.config, device, args, groups, f"GPU {rank}")
    result_queue.put((stats, failures, frames, videos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="Frame-pair JSONL from build_probe_pairs.py")
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local microsoft/xclip-base-patch16 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-neuron-count", type=int, default=768)
    parser.add_argument("--max-frame-pairs", type=int, default=0,
                        help="Debug-only cap; applied after deterministic manifest order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-youku", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on an unavailable/unreadable paired frame")
    args = parser.parse_args()
    if args.final_neuron_count <= 0 or args.max_frame_pairs < 0:
        parser.error("final-neuron-count must be positive and max-frame-pairs must be non-negative")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"XCLIP model directory not found: {args.model_path}")

    pairs = read_frame_pairs(args.pairs)
    if args.max_frame_pairs:
        pairs = pairs[:args.max_frame_pairs]
    groups = grouped_by_video(pairs)

    n_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    if n_gpus > 1:
        check = XCLIPVisionModel.from_pretrained(args.model_path, local_files_only=True).to("cuda:0")
        config = check.config
        del check
        torch.cuda.empty_cache()
        if int(config.hidden_size) != 768 or int(config.patch_size) != 16:
            raise ValueError(f"Expected XCLIP base patch16 (768/16), got {config.hidden_size}/{config.patch_size}")
        chunks = [groups[index::n_gpus] for index in range(n_gpus)]
        context, queue = mp.get_context("spawn"), mp.get_context("spawn").Queue()
        processes = [context.Process(target=_worker, args=(rank, args, chunks[rank], queue), daemon=False)
                     for rank in range(n_gpus)]
        for process in processes:
            process.start()
        stats, failures, frames, videos = None, [], 0, 0
        for _ in processes:
            worker_stats, worker_failures, worker_frames, worker_videos = queue.get()
            failures.extend(worker_failures)
            frames += worker_frames
            videos += worker_videos
            if stats is None:
                stats = worker_stats
            else:
                for layer in stats:
                    stats[layer] = _merge_stats(stats[layer], worker_stats[layer])
        for process in processes:
            process.join()
    else:
        device = torch.device(args.device)
        model = XCLIPVisionModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
        model.requires_grad_(False)
        config = model.config
        if int(config.hidden_size) != 768 or int(config.patch_size) != 16:
            raise ValueError(f"Expected XCLIP base patch16 (768/16), got {config.hidden_size}/{config.patch_size}")
        stats, failures, frames, videos = process_groups(model, config, device, args, groups, "Probing frame pairs")

    if videos < 2:
        raise RuntimeError(f"Only {videos} videos yielded valid pairs; at least two are required for a stable effect score")
    arrays, layers = {}, {}
    layer_count = int(config.num_hidden_layers)
    if args.final_neuron_count % layer_count:
        raise ValueError(
            f"final-neuron-count={args.final_neuron_count} must be divisible by "
            f"the {layer_count} XCLIP layers for equal per-layer selection"
        )
    neurons_per_layer = args.final_neuron_count // layer_count
    for layer, state in stats.items():
        signed = state.signed_effect()
        score = np.abs(signed)
        arrays[f"frame_layer_{layer:02d}_score"] = score.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_signed_effect"] = signed.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_mean_delta"] = state.mean.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_video_count"] = np.asarray(state.count, dtype=np.int32)
        if score.size < neurons_per_layer:
            raise RuntimeError(
                f"Layer {layer} has only {score.size} channels, fewer than "
                f"the requested {neurons_per_layer} per-layer neurons"
            )
        # One fake-sensitive neuron set, with a fixed equal allocation across
        # XCLIP layers: 12 layers * 64 channels = 768 channels.
        indices = np.argsort(score)[::-1][:neurons_per_layer].astype(np.int32)
        layers[str(layer)] = sorted(indices.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "xclip_neuron_scores.npz", **arrays)
    selector = {
        "schema_version": 3,
        "selection_task": "frame_level_fake_vs_real",
        "selection_unit": "per_video_mean_of_paired_fake_minus_real_frames",
        "frame_timestamp_rule": "timestamp=(frame_number-1)/fps; frame_1.jpg is t=0",
        "model": "microsoft/xclip-base-patch16",
        "hidden_size": int(config.hidden_size),
        "patch_size": int(config.patch_size),
        "image_size": int(config.image_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "final_neuron_count": args.final_neuron_count,
        "neurons_per_layer": neurons_per_layer,
        "selection_rule": "top absolute paired fake-vs-real sensitivity score within each layer",
        "processed_frame_pairs": frames,
        "processed_videos": videos,
        "layers": layers,
    }
    (args.output_dir / "xclip_neuron_indices.json").write_text(json.dumps(selector, indent=2), encoding="utf-8")
    (args.output_dir / "probe_failures.txt").write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"Processed {frames}/{len(pairs)} frame pairs from {videos} videos. Selector: {args.output_dir / 'xclip_neuron_indices.json'}")
    if failures:
        print(f"Skipped {len(failures)} invalid frame pair(s); see {args.output_dir / 'probe_failures.txt'}")


if __name__ == "__main__":
    main()
