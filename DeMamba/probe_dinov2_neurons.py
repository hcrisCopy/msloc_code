"""Select paired fake-sensitive DINOv2 ViT-B/14 patch-token neurons.

The encoder runs at 196px: with patch size 14 this produces the same 14x14
patch grid as the existing DeMamba head. For every one of 12 blocks, 64 of the
768 channels are selected using source-video-balanced fake-minus-real effects.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)
IMAGE_SIZE = 196


class RunningVectorStats:
    def __init__(self, width):
        self.count, self.mean, self.m2 = 0, np.zeros(width, np.float64), np.zeros(width, np.float64)

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
        floor = max(eps, float(np.median(nonzero)) * 0.05) if nonzero.size else 1.0
        return self.mean / np.sqrt(std * std + floor * floor)


def progress_signature(args):
    pairs_stat = args.pairs.stat()
    return json.dumps({
        "pairs": str(args.pairs.resolve()), "pairs_size": pairs_stat.st_size,
        "pairs_mtime_ns": pairs_stat.st_mtime_ns, "frame_root": str(args.frame_root.resolve()),
        "model": str(args.dinov2_hf_model_path.resolve()), "max_frame_pairs": args.max_frame_pairs,
        "crop_youku": args.crop_youku,
    }, sort_keys=True)


def save_progress(path, signature, next_group, stats, failures, processed_frames, processed_videos):
    arrays = {
        "signature": np.asarray(signature), "next_group": np.asarray(next_group),
        "failures": np.asarray(failures), "processed_frames": np.asarray(processed_frames),
        "processed_videos": np.asarray(processed_videos),
    }
    for layer, state in stats.items():
        arrays[f"count_{layer}"] = np.asarray(state.count)
        arrays[f"mean_{layer}"] = state.mean
        arrays[f"m2_{layer}"] = state.m2
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_progress(path, signature):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["signature"].item()) != signature:
            raise ValueError(f"Probe progress does not match current inputs: {path}")
        stats = {layer: RunningVectorStats(768) for layer in range(1, 13)}
        for layer, state in stats.items():
            state.count = int(saved[f"count_{layer}"].item())
            state.mean = saved[f"mean_{layer}"].copy()
            state.m2 = saved[f"m2_{layer}"].copy()
        return (int(saved["next_group"].item()), stats, saved["failures"].astype(str).tolist(),
                int(saved["processed_frames"].item()), int(saved["processed_videos"].item()))


def read_pairs(path):
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"fake_video", "normal_video", "frame_number", "normal_frame_number", "fake_frame_file", "normal_frame_file"}
    if not pairs:
        raise ValueError(f"No frame pairs in {path}")
    for pair in pairs:
        missing = required.difference(pair)
        if missing:
            raise ValueError(f"Invalid pair manifest record missing {sorted(missing)}")
    return pairs


def group_pairs(pairs):
    groups = {}
    for pair in pairs:
        groups.setdefault((str(pair["fake_video"]), str(pair["normal_video"])), []).append(pair)
    return [sorted(group, key=lambda item: int(item["frame_number"])) for group in groups.values()]


def read_image(frame_root, video_path, frame_number, frame_file, crop_youku):
    frame_file = Path(str(frame_file))
    match = FRAME_PATTERN.match(frame_file.name)
    if not match or int(match.group(1)) != int(frame_number):
        raise ValueError(f"Frame filename/number mismatch: {frame_file}")
    path = Path(frame_root) / Path(str(video_path)).with_suffix("") / frame_file
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unreadable paired frame: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_youku and "youku" in str(path).lower():
        height, width = image.shape[:2]
        image = image[:, int(width * .15):int(width * .85)] if width > height else image[int(height * .15):int(height * .85), :]
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return torch.from_numpy((image - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).contiguous()


def load_encoder(model_path, device):
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local Hugging Face DINOv2 directory not found: {model_path}")
    try:
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    except (ImportError, KeyError, OSError, ValueError) as error:
        raise RuntimeError(
            "Unable to load Hugging Face DINOv2. Ensure model.safetensors/config.json are present "
            "and install a compatible transformers version."
        ) from error
    shape = (getattr(model.config, "hidden_size", None), getattr(model.config, "patch_size", None),
             getattr(model.config, "num_hidden_layers", None), getattr(model.config, "num_register_tokens", 0))
    if shape != (768, 14, 12, 0):
        raise ValueError(f"Expected facebook/dinov2-base (768,14,12,0), got {shape}")
    return model.to(device).eval().requires_grad_(False)


def get_block_patch_states(model, batch):
    outputs = model(pixel_values=batch, output_hidden_states=True, return_dict=True)
    states = tuple(state[:, 1:, :] for state in outputs.hidden_states[1:])
    if len(states) != 12:
        raise RuntimeError(f"Expected 12 DINOv2 block outputs, got {len(states)}")
    return states


def process_groups(model, groups, args, device, stats, failures, processed_frames, processed_videos,
                   start_group, progress_path, signature):
    remaining = groups[start_group:]
    for group_index, group in enumerate(
            tqdm(remaining, desc="Probing paired DINOv2 neurons", unit="video", dynamic_ncols=True),
            start=start_group):
        loaded = []
        for pair in group:
            try:
                fake = read_image(args.frame_root, pair["fake_video"], pair["frame_number"], pair["fake_frame_file"], args.crop_youku)
                real = read_image(args.frame_root, pair["normal_video"], pair["normal_frame_number"], pair["normal_frame_file"], args.crop_youku)
                loaded.append((fake, real))
            except Exception as error:
                if args.strict:
                    raise
                failures.append(f"{pair.get('pair_id', '<unknown>')}: {type(error).__name__}: {error}")
        if not loaded:
            continue
        sums = {layer: np.zeros(768, np.float64) for layer in stats}
        valid = 0
        for first in range(0, len(loaded), args.image_batch_size):
            chunk = loaded[first:first + args.image_batch_size]
            count = len(chunk)
            batch = torch.stack([item for pair in chunk for item in pair]).to(device)
            batch = torch.cat((batch[0::2], batch[1::2]), dim=0)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                        enabled=args.amp and device.type == "cuda"):
                hidden_states = get_block_patch_states(model, batch)
            for layer, patch_tokens in enumerate(hidden_states, 1):
                if patch_tokens.shape != (2 * count, 196, 768):
                    raise RuntimeError(f"Unexpected DINOv2 patch shape at block {layer}: {tuple(patch_tokens.shape)}")
                fake = patch_tokens[:count].float().mean(dim=1)
                real = patch_tokens[count:].float().mean(dim=1)
                sums[layer] += (fake - real).sum(dim=0).cpu().numpy()
            valid += count
        for layer in stats:
            stats[layer].update(sums[layer] / valid)
        processed_frames += valid
        processed_videos += 1
        if args.resume:
            save_progress(progress_path, signature, group_index + 1, stats, failures,
                          processed_frames, processed_videos)
    return stats, failures, processed_frames, processed_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--dinov2-hf-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-neuron-count", type=int, default=768)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--max-frame-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-youku", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted probe or reuse completed neuron outputs")
    args = parser.parse_args()
    if args.final_neuron_count != 768:
        parser.error("The unchanged DeMamba head requires --final-neuron-count 768")
    if args.image_batch_size < 1 or args.max_frame_pairs < 0:
        parser.error("image-batch-size must be positive and max-frame-pairs non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "dinov2_neuron_scores.npz"
    selector_path = args.output_dir / "dinov2_neuron_indices.json"
    progress_path = args.output_dir / "dinov2_probe_progress.npz"
    if args.resume and score_path.is_file() and selector_path.is_file():
        print(f"[reuse] Completed DINOv2 neuron probe: {selector_path}")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pairs = read_pairs(args.pairs)
    if args.max_frame_pairs:
        pairs = pairs[:args.max_frame_pairs]
    groups = group_pairs(pairs)
    signature = progress_signature(args)
    stats = {layer: RunningVectorStats(768) for layer in range(1, 13)}
    failures, frames, videos, start_group = [], 0, 0, 0
    if args.resume and progress_path.is_file():
        start_group, stats, failures, frames, videos = load_progress(progress_path, signature)
        print(f"[resume] Continuing DINOv2 neuron probe at source video {start_group}/{len(groups)}")
    stats, failures, frames, videos = process_groups(
        load_encoder(args.dinov2_hf_model_path, device), groups, args, device, stats, failures,
        frames, videos, start_group, progress_path, signature)
    if videos < 2:
        raise RuntimeError("At least two source videos with valid paired frames are required")
    arrays, layers = {}, {}
    for layer, state in stats.items():
        signed = state.signed_effect()
        score = np.abs(signed)
        arrays[f"frame_layer_{layer:02d}_score"] = score.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_signed_effect"] = signed.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_mean_delta"] = state.mean.astype(np.float32)
        arrays[f"frame_layer_{layer:02d}_video_count"] = np.asarray(state.count, dtype=np.int32)
        layers[str(layer)] = sorted(np.argsort(score)[::-1][:64].astype(np.int32).tolist())
    np.savez_compressed(score_path, **arrays)
    selector = {
        "schema_version": 1, "backbone": "dinov2_vitb14", "selection_task": "frame_level_fake_vs_real",
        "selection_unit": "per_video_mean_of_paired_fake_minus_real_frames",
        "feature_location": "post_block_patch_tokens_only", "excluded_tokens": ["CLS"],
        "hidden_size": 768, "patch_size": 14, "input_size": IMAGE_SIZE, "patch_grid": [14, 14],
        "channels_per_layer": 64, "layers": layers,
        "processed_frame_pairs": frames, "processed_source_videos": videos, "failures": failures,
    }
    selector_path.write_text(json.dumps(selector, indent=2), encoding="utf-8")
    if progress_path.is_file():
        progress_path.unlink()
    print(f"Processed paired frames: {frames}; source videos: {videos}; failures: {len(failures)}")
    print(f"Selector: {selector_path}")


if __name__ == "__main__":
    main()
