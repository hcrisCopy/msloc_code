"""Visualize the spatial evidence of the XCLIP neurons selected by DeMamba.

This is an analysis-only script: it does not alter training, checkpoints, or
the neuron selector.  Given one fake/normal probe pair, it runs the *final*
XCLIP encoder, retains patch tokens, and writes two complementary views:

1. ``heatmaps/``: one fake--normal, key-neuron evidence heatmap per frame;
2. ``curve.png``: a temporal AIGC-evidence curve (mean of the strongest 5%
   patch evidences by default).

For a selected target q and selected coordinate (l, c), the per-patch base
evidence is

    ReLU(sign(mu[q,l,c]) * (h_fake - h_normal)[t,p,l,c] / d[q,l,c]),

where ``mu`` is the saved paired mean delta and
``d = |mu| / score`` reconstructs the stabilized delta scale used by the
probe score.  Scores are normalized within each layer and layers contribute
equally.  ``fake`` displays this base evidence.  ``r2f`` and ``f2r`` display,
respectively, its positive temporal increase and decrease, so their maps show
the transition evidence selected during probing.

Run from the repository root (``msloc_code``).  All defaults follow the server
layout where ``msloc_code`` and ``MSLoc_data`` are sibling directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import XCLIPVisionModel

from probe_xclip_neurons import CLIP_MEAN, CLIP_STD, frame_directory, load_window, read_pairs


TARGETS = ("fake", "r2f", "f2r")
SCORE_KEY = re.compile(r"^(fake|r2f|f2r)_layer_(\d+)_score$")


def _layer_number(value: str | int) -> int:
    text = str(value)
    return int(text.removeprefix("layer_"))


def load_probe_artifacts(score_path: Path, selector_path: Path):
    """Load score/mean vectors and target-specific members of the final union."""
    scores = {target: {} for target in TARGETS}
    means = {target: {} for target in TARGETS}
    with np.load(score_path, allow_pickle=False) as archive:
        for key in archive.files:
            match = SCORE_KEY.match(key)
            if not match:
                continue
            target, layer = match.group(1), int(match.group(2))
            mean_key = f"{target}_layer_{layer:02d}_mean_delta"
            if mean_key not in archive.files:
                raise KeyError(f"Missing {mean_key} in {score_path}")
            scores[target][layer] = np.asarray(archive[key], dtype=np.float32)
            means[target][layer] = np.asarray(archive[mean_key], dtype=np.float32)

    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    final_by_target = selector.get("final_by_target")
    if not isinstance(final_by_target, dict):
        raise ValueError(
            "Selector has no final_by_target section. Re-run the current "
            "probe_xclip_neurons.py before producing target-specific maps."
        )
    members = {target: {} for target in TARGETS}
    for target in TARGETS:
        for layer, indices in final_by_target.get(target, {}).items():
            number = _layer_number(layer)
            members[target][number] = np.asarray(sorted(set(map(int, indices))), dtype=np.int64)
    return scores, means, members


def load_final_encoder(model_path: Path, checkpoint: Path | None, device: torch.device):
    model = XCLIPVisionModel.from_pretrained(model_path, local_files_only=True)
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
        encoder_state = {}
        for key, value in state.items():
            for prefix in ("module.encoder.", "encoder."):
                if key.startswith(prefix):
                    encoder_state[key[len(prefix):]] = value
                    break
        if not encoder_state:
            raise ValueError(
                f"No encoder parameters found in {checkpoint}. Expected keys beginning "
                "with 'module.encoder.' or 'encoder.'."
            )
        incompatible = model.load_state_dict(encoder_state, strict=False)
        loaded = len(model.state_dict()) - len(incompatible.missing_keys)
        if loaded != len(model.state_dict()) or incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint encoder is incompatible with {model_path}: loaded {loaded}/"
                f"{len(model.state_dict())} tensors; missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        print(f"Loaded final XCLIP encoder from {checkpoint} ({loaded} tensors).")
    else:
        print("No --checkpoint supplied: using the initial pretrained XCLIP encoder.")
    return model.to(device).eval()


def select_pair(pairs, pair_id: str | None, pair_index: int | None):
    if pair_id is not None:
        matching = [pair for pair in pairs if pair.get("pair_id") == pair_id]
        if not matching:
            raise KeyError(f"pair_id {pair_id!r} is not present in the manifest")
        return matching[0]
    if pair_index is None or not 0 <= pair_index < len(pairs):
        raise IndexError(f"--pair-index must be in [0, {len(pairs) - 1}]")
    return pairs[pair_index]


def select_video_pair(pairs, video_name: str):
    """Choose one manifest record by fake-video file name for full-video mode."""
    requested = Path(video_name).stem
    matching = [
        pair for pair in pairs
        if Path(str(pair.get("fake_video", ""))).stem == requested
    ]
    if not matching:
        raise KeyError(f"video name {video_name!r} is not present as a fake video in the manifest")
    video_keys = {(pair["fake_video"], pair["normal_video"]) for pair in matching}
    if len(video_keys) != 1:
        choices = "\n".join(f"  {fake}" for fake, _ in sorted(video_keys))
        raise ValueError(
            f"video name {video_name!r} is ambiguous; use --pair-id instead. Candidates:\n{choices}"
        )
    return matching[0]


def list_videos(pairs):
    """Print one line per fake/normal video pair, suitable for --video-name."""
    grouped = {}
    for pair in pairs:
        key = (pair.get("fake_video"), pair.get("normal_video"))
        grouped.setdefault(key, []).append(pair)
    for (fake_video, normal_video), records in sorted(grouped.items()):
        name = Path(str(fake_video)).stem
        boundaries = sorted({
            float(record["boundary_time"])
            for record in records
            if record.get("target") in {"r2f", "f2r"} and record.get("boundary_time") is not None
        })
        boundary_text = ", ".join(f"{time:g}s" for time in boundaries) or "none"
        print(f"{name}\tpairs={len(records)}\tboundaries={boundary_text}\t{fake_video}")


def sample_frame_paths(frame_root: Path, video_path: str, start: float, end: float,
                       frames_per_window: int, reference_times: np.ndarray | None = None):
    """Return exactly the paths selected by probe_xclip_neurons.load_window."""
    directory = frame_directory(frame_root, video_path)
    files = sorted(directory.glob("frame_*.jpg"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No extracted frames found in {directory}")
    fps = 8.0
    first = max(0, int(start * fps))
    last = min(len(files) - 1, max(first, int(end * fps)))
    if reference_times is not None:
        indices = np.clip(np.rint(reference_times * fps).astype(np.int64), 0, len(files) - 1).tolist()
    elif last - first + 1 >= frames_per_window:
        step = max(1, (last - first) // frames_per_window)
        indices = list(range(first, last + 1, step))[:frames_per_window]
    else:
        indices = list(range(first, last + 1))
        indices.extend([indices[-1]] * (frames_per_window - len(indices)))
    return [files[int(index)] for index in indices]


def frame_index(path: Path) -> int:
    """Extract the original 8-fps frame index from ``frame_123.jpg``."""
    return int(path.stem.rsplit("_", 1)[1])


def all_matched_frame_paths(frame_root: Path, fake_video: str, normal_video: str):
    """Return every fake frame with its same-index normal counterpart."""
    def list_frames(video):
        directory = frame_directory(frame_root, video)
        paths = sorted(directory.glob("frame_*.jpg"), key=frame_index)
        if not paths:
            raise FileNotFoundError(f"No extracted frames found in {directory}")
        return paths

    fake_paths = list_frames(fake_video)
    normal_by_index = {frame_index(path): path for path in list_frames(normal_video)}
    missing = [path for path in fake_paths if frame_index(path) not in normal_by_index]
    if missing:
        print(f"Warning: skipping {len(missing)} fake frame(s) with no same-index normal counterpart.")
    fake_paths = [path for path in fake_paths if frame_index(path) in normal_by_index]
    if not fake_paths:
        raise ValueError("No temporally aligned fake/normal extracted frames remain")
    return fake_paths, [normal_by_index[frame_index(path)] for path in fake_paths]


def display_images(paths, crop_youku: bool):
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable frame: {path}")
        if crop_youku and "youku" in str(path).lower():
            height, width = image.shape[:2]
            image = image[:, int(width * .15):int(width * .85)] if width > height else image[int(height * .15):int(height * .85), :]
        images.append(image)
    return images


def preprocess_frame_paths(paths, image_size: int, crop_youku: bool):
    """Apply exactly the crop, resize and CLIP normalization used by the probe."""
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_youku and "youku" in str(path).lower():
            height, width = image.shape[:2]
            image = image[:, int(width * .15):int(width * .85)] if width > height else image[int(height * .15):int(height * .85), :]
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        images.append((image - CLIP_MEAN) / CLIP_STD)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


def target_evidence(hidden_states, scores, means, members, target: str, frames: int, patch_count: int):
    """Build [T, P] evidence before/after transition differencing for one target."""
    layer_maps = []
    for layer, indices_np in sorted(members[target].items()):
        if layer >= len(hidden_states):
            raise ValueError(f"Selector requests layer {layer}, model returned only {len(hidden_states) - 1} encoder layers")
        if layer not in scores[target] or layer not in means[target]:
            raise KeyError(f"No saved score/mean vector for {target} layer {layer}")
        indices = torch.from_numpy(indices_np).to(hidden_states[layer].device)
        fake = hidden_states[layer][:frames, 1:, :].reshape(frames, patch_count, -1)
        normal = hidden_states[layer][frames:, 1:, :].reshape(frames, patch_count, -1)
        delta = (fake - normal).index_select(-1, indices)

        score = torch.from_numpy(scores[target][layer][indices_np]).to(delta.device, dtype=delta.dtype)
        mean = torch.from_numpy(means[target][layer][indices_np]).to(delta.device, dtype=delta.dtype)
        # score = |mu| / sqrt(var + floor^2), hence |mu| / score recovers the
        # probe's stabilized scale without needing to serialize activations.
        scale = mean.abs() / score.clamp_min(1e-8)
        scale = scale.clamp_min(1e-6)
        signed = delta * mean.sign().where(mean != 0, torch.ones_like(mean)) / scale
        weights = score / score.sum().clamp_min(1e-8)
        layer_maps.append(torch.relu(signed).mul(weights).sum(dim=-1))

    if not layer_maps:
        raise ValueError(f"No final selected neurons are assigned to target {target!r}")
    base = torch.stack(layer_maps).mean(dim=0)  # equal weight for each XCLIP layer
    return base, event_from_base(base, target)


def event_from_base(base, target: str):
    """Convert persistent artifact evidence into target-specific event evidence."""
    event = torch.zeros_like(base)
    if target == "fake":
        event = base
    elif target == "r2f":
        event[1:] = torch.relu(base[1:] - base[:-1])
    else:  # f2r
        event[1:] = torch.relu(base[:-1] - base[1:])
    return event


def top_patch_curve(evidence: np.ndarray, ratio: float):
    count = max(1, math.ceil(evidence.shape[1] * ratio))
    return np.partition(evidence, -count, axis=1)[:, -count:].mean(axis=1)


def normalize_for_render(evidence: np.ndarray):
    positive = evidence[evidence > 0]
    if positive.size == 0:
        return np.zeros_like(evidence, dtype=np.float32)
    high = float(np.percentile(positive, 99))
    if high <= 0:
        return np.zeros_like(evidence, dtype=np.float32)
    return np.clip(evidence / high, 0, 1).astype(np.float32)


def configure_curve_axis(axis, sample_times: np.ndarray, indices: np.ndarray, max_ticks: int):
    """Use labels such as ``6.00(48)``: seconds followed by extracted frame id."""
    count = len(sample_times)
    positions = np.unique(np.rint(np.linspace(0, count - 1, min(max_ticks, count))).astype(int))
    axis.set_xticks(sample_times[positions])
    axis.set_xticklabels([f"{sample_times[i]:.2f}({int(indices[i])})" for i in positions], rotation=35, ha="right")
    axis.set_xlabel("video time (s) (extracted-frame index)")


def labelled_panel(image: np.ndarray, label: str):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (max(150, len(label) * 11), 28), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_outputs(output_dir: Path, pair, target: str, sample_times: np.ndarray, frame_indices: np.ndarray,
                   fake_images, normal_images, base: np.ndarray, event: np.ndarray, grid_size: int,
                   top_ratio: float, fps: float, max_curve_ticks: int):
    target_dir = output_dir / target
    heatmap_dir = target_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    render_values = normalize_for_render(event)
    frames = []
    for t, (fake, normal) in enumerate(zip(fake_images, normal_images)):
        height, width = fake.shape[:2]
        patch_map = render_values[t].reshape(grid_size, grid_size)
        full_map = cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(np.uint8(np.clip(full_map * 255, 0, 255)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(fake, .52, color, .48, 0)
        panel = np.hstack((labelled_panel(fake, "fake frame"), labelled_panel(normal, "matched normal"), labelled_panel(overlay, "key-neuron evidence")))
        cv2.putText(panel, f"t={sample_times[t]:.3f}s  frame={int(frame_indices[t])}", (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(heatmap_dir / f"frame_{int(frame_indices[t]):06d}.jpg"), panel)
        frames.append(panel)

    video_path = target_dir / "heatmap_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if writer.isOpened():
        for frame in frames:
            writer.write(frame)
        writer.release()
    else:
        print(f"Warning: OpenCV could not create {video_path}; individual JPEGs were still written.")

    curve = top_patch_curve(event, top_ratio)
    np.savez_compressed(
        target_dir / "evidence.npz",
        sample_times=sample_times,
        base_evidence=base,
        event_evidence=event,
        temporal_score=curve,
    )
    with (target_dir / "temporal_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "time_seconds", "aigc_evidence_top_patch_mean"])
        writer.writerows((int(frame_indices[t]), float(sample_times[t]), float(curve[t])) for t in range(len(curve)))

    figure, axis = plt.subplots(figsize=(8, 3.6), dpi=180)
    axis.plot(sample_times, curve, marker="o", color="#d62728", linewidth=2)
    if pair.get("boundary_time") is not None:
        axis.axvline(float(pair["boundary_time"]), color="#1f77b4", linestyle="--", label="annotated boundary")
        axis.legend()
    title = "AIGC key-neuron evidence" if target == "fake" else f"{target.upper()} transition evidence"
    axis.set_title(title)
    configure_curve_axis(axis, sample_times, frame_indices, max_curve_ticks)
    axis.set_ylabel(f"mean top {top_ratio:.0%} patch evidence")
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(target_dir / "curve.png", bbox_inches="tight")
    plt.close(figure)


def full_video_evidence(model, fake_paths, normal_paths, targets, scores, means, members, image_size,
                        patch_count, crop_youku, device, amp, batch_frames):
    """Run the final XCLIP on every aligned extracted frame, in bounded batches."""
    base_chunks = {target: [] for target in targets}
    total = len(fake_paths)
    for first in range(0, total, batch_frames):
        last = min(total, first + batch_frames)
        fake = preprocess_frame_paths(fake_paths[first:last], image_size, crop_youku)
        normal = preprocess_frame_paths(normal_paths[first:last], image_size, crop_youku)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda",
        ):
            outputs = model(torch.cat((fake, normal), dim=0).to(device), output_hidden_states=True)
        for target in targets:
            base, _ = target_evidence(outputs.hidden_states, scores, means, members, target,
                                      last - first, patch_count)
            base_chunks[target].append(base.float().cpu().numpy())
        print(f"Processed full video frames {first + 1}-{last}/{total}")
    bases = {target: np.concatenate(chunks, axis=0) for target, chunks in base_chunks.items()}
    events = {
        target: event_from_base(torch.from_numpy(base), target).numpy()
        for target, base in bases.items()
    }
    return bases, events


def render_full_video_outputs(output_dir: Path, boundary_events, target: str, sample_times: np.ndarray,
                              frame_indices: np.ndarray, fake_paths, normal_paths, base: np.ndarray,
                              event: np.ndarray, grid_size: int, top_ratio: float, fps: float,
                              max_curve_ticks: int, crop_youku: bool):
    """Write an overlay JPEG for every extracted frame plus a full-length curve and MP4."""
    target_dir = output_dir / target
    heatmap_dir = target_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    render_values = normalize_for_render(event)
    writer, video_size = None, None
    video_path = target_dir / "heatmap_video.mp4"

    for t, (fake_path, normal_path) in enumerate(zip(fake_paths, normal_paths)):
        fake = display_images([fake_path], crop_youku)[0]
        normal = display_images([normal_path], crop_youku)[0]
        height, width = fake.shape[:2]
        if normal.shape[:2] != (height, width):
            normal = cv2.resize(normal, (width, height), interpolation=cv2.INTER_CUBIC)
        patch_map = render_values[t].reshape(grid_size, grid_size)
        full_map = cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(np.uint8(np.clip(full_map * 255, 0, 255)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(fake, .52, color, .48, 0)
        panel = np.hstack((labelled_panel(fake, "fake frame"), labelled_panel(normal, "matched normal"), labelled_panel(overlay, "key-neuron evidence")))
        cv2.putText(panel, f"t={sample_times[t]:.3f}s  frame={int(frame_indices[t])}", (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(heatmap_dir / f"frame_{int(frame_indices[t]):06d}.jpg"), panel)
        if writer is None:
            video_size = (panel.shape[1], panel.shape[0])
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size)
            if not writer.isOpened():
                writer = None
                print(f"Warning: OpenCV could not create {video_path}; individual JPEGs were still written.")
        if writer is not None:
            writer.write(cv2.resize(panel, video_size) if (panel.shape[1], panel.shape[0]) != video_size else panel)
    if writer is not None:
        writer.release()

    curve = top_patch_curve(event, top_ratio)
    np.savez_compressed(
        target_dir / "evidence.npz",
        sample_times=sample_times,
        frame_indices=frame_indices,
        base_evidence=base,
        event_evidence=event,
        temporal_score=curve,
    )
    with (target_dir / "temporal_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["frame_index", "time_seconds", "aigc_evidence_top_patch_mean"])
        csv_writer.writerows((int(frame_indices[t]), float(sample_times[t]), float(curve[t])) for t in range(len(curve)))

    figure, axis = plt.subplots(figsize=(10, 4), dpi=180)
    axis.plot(sample_times, curve, color="#d62728", linewidth=1.4)
    boundary_colors = {"r2f": "#2ca02c", "f2r": "#1f77b4"}
    seen_boundary_labels = set()
    for boundary_time, boundary_target in boundary_events:
        label = boundary_target.upper()
        axis.axvline(
            float(boundary_time), color=boundary_colors.get(boundary_target, "#555555"),
            linestyle="--", linewidth=1.15,
            label=label if label not in seen_boundary_labels else None,
        )
        seen_boundary_labels.add(label)
    if seen_boundary_labels:
        axis.legend(title="annotated boundary")
    title = "AIGC key-neuron evidence" if target in {"fake", "combined"} else f"{target.upper()} transition evidence"
    axis.set_title(title)
    configure_curve_axis(axis, sample_times, frame_indices, max_curve_ticks)
    axis.set_ylabel(f"mean top {top_ratio:.0%} patch evidence")
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(target_dir / "curve.png", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/neuron_probe/train_pairs_full.jsonl"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pair-id", help="pair_id in --pairs")
    selection.add_argument("--pair-index", type=int, help="zero-based line index in --pairs")
    selection.add_argument("--video-name", help="Fake-video file name for --full-video, e.g. P02C05_115902976-part003")
    parser.add_argument("--list-pairs", action="store_true", help="Print the first 50 pairs and exit")
    parser.add_argument("--list-videos", action="store_true", help="Print video names accepted by --video-name and exit")
    parser.add_argument("--target", choices=("auto", *TARGETS, "all"), default="auto",
                        help="auto uses the pair target in window mode and all targets in full-video mode")
    parser.add_argument("--full-video", action="store_true",
                        help="Use the selected pair's fake/normal paths, but process every extracted frame instead of its 2-second window")
    parser.add_argument("--frame-root", type=Path, default=Path("../MSLoc_data/DeMamba/video_frames"))
    parser.add_argument("--model-path", type=Path, default=Path("../MSLoc_data/DeMamba/pretrained_weights/xclip-base-patch16"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/results/best_acc.pth"),
                        help="Final neuron-DeMamba checkpoint; use --no-checkpoint for the initial XCLIP")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--scores", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_scores.npz"))
    parser.add_argument("--selector", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/neuron_probe/xclip_neuron_indices.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("../MSLoc_data/DeMamba/full/method/visualizations/paired_neuron_evidence"))
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=32,
                        help="Number of fake and normal frames encoded together per full-video inference batch")
    parser.add_argument("--top-patch-ratio", type=float, default=.05)
    parser.add_argument("--max-curve-ticks", type=int, default=12,
                        help="Maximum x-axis labels; each label is time(frame_index), e.g. 6.00(48)")
    parser.add_argument("--crop-youku", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use fp16 inference on CUDA")
    parser.add_argument("--video-fps", type=float, default=8.0,
                        help="Playback FPS for the heatmap video; 8 matches the extracted-frame time base")
    args = parser.parse_args()
    if not 0 < args.top_patch_ratio <= 1:
        parser.error("--top-patch-ratio must be in (0, 1]")
    if args.frames_per_window <= 0 or args.batch_frames <= 0 or args.max_curve_ticks <= 0:
        parser.error("--frames-per-window, --batch-frames and --max-curve-ticks must be positive")
    if args.list_pairs:
        for index, pair in enumerate(read_pairs(args.pairs)[:50]):
            print(f"{index:5d}  {pair['pair_id']}  target={pair['target']}  window={pair['window']}")
        return
    if args.list_videos:
        list_videos(read_pairs(args.pairs))
        return
    if args.pair_id is None and args.pair_index is None and args.video_name is None:
        parser.error("Choose --pair-id, --pair-index, or --video-name (or use --list-pairs/--list-videos).")
    if args.video_name is not None and not args.full_video:
        parser.error("--video-name identifies a complete video; use it together with --full-video.")
    if args.no_checkpoint:
        checkpoint = None
    else:
        checkpoint = args.checkpoint

    pairs = read_pairs(args.pairs)
    pair = (select_video_pair(pairs, args.video_name)
            if args.video_name is not None else select_pair(pairs, args.pair_id, args.pair_index))
    start, end = map(float, pair["window"])
    device = torch.device(args.device)
    scores, means, members = load_probe_artifacts(args.scores, args.selector)
    model = load_final_encoder(args.model_path, checkpoint, device)
    config = model.config
    image_size, patch_size, hidden_size = int(config.image_size), int(config.patch_size), int(config.hidden_size)
    grid_size = image_size // patch_size
    patch_count = grid_size * grid_size

    if hidden_size != 768:
        raise ValueError(f"Expected the probe's 768-wide XCLIP, but model width is {hidden_size}")
    pair_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pair["pair_id"]))

    if args.full_video:
        # The selected pair only identifies the video.  Merge every probe record
        # for that fake/normal video so the complete curve carries every known
        # R2F and F2R annotation boundary, not only the selected window's one.
        video_pairs = [
            record for record in pairs
            if record.get("fake_video") == pair["fake_video"]
            and record.get("normal_video") == pair["normal_video"]
        ]
        if not video_pairs:
            raise RuntimeError(f"No records for selected video {pair['fake_video']}")
        boundary_events = sorted({
            (float(record["boundary_time"]), record["target"])
            for record in video_pairs
            if record.get("target") in {"r2f", "f2r"} and record.get("boundary_time") is not None
        })
        fake_paths, normal_paths = all_matched_frame_paths(args.frame_root, pair["fake_video"], pair["normal_video"])
        frame_indices = np.asarray([frame_index(path) for path in fake_paths], dtype=np.int64)
        sample_times = frame_indices.astype(np.float32) / 8.0
        targets = TARGETS if args.target in {"auto", "all"} else (args.target,)
        video_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(pair["fake_video"]).stem)
        pair_output = args.output_dir / f"{video_name}_full_video"
        pair_output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "visualization_mode": "full_video",
            "fake_video": pair["fake_video"],
            "normal_video": pair["normal_video"],
            "selected_pair_id": pair["pair_id"],
            "processed_frame_count": len(fake_paths),
            "boundary_events": [{"time": time, "target": target} for time, target in boundary_events],
            "pairs": video_pairs,
        }
        (pair_output / "video_pairs.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        bases, events = full_video_evidence(
            model, fake_paths, normal_paths, targets, scores, means, members, image_size,
            patch_count, args.crop_youku, device, args.amp, args.batch_frames,
        )
        for target in targets:
            render_full_video_outputs(
                pair_output, boundary_events, target, sample_times, frame_indices, fake_paths, normal_paths,
                bases[target], events[target], grid_size, args.top_patch_ratio, args.video_fps,
                args.max_curve_ticks, args.crop_youku,
            )
        if set(targets) == set(TARGETS):
            combined_base = np.maximum.reduce([bases[target] for target in TARGETS])
            combined_event = np.maximum.reduce([events[target] for target in TARGETS])
            render_full_video_outputs(
                pair_output, boundary_events, "combined", sample_times, frame_indices, fake_paths, normal_paths,
                combined_base, combined_event, grid_size, args.top_patch_ratio, args.video_fps,
                args.max_curve_ticks, args.crop_youku,
            )
        print(f"Wrote full-video heatmaps and temporal curves for {pair['fake_video']} to {pair_output}")
        return

    fake_tensor, sample_times = load_window(args.frame_root, pair["fake_video"], start, end, args.frames_per_window, image_size, args.crop_youku)
    normal_tensor, _ = load_window(args.frame_root, pair["normal_video"], start, end, args.frames_per_window, image_size, args.crop_youku, reference_times=sample_times)
    fake_paths = sample_frame_paths(args.frame_root, pair["fake_video"], start, end, args.frames_per_window)
    normal_paths = sample_frame_paths(args.frame_root, pair["normal_video"], start, end, args.frames_per_window, sample_times)
    frame_indices = np.asarray([frame_index(path) for path in fake_paths], dtype=np.int64)
    fake_images, normal_images = display_images(fake_paths, args.crop_youku), display_images(normal_paths, args.crop_youku)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
        outputs = model(torch.cat((fake_tensor, normal_tensor), dim=0).to(device), output_hidden_states=True)
    if args.target == "auto":
        targets = (pair["target"],)
    elif args.target == "all":
        targets = TARGETS
    else:
        targets = (args.target,)
    pair_output = args.output_dir / pair_name
    pair_output.mkdir(parents=True, exist_ok=True)
    (pair_output / "pair.json").write_text(json.dumps(pair, indent=2, ensure_ascii=False), encoding="utf-8")
    for target in targets:
        base, event = target_evidence(outputs.hidden_states, scores, means, members, target, args.frames_per_window, patch_count)
        render_outputs(pair_output, pair, target, sample_times, frame_indices, fake_images, normal_images,
                       base.float().cpu().numpy(), event.float().cpu().numpy(), grid_size,
                       args.top_patch_ratio, args.video_fps, args.max_curve_ticks)
    print(f"Wrote window heatmaps and temporal curves for {pair['pair_id']} to {pair_output}")


if __name__ == "__main__":
    main()
