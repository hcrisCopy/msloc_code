"""Render one heatmap for the final frame-level fake-sensitive neurons.

The frame probe writes one score vector per XCLIP layer under keys such as
``frame_layer_01_score`` and one final selector containing exactly 768
coordinates.  This script renders only those selected coordinates: each row
is an XCLIP layer, each column is the within-layer rank by sensitivity score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def selector_layers(selector: dict) -> dict[int, list[int]]:
    source = selector.get("layers", selector.get("selected_indices", selector))
    if not isinstance(source, dict):
        raise ValueError("Selector has no valid 'layers' mapping")
    layers = {}
    for raw_layer, raw_indices in source.items():
        layer = int(str(raw_layer).removeprefix("layer_"))
        indices = sorted({int(index) for index in raw_indices})
        if layer < 1 or not indices:
            raise ValueError(f"Invalid selector layer {raw_layer!r}")
        layers[layer] = indices
    return layers


def load_frame_scores(path: Path) -> dict[int, np.ndarray]:
    scores = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if not (key.startswith("frame_layer_") and key.endswith("_score")):
                continue
            layer_text = key[len("frame_layer_"):-len("_score")]
            scores[int(layer_text)] = np.asarray(data[key], dtype=np.float64)
    if not scores:
        raise ValueError(
            f"No frame-level score vectors found in {path}. "
            "Re-run the frame-level probe_xclip_neurons.py first."
        )
    return scores


def render(selector: dict, scores: dict[int, np.ndarray], output_dir: Path):
    layers = selector_layers(selector)
    expected = int(selector.get("final_neuron_count", sum(map(len, layers.values()))))
    actual = sum(map(len, layers.values()))
    if actual != expected:
        raise ValueError(f"Selector declares {expected} neurons but contains {actual}")

    layer_count = int(selector.get("num_hidden_layers", max(max(layers), max(scores))))
    ordered_layers = list(range(1, layer_count + 1))
    ranked = {}
    for layer in ordered_layers:
        indices = layers.get(layer, [])
        if not indices:
            ranked[layer] = []
            continue
        if layer not in scores:
            raise KeyError(f"Selected layer {layer} has no frame score vector")
        score = scores[layer]
        if min(indices) < 0 or max(indices) >= score.size:
            raise IndexError(f"Selector index for layer {layer} is outside its score vector")
        ranked[layer] = sorted(indices, key=lambda index: float(score[index]), reverse=True)

    width = max(len(indices) for indices in ranked.values())
    matrix = np.full((layer_count, width), np.nan, dtype=np.float64)
    for row, layer in enumerate(ordered_layers):
        indices = ranked[layer]
        if indices:
            matrix[row, :len(indices)] = scores[layer][indices]

    figure, axis = plt.subplots(figsize=(max(9, width * .12), max(4, layer_count * .50)), dpi=180)
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="magma", vmin=0.0)
    axis.set_yticks(np.arange(layer_count), [f"layer {layer:02d}" for layer in ordered_layers])
    axis.set_xlabel("rank among final selected neurons in this layer")
    axis.set_title(f"Frame-level fake-sensitive neurons (final {actual})")
    figure.colorbar(image, ax=axis, label="paired fake-vs-real sensitivity score")
    figure.tight_layout()
    output = output_dir / "frame_level_selected_neurons_heatmap.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selector = json.loads(args.selector.read_text(encoding="utf-8"))
    if selector.get("selection_task") not in {None, "frame_level_fake_vs_real"}:
        raise ValueError("This visualizer accepts only the new frame-level selector")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = render(selector, load_frame_scores(args.scores), args.output_dir)
    print(f"Wrote heatmap: {output}")


if __name__ == "__main__":
    main()
