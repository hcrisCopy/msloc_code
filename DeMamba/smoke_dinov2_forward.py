"""Run one small DINOv2-NeuronDeMamba training and inference step.

This is a wiring check, not an accuracy experiment.  It consumes the first
fake frame in a tiny probe manifest, repeats it into an eight-frame video
window, then verifies model construction, forward propagation, backward
propagation, and one optimizer update.  In SFT mode it additionally verifies
that a DINOv2 encoder parameter receives a non-zero gradient and is updated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_dinov2_neurons import read_image
from util import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--neuron-indices", type=Path, required=True)
    parser.add_argument("--dinov2-hf-model-path", type=Path, required=True)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--tuning-mode", choices=("lp", "sft"), default="sft")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-youku", action="store_true")
    args = parser.parse_args()
    if args.frames_per_window < 1:
        parser.error("--frames-per-window must be positive")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a visible CUDA GPU")

    records = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"No records in {args.pairs}")
    record = records[0]
    image = read_image(
        args.frame_root, record["fake_video"], record["frame_number"], record["fake_frame_file"], args.crop_youku
    )
    # [B=1, T, C, H, W], exactly the tensor format sent by the data loader.
    video = image.unsqueeze(0).repeat(args.frames_per_window, 1, 1, 1).unsqueeze(0).cuda(non_blocking=True)
    if tuple(video.shape) != (1, args.frames_per_window, 3, 196, 196):
        raise RuntimeError(f"Unexpected smoke-test tensor shape: {tuple(video.shape)}")

    model = build_model(
        "DINOv2_NeuronDeMamba_4",
        neuron_indices_path=str(args.neuron_indices),
        dinov2_hf_model_path=str(args.dinov2_hf_model_path),
    ).cuda()
    if args.tuning_mode == "lp":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    # CLS participates in self-attention at every block, so it is a compact
    # and reliable encoder parameter on which to prove the SFT update.
    encoder_probe_parameter = model.encoder.embeddings.cls_token
    if args.tuning_mode == "sft" and not encoder_probe_parameter.requires_grad:
        raise RuntimeError("SFT requested but DINOv2 encoder parameters are frozen")
    encoder_before = encoder_probe_parameter.detach().clone() if args.tuning_mode == "sft" else None
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=1e-6)
    target = torch.tensor([1], device="cuda", dtype=torch.long)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(video)
    if tuple(logits.shape) != (1, 4) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Unexpected non-finite training logits: shape={tuple(logits.shape)}")
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    if args.tuning_mode == "sft":
        gradient = encoder_probe_parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
            raise RuntimeError("SFT failed: DINOv2 CLS token did not receive a finite non-zero gradient")
    optimizer.step()
    if args.tuning_mode == "sft" and torch.equal(encoder_before, encoder_probe_parameter.detach()):
        raise RuntimeError("SFT failed: DINOv2 CLS token was not updated by optimizer.step()")

    model.eval()
    with torch.inference_mode():
        inference_logits = model(video)
    if tuple(inference_logits.shape) != (1, 4) or not torch.isfinite(inference_logits).all():
        raise RuntimeError(f"Unexpected non-finite inference logits: shape={tuple(inference_logits.shape)}")
    print(
        f"[ok] DINOv2 {args.tuning_mode.upper()} smoke test passed | "
        f"input={tuple(video.shape)} | logits={tuple(inference_logits.shape)} | loss={loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
