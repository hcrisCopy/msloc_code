#!/usr/bin/env python3
"""Build a deterministic fake/real annotation subset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fake", type=int, default=2)
    parser.add_argument("--real", type=int, default=5)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.fake < 0 or args.real < 0 or args.fake + args.real == 0:
        parser.error("--fake and --real must be non-negative with a positive sum")

    source = Path(args.input)
    destination = Path(args.output)
    if destination.exists() and not args.clean:
        raise FileExistsError(f"{destination} already exists; pass --clean to replace it")

    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("annotations") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("annotation JSON must be a list or contain an annotations list")

    fake = [record for record in records if record.get("type") == "fake"][: args.fake]
    real = [record for record in records if record.get("type") == "real"][: args.real]
    if len(fake) != args.fake or len(real) != args.real:
        raise ValueError(
            f"requested fake={args.fake}, real={args.real}; "
            f"found fake={len(fake)}, real={len(real)}"
        )

    # Interleave classes so downstream MAX_SAMPLES does not select only fake
    # records merely because fake examples were written first.
    selected = []
    for index in range(max(len(fake), len(real))):
        if index < len(fake):
            selected.append(fake[index])
        if index < len(real):
            selected.append(real[index])
    output = dict(payload) if isinstance(payload, dict) else selected
    if isinstance(output, dict):
        output["annotations"] = selected

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Smoke annotations: total={len(selected)}, fake={len(fake)}, real={len(real)}")
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
