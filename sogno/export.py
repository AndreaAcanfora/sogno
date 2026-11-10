"""Exports a trained dreamer for the browser demo: `model.json` (config and tensor table) and `weights.bin`.

    sogno-export --model runs/sabbia --out web/model

The weights are stored as float16, half the download, and the page widens them back to float32. With
`--golden FILE` it also writes a test case for the browser engine: an input and the logits this model
gives for it, computed with the same float16-rounded weights the page will use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import checkpoint
from .model import Dreamer

FRAME = (64, 96)  # sabbia's training frames, height x width, if the checkpoint does not say


def tensors(model: Dreamer) -> list[tuple[str, np.ndarray]]:
    """Every parameter as float16, in a stable order."""
    return [(name, np.array(value.astype(mx.float16))) for name, value in sorted(tree_flatten(model.parameters()))]


def export(model: Dreamer, info: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table, chunks, offset = {}, [], 0
    for name, value in tensors(model):
        table[name] = {"shape": list(value.shape), "offset": offset}
        chunks.append(value.tobytes())
        offset += value.size
    (out / "weights.bin").write_bytes(b"".join(chunks))
    data = info.get("data", {})
    meta = {
        "config": model.config.to_dict(),
        "frame": [data.get("height", FRAME[0]), data.get("width", FRAME[1])],
        "stride": data.get("stride", 2),
        "dtype": "float16",
        "tensors": table,
        "trained": {k: info[k] for k in ("steps", "accuracy", "changed_accuracy") if k in info},
        # Each class's sabbia element and range of remaining life, to draw and count them.
        "elements": data.get("elements", list(range(model.config.classes))),
        "lives": data.get("lives", [[0, 0]] * model.config.classes),
    }
    (out / "model.json").write_text(json.dumps(meta, indent=1) + "\n")


def golden(model: Dreamer, path: Path, frame: tuple[int, int] = FRAME, seed: int = 0) -> None:
    """A random input and the model's logits for it, with float16-rounded weights.

    Layout, all little-endian: `u8 [history + 1, H, W]` (past frames, oldest first, then the partial
    frame where `classes` marks a hidden cell), then `f32 [H, W, classes]` logits.
    """
    rounded = tree_unflatten([(name, mx.array(value).astype(mx.float32)) for name, value in tensors(model)])
    model.update(rounded)
    c, k = model.config.classes, model.config.history
    rng = np.random.default_rng(seed)
    h, w = frame
    past = rng.integers(0, c, size=(1, k, h, w), dtype=np.uint8)
    partial = rng.integers(0, c, size=(1, h, w), dtype=np.uint8)
    partial[rng.random((1, h, w)) < 0.6] = c
    logits = np.array(model(mx.array(past), mx.array(partial)).astype(mx.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.concatenate([past[0], partial]).tobytes() + logits[0].astype("<f4").tobytes())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint folder written by sogno-train")
    parser.add_argument("--out", default="web/model")
    parser.add_argument("--golden", help="also write a browser test case to this file")
    args = parser.parse_args(argv)

    model, info = checkpoint.load(args.model)
    out = Path(args.out)
    export(model, info, out)
    size = (out / "weights.bin").stat().st_size
    print(f"{args.model} -> {out}: {size / 1e6:.1f} MB of float16 weights")
    if args.golden:
        data = info.get("data", {})
        golden(model, Path(args.golden), (data.get("height", FRAME[0]), data.get("width", FRAME[1])))
        print(f"test case -> {args.golden}")


if __name__ == "__main__":
    main()
