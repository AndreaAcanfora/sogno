"""Lets a trained dreamer play a recorded episode and draws it next to the real game.

    sogno-dream --model runs/sabbia --data data/sabbia --episode -1 --out dream.gif

Both worlds start from the same frames and get the same brush strokes. The real one runs sabbia's
rules; the dream only ever sees its own previous frames. The GIF shows them side by side, and the
table at the end compares how much of each material there is.

With `--score N` it dreams the last N (held-out) episodes at once and prints how far the dream's mix
of materials (whatever their age) drifts from the real one over time: 0 is the same amount of every material, 1 is nothing
in common. That is the number to watch, since cell-by-cell agreement is not expected.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
from PIL import Image

from . import checkpoint
from .data import Episodes, apply_paint
from .model import Dreamer, sample
from .palette import NAMES, class_palette, rgb


def dream(
    model: Dreamer, data: Episodes, episode: int, frames: int, passes: int, temperature: float
) -> tuple[list, list]:
    """Real and dreamed post-paint frames for `episode`, starting after `history` real frames."""
    k = model.config.history
    real = [data.painted(episode, t) for t in range(k)]
    dreamt = [f.copy() for f in real]
    for t in range(k - 1, min(k - 1 + frames, data.frames - 1)):
        past = mx.array(np.stack(dreamt[-k:])[None])
        nxt = np.array(sample(model, past, passes, temperature)[0])
        cells, elements = data.paints(episode, t + 1)
        apply_paint(nxt, cells, elements, strict=False)
        dreamt.append(nxt)
        real.append(data.painted(episode, t + 1))
    return real, dreamt


def materials(data: Episodes, frame: np.ndarray) -> np.ndarray:
    """How many cells of each sabbia element `frame` holds, whatever their age."""
    return np.bincount(data.elements[frame].ravel(), minlength=len(NAMES))


def drift(
    model: Dreamer, data: Episodes, episodes: range, frames: int, passes: int, temperature: float
) -> dict[int, float]:
    """Mean material-mix drift between real and dreamed worlds, after 10, 30, 75 and 150 frames."""
    k = model.config.history
    eps = list(episodes)
    dreamt = [np.stack([data.painted(e, t) for e in eps]) for t in range(k)]
    out = {}
    cells = data.height * data.width
    for t in range(k - 1, min(k - 1 + frames, data.frames - 1)):
        nxt = np.array(sample(model, mx.array(np.stack(dreamt[-k:], axis=1)), passes, temperature))
        for frame, e in zip(nxt, eps):
            apply_paint(frame, *data.paints(e, t + 1), strict=False)
        dreamt = [*dreamt[1 - k :], nxt]
        real = np.stack([data.painted(e, t + 1) for e in eps])
        n = t + 2 - k
        if n in (10, 30, 75, 150) or n == frames:
            count = [np.stack([materials(data, f) for f in w]) for w in (real, nxt)]
            out[n] = float((np.abs(count[0] - count[1]).sum(1) / (2 * cells)).mean())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episode", type=int, default=-1, help="negative counts from the end (held-out episodes)")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--passes", type=int, default=4, help="model calls per dreamed frame")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="dream.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", type=int, default=0, help="score drift on this many held-out episodes")
    args = parser.parse_args(argv)

    mx.random.seed(args.seed)
    model, _ = checkpoint.load(args.model)
    data = Episodes(args.data)
    if args.score:
        episodes = range(max(data.episodes - args.score, 0), data.episodes)
        scores = drift(model, data, episodes, args.frames, args.passes, args.temperature)
        print(f"material drift over {len(episodes)} episodes, temperature {args.temperature}, {args.passes} passes")
        print("  ".join(f"frame {n}: {v:.3f}" for n, v in scores.items()))
        return
    episode = args.episode % data.episodes
    real, dreamt = dream(model, data, episode, args.frames, args.passes, args.temperature)

    gap = np.full((data.height, 2, 3), 255, np.uint8)
    palette = class_palette(data.elements, data.lives)
    images = []
    for a, b in zip(real, dreamt):
        frame = np.concatenate([rgb(a, palette), gap, rgb(b, palette)], axis=1)
        frame = frame.repeat(args.scale, axis=0).repeat(args.scale, axis=1)
        images.append(Image.fromarray(frame))
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=1000 // args.fps, loop=0)
    print(f"episode {episode}: {len(images)} frames, real on the left, dream on the right -> {args.out}")

    agree = np.mean([np.mean(a == b) for a, b in zip(real, dreamt)])
    print(f"cells that match the real game: {agree:.1%} (they drift apart: both worlds are random)")
    print(f"{'material':10s} {'real':>7s} {'dream':>7s}")
    last_real, last_dream = materials(data, real[-1]), materials(data, dreamt[-1])
    for i, name in enumerate(NAMES):
        if i and (last_real[i] or last_dream[i]):
            print(f"{name:10s} {last_real[i]:7d} {last_dream[i]:7d}")


if __name__ == "__main__":
    main()
