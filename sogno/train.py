"""Trains a dreamer on recorded episodes.

    sogno-train --data data/sabbia --out runs/sabbia

Runs on the GPU of an Apple Silicon Mac (MLX picks it on its own), or on the CPU elsewhere.
"""

from __future__ import annotations

import argparse
import math
import time
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from . import checkpoint
from .data import Batch, Episodes, apply_paint
from .model import Config, Dreamer, loss, sample


def dream_past(model: Dreamer, batch: Batch, passes: int, temperature: float) -> np.ndarray:
    """Rolls `batch.past` forward through the frames the model must dream itself (the pushforward
    trick): sampled exactly as at play time, painter strokes included, with no gradient through it."""
    past = batch.past
    for strokes in batch.paints:
        nxt = np.array(sample(model, mx.array(past), passes, temperature))
        for frame, (cells, elements) in zip(nxt, strokes):
            apply_paint(frame, cells, elements, strict=False)
        past = np.concatenate([past[:, 1:], nxt[:, None]], axis=1)
    return past


def evaluate(model: Dreamer, data: Episodes, episodes: range, batches: int, size: int, seed: int) -> dict:
    """One-pass accuracy from a fully masked frame: on every cell, and on the cells that changed."""
    rng = np.random.default_rng(seed)
    c = model.config.classes
    hits = cells = changed_hits = changed = 0
    for _ in range(batches):
        batch = data.batch(rng, size, model.config.history, episodes, flip=False)
        past, target = mx.array(batch.past), mx.array(batch.target)
        guess = mx.argmax(model(past, mx.full(target.shape, c, dtype=mx.uint8)), axis=-1)
        right = guess == target
        moved = target != past[:, -1]
        hits += right.sum().item()
        cells += target.size
        changed_hits += (right & moved).sum().item()
        changed += moved.sum().item()
    return {"accuracy": hits / cells, "changed_accuracy": changed_hits / max(changed, 1)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder written by sogno-gen")
    parser.add_argument("--out", required=True, help="folder for the checkpoint")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--width", type=int, default=32, help="channels at full resolution")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=1, help="residual blocks per level")
    parser.add_argument("--history", type=int, default=2, help="past frames the model sees")
    parser.add_argument("--changed-weight", type=float, default=10.0)
    parser.add_argument(
        "--unroll",
        type=int,
        default=0,
        help="up to this many frames the model dreams before each target (fights drift in long dreams)",
    )
    parser.add_argument("--unroll-passes", type=int, default=4)
    parser.add_argument("--unroll-temperature", type=float, default=1.0)
    parser.add_argument("--holdout", type=int, default=10, help="episodes kept for evaluation")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", help="checkpoint folder to start from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = Episodes(args.data)
    train_episodes, eval_episodes = data.split(args.holdout)
    if args.resume:
        model, _ = checkpoint.load(args.resume)
        model.train()
    else:
        config = Config(
            classes=data.classes, history=args.history, width=args.width, levels=args.levels, blocks=args.blocks
        )
        model = Dreamer(config)
    side = 2 ** (model.config.levels - 1)
    if data.height % side or data.width % side:
        raise SystemExit(f"frames are {data.width}x{data.height}: sides must be multiples of {side}")
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(
        f"{params / 1e6:.2f}M parameters on {mx.default_device()}; {len(train_episodes)} training episodes "
        f"of {data.frames} frames, {data.width}x{data.height}"
    )

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-7, args.lr, args.warmup),
            optim.cosine_decay(args.lr, max(args.steps - args.warmup, 1), args.lr / 20),
        ],
        [args.warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    state = [model.state, optimizer.state, mx.random.state]

    def objective(past, target):
        value, _ = loss(model, past, target, args.changed_weight)
        return value

    grad = nn.value_and_grad(model, objective)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(past, target):
        value, grads = grad(past, target)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return value

    info = {"data": data.meta, "steps": 0}
    started = last = time.perf_counter()
    running = None
    for i in range(1, args.steps + 1):
        unroll = int(rng.integers(0, args.unroll + 1)) if args.unroll else 0
        batch = data.batch(rng, args.batch, model.config.history, train_episodes, unroll=unroll)
        past = dream_past(model, batch, args.unroll_passes, args.unroll_temperature) if unroll else batch.past
        value = step(mx.array(past), mx.array(batch.target))
        mx.eval(state)
        value = value.item()
        if not math.isfinite(value):
            raise SystemExit(f"loss is {value} at step {i}: try a lower --lr")
        running = value if running is None else 0.98 * running + 0.02 * value
        if i % args.log_every == 0:
            now = time.perf_counter()
            rate = args.log_every / (now - last)
            last = now
            eta = (args.steps - i) / rate
            print(
                f"step {i:6d}  loss {running:.4f}  lr {schedule(optimizer.step).item():.2e}  "
                f"{rate:.1f} it/s  eta {eta / 60:.0f} min",
                flush=True,
            )
        if i % args.eval_every == 0 or i == args.steps:
            model.eval()
            metrics = evaluate(model, data, eval_episodes, batches=8, size=args.batch, seed=args.seed + 1)
            model.train()
            info.update(steps=i, loss=running, minutes=(time.perf_counter() - started) / 60, **metrics)
            checkpoint.save(model, args.out, info)
            print(
                f"eval  step {i}  accuracy {metrics['accuracy']:.4f}  "
                f"changed cells {metrics['changed_accuracy']:.4f}  -> {args.out}",
                flush=True,
            )


if __name__ == "__main__":
    main()
