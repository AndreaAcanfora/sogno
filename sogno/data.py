"""Training data: episodes recorded by `sogno-gen` (see gen/src/main.rs for the format)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMPTY, WALL = 0, 1


def apply_paint(state: np.ndarray, cells: np.ndarray, elements: np.ndarray, strict: bool = True) -> None:
    """Applies painted cells to a flat-indexable `state`, in place.

    With `strict`, cells are set exactly as recorded (the real world). Without it, they follow
    sabbia's brush rule on whatever is there now (used on dreamed worlds): the eraser and walls
    overwrite, every other material only fills empty cells.
    """
    flat = state.reshape(-1)
    if not strict:
        keep = (elements == EMPTY) | (elements == WALL) | (flat[cells] == EMPTY)
        cells, elements = cells[keep], elements[keep]
    flat[cells] = elements


@dataclass
class Batch:
    past: np.ndarray  # u8 [B, history, H, W]: the world after the painter acted, oldest first
    target: np.ndarray  # u8 [B, H, W]: the world `stride` steps after the last frame of `past`
    # With `unroll`: for each of the frames the model must dream before `target`, the painter's
    # strokes for every sample, as (cells, elements). See `batch`.
    paints: list = field(default_factory=list)


class Episodes:
    def __init__(self, path: str | Path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        self.states = np.load(path / "states.npy", mmap_mode="r")
        paints = np.load(path / "paints.npy")
        self.classes = int(self.meta["classes"])
        self.episodes, frames_plus_one, self.height, self.width = self.states.shape
        self.frames = frames_plus_one - 1
        keys = paints[:, 0].astype(np.int64) * self.frames + paints[:, 1]
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._cells = paints[order, 2]
        self._elements = paints[order, 3].astype(np.uint8)

    def paints(self, episode: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """The cells the painter changed at `frame`, and what it painted there."""
        key = episode * self.frames + frame
        lo, hi = np.searchsorted(self._keys, [key, key + 1])
        return self._cells[lo:hi], self._elements[lo:hi]

    def painted(self, episode: int, frame: int) -> np.ndarray:
        """The world at `frame` after the painter acted: the model's input for that frame."""
        state = np.array(self.states[episode, frame])
        apply_paint(state, *self.paints(episode, frame))
        return state

    def split(self, holdout: int) -> tuple[range, range]:
        """Episode ids for training and for evaluation (the last `holdout` episodes)."""
        holdout = min(max(holdout, 1), self.episodes - 1)
        return range(self.episodes - holdout), range(self.episodes - holdout, self.episodes)

    def batch(
        self, rng: np.random.Generator, size: int, history: int, episodes: range, flip: bool = True, unroll: int = 0
    ) -> Batch:
        """Random training pairs. With `unroll` > 0, `past` ends `unroll` frames before the one `target`
        follows: the trainer dreams those frames itself (applying `paints`), so the model learns to
        predict the real next frame from its own, imperfect, past."""
        past = np.empty((size, history, self.height, self.width), np.uint8)
        target = np.empty((size, self.height, self.width), np.uint8)
        paints: list = [[None] * size for _ in range(unroll)]
        for i in range(size):
            episode = int(rng.integers(episodes.start, episodes.stop))
            frame = int(rng.integers(history - 1 + unroll, self.frames))
            start = frame - unroll - history + 1
            for j in range(history):
                past[i, j] = self.painted(episode, start + j)
            target[i] = self.states[episode, frame + 1]
            for s in range(unroll):
                paints[s][i] = self.paints(episode, frame - unroll + 1 + s)
            # sabbia alternates its scan direction every step, so the physics is mirror-symmetric.
            if flip and rng.random() < 0.5:
                past[i] = past[i, :, :, ::-1]
                target[i] = target[i, :, ::-1]
                for s in range(unroll):
                    cells, elements = paints[s][i]
                    y, x = np.divmod(cells, self.width)
                    paints[s][i] = (y * self.width + (self.width - 1 - x), elements)
        return Batch(past, target, paints)
