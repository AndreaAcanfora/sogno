"""The dreamer: a small U-Net that predicts the next frame one cell class at a time.

Frames are grids of discrete classes (sabbia's 16 elements, or the Game Boy's 4 shades), so instead of
denoising pixels the model fills in a masked frame, MaskGIT-style. It sees the last `history` frames
and a partly revealed next frame, and gives a distribution over classes for every cell. At inference
it starts from a fully masked frame and reveals it in a few passes, most confident cells first. Each
pass conditions on the cells already chosen, which is what keeps two grains of sand from both moving
into the same hole.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class Config:
    classes: int = 16
    history: int = 2
    width: int = 32  # channels at full resolution; doubled once per level
    levels: int = 3  # full, 1/2 and 1/4 resolution: frame sides must be multiples of 2**(levels - 1)
    blocks: int = 1  # residual blocks per level

    def to_dict(self) -> dict:
        return asdict(self)


def one_hot(index: mx.array, classes: int) -> mx.array:
    return (index[..., None] == mx.arange(classes)).astype(mx.float32)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, cin // 4), cin, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, cout // 4), cout, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class Dreamer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        c, mask = config.classes, 1
        channels = [config.width * (2**i) for i in range(config.levels)]
        self.stem = nn.Conv2d(config.history * c + c + mask, channels[0], 3, padding=1)
        self.down = []
        for i, ch in enumerate(channels):
            cin = channels[i - 1] if i else channels[0]
            level = {"blocks": [ResBlock(cin if b == 0 else ch, ch) for b in range(config.blocks)]}
            if i < len(channels) - 1:
                level["pool"] = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            self.down.append(level)
        self.middle = [ResBlock(channels[-1], channels[-1]) for _ in range(2)]
        self.up = []
        for i in reversed(range(len(channels) - 1)):
            self.up.append(
                {
                    "lift": nn.Conv2d(channels[i + 1], channels[i], 3, padding=1),
                    "blocks": [
                        ResBlock(channels[i] * 2 if b == 0 else channels[i], channels[i]) for b in range(config.blocks)
                    ],
                }
            )
        self.norm = nn.GroupNorm(min(32, channels[0] // 4), channels[0], pytorch_compatible=True)
        self.head = nn.Conv2d(channels[0], c, 1)

    def __call__(self, past: mx.array, partial: mx.array) -> mx.array:
        """Logits `[B, H, W, classes]` for the next frame.

        `past` is `[B, history, H, W]` class ids, oldest first; `partial` is `[B, H, W]` class ids where
        the value `classes` marks a cell still to be predicted.
        """
        c = self.config.classes
        b, k, h, w = past.shape
        frames = one_hot(past, c).transpose(0, 2, 3, 1, 4).reshape(b, h, w, k * c)
        x = mx.concatenate([frames, one_hot(partial, c + 1)], axis=-1)
        x = self.stem(x)
        skips = []
        for level in self.down:
            for block in level["blocks"]:
                x = block(x)
            if "pool" in level:
                skips.append(x)
                x = level["pool"](x)
        for block in self.middle:
            x = block(x)
        for level in self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
            x = level["lift"](x)
            x = mx.concatenate([x, skips.pop()], axis=-1)
            for block in level["blocks"]:
                x = block(x)
        return self.head(nn.silu(self.norm(x)))


def mask_ratio(t: mx.array | float) -> mx.array | float:
    """Fraction of cells still masked at progress `t` in [0, 1] (cosine schedule)."""
    if isinstance(t, mx.array):
        return mx.cos(t * math.pi / 2)
    return math.cos(t * math.pi / 2)


def loss(model: Dreamer, past: mx.array, target: mx.array, changed_weight: float = 10.0) -> tuple[mx.array, mx.array]:
    """Masked-cell cross-entropy. Cells that change from the last frame weigh more: most of a sabbia
    frame sits still, and copying it is easy; the physics is in the few cells that move."""
    c = model.config.classes
    b = target.shape[0]
    ratio = mask_ratio(mx.random.uniform(shape=(b, 1, 1)))
    masked = mx.random.uniform(shape=target.shape) < ratio
    partial = mx.where(masked, c, target)
    logits = model(past, partial)
    ce = nn.losses.cross_entropy(logits, target, reduction="none")
    changed = (target != past[:, -1]).astype(mx.float32)
    weight = masked.astype(mx.float32) * (1 + changed_weight * changed)
    value = (ce * weight).sum() / mx.maximum(weight.sum(), 1)
    return value, ce


def sample(model: Dreamer, past: mx.array, passes: int = 4, temperature: float = 1.0) -> mx.array:
    """Dreams the next frame: `[B, H, W]` class ids, revealed over `passes` model calls."""
    c = model.config.classes
    b, _, h, w = past.shape
    cells = h * w
    partial = mx.full((b, h, w), c, dtype=mx.uint8)
    for i in range(passes):
        logits = model(past, partial).reshape(b, cells, c).astype(mx.float32)
        if temperature > 0:
            choice = mx.random.categorical(logits / temperature, axis=-1)
        else:
            choice = mx.argmax(logits, axis=-1)
        logp = nn.log_softmax(logits, axis=-1)
        confidence = mx.take_along_axis(logp, choice[..., None], axis=-1)[..., 0]
        if temperature > 0:
            # A little Gumbel noise, fading with each pass, so the first cells revealed are not
            # always the same ones (as in MaskGIT).
            gumbel = -mx.log(-mx.log(mx.random.uniform(shape=confidence.shape, low=1e-9, high=1.0)))
            confidence = confidence + temperature * (1 - (i + 1) / passes) * gumbel
        flat = partial.reshape(b, cells)
        masked = flat == c
        confidence = mx.where(masked, confidence, mx.inf)
        keep_masked = math.floor(mask_ratio((i + 1) / passes) * cells)
        if keep_masked > 0:
            threshold = mx.sort(confidence, axis=-1)[:, keep_masked - 1 : keep_masked]
            reveal = masked & (confidence > threshold)
        else:
            reveal = masked
        flat = mx.where(reveal, choice.astype(mx.uint8), flat)
        partial = flat.reshape(b, h, w)
    return partial
