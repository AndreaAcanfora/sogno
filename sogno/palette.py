"""Colours for drawing frames: sabbia's elements (from sabbia/src/element.rs)."""

import numpy as np

SABBIA = np.array(
    [
        0x10121A,
        0x6B6F7A,
        0xE2BF6E,
        0x2F7DE1,
        0xFF8A1C,
        0x555860,
        0xB8C4D6,
        0x7A4E2A,
        0x4A3A2A,
        0xF04A14,
        0x7CE83A,
        0xA8E4F5,
        0x2FA34A,
        0xC9E8EF,
        0x8A8680,
        0xC83A10,
    ]
)

NAMES = [
    "empty",
    "wall",
    "sand",
    "water",
    "fire",
    "smoke",
    "steam",
    "wood",
    "oil",
    "lava",
    "acid",
    "ice",
    "plant",
    "glass",
    "stone",
    "ember",
]


def rgb(frame: np.ndarray, palette: np.ndarray = SABBIA) -> np.ndarray:
    """`[H, W]` class ids to `[H, W, 3]` u8 colours."""
    colours = np.stack([(palette >> 16) & 0xFF, (palette >> 8) & 0xFF, palette & 0xFF], axis=-1).astype(np.uint8)
    return colours[frame]
