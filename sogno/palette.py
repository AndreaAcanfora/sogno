"""Colours for drawing frames: sabbia's elements (from sabbia/src/element.rs), and its classes."""

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


FIRE, EMBER = 4, 15


def class_palette(elements, lives) -> np.ndarray:
    """A colour per class (see `Episodes.elements`): its element's colour, except that flames and
    embers are shaded by how long they have left, the way sabbia draws them (world.rs, `color_of`)."""
    out = SABBIA[np.asarray(elements)].copy()
    for i, (element, (low, high)) in enumerate(zip(elements, lives)):
        life = (low + high) // 2
        if element == FIRE:
            heat = min(life, 40)
            out[i] = (255 << 16) | ((60 + heat * 4 + 15) << 8) | heat
        elif element == EMBER:
            glow = min(life, 140) // 2
            out[i] = ((150 + min(glow, 105)) << 16) | ((30 + glow // 2 + 12) << 8) | 10
    return out
