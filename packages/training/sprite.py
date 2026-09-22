"""Final sprite decoding: threshold the logits, then clear isolated foreground pixels."""

from __future__ import annotations

import numpy as np

SIDE = 16
PIXELS = SIDE * SIDE


def despeckle(pixels: np.ndarray) -> np.ndarray:
    """Clear every foreground pixel that has no foreground pixel among its 8 neighbours.

    One pass: neighbours are counted on the input image, not on the partially cleaned one.
    Pixels outside the 16x16 grid count as background. Accepts (N, 256) or (256,) 0/1
    pixels (row-major) and returns uint8 of the same shape.
    """
    px = np.asarray(pixels)
    if px.ndim not in (1, 2) or px.shape[-1] != PIXELS:
        raise ValueError(f"expected (N, {PIXELS}) or ({PIXELS},) pixels, got shape {px.shape}")
    fg = px.reshape(-1, SIDE, SIDE).astype(bool)
    padded = np.pad(fg, ((0, 0), (1, 1), (1, 1))).astype(np.uint8)
    neighbours = np.zeros(fg.shape, dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            neighbours += padded[:, dy : dy + SIDE, dx : dx + SIDE]
    return (fg & (neighbours > 0)).astype(np.uint8).reshape(px.shape)


def logits_to_sprites(logits: np.ndarray) -> np.ndarray:
    """(N, 256) or (256,) float logits -> final uint8 pixels: `logit > 0`, then despeckle."""
    return despeckle(np.asarray(logits) > 0)
