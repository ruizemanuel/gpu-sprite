"""Deterministic seed -> latent, bit-identical to packages/core/src/seed.ts."""

import numbers
from collections.abc import Iterable

import numpy as np

LATENT_DIM = 32
SEED_VERSION = 1
_MASK = 0xFFFFFFFF
_GOLDEN = 0x9E3779B9
_DRAWS_PER_DIM = 12
_TWO_32 = 4294967296.0


def to_uint32_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int, got {seed!r}")
    seed = int(seed)
    if abs(seed) > 2**53 - 1:
        raise TypeError("seed must be a JavaScript safe integer")
    return seed & _MASK


def seed_to_latent(seed: int) -> np.ndarray:
    x = to_uint32_seed(seed)
    if x == 0:
        x = _GOLDEN
    z = np.empty(LATENT_DIM, dtype=np.float32)
    for i in range(LATENT_DIM):
        acc = 0.0
        for _ in range(_DRAWS_PER_DIM):
            x ^= (x << 13) & _MASK
            x ^= x >> 17
            x ^= (x << 5) & _MASK
            acc += x / _TWO_32
        z[i] = np.float32(acc - 6.0)
    return z


def latents_for_seeds(seeds: Iterable[int]) -> np.ndarray:
    return np.stack([seed_to_latent(s) for s in seeds]).astype(np.float32)
