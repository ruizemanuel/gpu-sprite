"""Quality-gate metrics for generated 1-bit sprites."""

from __future__ import annotations

from collections import deque

import numpy as np

THRESHOLDS = {"uniqueness": 0.95, "hamming_median": 8, "components_fraction": 0.80}
SIDE = 16


def uniqueness(samples: np.ndarray) -> float:
    rows = {np.packbits(s.astype(np.uint8)).tobytes() for s in samples}
    return len(rows) / len(samples)


def nearest_hamming(samples: np.ndarray, train: np.ndarray) -> np.ndarray:
    s = samples.astype(np.int32)
    t = train.astype(np.int32)
    d = s.sum(1)[:, None] + t.sum(1)[None, :] - 2 * (s @ t.T)
    return d.min(1)


def count_components(sprite: np.ndarray) -> int:
    fg = sprite.reshape(SIDE, SIDE).astype(bool)
    seen = np.zeros_like(fg)
    count = 0
    for y in range(SIDE):
        for x in range(SIDE):
            if not fg[y, x] or seen[y, x]:
                continue
            count += 1
            queue = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < SIDE and 0 <= nx < SIDE and fg[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
    return count


def components_fraction(samples: np.ndarray, max_components: int = 2) -> float:
    ok = sum(1 for s in samples if count_components(s) <= max_components)
    return ok / len(samples)


def density(samples: np.ndarray) -> np.ndarray:
    return samples.reshape(len(samples), -1).astype(np.float32).mean(1)


def gate_checks(samples: np.ndarray, train: np.ndarray, train_stats: dict, recon_acc: float, recon_floor: float) -> dict:
    dens = density(samples)
    p5, p95 = float(np.percentile(dens, 5)), float(np.percentile(dens, 95))
    checks = {
        "uniqueness": {"value": uniqueness(samples), "threshold": THRESHOLDS["uniqueness"], "op": ">="},
        "hamming_median": {"value": float(np.median(nearest_hamming(samples, train))), "threshold": THRESHOLDS["hamming_median"], "op": ">="},
        "components_fraction": {"value": components_fraction(samples, 2), "threshold": THRESHOLDS["components_fraction"], "op": ">="},
        "density": {"value": [p5, p95], "threshold": [float(train_stats["density_p5"]), float(train_stats["density_p95"])], "op": "within"},
        "reconstruction": {"value": float(recon_acc), "threshold": float(recon_floor), "op": ">="},
    }
    for c in checks.values():
        if c["op"] == "within":
            lo, hi = c["threshold"]
            c["passed"] = bool(lo <= c["value"][0] and c["value"][1] <= hi)
        else:
            c["passed"] = bool(c["value"] >= c["threshold"])
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}
