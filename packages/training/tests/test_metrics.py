import numpy as np

import metrics


def sprite(pixels):
    s = np.zeros((16, 16), dtype=np.uint8)
    for y, x in pixels:
        s[y, x] = 1
    return s


def test_uniqueness():
    a = sprite([(0, 0)]).reshape(-1)
    b = sprite([(1, 1)]).reshape(-1)
    assert metrics.uniqueness(np.stack([a, b, a])) == 2 / 3


def test_nearest_hamming():
    train = np.stack([sprite([(0, 0)]).reshape(-1), sprite([(0, 0), (0, 1), (0, 2)]).reshape(-1)])
    samples = np.stack([sprite([(0, 0), (0, 1)]).reshape(-1), sprite([(5, 5)]).reshape(-1)])
    assert metrics.nearest_hamming(samples, train).tolist() == [1, 2]


def test_count_components_uses_8_connectivity():
    assert metrics.count_components(sprite([]).astype(bool)) == 0
    assert metrics.count_components(sprite([(0, 0), (1, 1)]).astype(bool)) == 1
    assert metrics.count_components(sprite([(0, 0), (2, 2)]).astype(bool)) == 2
    assert metrics.count_components(sprite([(0, 0), (0, 15), (15, 0)]).astype(bool)) == 3


def test_components_fraction():
    samples = np.stack([
        sprite([(0, 0)]).reshape(-1),
        sprite([(0, 0), (4, 4)]).reshape(-1),
        sprite([(0, 0), (4, 4), (8, 8)]).reshape(-1),
        sprite([(0, 0), (4, 4), (8, 8), (12, 12)]).reshape(-1),
    ])
    assert metrics.components_fraction(samples, 2) == 0.5


def test_gate_checks_pass_and_fail():
    rng = np.random.default_rng(1)
    train = (rng.random((50, 256)) > 0.6).astype(np.uint8)
    samples = (rng.random((100, 256)) > 0.6).astype(np.uint8)
    stats = {"density_p5": 0.0, "density_p95": 1.0}
    result = metrics.gate_checks(samples, train, stats, recon_acc=0.95, recon_floor=0.9)
    assert set(result["checks"]) == {"uniqueness", "hamming_median", "components_fraction", "density", "reconstruction"}
    assert result["checks"]["uniqueness"]["passed"] is True
    assert result["checks"]["reconstruction"]["passed"] is True
    assert result["checks"]["components_fraction"]["passed"] is False  # random noise has many components
    assert result["passed"] is False
    tight = metrics.gate_checks(samples, train, stats, recon_acc=0.5, recon_floor=0.9)
    assert tight["checks"]["reconstruction"]["passed"] is False
