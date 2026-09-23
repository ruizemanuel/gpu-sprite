import numpy as np
import pytest

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


def noise_samples():
    rng = np.random.default_rng(1)
    train = (rng.random((50, 256)) > 0.6).astype(np.uint8)
    samples = (rng.random((100, 256)) > 0.6).astype(np.uint8)
    return samples, train


def test_gate_checks_pass_and_fail():
    samples, train = noise_samples()
    stats = {"density_p1": 0.0, "density_p99": 1.0, "components_fraction_real": 0.8}
    result = metrics.gate_checks(samples, train, stats, recon_acc=0.95, recon_floor=0.9)
    assert set(result["checks"]) == {"uniqueness", "hamming_median", "components_fraction", "density", "reconstruction"}
    assert result["checks"]["uniqueness"]["passed"] is True
    assert result["checks"]["reconstruction"]["passed"] is True
    assert result["checks"]["components_fraction"]["passed"] is False  # random noise has many components
    assert result["passed"] is False
    tight = metrics.gate_checks(samples, train, stats, recon_acc=0.5, recon_floor=0.9)
    assert tight["checks"]["reconstruction"]["passed"] is False


def test_fixed_thresholds_are_only_uniqueness_and_hamming():
    assert metrics.THRESHOLDS == {"uniqueness": 0.95, "hamming_median": 8}


def test_gate_thresholds_come_from_stats():
    samples, train = noise_samples()
    # density_p5/p95 would fail this noise; the gate must use p1/p99 instead.
    stats = {"density_p1": 0.0, "density_p99": 1.0, "density_p5": 0.5, "density_p95": 0.5, "components_fraction_real": 0.0}
    result = metrics.gate_checks(samples, train, stats, recon_acc=0.95, recon_floor=0.9)
    comp = result["checks"]["components_fraction"]
    assert comp["threshold"] == 0.0 and comp["passed"] is True
    dens = result["checks"]["density"]
    assert dens["threshold"] == [0.0, 1.0] and dens["passed"] is True
    assert result["passed"] is True


@pytest.mark.parametrize("stats", [
    {"density_p1": 0.0, "density_p99": 1.0},
    {"density_p1": 0.0, "density_p99": 1.0, "components_fraction_real": None},
    # Stats built before the real-sprite reference must be rebuilt, not silently reused.
    {"density_p1": 0.0, "density_p99": 1.0, "components_fraction_val": 0.5},
])
def test_gate_requires_components_fraction_real(stats):
    samples, train = noise_samples()
    with pytest.raises(ValueError, match="components_fraction_real"):
        metrics.gate_checks(samples, train, stats, recon_acc=0.95, recon_floor=0.9)
