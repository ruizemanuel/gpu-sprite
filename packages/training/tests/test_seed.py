import json
from pathlib import Path

import numpy as np
import pytest

from seed import LATENT_DIM, SEED_VERSION, latents_for_seeds, seed_to_latent

FIXTURE = Path(__file__).resolve().parents[2] / "core" / "test" / "fixtures" / "seed-vectors.json"


def test_matches_typescript_fixture():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["seedVersion"] == SEED_VERSION
    assert fixture["latentDim"] == LATENT_DIM
    for entry in fixture["vectors"]:
        z = seed_to_latent(entry["seed"])
        assert z.dtype == np.float32
        assert z.tolist() == entry["z"], f"seed {entry['seed']}"


def test_seed_zero_is_remapped():
    assert seed_to_latent(0).tolist() == seed_to_latent(0x9E3779B9).tolist()


def test_negative_seed_wraps():
    assert seed_to_latent(-1).tolist() == seed_to_latent(4294967295).tolist()


def test_rejects_non_int():
    with pytest.raises(TypeError):
        seed_to_latent(1.5)
    with pytest.raises(TypeError):
        seed_to_latent(True)


def test_batch_helper():
    zs = latents_for_seeds([1, 2, 3])
    assert zs.shape == (3, LATENT_DIM)
    assert zs.dtype == np.float32
    assert zs[1].tolist() == seed_to_latent(2).tolist()


def test_batch_helper_rejects_non_int():
    with pytest.raises(TypeError):
        latents_for_seeds([1, 1.5])


def test_numpy_integer_scalar_accepted():
    assert seed_to_latent(np.int64(5)).tolist() == seed_to_latent(5).tolist()
