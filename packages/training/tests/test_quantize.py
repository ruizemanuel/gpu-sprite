import numpy as np
import pytest
import torch

import model
import quantize
from seed import latents_for_seeds


def random_specs(seed=0):
    torch.manual_seed(seed)
    return model.Decoder(latent=32, hidden=(128,)).layer_specs()


def test_quantize_tensor_bounds_and_error():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 32)).astype(np.float32)
    q, scale = quantize.quantize_tensor(w)
    assert q.dtype == np.int8 and scale.dtype == np.float32
    assert quantize.QMAX == 31
    assert q.min() >= -31 and q.max() <= 31
    assert np.abs(q).max() == 31  # the largest |w| lands on the end of the range
    assert np.abs(q.astype(np.float32) * scale - w).max() <= scale / 2 + 1e-6


def test_quantize_zero_tensor():
    q, scale = quantize.quantize_tensor(np.zeros((4, 4), np.float32))
    assert scale == np.float32(1.0)
    assert not q.any()


def test_quantize_decoder_layout():
    spec = quantize.quantize_decoder(random_specs())
    assert spec["format"] == 2 and spec["seedVersion"] == 1 and spec["latent"] == 32
    assert [(l["in"], l["out"], l["activation"]) for l in spec["layers"]] == [(32, 128, "relu"), (128, 256, "none")]
    assert spec["layers"][0]["q"].shape == (128, 32)


def test_quantize_decoder_rounds_biases_to_three_decimals():
    specs = random_specs()
    qspec = quantize.quantize_decoder(specs)
    for layer, s in zip(qspec["layers"], specs):
        assert layer["bias"].dtype == np.float32
        assert np.array_equal(layer["bias"], quantize.round_bias(s["bias"]))
        for b in layer["bias"]:
            text = quantize.f32_repr(b)
            assert len(text.partition(".")[2]) <= 3, text
            assert np.float32(float(text)) == b


@pytest.mark.parametrize("x, text", [
    (0.46607, "0.466"), (-0.57072, "-0.571"), (0.99962, "1"), (-0.00037, "-0"), (0.0, "0"), (-0.6308599, "-0.631"),
])
def test_round_bias_edge_values(x, text):
    b = quantize.round_bias(np.array([x], np.float32))
    assert b.dtype == np.float32
    assert quantize.f32_repr(b[0]) == text


def test_pack_weights_one_character_per_weight():
    qspec = quantize.quantize_decoder(random_specs())
    packed = quantize.pack_weights(qspec)
    expected = np.concatenate([l["q"].reshape(-1) for l in qspec["layers"]])
    assert len(packed) == expected.size == 32 * 128 + 128 * 256
    assert set(packed) <= set(quantize.WEIGHT_ALPHABET)
    decoded = np.array([quantize.WEIGHT_ALPHABET.index(c) - quantize.QMAX for c in packed])
    assert np.array_equal(decoded, expected)


def test_weight_alphabet_ends_encode_the_range_ends():
    assert quantize.WEIGHT_ALPHABET == "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
    assert len(set(quantize.WEIGHT_ALPHABET)) == 2 * quantize.QMAX + 1 == 63
    qspec = {"layers": [{"q": np.array([[-31, 0, 31]], np.int8)}, {"q": np.array([[-1, 1]], np.int8)}]}
    assert quantize.pack_weights(qspec) == "Af+eg"


def test_pack_weights_rejects_out_of_range_values():
    for bad in (-32, 32):
        qspec = {"layers": [{"q": np.array([[bad]], np.int8)}]}
        with pytest.raises(ValueError):
            quantize.pack_weights(qspec)


def test_forward_numpy_matches_torch_on_dequantized_weights():
    specs = random_specs()
    qspec = quantize.quantize_decoder(specs)
    z = latents_for_seeds(range(8))
    ours = quantize.forward_numpy(qspec, z)
    h = torch.from_numpy(z)
    for layer in qspec["layers"]:
        w = torch.from_numpy(quantize.dequantize_layer(layer))
        h = h @ w.T + torch.from_numpy(layer["bias"])
        if layer["activation"] == "relu":
            h = torch.relu(h)
    assert ours.shape == (8, 256)
    assert np.abs(ours - h.numpy()).max() < 1e-4


def test_forward_sequential_matches_vectorized():
    qspec = quantize.quantize_decoder(random_specs())
    z = latents_for_seeds([5])
    a = quantize.forward_sequential(qspec, z[0])
    b = quantize.forward_numpy(qspec, z)[0]
    assert a.dtype == np.float32
    assert np.abs(a - b).max() < 1e-4


def test_f32_repr_round_trips():
    for x in [0.1, 1 / 3, 123.456, -2.5e-5, 0.0]:
        s = quantize.f32_repr(np.float32(x))
        assert np.float32(float(s)) == np.float32(x)
