import json

import numpy as np
import torch

import model
import train


def synthetic(n=128, seed=0):
    """Four fixed patterns with a little noise, so a VAE can learn them quickly."""
    rng = np.random.default_rng(seed)
    base = rng.random((4, 256)) > 0.75
    out = []
    for i in range(n):
        p = base[i % 4].copy()
        flip = rng.random(256) < 0.02
        p[flip] = ~p[flip]
        out.append(p)
    return np.stack(out).astype(np.uint8)


def test_decoder_layer_specs_shapes():
    dec = model.Decoder(latent=8, hidden=(16,))
    specs = dec.layer_specs()
    assert [s["weight"].shape for s in specs] == [(16, 8), (256, 16)]
    assert [s["activation"] for s in specs] == ["relu", "none"]
    assert specs[0]["weight"].dtype == np.float32


def test_decoder_forward_matches_numpy():
    dec = model.Decoder(latent=8, hidden=(16,))
    z = torch.randn(3, 8)
    expected = dec(z).detach().numpy()
    h = z.numpy()
    for s in dec.layer_specs():
        h = h @ s["weight"].T + s["bias"]
        if s["activation"] == "relu":
            h = np.maximum(h, 0)
    assert np.allclose(h, expected, atol=1e-5)


def test_vae_loss_components_are_positive():
    x = torch.rand(4, 256).round()
    logits = torch.zeros(4, 256)
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    loss, recon, kl = model.vae_loss(logits, x, mu, logvar, beta=1.0)
    assert abs(kl.item()) < 1e-6
    assert abs(recon.item() - 256 * np.log(2)) < 1e-3
    assert abs(loss.item() - recon.item()) < 1e-6


def test_training_reduces_loss_and_writes_artifacts(tmp_path):
    x = synthetic()
    config = train.default_config(run="t", epochs=15, batch=32, lr=5e-3, latent=8, enc_hidden=32, dec_hidden=(32,), seed=1)
    report = train.train(config, x[:96], x[96:], tmp_path)
    assert len(report["epochs"]) == 15
    assert report["epochs"][-1]["train_loss"] < report["epochs"][0]["train_loss"]
    assert (tmp_path / "best.pt").exists()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["best_epoch"] == report["best_epoch"]
    vae, cfg, extra = model.load_checkpoint(tmp_path / "best.pt")
    assert cfg["latent"] == 8
    assert isinstance(vae, model.VAE)
    assert 0.0 <= extra["val_acc"] <= 1.0
