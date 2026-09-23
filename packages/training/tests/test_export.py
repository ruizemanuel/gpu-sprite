import hashlib
import json

import numpy as np
import pytest
import torch

import evaluate
import export
import metrics
import model
import quantize
import sprite
import train
from tests.test_train import synthetic


def random_qspec(seed=0):
    torch.manual_seed(seed)
    return quantize.quantize_decoder(model.Decoder(latent=32, hidden=(128,)).layer_specs())


def test_render_weights_ts_structure():
    qspec = random_qspec()
    src = export.render_weights_ts(qspec, "test@abc123")
    assert src.startswith("// GENERATED")
    assert 'import type { ModelSpec } from "./types.ts";' in src
    assert 'checkpoint: "test@abc123"' in src
    assert src.count('"activation": "relu"') == 1 and src.count('"activation": "none"') == 1
    assert 'weights: "' in src
    payload = src.split('weights: "')[1].split('"')[0]
    assert len(payload) == ((32 * 128 + 128 * 256 + 2) // 3) * 4


def test_build_parity_entries():
    qspec = random_qspec()
    parity = export.build_parity(qspec, range(4))
    assert parity["seedVersion"] == 1 and parity["pixels"] == 256 and len(parity["entries"]) == 4
    e = parity["entries"][1]
    assert e["seed"] == 1 and len(e["z"]) == 32 and len(e["logits"]) == 256 and len(e["bits"]) == 64
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(e["bits"]), dtype=np.uint8))
    assert np.array_equal(bits, (np.array(e["logits"]) > 0).astype(np.uint8))
    assert len(e["sprite"]) == 64
    final = np.unpackbits(np.frombuffer(bytes.fromhex(e["sprite"]), dtype=np.uint8))
    assert np.array_equal(final, sprite.logits_to_sprites(np.array(e["logits"])))
    assert e["minMargin"] == pytest.approx(float(np.abs(e["logits"]).min()))


def synthetic_dataset_and_checkpoint(tmp_path):
    """A tiny synthetic data dir and a 5-epoch checkpoint trained on it."""
    x = synthetic(160)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "train.npy", x[:128])
    np.save(data_dir / "val.npy", x[128:])
    dens = x[:128].mean(1)
    (data_dir / "stats.json").write_text(json.dumps({
        "density_p1": float(np.percentile(dens, 1)),
        "density_p5": float(np.percentile(dens, 5)),
        "density_p95": float(np.percentile(dens, 95)),
        "density_p99": float(np.percentile(dens, 99)),
        "components_fraction_real": metrics.components_fraction(x, 2),
    }))
    config = train.default_config(run="t", epochs=5, batch=32, lr=5e-3, latent=8, enc_hidden=32, dec_hidden=(32,), seed=1)
    run_dir = tmp_path / "run"
    train.train(config, x[:128], x[128:], run_dir)
    return x, data_dir, run_dir


def test_evaluate_checkpoint_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", tmp_path / "no-active")
    _, data_dir, run_dir = synthetic_dataset_and_checkpoint(tmp_path)
    result = evaluate.evaluate_checkpoint(run_dir / "best.pt", data_dir, n_seeds=50, out_dir=run_dir)
    assert set(result["checks"]) >= {"uniqueness", "hamming_median", "components_fraction", "density", "reconstruction"}
    assert (run_dir / "gate.json").exists() and (run_dir / "samples.png").exists()
    assert result["recon_floor_source"] == "first-promotion"
    assert result["recon_floor"] == pytest.approx(result["recon_acc"] - evaluate.FIRST_PROMOTION_MARGIN)


def test_evaluate_checkpoint_uses_the_active_floor_only_for_the_same_dataset(tmp_path, monkeypatch):
    active = tmp_path / "active"
    active.mkdir()
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", active)
    x, data_dir, run_dir = synthetic_dataset_and_checkpoint(tmp_path)
    same = {
        "train_sha256": hashlib.sha256(x[:128].tobytes()).hexdigest(),
        "val_sha256": hashlib.sha256(x[128:].tobytes()).hexdigest(),
    }
    (active / "report.json").write_text(json.dumps({"floors": {"reconstruction": 0.123}, "data": same}))
    result = evaluate.evaluate_checkpoint(run_dir / "best.pt", data_dir, n_seeds=50, out_dir=run_dir)
    assert (result["recon_floor"], result["recon_floor_source"]) == (0.123, "active")
    other = {**same, "train_sha256": "0" * 64}
    (active / "report.json").write_text(json.dumps({"floors": {"reconstruction": 0.123}, "data": other}))
    result = evaluate.evaluate_checkpoint(run_dir / "best.pt", data_dir, n_seeds=50, out_dir=run_dir)
    assert result["recon_floor_source"] == "first-promotion"


def test_skip_gate_refuses_src_path(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit) as exc:
        export.main(["--checkpoint", str(ckpt), "--skip-gate"])
    assert exc.value.code == 2
    assert "src/" in capsys.readouterr().err


def test_export_refuses_reduced_seeds(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit) as exc:
        export.main(["--checkpoint", str(ckpt), "--seeds", "10"])
    assert exc.value.code == 2
    assert "--seeds" in capsys.readouterr().err


def test_export_refuses_nonstandard_data_dir(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit) as exc:
        export.main(["--checkpoint", str(ckpt), "--data-dir", str(tmp_path / "other-data")])
    assert exc.value.code == 2
    assert "--data-dir" in capsys.readouterr().err


def test_export_refuses_nonstandard_out(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit) as exc:
        export.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "candidate.ts")])
    assert exc.value.code == 2
    assert "--out" in capsys.readouterr().err


ACTIVE_DATA = {"train_sha256": "a" * 64, "val_sha256": "b" * 64}


def test_current_floor_same_dataset_uses_active_floor_over_a_higher_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"floors": {"reconstruction": 0.9}, "data": ACTIVE_DATA}))
    assert evaluate.current_floor(0.95, "a" * 64, "b" * 64) == (0.9, "active")


@pytest.mark.parametrize("train_sha256, val_sha256", [("c" * 64, "b" * 64), ("a" * 64, "c" * 64), ("c" * 64, "d" * 64)])
def test_current_floor_new_dataset_starts_its_own_floor(tmp_path, monkeypatch, train_sha256, val_sha256):
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"floors": {"reconstruction": 0.99}, "data": ACTIVE_DATA}))
    floor, source = evaluate.current_floor(0.95, train_sha256, val_sha256)
    assert floor == pytest.approx(0.94)
    assert source == "first-promotion"


def test_current_floor_active_report_without_dataset_hashes_is_first_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"floors": {"reconstruction": 0.9}}))
    assert evaluate.current_floor(0.95, "a" * 64, "b" * 64) == (pytest.approx(0.94), "first-promotion")


def test_current_floor_without_active_report_is_measured_minus_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ACTIVE_DIR", tmp_path)
    floor, source = evaluate.current_floor(0.95, "a" * 64, "b" * 64)
    assert floor == pytest.approx(0.94)
    assert source == "first-promotion"
