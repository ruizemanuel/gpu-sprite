import json

import numpy as np
import pytest
import torch

import evaluate
import export
import model
import quantize
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
    assert e["minMargin"] == pytest.approx(float(np.abs(e["logits"]).min()))


def test_evaluate_checkpoint_end_to_end(tmp_path):
    x = synthetic(160)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "train.npy", x[:128])
    np.save(data_dir / "val.npy", x[128:])
    dens = x[:128].mean(1)
    (data_dir / "stats.json").write_text(json.dumps({"density_p5": float(np.percentile(dens, 5)), "density_p95": float(np.percentile(dens, 95))}))
    config = train.default_config(run="t", epochs=5, batch=32, lr=5e-3, latent=8, enc_hidden=32, dec_hidden=(32,), seed=1)
    run_dir = tmp_path / "run"
    train.train(config, x[:128], x[128:], run_dir)
    result = evaluate.evaluate_checkpoint(run_dir / "best.pt", data_dir, n_seeds=50, out_dir=run_dir)
    assert set(result["checks"]) >= {"uniqueness", "hamming_median", "components_fraction", "density", "reconstruction"}
    assert (run_dir / "gate.json").exists() and (run_dir / "samples.png").exists()
    assert result["recon_floor_source"] in ("first-promotion", "active")


def test_skip_gate_refuses_src_path(tmp_path):
    with pytest.raises(SystemExit):
        export.main(["--checkpoint", str(tmp_path / "missing.pt"), "--skip-gate"])
