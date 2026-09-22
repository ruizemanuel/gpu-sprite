"""Quality gate for a checkpoint's int8 decoder. Usage: python evaluate.py --checkpoint runs/baseline/best.pt"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import metrics
import model
import quantize
from data import DATA_DIR, load_dataset, render_contact
from seed import latents_for_seeds

HERE = Path(__file__).resolve().parent
ACTIVE_DIR = HERE / "active"
FIRST_PROMOTION_MARGIN = 0.01


def current_floor(measured: float) -> tuple[float, str]:
    report = ACTIVE_DIR / "report.json"
    if report.exists():
        floors = json.loads(report.read_text()).get("floors", {})
        if "reconstruction" in floors:
            return float(floors["reconstruction"]), "active"
    return float(measured) - FIRST_PROMOTION_MARGIN, "first-promotion"


def evaluate_checkpoint(ckpt: Path, data_dir: Path = DATA_DIR, n_seeds: int = 1000, recon_floor: float | None = None, out_dir: Path | None = None) -> dict:
    vae, config, extra = model.load_checkpoint(ckpt)
    train, val, stats = load_dataset(data_dir)
    qspec = quantize.quantize_decoder(vae.decoder.layer_specs())
    objective = extra.get("objective", config.get("objective", "vae"))

    with torch.no_grad():
        mu, _ = vae.encoder(torch.from_numpy(val.astype(np.float32)))
    recon_logits = quantize.forward_numpy(qspec, mu.numpy())
    recon_acc = float(((recon_logits > 0).astype(np.uint8) == val).mean())
    if recon_floor is None:
        floor, source = current_floor(recon_acc)
    else:
        floor, source = float(recon_floor), "override"

    # A sequential generator of width `latent` yields exactly this prefix of the 32-dim seed stream.
    z = latents_for_seeds(range(n_seeds))[:, : qspec["latent"]]
    samples = (quantize.forward_numpy(qspec, z) > 0).astype(np.uint8)
    result = metrics.gate_checks(samples, train, stats, recon_acc, floor)
    if objective == "gan":
        result["checks"]["reconstruction"] = {"value": None, "threshold": None, "op": "n/a", "passed": True, "note": "GAN objective: encoder untrained, reconstruction not applicable"}
        result["passed"] = all(c["passed"] for c in result["checks"].values())
    result.update({
        "checkpoint": str(ckpt),
        "objective": objective,
        "seeds": n_seeds,
        "recon_acc": recon_acc,
        "recon_floor": floor,
        "recon_floor_source": source,
        "decoder_parameters": int(sum(int(l["q"].size) + int(l["bias"].size) for l in qspec["layers"])),
        "config": {**config, "dec_hidden": list(config["dec_hidden"])},
    })
    out_dir = out_dir or ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    render_contact(samples[:256].reshape(-1, 16, 16).astype(bool), 16, out_dir / "samples.png", scale=4, labels=False)
    (out_dir / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def print_result(result: dict) -> None:
    for name, c in result["checks"].items():
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {mark} {name}: {c['value']} ({c['op']} {c['threshold']})")
    print("GATE", "PASSED" if result["passed"] else "FAILED", f"recon_acc={result['recon_acc']:.4f} floor={result['recon_floor']:.4f} ({result['recon_floor_source']})")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--recon-floor", type=float, default=None)
    a = p.parse_args(argv)
    result = evaluate_checkpoint(Path(a.checkpoint), Path(a.data_dir), a.seeds, a.recon_floor)
    print_result(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
