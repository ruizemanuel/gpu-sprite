"""Quality gate for a checkpoint's int8 decoder. Usage: python evaluate.py --checkpoint runs/<name>/best.pt"""

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
import sprite
from data import DATA_DIR, dataset_sha256, load_dataset, render_contact
from seed import latents_for_seeds

HERE = Path(__file__).resolve().parent
ACTIVE_DIR = HERE / "active"
FIRST_PROMOTION_MARGIN = 0.01


def dataset_key(train_sha256: str, val_sha256: str) -> str:
    """Key of a dataset in `floors_by_dataset`."""
    return f"{train_sha256}:{val_sha256}"


def recorded_floors(report: dict) -> dict[str, float]:
    """Reconstruction floors a report knows, by dataset key: its `floors_by_dataset` map, plus, for
    reports that predate the map, its single `floors.reconstruction` keyed by its own `data` hashes."""
    floors: dict[str, float] = {}
    legacy = report.get("floors", {}).get("reconstruction")
    data = report.get("data", {})
    if legacy is not None and data.get("train_sha256") and data.get("val_sha256"):
        floors[dataset_key(data["train_sha256"], data["val_sha256"])] = float(legacy)
    floors.update({k: float(v) for k, v in report.get("floors_by_dataset", {}).items()})
    return floors


def current_floor(measured: float, train_sha256: str, val_sha256: str) -> tuple[float, str]:
    """The floor recorded for this dataset in the active report ("active"). Otherwise the measured
    accuracy minus the margin: "new-dataset" when the report records floors for other datasets (or
    an unkeyed one), "first-promotion" when there is no active report or it records no floor."""
    measured_floor = float(measured) - FIRST_PROMOTION_MARGIN
    path = ACTIVE_DIR / "report.json"
    if not path.exists():
        return measured_floor, "first-promotion"
    report = json.loads(path.read_text())
    floors = recorded_floors(report)
    key = dataset_key(train_sha256, val_sha256)
    if key in floors:
        return floors[key], "active"
    if floors or "reconstruction" in report.get("floors", {}):
        return measured_floor, "new-dataset"
    return measured_floor, "first-promotion"


def evaluate_checkpoint(ckpt: Path, data_dir: Path = DATA_DIR, n_seeds: int = 1000, recon_floor: float | None = None, out_dir: Path | None = None) -> dict:
    vae, config, extra = model.load_checkpoint(ckpt)
    train, val, stats = load_dataset(data_dir)
    qspec = quantize.quantize_decoder(vae.decoder.layer_specs())
    objective = extra.get("objective", config.get("objective", "vae"))

    with torch.no_grad():
        mu, _ = vae.encoder(torch.from_numpy(val.astype(np.float32)))
    recon_logits = quantize.forward_numpy(qspec, mu.numpy())
    recon_acc = float((sprite.logits_to_sprites(recon_logits) == val).mean())
    if recon_floor is None:
        floor, source = current_floor(recon_acc, *dataset_sha256(train, val))
    else:
        floor, source = float(recon_floor), "override"

    # A sequential generator of width `latent` yields exactly this prefix of the 32-dim seed stream.
    z = latents_for_seeds(range(n_seeds))[:, : qspec["latent"]]
    samples = sprite.logits_to_sprites(quantize.forward_numpy(qspec, z))
    result = metrics.gate_checks(samples, train, stats, recon_acc, floor)
    if objective == "gan":
        result["checks"]["reconstruction"] = {"value": None, "threshold": None, "op": "n/a", "passed": True, "note": "GAN objective: encoder untrained, reconstruction not applicable"}
        result["passed"] = all(c["passed"] for c in result["checks"].values())
    result.update({
        "checkpoint": Path(ckpt).as_posix(),
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
