"""Train the sprite VAE on CPU. Usage: python train.py --run baseline [--epochs 300]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import model
from data import DATA_DIR, load_dataset

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"


def default_config(**overrides) -> dict:
    config = {
        "run": "baseline",
        "objective": "vae",
        "epochs": 300,
        "batch": 64,
        "lr": 1e-3,
        "beta": 1.0,
        "warmup": 0.3,
        "latent": 32,
        "enc_hidden": 128,
        "dec_hidden": (128,),
        "seed": 1337,
    }
    config.update(overrides)
    config["dec_hidden"] = tuple(int(h) for h in config["dec_hidden"])
    return config


def parse_args(argv: list[str]) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--objective", choices=["vae", "gan"], default="vae")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.3)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--enc-hidden", type=int, default=128)
    p.add_argument("--dec-hidden", default="128", help="comma-separated hidden widths of the decoder")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    a = p.parse_args(argv)
    return default_config(
        run=a.run, objective=a.objective, epochs=a.epochs, batch=a.batch, lr=a.lr, beta=a.beta, warmup=a.warmup, latent=a.latent,
        enc_hidden=a.enc_hidden, dec_hidden=[int(h) for h in a.dec_hidden.split(",") if h], seed=a.seed,
        data_dir=a.data_dir,
    )


@torch.no_grad()
def validate(vae: model.VAE, val_x: torch.Tensor) -> tuple[float, float]:
    mu, _ = vae.encoder(val_x)
    logits = vae.decoder(mu)
    recon = torch.nn.functional.binary_cross_entropy_with_logits(logits, val_x, reduction="none").sum(1).mean()
    acc = ((logits > 0).float() == val_x).float().mean()
    return float(recon), float(acc)


def train_gan(config: dict, train_np: np.ndarray, val_np: np.ndarray, run_dir: Path, log=print) -> dict:
    import metrics
    from seed import latents_for_seeds

    torch.set_num_threads(1)  # tiny MLPs: thread sync overhead dominates, one thread is much faster on CPU
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    vae = model.build_vae(config)  # encoder stays untrained; decoder is the generator
    gen = vae.decoder
    disc = model.Discriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=float(config["lr"]), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(config["lr"]), betas=(0.5, 0.999))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    train_x = torch.from_numpy(train_np.astype(np.float32))
    epochs, batch, latent = int(config["epochs"]), int(config["batch"]), int(config["latent"])
    # A sequential generator of width `latent` yields exactly this prefix of the 32-dim seed stream.
    probe_z = torch.from_numpy(latents_for_seeds(range(200))[:, :latent])
    best_score, best_epoch, history = -1.0, -1, []
    start = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(train_x))
        sums = np.zeros(2)
        steps = 0
        for i in range(0, len(train_x), batch):
            real = train_x[perm[i : i + batch]]
            z = torch.randn(len(real), latent)
            probs = torch.sigmoid(gen(z))
            fake = (probs > 0.5).float() + probs - probs.detach()  # straight-through hard samples
            loss_d = bce(disc(real), torch.ones(len(real))) + bce(disc(fake.detach()), torch.zeros(len(real)))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            loss_g = bce(disc(fake), torch.ones(len(real)))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            sums += (loss_d.item(), loss_g.item())
            steps += 1
        with torch.no_grad():
            samples = (gen(probe_z) > 0).numpy().astype(np.uint8)
        score = metrics.components_fraction(samples, 2) + metrics.uniqueness(samples)
        row = {"epoch": epoch, "loss_d": sums[0] / steps, "loss_g": sums[1] / steps, "probe_score": score}
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_checkpoint(run_dir / "best.pt", vae, config, {"epoch": epoch, "objective": "gan", "probe_score": score, "val_recon": None, "val_acc": None})
        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"epoch {epoch:4d} loss_d {row['loss_d']:.3f} loss_g {row['loss_g']:.3f} probe {score:.3f}")
    report = {
        "config": {**config, "dec_hidden": list(config["dec_hidden"])},
        "train_examples": int(len(train_np)),
        "val_examples": int(len(val_np)),
        "seconds": time.time() - start,
        "best_epoch": best_epoch,
        "best_probe_score": best_score,
        "decoder_parameters": int(sum(p.numel() for p in gen.parameters())),
        "epochs": history,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(config: dict, train_np: np.ndarray, val_np: np.ndarray, run_dir: Path, log=print) -> dict:
    if config.get("objective") == "gan":
        return train_gan(config, train_np, val_np, run_dir, log)
    torch.set_num_threads(1)  # tiny MLPs: thread sync overhead dominates, one thread is much faster on CPU
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    vae = model.build_vae(config)
    opt = torch.optim.Adam(vae.parameters(), lr=float(config["lr"]))
    train_x = torch.from_numpy(train_np.astype(np.float32))
    val_x = torch.from_numpy(val_np.astype(np.float32))
    epochs = int(config["epochs"])
    batch = int(config["batch"])
    warmup_epochs = max(1, round(float(config["warmup"]) * epochs))
    best = float("inf")
    best_epoch = -1
    history = []
    start = time.time()
    for epoch in range(epochs):
        beta_t = float(config["beta"]) * min(1.0, (epoch + 1) / warmup_epochs)
        vae.train()
        perm = torch.randperm(len(train_x))
        sums = np.zeros(3)
        steps = 0
        for i in range(0, len(train_x), batch):
            xb = train_x[perm[i : i + batch]]
            logits, mu, logvar = vae(xb)
            loss, recon, kl = model.vae_loss(logits, xb, mu, logvar, beta_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += (loss.item(), recon.item(), kl.item())
            steps += 1
        vae.eval()
        val_recon, val_acc = validate(vae, val_x)
        row = {
            "epoch": epoch,
            "beta": beta_t,
            "train_loss": sums[0] / steps,
            "train_recon": sums[1] / steps,
            "train_kl": sums[2] / steps,
            "val_recon": val_recon,
            "val_acc": val_acc,
        }
        history.append(row)
        if val_recon < best:
            best, best_epoch = val_recon, epoch
            model.save_checkpoint(run_dir / "best.pt", vae, config, {"epoch": epoch, "val_recon": val_recon, "val_acc": val_acc})
        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"epoch {epoch:4d} beta {beta_t:.2f} loss {row['train_loss']:.2f} recon {row['train_recon']:.2f} kl {row['train_kl']:.2f} val_recon {val_recon:.2f} val_acc {val_acc:.4f}")
    report = {
        "config": {**config, "dec_hidden": list(config["dec_hidden"])},
        "train_examples": int(len(train_np)),
        "val_examples": int(len(val_np)),
        "seconds": time.time() - start,
        "best_epoch": best_epoch,
        "best_val_recon": best,
        "best_val_acc": history[best_epoch]["val_acc"],
        "decoder_parameters": int(sum(p.numel() for p in vae.decoder.parameters())),
        "epochs": history,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    train_np, val_np, _ = load_dataset(Path(config.pop("data_dir")))
    run_dir = RUNS_DIR / config["run"]
    report = train(config, train_np, val_np, run_dir)
    if config["objective"] == "gan":
        best = f"probe_score {report['best_probe_score']:.3f}"
    else:
        best = f"val_recon {report['best_val_recon']:.3f} val_acc {report['best_val_acc']:.4f}"
    print(f"best epoch {report['best_epoch']} {best} decoder params {report['decoder_parameters']} in {report['seconds']:.1f}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
