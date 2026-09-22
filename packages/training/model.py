"""VAE for 16x16 1-bit sprites. Only the decoder ships."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PIXELS = 256


class Encoder(nn.Module):
    def __init__(self, latent: int = 32, hidden: int = 128):
        super().__init__()
        self.latent = latent
        self.net = nn.Sequential(nn.Linear(PIXELS, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return h[:, : self.latent], h[:, self.latent :]


class Decoder(nn.Module):
    def __init__(self, latent: int = 32, hidden: tuple[int, ...] = (128,)):
        super().__init__()
        dims = [latent, *hidden, PIXELS]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        last = len(self.linears) - 1
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < last:
                h = F.relu(h)
        return h

    def layer_specs(self) -> list[dict]:
        last = len(self.linears) - 1
        return [
            {
                "weight": lin.weight.detach().cpu().numpy().astype(np.float32),
                "bias": lin.bias.detach().cpu().numpy().astype(np.float32),
                "activation": "relu" if i < last else "none",
            }
            for i, lin in enumerate(self.linears)
        ]


class VAE(nn.Module):
    def __init__(self, latent: int = 32, enc_hidden: int = 128, dec_hidden: tuple[int, ...] = (128,)):
        super().__init__()
        self.encoder = Encoder(latent, enc_hidden)
        self.decoder = Decoder(latent, dec_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


def vae_loss(logits, x, mu, logvar, beta: float):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return recon + beta * kl, recon, kl


def build_vae(config: dict) -> VAE:
    return VAE(int(config["latent"]), int(config["enc_hidden"]), tuple(int(h) for h in config["dec_hidden"]))


def save_checkpoint(path: Path, vae: VAE, config: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": vae.state_dict(), **extra}, path)


def load_checkpoint(path: Path) -> tuple[VAE, dict, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vae = build_vae(blob["config"])
    vae.load_state_dict(blob["state_dict"])
    vae.eval()
    extra = {k: v for k, v in blob.items() if k not in ("config", "state_dict")}
    return vae, blob["config"], extra


class Discriminator(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PIXELS, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
