"""Int8 per-tensor symmetric quantization and numpy reference forward passes."""

from __future__ import annotations

import base64

import numpy as np

from seed import LATENT_DIM, SEED_VERSION

FORMAT = 1


def quantize_tensor(w: np.ndarray) -> tuple[np.ndarray, np.float32]:
    w = w.astype(np.float32)
    amax = float(np.abs(w).max()) if w.size else 0.0
    scale = np.float32(amax / 127.0) if amax > 0 else np.float32(1.0)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_decoder(specs: list[dict]) -> dict:
    layers = []
    for s in specs:
        q, scale = quantize_tensor(s["weight"])
        layers.append({
            "in": int(s["weight"].shape[1]),
            "out": int(s["weight"].shape[0]),
            "activation": s["activation"],
            "scale": scale,
            "bias": s["bias"].astype(np.float32),
            "q": q,
        })
    return {"format": FORMAT, "seedVersion": SEED_VERSION, "latent": layers[0]["in"] if layers else LATENT_DIM, "layers": layers}


def dequantize_layer(layer: dict) -> np.ndarray:
    return layer["q"].astype(np.float32) * np.float32(layer["scale"])


def forward_numpy(qspec: dict, z: np.ndarray) -> np.ndarray:
    h = z.astype(np.float32)
    for layer in qspec["layers"]:
        h = h @ dequantize_layer(layer).T + layer["bias"]
        if layer["activation"] == "relu":
            h = np.maximum(h, np.float32(0))
    return h.astype(np.float32)


def forward_sequential(qspec: dict, z: np.ndarray) -> np.ndarray:
    """Same operation order as packages/core/src/model/cpu.ts: float32 accumulator, sequential over inputs."""
    x = z.astype(np.float32)
    for layer in qspec["layers"]:
        w = dequantize_layer(layer)
        bias = layer["bias"]
        out = np.empty(layer["out"], dtype=np.float32)
        for o in range(layer["out"]):
            acc = np.float32(bias[o])
            row = w[o]
            for i in range(layer["in"]):
                acc = np.float32(acc + np.float32(row[i] * x[i]))
            if layer["activation"] == "relu" and acc < 0:
                acc = np.float32(0)
            out[o] = acc
        x = out
    return x


def pack_weights_base64(qspec: dict) -> str:
    raw = np.concatenate([layer["q"].reshape(-1) for layer in qspec["layers"]]).astype(np.int8).tobytes()
    return base64.b64encode(raw).decode("ascii")


def f32_repr(x) -> str:
    """Shortest decimal that round-trips to the same float32 (JS parses it then Math.fround)."""
    return np.format_float_positional(np.float32(x), unique=True, trim="-")
