"""Int6 per-tensor symmetric quantization, bias rounding, the weight payload and numpy reference forward passes."""

from __future__ import annotations

import numpy as np

from seed import LATENT_DIM, SEED_VERSION

FORMAT = 2
QMAX = 31  # int6: q in [-31, 31]
BIAS_DECIMALS = 3
# One payload character per weight: character i encodes q = i - QMAX (the base64 alphabet without "/").
# Shared with packages/core/src/model/decode.ts.
WEIGHT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def quantize_tensor(w: np.ndarray) -> tuple[np.ndarray, np.float32]:
    w = w.astype(np.float32)
    amax = float(np.abs(w).max()) if w.size else 0.0
    scale = np.float32(amax / QMAX) if amax > 0 else np.float32(1.0)
    q = np.clip(np.rint(w / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale


def round_bias(b: np.ndarray) -> np.ndarray:
    """Biases rounded to BIAS_DECIMALS decimals, as float32, so `f32_repr` prints at most that many."""
    return np.round(b.astype(np.float64), BIAS_DECIMALS).astype(np.float32)


def quantize_decoder(specs: list[dict]) -> dict:
    layers = []
    for s in specs:
        q, scale = quantize_tensor(s["weight"])
        layers.append({
            "in": int(s["weight"].shape[1]),
            "out": int(s["weight"].shape[0]),
            "activation": s["activation"],
            "scale": scale,
            "bias": round_bias(s["bias"]),
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


def pack_weights(qspec: dict) -> str:
    """Every layer's q, concatenated in layer order, each row-major (out, in), one WEIGHT_ALPHABET character per weight."""
    q = np.concatenate([layer["q"].reshape(-1) for layer in qspec["layers"]]).astype(np.int64)
    if q.size and (q.min() < -QMAX or q.max() > QMAX):
        raise ValueError(f"quantized weight out of range [{-QMAX}, {QMAX}]: {q.min()}..{q.max()}")
    return "".join(WEIGHT_ALPHABET[v] for v in q + QMAX)


def f32_repr(x) -> str:
    """Shortest decimal that round-trips to the same float32 (JS parses it then Math.fround)."""
    return np.format_float_positional(np.float32(x), unique=True, trim="-")
