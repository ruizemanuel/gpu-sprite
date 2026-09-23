"""Build the 1-bit sprite dataset from Kenney's 1-Bit Pack (CC0)."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import metrics

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CACHE_DIR = DATA_DIR / "cache"
TILE = 16
VAL_BUCKET = 0  # sha1(canonical) % 10 == VAL_BUCKET -> validation


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download(url: str, sha256: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and verify_sha256(dest, sha256):
        return dest
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    if not verify_sha256(dest, sha256):
        raise RuntimeError(f"sha256 mismatch for {dest}")
    return dest


def is_one_bit(rgba: np.ndarray) -> bool:
    """Every pixel must be transparent, black or white."""
    alpha = rgba[..., 3]
    rgb = rgba[..., :3].astype(np.int32)
    transparent = alpha == 0
    black = (alpha > 0) & (rgb.max(-1) <= 8)
    white = (alpha > 0) & (rgb.min(-1) >= 200)
    return bool(np.all(transparent | black | white))


def binarize(rgba: np.ndarray) -> np.ndarray:
    lum = rgba[..., :3].astype(np.int32).sum(-1) / 3.0
    return (rgba[..., 3] > 0) & (lum > 127)


def slice_tiles(fg: np.ndarray, tile: int = TILE) -> np.ndarray:
    h, w = fg.shape
    if h % tile or w % tile:
        raise ValueError(f"sheet {w}x{h} is not a multiple of {tile}")
    rows, cols = h // tile, w // tile
    return fg.reshape(rows, tile, cols, tile).transpose(0, 2, 1, 3).reshape(rows * cols, tile, tile)


def index_ranges(ranges: list[dict]) -> set[int]:
    """Tile indices covered by a list of inclusive `{from, to}` ranges."""
    out: set[int] = set()
    for rng in ranges:
        out.update(range(int(rng["from"]), int(rng["to"]) + 1))
    return out


def filter_tiles(tiles: np.ndarray, exclude: list[dict], include: list[dict] | None = None) -> list[int]:
    """Indices of tiles inside `include` (the whole sheet when it is None or empty), then minus
    empty and full tiles and the `exclude` ranges."""
    excluded = index_ranges(exclude)
    included = index_ranges(include) if include else None
    kept = []
    for i, t in enumerate(tiles):
        if included is not None and i not in included:
            continue
        n = int(t.sum())
        if n == 0 or n == t.size or i in excluded:
            continue
        kept.append(i)
    return kept


def canonical(tile: np.ndarray) -> bytes:
    a = np.packbits(tile.astype(np.uint8)).tobytes()
    b = np.packbits(np.fliplr(tile).astype(np.uint8)).tobytes()
    return min(a, b)


def dedupe(tiles: np.ndarray, indices: list[int]) -> list[int]:
    seen: set[bytes] = set()
    out = []
    for i in indices:
        key = canonical(tiles[i])
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def split(tiles: np.ndarray, indices: list[int]) -> tuple[list[int], list[int]]:
    train, val = [], []
    for i in indices:
        bucket = int(hashlib.sha1(canonical(tiles[i])).hexdigest(), 16) % 10
        (val if bucket == VAL_BUCKET else train).append(i)
    return train, val


def augment(tiles: np.ndarray) -> np.ndarray:
    out = [t for t in tiles]
    for t in tiles:
        f = np.fliplr(t)
        if not np.array_equal(f, t):
            out.append(f)
    if not out:
        return np.zeros((0, TILE, TILE), dtype=bool)
    return np.stack(out)


def render_contact(tiles: np.ndarray, columns: int, path: Path, scale: int = 3, labels: bool = True) -> None:
    n = len(tiles)
    rows = (n + columns - 1) // columns
    pad = 40 if labels else 0
    cell = TILE * scale
    img = Image.new("RGB", (pad + columns * cell, pad + rows * cell), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        tile_img = Image.fromarray((t.astype(np.uint8) * 255)).resize((cell, cell), Image.NEAREST).convert("RGB")
        x, y = pad + c * cell, pad + r * cell
        img.paste(tile_img, (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(90, 90, 90))
    if labels:
        for c in range(columns):
            draw.text((pad + c * cell + 2, 4), str(c), fill=(255, 255, 0))
        for r in range(rows):
            draw.text((2, pad + r * cell + 2), f"{r}\n{r * columns}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build(zip_path: Path, manifest: dict, out_dir: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        rgba = np.array(Image.open(io.BytesIO(z.read(manifest["member"]))).convert("RGBA"))
    if not is_one_bit(rgba):
        raise RuntimeError("sheet is not 1-bit")
    tile = int(manifest.get("tile", TILE))
    fg = binarize(rgba)
    tiles = slice_tiles(fg, tile)
    if "columns" in manifest and "rows" in manifest:
        expected = int(manifest["columns"]) * int(manifest["rows"])
        if len(tiles) != expected:
            raise RuntimeError(f"expected {expected} tiles, got {len(tiles)}")
    columns = fg.shape[1] // tile
    render_contact(tiles, columns, out_dir / "contact.png")

    include = manifest.get("include")
    kept = filter_tiles(tiles, manifest.get("exclude", []), include)
    not_included = len(set(range(len(tiles))) - index_ranges(include)) if include else 0
    unique = dedupe(tiles, kept)
    train_idx, val_idx = split(tiles, unique)
    train = augment(tiles[train_idx])
    val = augment(tiles[val_idx])
    train_flat = train.reshape(len(train), tile * tile).astype(np.uint8)
    val_flat = val.reshape(len(val), tile * tile).astype(np.uint8)
    real_flat = np.concatenate([train_flat, val_flat])
    density = train_flat.mean(axis=1)
    stats = {
        "total_tiles": int(len(tiles)),
        "kept_after_filter": int(len(kept)),
        "unique": int(len(unique)),
        # Empty, full and exclude-range tiles inside `include`, so total = kept + excluded + not_included.
        "excluded": int(len(tiles) - len(kept) - not_included),
        "not_included": int(not_included),
        "duplicates": int(len(kept) - len(unique)),
        "train": int(len(train_flat)),
        "val": int(len(val_flat)),
        "density_mean": float(density.mean()),
        "density_p5": float(np.percentile(density, 5)),
        "density_p95": float(np.percentile(density, 95)),
        # Quality-gate thresholds are calibrated on the dataset itself (see metrics.gate_checks).
        "density_p1": float(np.percentile(density, 1)),
        "density_p99": float(np.percentile(density, 99)),
        # Coherence reference: share of all real sprites (train + validation, raw pixels) with <= 2 components.
        "components_fraction_real": metrics.components_fraction(real_flat, 2) if len(real_flat) else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train_flat)
    np.save(out_dir / "val.npy", val_flat)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    train = np.load(data_dir / "train.npy")
    val = np.load(data_dir / "val.npy")
    stats = json.loads((data_dir / "stats.json").read_text())
    return train, val, stats


def main(argv: list[str]) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    zip_path = download(manifest["url"], manifest["sha256"], CACHE_DIR / "kenney_1-bit-pack.zip")
    stats = build(zip_path, manifest, DATA_DIR)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
