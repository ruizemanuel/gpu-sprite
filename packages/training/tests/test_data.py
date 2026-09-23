import hashlib
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

import data
import metrics
import sprite


def make_sheet(tiles, columns):
    """tiles: list of (16,16) bool arrays -> RGBA PNG bytes (black bg, white fg)."""
    rows = (len(tiles) + columns - 1) // columns
    img = np.zeros((rows * 16, columns * 16, 4), dtype=np.uint8)
    img[..., 3] = 255
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        block = img[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        block[t, :3] = (249, 250, 251)
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def tile_with(pixels):
    t = np.zeros((16, 16), dtype=bool)
    for y, x in pixels:
        t[y, x] = True
    return t


def test_binarize_and_slice_roundtrip():
    tiles = [tile_with([(0, 0)]), tile_with([(15, 15)]), tile_with([(3, 4), (3, 5)]), tile_with([])]
    fg = data.binarize(np.array(Image.open(io.BytesIO(make_sheet(tiles, 2))).convert("RGBA")))
    sliced = data.slice_tiles(fg, 16)
    assert sliced.shape == (4, 16, 16)
    for a, b in zip(sliced, tiles):
        assert np.array_equal(a, b)


def test_is_one_bit_rejects_gray():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    assert data.is_one_bit(rgba)
    rgba[0, 0, :3] = 120
    assert not data.is_one_bit(rgba)


def test_filter_drops_empty_full_and_excluded():
    tiles = np.stack([tile_with([]), np.ones((16, 16), bool), tile_with([(1, 1)]), tile_with([(2, 2)])])
    kept = data.filter_tiles(tiles, [{"from": 3, "to": 3, "reason": "test"}])
    assert kept == [2]


def test_filter_include_keeps_only_included_tiles():
    tiles = np.stack([
        tile_with([]),              # 0: empty, inside include
        np.ones((16, 16), bool),    # 1: full, inside include
        tile_with([(1, 1)]),        # 2: kept
        tile_with([(2, 2)]),        # 3: excluded, inside include
        tile_with([(3, 3)]),        # 4: kept
        tile_with([(4, 4)]),        # 5: outside include
        tile_with([(5, 5)]),        # 6: kept (second range)
        tile_with([(6, 6)]),        # 7: outside include
    ])
    exclude = [{"from": 3, "to": 3, "reason": "test"}]
    include = [{"from": 0, "to": 4, "reason": "test"}, {"from": 6, "to": 6, "reason": "test"}]
    assert data.filter_tiles(tiles, exclude, include) == [2, 4, 6]
    # No include list (or an empty one) means the whole sheet.
    assert data.filter_tiles(tiles, exclude) == [2, 4, 5, 6, 7]
    assert data.filter_tiles(tiles, exclude, []) == [2, 4, 5, 6, 7]


def test_dedupe_treats_flip_as_duplicate():
    a = tile_with([(0, 0), (1, 1)])
    b = np.fliplr(a)
    c = tile_with([(5, 5)])
    tiles = np.stack([a, b, c, a.copy()])
    assert data.dedupe(tiles, [0, 1, 2, 3]) == [0, 2]


def test_split_is_stable_and_by_hash():
    rng = np.random.default_rng(0)
    tiles = rng.random((200, 16, 16)) > 0.7
    idx = list(range(200))
    train1, val1 = data.split(tiles, idx)
    train2, val2 = data.split(tiles, idx)
    assert train1 == train2 and val1 == val2
    assert set(train1).isdisjoint(val1)
    assert len(train1) + len(val1) == 200
    assert 5 <= len(val1) <= 40


def test_augment_adds_only_asymmetric_flips():
    sym = tile_with([(0, 0), (0, 15)])
    asym = tile_with([(0, 0)])
    out = data.augment(np.stack([sym, asym]))
    assert out.shape == (3, 16, 16)
    assert np.array_equal(out[2], np.fliplr(asym))


def test_verify_sha256(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"hello")
    good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert data.verify_sha256(tmp_path / "x.bin", good)
    assert not data.verify_sha256(tmp_path / "x.bin", "00" * 32)


def sheet_zip(tmp_path, tiles, columns):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("Tilesheet/monochrome_packed.png", make_sheet(tiles, columns))
    zip_path = tmp_path / "pack.zip"
    zip_path.write_bytes(zbuf.getvalue())
    return zip_path


def test_end_to_end_build(tmp_path):
    tiles = [tile_with([(y, y)]) for y in range(16)] + [tile_with([(0, x)]) for x in range(16)]
    zip_path = sheet_zip(tmp_path, tiles, 8)
    manifest = {"member": "Tilesheet/monochrome_packed.png", "tile": 16, "columns": 8, "rows": 4, "exclude": []}
    stats = data.build(zip_path, manifest, tmp_path)
    train = np.load(tmp_path / "train.npy")
    val = np.load(tmp_path / "val.npy")
    assert train.dtype == np.uint8 and train.shape[1] == 256
    assert stats["train"] == len(train) and stats["val"] == len(val)
    assert {"density_p1", "density_p99", "components_fraction_real"} <= set(stats)
    assert "components_fraction_val" not in stats
    assert stats["density_p1"] <= stats["density_p5"] <= stats["density_p95"] <= stats["density_p99"]
    # The coherence reference is the raw share over every real sprite of the dataset (train + validation).
    assert stats["components_fraction_real"] == metrics.components_fraction(np.concatenate([train, val]), 2)
    assert stats["not_included"] == 0
    assert stats["total_tiles"] == stats["kept_after_filter"] + stats["excluded"] + stats["not_included"]
    assert (tmp_path / "contact.png").exists()


def two_by_two(t, r, c):
    t[r:r + 2, c:c + 2] = True


def test_components_fraction_real_is_raw_and_over_train_and_val(tmp_path):
    # 20 tiles with one blob (1 component) and 10 with two blobs and an isolated pixel (3 components
    # raw, 2 after cleanup). Every tile is asymmetric, so it is written twice (itself and its flip)
    # whichever split it lands in: the raw share over train + val is 20/30 by construction.
    positions = [(r, c) for r in (0, 3, 6, 9, 12) for c in (0, 2, 4, 6)]
    tiles = []
    for r, c in positions:
        t = np.zeros((16, 16), bool)
        two_by_two(t, r, c)
        tiles.append(t)
    for r, c in positions[:10]:
        t = np.zeros((16, 16), bool)
        two_by_two(t, r, c)
        two_by_two(t, r, c + 8)
        t[15, 15] = True
        tiles.append(t)
    manifest = {"member": "Tilesheet/monochrome_packed.png", "tile": 16, "columns": 6, "rows": 5, "exclude": []}
    stats = data.build(sheet_zip(tmp_path, tiles, 6), manifest, tmp_path)
    val = np.load(tmp_path / "val.npy")
    real = np.concatenate([np.load(tmp_path / "train.npy"), val])
    assert stats["unique"] == 30 and len(real) == 60
    assert stats["components_fraction_real"] == pytest.approx(20 / 30)
    # The fixture tells the candidates apart: validation only, or cleaned pixels, give other values.
    assert len(val) > 0 and metrics.components_fraction(val, 2) != pytest.approx(20 / 30)
    assert metrics.components_fraction(sprite.despeckle(real), 2) == 1.0


def test_dataset_sha256_hashes_the_raw_array_bytes():
    train = (np.arange(512).reshape(2, 256) % 2).astype(np.uint8)
    val = np.ones((1, 256), dtype=np.uint8)
    expected = (hashlib.sha256(train.tobytes()).hexdigest(), hashlib.sha256(val.tobytes()).hexdigest())
    assert data.dataset_sha256(train, val) == expected


def test_end_to_end_build_with_include(tmp_path):
    # 0-15: a single pixel on the diagonal; 16-31: a single pixel on the top row (16 duplicates 0).
    tiles = [tile_with([(y, y)]) for y in range(16)] + [tile_with([(0, x)]) for x in range(16)]
    zip_path = sheet_zip(tmp_path, tiles, 8)
    manifest = {
        "member": "Tilesheet/monochrome_packed.png", "tile": 16, "columns": 8, "rows": 4,
        "include": [{"from": 0, "to": 7, "reason": "test"}, {"from": 16, "to": 19, "reason": "test"}],
        "exclude": [{"from": 2, "to": 2, "reason": "test"}, {"from": 24, "to": 31, "reason": "outside include"}],
    }
    stats = data.build(zip_path, manifest, tmp_path)
    train = np.load(tmp_path / "train.npy")
    val = np.load(tmp_path / "val.npy")
    included = [tiles[i] for i in [0, 1, 3, 4, 5, 6, 7, 16, 17, 18, 19]]
    allowed = {np.packbits(t.astype(np.uint8)).tobytes() for t in included}
    allowed |= {np.packbits(np.fliplr(t).astype(np.uint8)).tobytes() for t in included}
    written = np.concatenate([train, val])
    assert len(written) > 0
    assert all(np.packbits(row).tobytes() in allowed for row in written)
    assert stats["not_included"] == 20
    assert stats["kept_after_filter"] == 11
    assert stats["excluded"] == 1
    assert stats["duplicates"] == 1 and stats["unique"] == 10
    assert stats["total_tiles"] == stats["kept_after_filter"] + stats["excluded"] + stats["not_included"] == 32
