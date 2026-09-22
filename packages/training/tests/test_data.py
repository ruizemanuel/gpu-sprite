import io
import zipfile

import numpy as np
from PIL import Image

import data


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


def test_end_to_end_build(tmp_path):
    tiles = [tile_with([(y, y)]) for y in range(16)] + [tile_with([(0, x)]) for x in range(16)]
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("Tilesheet/monochrome_packed.png", make_sheet(tiles, 8))
    zip_path = tmp_path / "pack.zip"
    zip_path.write_bytes(zbuf.getvalue())
    manifest = {"member": "Tilesheet/monochrome_packed.png", "tile": 16, "columns": 8, "rows": 4, "exclude": []}
    stats = data.build(zip_path, manifest, tmp_path)
    train = np.load(tmp_path / "train.npy")
    val = np.load(tmp_path / "val.npy")
    assert train.dtype == np.uint8 and train.shape[1] == 256
    assert stats["train"] == len(train) and stats["val"] == len(val)
    assert (tmp_path / "contact.png").exists()
