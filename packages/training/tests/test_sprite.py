import numpy as np

import sprite


def grid(pixels):
    s = np.zeros((16, 16), dtype=np.uint8)
    for y, x in pixels:
        s[y, x] = 1
    return s.reshape(-1)


def test_isolated_pixel_is_cleared():
    out = sprite.despeckle(grid([(7, 7)]))
    assert out.sum() == 0


def test_pairs_are_kept():
    for pair in ([(5, 5), (5, 6)], [(5, 5), (6, 5)], [(5, 5), (6, 6)], [(5, 6), (6, 5)]):
        px = grid(pair)
        assert np.array_equal(sprite.despeckle(px), px), pair


def test_isolated_corner_and_edge_pixels_are_cleared():
    for p in ((0, 0), (0, 15), (15, 0), (15, 15), (0, 7), (7, 0), (15, 7), (7, 15)):
        assert sprite.despeckle(grid([p])).sum() == 0, p


def test_edge_pixel_with_neighbour_is_kept():
    px = grid([(0, 0), (1, 1), (15, 14), (15, 15)])
    assert np.array_equal(sprite.despeckle(px), px)


def test_single_pass_on_raw_image():
    # Clearing the isolated pixel must not cascade into its former neighbourhood.
    px = grid([(3, 3), (3, 4), (10, 10)])
    assert np.array_equal(sprite.despeckle(px), grid([(3, 3), (3, 4)]))


def test_does_not_wrap_around_edges():
    # (0, 15) and (1, 0) are adjacent in the flat row-major array but not on the grid.
    assert sprite.despeckle(grid([(0, 15), (1, 0)])).sum() == 0


def test_logit_exactly_zero_is_background():
    logits = np.full(256, -1.0, dtype=np.float32)
    logits[grid([(4, 4), (4, 5)]).astype(bool)] = 1.0
    logits[4 * 16 + 6] = 0.0
    out = sprite.logits_to_sprites(logits)
    assert out[4 * 16 + 6] == 0
    assert np.array_equal(out, grid([(4, 4), (4, 5)]))


def test_batch_and_single_shapes_return_uint8():
    batch = np.stack([grid([(2, 2)]), grid([(2, 2), (2, 3)]), grid([])]).astype(bool)
    out = sprite.despeckle(batch)
    assert out.shape == (3, 256) and out.dtype == np.uint8
    assert out.sum(1).tolist() == [0, 2, 0]
    single = sprite.despeckle(grid([(2, 2), (2, 3)]))
    assert single.shape == (256,) and single.dtype == np.uint8
    logits = np.where(batch, 1.0, -1.0).astype(np.float32)
    lb = sprite.logits_to_sprites(logits)
    assert lb.shape == (3, 256) and lb.dtype == np.uint8
    assert np.array_equal(lb, out)
    ls = sprite.logits_to_sprites(logits[1])
    assert ls.shape == (256,) and ls.dtype == np.uint8
    assert np.array_equal(ls, out[1])
