"""Checks of the pieces that decide the numbers: geometry, labels, clean-up and scoring."""
import numpy as np
import pytest

from urubamba import config as C
from urubamba.evaluate import boundaries, confusion, mcc, scores
from urubamba.features import BASE, FULL, GROUPS, illumination
from urubamba.grid import grid, x_centres
from urubamba.labels import ndsi_rule, seeds
from urubamba.postprocess import cleanup, guided


def test_grid_matches_box():
    _, (rows, cols) = grid()
    assert (rows, cols) == (725, 1100)
    x = x_centres(cols)
    assert x[0] == C.BOUNDS[0] + 10 and x[-1] == C.BOUNDS[2] - 10


def test_base_is_prefix_of_full_and_groups_cover_everything():
    assert FULL[:len(BASE)] == BASE
    covered = sorted(i for cols in GROUPS.values() for i in cols)
    assert covered == list(range(len(FULL)))


def test_incidence_north_slope_faces_northern_sun():
    # Southern hemisphere: sun to the north (azimuth 0). Elevation rises southwards,
    # so the slope faces north and must be better lit than flat ground.
    rows = np.arange(50, dtype="float32")[:, None]
    dem = np.repeat(rows * 10, 50, axis=1)          # +10 m per row (southwards)
    aspect_sin, aspect_cos, inc = illumination(dem, 0.0, 45.0)
    flat = np.sin(np.radians(45))
    assert np.allclose(aspect_cos[5:-5, 5:-5], 1, atol=1e-6)   # facing north
    assert np.all(inc[5:-5, 5:-5] > flat)
    _, _, inc_south_sun = illumination(dem, 180.0, 45.0)
    assert np.all(inc_south_sun[5:-5, 5:-5] < flat)


def _cube(ndsi, nir, red, swir):
    cube = np.zeros((1, len(ndsi), 7), "float32")
    cube[0, :, 2], cube[0, :, 3], cube[0, :, 4], cube[0, :, 6] = red, nir, swir, ndsi
    return cube


def test_seed_rules():
    cube = _cube(ndsi=[.8, .8, .1, .5, .1], nir=[.3, .3, .3, .05, .05], red=[.6, .15, .1, .3, .02],
                 swir=[.1, .1, .1, .1, .1])
    valid = np.ones((1, 5), bool)
    pos, neg = seeds(cube, valid)
    assert pos[0].tolist() == [True, False, False, False, False]      # ratio < 2 rejects the 2nd
    assert neg[0].tolist() == [False, False, True, True, True]        # dark pixels are negative
    _, neg_p = seeds(cube, valid, prudent=True)
    assert neg_p[0].tolist() == [False, False, True, False, True]     # dark NDSI 0.5 is now ambiguous
    assert ndsi_rule(cube, valid)[0].tolist() == [True, True, False, False, False]


def test_cleanup_removes_patches_below_one_hectare():
    pred = np.zeros((40, 40), bool)
    pred[2:4, 2:4] = True        # 4 cells = 0.16 ha
    pred[10:20, 10:20] = True    # 100 cells = 4 ha
    out = cleanup(pred, np.ones_like(pred))
    assert not out[2:4, 2:4].any() and out[12:18, 12:18].all()


def test_guided_keeps_constant_score():
    score = np.full((20, 20), .7, "float32")
    ndsi = np.random.default_rng(0).uniform(-1, 1, (20, 20)).astype("float32")
    assert np.allclose(guided(score, ndsi, np.ones((20, 20), bool)), .7)


def test_scores_and_mcc():
    truth = np.array([1, 1, 0, 0, 1, 0], bool)
    pred = np.array([1, 0, 1, 0, 1, 0], bool)
    conf = confusion(truth, pred, np.ones(6, bool))
    assert conf.tolist() == [2, 1, 1, 2]
    iou, p, r, f1 = scores(conf)
    assert iou == pytest.approx(.5) and p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3) and mcc(conf) == pytest.approx(1 / 3)


def test_identical_maps_have_perfect_boundaries():
    truth = np.zeros((30, 30), bool)
    truth[8:20, 8:20] = True
    median, p95, f20, f40 = boundaries(truth, truth.copy(), np.ones_like(truth))
    assert median == 0 and p95 == 0 and f20 == 1 and f40 == 1
