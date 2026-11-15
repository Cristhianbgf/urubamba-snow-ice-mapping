"""Spectral pseudo-labels and the single-rule NDSI baseline.

Positive: NDSI >= 0.55, red/SWIR1 >= 2 and NIR >= 0.15.
Base negative: NDSI <= 0.15 or NIR <= 0.08.
Prudent negative: NDSI <= 0.15 only, so that a dark pixel is no longer labelled
background just because it is dark (shaded ice can be dark).
Pixels that satisfy neither rule are never used for fitting.
"""
import numpy as np

from .config import NDSI_THRESHOLD, NEGATIVE_SEED_BASE, NEGATIVE_SEED_PRUDENT, POSITIVE_SEED


def seeds(cube, valid, prudent=False):
    ndsi, nir = cube[..., 6], cube[..., 3]
    ratio = cube[..., 2] / np.maximum(cube[..., 4], 1e-6)
    pos = (valid & (ndsi >= POSITIVE_SEED["ndsi_min"]) & (ratio >= POSITIVE_SEED["red_swir_min"])
           & (nir >= POSITIVE_SEED["nir_min"]))
    if prudent:
        neg = valid & (ndsi <= NEGATIVE_SEED_PRUDENT["ndsi_max"])
    else:
        neg = valid & ((ndsi <= NEGATIVE_SEED_BASE["ndsi_max"]) | (nir <= NEGATIVE_SEED_BASE["nir_max"]))
    return pos, neg


def ndsi_rule(cube, valid):
    """Classic threshold map: NDSI >= 0.40 and NIR >= 0.11."""
    return valid & (cube[..., 6] >= NDSI_THRESHOLD["ndsi_min"]) & (cube[..., 3] >= NDSI_THRESHOLD["nir_min"])
