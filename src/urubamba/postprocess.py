"""Spatial clean-up of binary maps and the edge-preserving score filter."""
import numpy as np
from scipy.ndimage import label, median_filter, uniform_filter

from .config import CLEANUP, GUIDED


def cleanup(pred, valid, majority=True):
    """3 x 3 majority (only where every neighbour is valid) and removal of patches < 1 ha.

    Small real ice bodies can disappear; the trade-off is reported in the edge metrics.
    """
    if majority:
        full = uniform_filter(valid.astype("float32"), CLEANUP["majority_window"], mode="constant") > 0.999
        pred = np.where(full, median_filter(pred, size=CLEANUP["majority_window"]), pred)
    labels, _ = label(pred & valid, structure=np.ones((3, 3)))
    keep = np.bincount(labels.ravel()) >= CLEANUP["min_component_px"]
    keep[0] = False
    return keep[labels] & valid


def guided(score, ndsi, valid):
    """Bilateral smoothing of the score with NDSI as range guide, blended 1:1 with the raw score.

    Neighbours with a different NDSI signature get little weight, so mixed borders are
    smoothed less than homogeneous areas.
    """
    r = GUIDED["radius_px"]
    h, w = valid.shape
    sp = np.pad(score, r, mode="edge")
    nd = np.pad(ndsi, r, mode="edge")
    vp = np.pad(valid, r, constant_values=False)
    total = np.zeros_like(score)
    weight = np.zeros_like(score)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sl = np.s_[r + dy:r + dy + h, r + dx:r + dx + w]
            ww = np.exp(-(dy * dy + dx * dx) / (2 * GUIDED["spatial_sigma_px"] ** 2)
                        - ((nd[sl] - ndsi) / GUIDED["ndsi_sigma"]) ** 2 / 2)
            ww = np.where(vp[sl] & valid & np.isfinite(ww), ww, 0)
            total += ww * np.nan_to_num(sp[sl])
            weight += ww
    smooth = total / np.maximum(weight, 1e-8)
    return GUIDED["blend"] * score + (1 - GUIDED["blend"]) * smooth
