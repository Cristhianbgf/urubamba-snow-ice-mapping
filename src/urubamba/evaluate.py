"""Scoring of the frozen maps against the 2020 reference outlines.

This is the only module that reads the reference. Evaluation support: every valid
pixel of the target scene (scene classification accepted and DEM available).

Metrics: IoU of the positive class (not mIoU), precision, recall, F1, MCC, boundary
F1 at 20 and 40 m, 95th percentile of the symmetric boundary distance, area bias,
and average precision (AP) of the continuous scores before thresholding.
Intervals: 2 000 paired resamples of 1 km blocks (50 x 50 cells). They describe
spatial variation for this scene and reference; they do not correct for model
selection, reference error or the date offset of the Landsat composite.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import config as C
from . import data
from .grid import read, save_raster, x_centres
from .models import RASTERS, TABLES

ABLATIONS = [  # candidate, control, question
    ("ET_relieve", "ET_base", "Añadir relieve y contexto"),
    ("ET_prudente", "ET_relieve", "Semillas negativas prudentes"),
    ("HGB_prudente", "ET_prudente", "Boosting frente a Extra Trees"),
    ("ET_prudente_filtro", "ET_prudente", "Filtro espacial"),
    ("Ensamble_ET_filtro", "ET_base_filtro", "Ensamble frente a la base"),
    ("ET_base_guiado", "ET_base_filtro", "Guiado frente a mayoría"),
]
SCORED = ["ET_base", "ET_relieve", "ET_prudente", "HGB_prudente", "Ensamble_ET"]


def confusion(truth, pred, support):
    return np.array([np.count_nonzero(truth & pred & support), np.count_nonzero(~truth & pred & support),
                     np.count_nonzero(truth & ~pred & support), np.count_nonzero(~truth & ~pred & support)])


def scores(matrix):
    """IoU, precision, recall and F1 from [tp, fp, fn, tn] (vectorised on the last axis)."""
    tp, fp, fn, _ = np.moveaxis(np.asarray(matrix, dtype="float64"), -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fp + fn), tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


def mcc(conf):
    tp, fp, fn, tn = map(float, conf)
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def boundaries(truth, pred, support):
    """Median and P95 symmetric boundary distance (m) and boundary F1 at 20 and 40 m."""
    interior = binary_erosion(support, iterations=2)
    edges = [(m & ~binary_erosion(m)) & interior for m in (truth, pred)]
    if not all(e.any() for e in edges):
        return np.nan, np.nan, np.nan, np.nan
    dist = [distance_transform_edt(~e, sampling=C.RESOLUTION)[o] for e, o in (edges, edges[::-1])]
    f1 = []
    for tol in (20, 40):
        p, r = (np.mean(d <= tol) for d in dist)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    both = np.concatenate(dist)
    return np.median(both), np.percentile(both, 95), *f1


def evaluate():
    truth = data.load_reference()
    support = read(RASTERS / "valid_target.tif").astype(bool)
    save_raster(RASTERS / "referencia_2020.tif", truth.astype("uint8"))
    methods = {p.stem.removeprefix("mascara_"): read(p) == 1 for p in sorted(RASTERS.glob("mascara_*.tif"))}
    b = C.BOOTSTRAP["block_px"]
    blocks = [np.s_[r:r + b, c:c + b] for r in range(0, support.shape[0], b)
              for c in range(0, support.shape[1], b) if support[r:r + b, c:c + b].any()]
    draws = np.random.default_rng(C.SEED).integers(len(blocks), size=(C.BOOTSTRAP["draws"], len(blocks)))
    incidence = read(RASTERS / "variable_incidencia.tif")
    slope = read(RASTERS / "variable_pendiente.tif")
    cube, _ = data.load_scene(C.TARGET_SCENE)
    x = x_centres(support.shape[1])
    regimes = {"Incidencia < 0,3": incidence < .3, "Incidencia ≥ 0,3": incidence >= .3,
               "Pendiente > 30°": slope > 30, "Pendiente ≤ 30°": slope <= 30,
               "NIR < 0,15": cube[..., 3] < .15, "NIR ≥ 0,15": cube[..., 3] >= .15}
    for name, lo, hi in zip(C.SECTOR_NAMES, C.SECTOR_EDGES, C.SECTOR_EDGES[1:]):
        regimes[name] = (x >= lo) & (x < hi)
    boots, rows, by_regime = {}, [], []
    for name, pred in methods.items():
        conf = confusion(truth, pred, support)
        iou, precision, recall, f1 = scores(conf)
        median_d, p95_d, bf20, bf40 = boundaries(truth, pred, support)
        pieces = np.array([confusion(truth[s], pred[s], support[s]) for s in blocks])
        boots[name] = scores(pieces[draws].sum(axis=1))[0]
        lo, hi = np.nanpercentile(boots[name], [2.5, 97.5])
        tp, fp, fn, tn = conf
        rows.append(dict(metodo=name, iou=iou, f1=f1, precision=precision, recall=recall, mcc=mcc(conf),
                         f1_borde_20m=bf20, f1_borde_40m=bf40, distancia_mediana_m=median_d,
                         distancia_p95_m=p95_d, area_km2=(tp + fp) * C.CELL_AREA_KM2,
                         referencia_km2=(tp + fn) * C.CELL_AREA_KM2,
                         sesgo_area_km2=(fp - fn) * C.CELL_AREA_KM2, iou_lo=lo, iou_hi=hi,
                         n_pixeles=int(support.sum()), tp=tp, fp=fp, fn=fn, tn=tn))
        for regime, mask in regimes.items():
            sub = support & mask
            j, p, r, f = scores(confusion(truth, pred, sub))
            by_regime.append(dict(metodo=name, regimen=regime, iou=j, precision=p, recall=r, f1=f,
                                  n=int(sub.sum()), n_positivos=int((truth & sub).sum())))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "metricas.csv", index=False)
    pd.DataFrame(by_regime).to_csv(TABLES / "regimenes.csv", index=False)
    iou = table.set_index("metodo").iou
    pairs = []
    for cand, ctrl, question in ABLATIONS:
        diff = boots[cand] - boots[ctrl]
        pairs.append(dict(pregunta=question, candidato=cand, control=ctrl, delta_iou=iou[cand] - iou[ctrl],
                          lo=np.nanpercentile(diff, 2.5), hi=np.nanpercentile(diff, 97.5)))
    pd.DataFrame(pairs).to_csv(TABLES / "ablaciones.csv", index=False)
    aps, curves = [], []
    gt = truth[support]
    continuous = {name: read(RASTERS / f"puntaje_{name}.tif") for name in SCORED}
    continuous["NDSI"] = read(RASTERS / "variable_ndsi.tif")
    for name, score in continuous.items():
        s = score[support]
        ap = average_precision_score(gt, s)
        p, r, _ = precision_recall_curve(gt, s)
        keep = np.unique(np.r_[0, np.flatnonzero(np.diff(r) != 0), len(p) - 1])
        assert np.isclose(-np.sum(np.diff(r[keep]) * p[keep][:-1]), ap, atol=1e-10)
        aps.append(dict(metodo=name, AP=ap, prevalencia=gt.mean()))
        curves.extend(dict(metodo=name, precision=float(p[i]), recall=float(r[i])) for i in keep)
    pd.DataFrame(aps).to_csv(TABLES / "average_precision.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / "precision_recall.csv", index=False)
    print(table[["metodo", "iou", "f1", "f1_borde_20m", "iou_lo", "iou_hi"]]
          .sort_values("iou", ascending=False).to_string(index=False), flush=True)
    return table
