"""Training on spectral pseudo-labels with spatially excluded sectors.

The study box is split into three sectors (west, centre, east). For each sector the
models are fitted on the four 2020 fitting dates, excluding that sector and a 500 m
margin, and then predict that sector on the target date (27 August 2020). The target
date never contributes pseudo-labels. Rotating the excluded sector yields a complete map.

Methods
-------
ET_base       Extra Trees, 13 variables, base negative seeds
ET_relieve    Extra Trees, 20 variables, base negative seeds
ET_prudente   Extra Trees, 20 variables, prudent negative seeds
HGB_prudente  HistGradientBoosting, 20 variables, prudent seeds; the number of
              iterations is chosen by the minimum log-loss on 3 km internal blocks
              (200 m margin) and the model is then refitted on all allowed samples
Ensamble_ET   mean of the ET_base and ET_prudente scores (votes, not calibrated
              probabilities)
NDSI          single threshold rule, no training

Each score is thresholded at 0.50 and kept raw and after ``cleanup`` ("_filtro");
ET_base and ET_prudente also get the edge-preserving filter ("_guiado").
"""
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

from . import config as C
from . import data, features, labels
from .grid import save_raster, x_centres
from .postprocess import cleanup, guided

RASTERS = C.OUTPUTS / "rasters"
MODELS = C.OUTPUTS / "models"
TABLES = C.RESULTS / "tables"

METHODS = {  # name: (feature columns, prudent negatives, learner)
    "ET_base": (len(features.BASE), False, "et"),
    "ET_relieve": (len(features.FULL), False, "et"),
    "ET_prudente": (len(features.FULL), True, "et"),
    "HGB_prudente": (len(features.FULL), True, "hgb"),
}


def configuration():
    keys = ["SEED", "FIT_SCENES", "TARGET_SCENE", "SECTOR_EDGES", "SPATIAL_GAP_M",
            "SAMPLES_PER_CLASS_AND_DATE", "INNER_VALIDATION", "POSITIVE_SEED",
            "NEGATIVE_SEED_BASE", "NEGATIVE_SEED_PRUDENT", "NDSI_THRESHOLD", "EXTRA_TREES",
            "BOOSTING", "THRESHOLD", "CLEANUP", "GUIDED", "BOOTSTRAP", "DEM_GRANULE"]
    out = {k: getattr(C, k) for k in keys}
    out["features_base"] = features.BASE
    out["features_full"] = features.FULL
    return out


def prepare(scene, dem):
    cube, valid = data.load_scene(scene)
    valid &= np.isfinite(dem)
    return cube, valid, features.full(cube, valid, dem, data.sun_geometry(scene))


def sample(dev, domain, seed, prudent, n_cols, n=C.SAMPLES_PER_CLASS_AND_DATE):
    """Up to ``n`` pixels per class and date, without replacement, inside ``domain``."""
    rng = np.random.default_rng(seed)
    xs, ys, audit = [], [], []
    for scene, cube, valid, feats in dev:
        for cls, mask in zip((1, 0), labels.seeds(cube, valid, prudent)):
            pool = np.flatnonzero(mask & domain)
            ids = rng.choice(pool, min(n, len(pool)), replace=False)
            xs.append(feats.reshape(-1, feats.shape[-1])[ids, :n_cols])
            ys.append(np.full(len(ids), cls))
            audit.append(dict(scene=scene, label=cls, available=len(pool), used=len(ids)))
    return np.concatenate(xs), np.concatenate(ys), audit


def _boosting(dev, domain, inner_train, inner_val, fold, n_cols, curves, stats, importance):
    seed = C.SEED + fold
    tx, ty, _ = sample(dev, domain & inner_train, seed, True, n_cols)
    vx, vy, _ = sample(dev, domain & inner_val, seed + 100, True, n_cols, n=C.INNER_VALIDATION["samples"])
    assert len(np.unique(ty)) == len(np.unique(vy)) == 2, "both classes needed for inner fit and validation"
    candidate = HistGradientBoostingClassifier(**C.BOOSTING, early_stopping=False, random_state=seed).fit(tx, ty)
    losses = []
    for it, (tp, vp) in enumerate(zip(candidate.staged_predict_proba(tx), candidate.staged_predict_proba(vx)), 1):
        losses.append(log_loss(vy, vp))
        curves.append(dict(sector=fold, iteration=it, train_loss=log_loss(ty, tp), val_loss=losses[-1],
                           train_f1=f1_score(ty, tp[:, 1] >= .5), val_f1=f1_score(vy, vp[:, 1] >= .5),
                           n_train=len(ty), n_val=len(vy), n_positive_train=int(ty.sum()),
                           n_positive_val=int(vy.sum())))
    chosen = int(np.argmin(losses) + 1)
    args = dict(C.BOOSTING, max_iter=chosen)
    # Grouped permutation importance on the inner validation set: dependence on the
    # pseudo-label rules, not accuracy against glacier outlines.
    checkpoint = HistGradientBoostingClassifier(**args, early_stopping=False, random_state=seed).fit(tx, ty)
    reference_loss = log_loss(vy, checkpoint.predict_proba(vx))
    rng = np.random.default_rng(seed)
    for group, cols in features.GROUPS.items():
        for rep in range(3):
            shuffled = vx.copy()
            shuffled[:, cols] = vx[rng.permutation(len(vx))][:, cols]
            importance.append(dict(sector=fold, grupo=group, repeticion=rep,
                                   delta_log_loss=log_loss(vy, checkpoint.predict_proba(shuffled)) - reference_loss))
    stats.append(dict(sector=fold, selected_iteration=chosen, inner_train=len(ty), inner_val=len(vy),
                      validation_logloss=min(losses)))
    return HistGradientBoostingClassifier(**args, early_stopping=False, random_state=seed)


def fit():
    for folder in (RASTERS, MODELS, TABLES):
        folder.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "configuration.json").write_text(
        json.dumps(configuration(), ensure_ascii=False, indent=2), encoding="utf-8")
    dem = data.load_dem()
    dev = [(scene, *prepare(scene, dem)) for scene in C.FIT_SCENES]
    cube, valid, feats = prepare(C.TARGET_SCENE, dem)
    save_raster(RASTERS / "valid_target.tif", valid.astype("uint8"))
    for name, layer in (("incidencia", feats[..., 15]), ("pendiente", feats[..., 10]), ("ndsi", cube[..., 6])):
        save_raster(RASTERS / f"variable_{name}.tif", np.where(valid, layer, -9999).astype("float32"), -9999)
    pos, neg = labels.seeds(cube, valid, prudent=True)
    save_raster(RASTERS / "pseudoetiquetas_objetivo_ilustracion.tif",
                np.where(valid, np.where(pos, 1, np.where(neg, 0, 2)), 255).astype("uint8"), 255)

    x = np.broadcast_to(x_centres(valid.shape[1]), valid.shape)
    rows, cols = np.indices(valid.shape)
    b = C.INNER_VALIDATION["block_px"]
    inner_val = ((cols // b + 2 * (rows // b)) % C.INNER_VALIDATION["modulo"]) == 0
    inner_train = distance_transform_edt(~inner_val, sampling=C.RESOLUTION) > C.INNER_VALIDATION["gap_m"]
    scores = {name: np.zeros(valid.shape, "float32") for name in METHODS}
    audit, curves, stats, importance = [], [], [], []
    for fold, (left, right) in enumerate(zip(C.SECTOR_EDGES, C.SECTOR_EDGES[1:])):
        domain = (x < left - C.SPATIAL_GAP_M) | (x >= right + C.SPATIAL_GAP_M)
        test = valid & (x >= left) & (x < right)
        assert not np.any(domain & test)
        assert not np.any(inner_train & inner_val)
        for name, (n_cols, prudent, learner) in METHODS.items():
            start = time.monotonic()
            xx, yy, used = sample(dev, domain, C.SEED + fold, prudent, n_cols)
            if learner == "et":
                model = ExtraTreesClassifier(**C.EXTRA_TREES, n_jobs=C.N_JOBS, random_state=C.SEED + fold)
            else:
                model = _boosting(dev, domain, inner_train, inner_val, fold, n_cols, curves, stats, importance)
            model.fit(xx, yy)
            scores[name][test] = model.predict_proba(feats[test][:, :n_cols])[:, 1]
            joblib.dump(model, MODELS / f"{name}_sector_{fold}.joblib", compress=3)
            audit.extend(dict(sector=fold, metodo=name, **r) for r in used)
            print(f"{name} | sector {fold} | {len(yy)} muestras | {time.monotonic() - start:.1f} s", flush=True)
    del dev
    scores["Ensamble_ET"] = 0.5 * (scores["ET_base"] + scores["ET_prudente"])
    masks = {"NDSI": labels.ndsi_rule(cube, valid)}
    masks["NDSI_filtro"] = cleanup(masks["NDSI"], valid)
    for name, score in scores.items():
        save_raster(RASTERS / f"puntaje_{name}.tif", np.where(valid, score, -9999).astype("float32"), -9999)
        masks[name] = valid & (score >= C.THRESHOLD)
        masks[f"{name}_filtro"] = cleanup(masks[name], valid)
    for name in ("ET_base", "ET_prudente"):
        masks[f"{name}_guiado"] = cleanup(guided(scores[name], cube[..., 6], valid) >= C.THRESHOLD,
                                          valid, majority=False)
    for name, mask in masks.items():
        save_raster(RASTERS / f"mascara_{name}.tif", np.where(valid, mask, 255).astype("uint8"), 255)
    pd.DataFrame(audit).to_csv(TABLES / "muestras.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / "entrenamiento_boosting.csv", index=False)
    pd.DataFrame(stats).to_csv(TABLES / "seleccion_iteraciones.csv", index=False)
    pd.DataFrame(importance).to_csv(TABLES / "importancia_grupos.csv", index=False)
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(RASTERS.glob("mascara_*.tif")) + sorted(RASTERS.glob("puntaje_*.tif"))}
    (C.RESULTS / "predicciones_congeladas.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
