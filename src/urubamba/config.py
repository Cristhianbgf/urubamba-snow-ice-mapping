"""Study area, scenes, folders and every fixed parameter of the experiment.

All settings live here and are written to ``results/configuration.json``
before any prediction is scored, so they cannot be tuned after looking at
the reference outlines.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"          # rasters and fitted models (not versioned)
RESULTS = ROOT / "results"          # tables and figures (versioned)
FIGURES = RESULTS / "figures"

CRS = "EPSG:32718"
BOUNDS = (808000, 8530000, 830000, 8544500)   # left, bottom, right, top (m)
RESOLUTION = 20
CELL_AREA_KM2 = RESOLUTION * RESOLUTION / 1e6

STAC_ITEMS = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"
FIT_SCENES = ("S2A_18LZL_20200529_1_L2A", "S2A_18LZL_20200618_0_L2A",
              "S2A_18LZL_20200718_0_L2A", "S2A_18LZL_20200807_1_L2A")
TARGET_SCENE = "S2A_18LZL_20200827_1_L2A"
SCENES = FIT_SCENES + (TARGET_SCENE,)
BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08",
         "swir16": "B11", "swir22": "B12", "scl": "SCL"}
SPECTRAL = ("B02", "B03", "B04", "B08", "B11", "B12")
SCL_VALID = (2, 4, 5, 6, 11)   # dark area, vegetation, bare soil, water, snow/ice

DEM_GRANULE = "AP_27628_PLR_F6920_RT1"   # ALOS PALSAR RTC high-resolution DEM (ASF)
DEM_FILE = DATA / "dem" / f"{DEM_GRANULE}.dem.tif"

REFERENCE_DOI = "https://doi.org/10.5285/d4dedc2d-1f7f-45c8-8b04-41b8b099cef9"
REFERENCE_FILES = [f"Urubamba_2020.{ext}" for ext in ("shp", "shx", "dbf", "prj")]
REFERENCE_URL = ("https://ramadda.data.bas.ac.uk/repository/entry/get/{name}?entryid=synth%3A"
                 "d4dedc2d-1f7f-45c8-8b04-41b8b099cef9%3A{path}")

SEED = 20261114
SECTOR_EDGES = (808000, 815000, 822500, 830000)   # west / centre / east
SECTOR_NAMES = ("Oeste", "Centro", "Este")
SPATIAL_GAP_M = 500
SAMPLES_PER_CLASS_AND_DATE = 3500
INNER_VALIDATION = dict(block_px=150, modulo=5, gap_m=200, samples=1200)

POSITIVE_SEED = dict(ndsi_min=0.55, red_swir_min=2.0, nir_min=0.15)
NEGATIVE_SEED_BASE = dict(ndsi_max=0.15, nir_max=0.08)   # NDSI <= 0.15 OR NIR <= 0.08
NEGATIVE_SEED_PRUDENT = dict(ndsi_max=0.15)              # NDSI <= 0.15 only
NDSI_THRESHOLD = dict(ndsi_min=0.40, nir_min=0.11)       # single-rule baseline

EXTRA_TREES = dict(n_estimators=200, min_samples_leaf=5, max_features=0.7)
BOOSTING = dict(max_iter=240, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0)
THRESHOLD = 0.5
CLEANUP = dict(majority_window=3, min_component_px=25)
GUIDED = dict(radius_px=2, spatial_sigma_px=1.0, ndsi_sigma=0.10, blend=0.5)

BOOTSTRAP = dict(block_px=50, draws=2000)

N_JOBS = 4
