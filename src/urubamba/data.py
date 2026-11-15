"""Inputs: Sentinel-2 L2A crops, the ALOS PALSAR DEM and the 2020 reference outlines.

* Sentinel-2 bands are read as cloud-optimised GeoTIFFs from the public Earth Search
  STAC catalogue and cropped to the study box. Reflectance uses the scale and offset
  published with each asset.
* The DEM is the ALOS PALSAR radiometrically terrain corrected product
  ``AP_27628_PLR_F6920_RT1`` (Alaska Satellite Facility). ASF requires a free NASA
  Earthdata login, so it is not downloaded automatically: place the ``.dem.tif`` file
  in ``data/dem/``.
* The reference is the 2020 Cordillera Urubamba outline set of Taylor et al. (2022),
  UK Polar Data Centre, Open Government Licence v3.0. It is only read in ``evaluate``.
"""
import base64
import json
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .config import (BANDS, BOUNDS, CRS, DATA, DEM_FILE, DEM_GRANULE, REFERENCE_FILES,
                     REFERENCE_URL, SCENES, SCL_VALID, SPECTRAL, STAC_ITEMS)
from .grid import align, grid, index

S2 = DATA / "sentinel2"
REFERENCE = DATA / "reference" / "Urubamba_2020.shp"


def scene_folder(scene):
    return S2 / scene


def _crop(href, resolution, categorical):
    transform, shape = grid(resolution)
    with rasterio.open(href) as src:
        win = from_bounds(*transform_bounds(CRS, src.crs, *BOUNDS), transform=src.transform)
        c0, r0 = np.floor([win.col_off, win.row_off]).astype(int) - 2
        c1, r1 = np.ceil([win.col_off + win.width, win.row_off + win.height]).astype(int) + 2
        window = Window(c0, r0, c1 - c0, r1 - r0).intersection(Window(0, 0, src.width, src.height))
        values = src.read(1, window=window)
        out = np.zeros(shape, dtype="uint8" if categorical else "float32")
        reproject(values, out, src_transform=src.window_transform(window), src_crs=src.crs,
                  src_nodata=0, dst_transform=transform, dst_crs=CRS, dst_nodata=0,
                  resampling=Resampling.nearest if categorical else Resampling.average)
    return out


def download_scene(scene):
    """Crop the six optical bands and the scene classification of one acquisition."""
    folder = scene_folder(scene)
    folder.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{STAC_ITEMS}/{scene}", timeout=60) as response:
        item = json.load(response)
    (folder / "escena_stac.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3"):
        for key, band in BANDS.items():
            path = folder / f"{band}.tif"
            if path.exists():
                continue
            asset = item["assets"][key]
            categorical = band == "SCL"
            native = 20 if categorical else asset["gsd"]
            values = _crop(asset["href"], native, categorical)
            nodata = 0 if categorical else -9999
            if not categorical:
                meta = asset["raster:bands"][0]
                values = np.where(values != 0, values * meta["scale"] + meta["offset"],
                                  nodata).astype("float32")
            transform, (height, width) = grid(native)
            with rasterio.open(path, "w", driver="GTiff", crs=CRS, transform=transform,
                               height=height, width=width, count=1, dtype=values.dtype,
                               nodata=nodata, compress="deflate", tiled=True) as dst:
                dst.write(values, 1)
            print(f"{scene} {band}", flush=True)


def download_reference():
    """Fetch the 2020 Urubamba outlines (four shapefile parts) from the UK Polar Data Centre."""
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        target = REFERENCE.parent / name
        if target.exists():
            continue
        path = base64.b64encode(f"/Glacier_Boundaries/Urubamba/{name}".encode()).decode()
        with urlopen(REFERENCE_URL.format(name=name, path=quote(path, safe="")), timeout=60) as r:
            target.write_bytes(r.read())
        print(f"reference {name}", flush=True)


def download_all():
    for scene in SCENES:
        download_scene(scene)
    download_reference()
    if not DEM_FILE.exists():
        print(f"Missing DEM: download {DEM_GRANULE} from https://search.asf.alaska.edu "
              f"(Earthdata login) and copy {DEM_FILE.name} to {DEM_FILE.parent}")


def load_scene(scene):
    """Six reflectance bands plus NDSI on the 20 m grid, and the valid-pixel mask.

    Optical bands are block-averaged to 20 m; the scene classification is taken by
    nearest neighbour so its codes are never interpolated.
    """
    folder = scene_folder(scene)
    bands = [align(folder / f"{b}.tif", Resampling.average) for b in SPECTRAL]
    ndsi = index(bands[1], bands[4])
    cube = np.stack([*bands, ndsi], axis=-1).astype("float32")
    scl = align(folder / "SCL.tif", Resampling.nearest)
    valid = np.isin(scl, SCL_VALID) & np.isfinite(cube).all(axis=-1)
    return cube, valid


def sun_geometry(scene):
    props = json.loads((scene_folder(scene) / "escena_stac.json").read_text())["properties"]
    return props["view:sun_azimuth"], props["view:sun_elevation"]


def load_dem():
    """Ellipsoidal heights of the ASF RTC DEM, averaged to the 20 m grid."""
    return align(DEM_FILE, Resampling.average)


def load_reference():
    """Reference outlines burnt on the grid (cell centre inside a polygon)."""
    import geopandas as gpd
    from shapely.geometry import box
    outlines = gpd.read_file(REFERENCE).to_crs(CRS)
    outlines = outlines[outlines.intersects(box(*BOUNDS))]
    transform, shape = grid()
    return rasterize(((g, 1) for g in outlines.geometry), out_shape=shape,
                     transform=transform, dtype="uint8").astype(bool)
