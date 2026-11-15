"""The common 20 m grid: reprojection, spectral indices and raster input/output."""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from .config import BOUNDS, CRS, RESOLUTION

NODATA = -9999.0


def grid(resolution=RESOLUTION):
    """Affine transform and (rows, cols) of the study grid."""
    left, bottom, right, top = BOUNDS
    shape = (int((top - bottom) // resolution), int((right - left) // resolution))
    return from_origin(left, top, resolution, resolution), shape


def align(path, resampling=Resampling.average, resolution=RESOLUTION):
    """Read any raster on the study grid; cells without data become NaN."""
    transform, (height, width) = grid(resolution)
    with rasterio.open(path) as src, WarpedVRT(
            src, crs=CRS, transform=transform, width=width, height=height,
            resampling=resampling, dtype="float32", nodata=NODATA) as vrt:
        return vrt.read(1, masked=True).filled(np.nan)


def index(a, b):
    """Normalised difference (a - b) / (a + b); NaN where the sum is ~0."""
    total = a + b
    return np.divide(a - b, total, out=np.full_like(total, np.nan), where=total > 1e-6)


def x_centres(width):
    """Easting of every column centre (m)."""
    return BOUNDS[0] + (np.arange(width) + 0.5) * RESOLUTION


def save_raster(path, values, nodata=None):
    transform, (height, width) = grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                       crs=CRS, transform=transform, dtype=values.dtype, nodata=nodata,
                       tiled=True, compress="deflate") as dst:
        dst.write(values, 1)


def read(path):
    with rasterio.open(path) as src:
        return src.read(1)
