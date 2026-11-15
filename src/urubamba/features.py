"""Per-pixel predictors.

``base`` (13 variables): six reflectances, NDSI, NDVI, NDWI, elevation, slope and the
3 x 3 mean and standard deviation of NDSI.

``full`` (20 variables) adds slope aspect (sine, cosine), local solar incidence,
3 x 3 DEM standard deviation (a roughness proxy, not Riley's TRI), NDGI and the
7 x 7 mean and standard deviation of NDSI. Incidence uses the sun position of each
acquisition; it describes local illumination, not cast shadows, and no topographic
correction is applied to reflectance.
"""
import numpy as np
from scipy.ndimage import uniform_filter

from .config import RESOLUTION
from .grid import index

BASE = ["B02", "B03", "B04", "B08", "B11", "B12", "NDSI", "NDVI", "NDWI",
        "Elevación", "Pendiente", "NDSI media 3", "NDSI desv. 3"]
FULL = BASE + ["Aspecto seno", "Aspecto coseno", "Incidencia solar", "Rugosidad 3",
               "NDGI", "NDSI media 7", "NDSI desv. 7"]
GROUPS = {"Espectro e índices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 17],
          "Relieve e iluminación": [9, 10, 13, 14, 15, 16],
          "Contexto espacial": [11, 12, 18, 19]}


def _local_stats(values, valid, size):
    """Mean and standard deviation over valid neighbours only."""
    weight = np.maximum(uniform_filter(valid.astype("float32"), size, mode="constant"), 1e-6)
    mean = uniform_filter(np.where(valid, values, 0), size, mode="constant") / weight
    second = uniform_filter(np.where(valid, values ** 2, 0), size, mode="constant") / weight
    return mean, np.sqrt(np.maximum(0, second - mean ** 2))


def slope_degrees(dem):
    dy, dx = np.gradient(dem, RESOLUTION)
    return np.degrees(np.arctan(np.hypot(dy, dx)))


def illumination(dem, sun_azimuth, sun_elevation):
    """Aspect (sine, cosine) and cosine of the local solar incidence angle.

    Array rows grow southwards while northing grows northwards, so the row gradient
    changes sign before building the surface normal.
    """
    dz_south, dz_east = np.gradient(dem, RESOLUTION)
    dz_north = -dz_south
    magnitude = np.hypot(dz_east, dz_north)
    aspect_sin = np.divide(-dz_east, magnitude, out=np.zeros_like(dem), where=magnitude > 1e-6)
    aspect_cos = np.divide(-dz_north, magnitude, out=np.zeros_like(dem), where=magnitude > 1e-6)
    az, el = np.radians([sun_azimuth, sun_elevation])
    incidence = ((-dz_east * np.sin(az) * np.cos(el) - dz_north * np.cos(az) * np.cos(el)
                  + np.sin(el)) / np.sqrt(1 + magnitude ** 2))
    return aspect_sin, aspect_cos, np.clip(incidence, -1, 1)


def base(cube, valid, dem):
    mean3, std3 = _local_stats(cube[..., 6], valid, 3)
    stack = np.dstack([cube, index(cube[..., 3], cube[..., 2]), index(cube[..., 1], cube[..., 3]),
                       dem, slope_degrees(dem), mean3, std3])
    return np.nan_to_num(stack).astype("float32")


def full(cube, valid, dem, sun):
    aspect_sin, aspect_cos, incidence = illumination(dem, *sun)
    rough = np.sqrt(np.maximum(0, uniform_filter(dem ** 2, 3) - uniform_filter(dem, 3) ** 2))
    mean7, std7 = _local_stats(cube[..., 6], valid, 7)
    extra = np.dstack([aspect_sin, aspect_cos, incidence, rough,
                       index(cube[..., 1], cube[..., 2]), mean7, std7])
    return np.concatenate([base(cube, valid, dem), np.nan_to_num(extra).astype("float32")], axis=-1)
