"""RADOLAN grid definitions and coordinate transforms.

Two grid generations exist (see DWD "RADOLAN-RADVOR-Kompositformat" spec):

* Classic (``VS 1/2``): 900x900 national composite, 1 km, spherical earth
  (R = 6370.04 km), polar stereographic, standard parallel 60N, lon_0 = 10E.
  The LL corner sits at (-465.696, -465.696) km in that projection
  (wradlib's ``RADOLAN_GRID``); pixel centers are at half-integer offsets.
* HDF5/ODIM (``H5rad 2.3``): 1100x1200 (1200 rows x 1100 cols), WGS84
  ellipsoid, projection parameters carried in the file's ``where/projdef``
  attribute. Pixel centers are at integer (col, -row) km.

Both are handled through :class:`RadolanGrid`, which exposes a pyproj
transformer from WGS84 lon/lat to grid pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

# Classic grid: sphere-based projection with false offsets derived so that the
# spec's documented LL corner (46.9526N / 3.5889E) maps to (-465.696, -465.696) km.
_CLASSIC_SPHERE = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6370040 +b=6370040 +no_defs"
# Spec corner coordinates (DWD format description, rounded to 4 decimals).
_SPEC_LL = (3.5889, 46.9526)  # lon, lat
_SPEC_LL_KM = (-465.696, -465.696)


def _classic_proj4() -> str:
    raw = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(_CLASSIC_SPHERE),
                               always_xy=True)
    x, y = raw.transform(*_SPEC_LL)
    x0 = _SPEC_LL_KM[0] * 1000.0 - x
    y0 = _SPEC_LL_KM[1] * 1000.0 - y
    return f"{_CLASSIC_SPHERE} +x_0={x0:.3f} +y_0={y0:.3f} +units=m"


@dataclass(frozen=True)
class RadolanGrid:
    """A RADOLAN composite grid.

    Attributes:
        proj4: PROJ string of the grid projection.
        ncols: number of columns (x).
        nrows: number of rows (y, row 0 = north).
        x_ll: x coordinate of the LL corner of the grid [m].
        y_ll: y coordinate of the LL corner of the grid [m].
        cell: cell size [m].
    """

    proj4: str
    ncols: int
    nrows: int
    x_ll: float
    y_ll: float
    cell: float = 1000.0

    @property
    def transformer(self) -> Transformer:
        return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(self.proj4),
                                    always_xy=True)

    def lonlat_to_pixels(self, lon: float, lat: float) -> tuple[float, float]:
        """Return fractional (col, row) pixel coordinates for a lon/lat point."""
        x, y = self.transformer.transform(lon, lat)
        col = (x - self.x_ll) / self.cell - 0.5
        row = self.nrows - 1 - ((y - self.y_ll) / self.cell - 0.5)
        return col, row

    def pixel_centers_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Lon/lat of every pixel center. Returns (lon[nrows, ncols], lat[...])."""
        cols = np.arange(self.ncols)
        rows = np.arange(self.nrows)
        cc, rr = np.meshgrid(cols, rows)
        x = self.x_ll + (cc + 0.5) * self.cell
        y = self.y_ll + (self.nrows - 1 - rr + 0.5) * self.cell
        inv = Transformer.from_crs(CRS.from_proj4(self.proj4), CRS.from_epsg(4326),
                                   always_xy=True)
        lon, lat = inv.transform(x, y)
        return lon, lat


def classic_grid() -> RadolanGrid:
    """The 900x900 national composite grid (RADKLIM bin products)."""
    return RadolanGrid(proj4=_classic_proj4(), ncols=900, nrows=900,
                       x_ll=-465_696.0, y_ll=-465_696.0)


def grid_from_odim_where(where_attrs: dict) -> RadolanGrid:
    """Build a grid from an ODIM HDF5 ``where`` group's attributes."""

    def _f(key: str) -> float:
        v = where_attrs[key]
        return float(v.decode() if isinstance(v, bytes) else v)

    return RadolanGrid(
        proj4=str(where_attrs["projdef"].decode()
                   if isinstance(where_attrs["projdef"], bytes)
                   else where_attrs["projdef"]),
        ncols=int(_f("xsize")),
        nrows=int(_f("ysize")),
        # ODIM projdef carries x_0/y_0 such that pixel (col=0, row=0) center is
        # at (0, 0) m; grid corners therefore sit half a cell outside.
        x_ll=-_f("xscale") / 2.0,
        y_ll=-(_f("ysize") - 1) * _f("xscale") - _f("xscale") / 2.0,
        cell=_f("xscale"),
    )