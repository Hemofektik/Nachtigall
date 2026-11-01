"""Readers for DWD RADOLAN composite files.

Supported formats:

* Classic binary (RADKLIM ``bin`` products, ``raa01-xx_10000-...-bin[.gz]``):
  ASCII header terminated by ``0x03``, then 16-bit little-endian values.
  Data bits = value & 0x0FFF; bit 13 = secondary dataset, bit 14 = error
  (2500), bit 15 = sign, bit 16 = clutter (2490) per the DWD spec.
  Payload rows are stored **south-first** and are flipped to north-up on
  read (ODIM HDF5 is already north-up); classic YW values are 0.01 mm/5min
  accumulations and are converted to mm/h so both readers agree.
* ODIM HDF5 (current opendata products, ``...-bin.hdf5``): ``dataset1/data1``
  uint16 array with ``gain``/``offset``/``nodata`` attributes.

Both readers return physical values (mm/h for RATE/YW, mm for ACRR/RW)
as float32 plus a validity mask (False = nodata/outside coverage).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pyproj import CRS, Transformer

from .grid import RadolanGrid, classic_grid, grid_from_odim_where

_DATA_BITS = 0x0FFF
_SECONDARY_BIT = 1 << 12
_ERROR_BIT = 1 << 13
_SIGN_BIT = 1 << 14
_CLUTTER_BIT = 1 << 15

_FILENAME_TS = re.compile(r"-(\d{10})-dwd")

# RADOLAN grid definitions as (nrows, ncols) -> (j_0, i_0) km offsets from
# the projection anchor (wradlib's griddefs table). The national composite
# is rx; RADKLIM reproc YW ships on the extended normal_wx grid, which is
# offset +80 km east / -100 km south of rx and must be cropped to it.
_GRIDDEFS = {
    (900, 900): (450, 450),    # rx (national composite)
    (1100, 900): (370, 550),   # normal_wx (RADKLIM reproc YW)
}


@dataclass
class RadolanFrame:
    """One decoded composite frame."""

    data: np.ndarray  # float32 physical values, NaN where invalid
    valid: np.ndarray  # bool mask of usable pixels
    grid: RadolanGrid
    timestamp: str  # YYYYMMDDHHMM (UTC)
    product: str  # e.g. "YW", "RW"
    quantity: str  # e.g. "RATE", "ACRR"


def _timestamp_from_name(path: Path) -> str:
    m = _FILENAME_TS.search(path.name)
    if not m:
        raise ValueError(f"no RADOLAN timestamp in filename: {path.name}")
    return m.group(1)


def read_bin(path: Path) -> RadolanFrame:
    """Read a classic RADOLAN binary file (optionally gzipped).

    The ``GP`` header token gives the payload grid as ``nrow x ncol``
    (wradlib convention). RADKLIM reproc YW ships on the extended 1100x900
    ``normal_wx`` grid; it is cropped to the national 900x900 ``rx`` window
    via the grid origins (national row r <- payload row r+100, national
    col n <- payload col n-80; the 80 westernmost national columns lie
    outside the extended grid and stay invalid).

    Payload rows are stored south-first; the frame is flipped to north-up
    (row 0 = north) so it matches the ODIM HDF5 reader, ``grid.py`` and the
    static/wind/cloud channels. Classic YW values are 0.01 mm/5min
    accumulations and are scaled x12 to mm/h to match HDF5 RATE.
    """
    raw = path.read_bytes() if path.suffix != ".gz" else gzip.decompress(path.read_bytes())
    # ASCII header is terminated by 0x03 (ETX); a duplicate 0x03 may follow.
    end = raw.find(b"\x03")
    if end < 0:
        raise ValueError(f"no ASCII header in {path}")
    header = raw[:end].decode("ascii", errors="replace")

    m = re.search(r"GP\s*(\d+)x\s*(\d+)", header)
    if not m:
        raise ValueError(f"{path.name}: no GP grid token in header")
    nrow, ncol = int(m.group(1)), int(m.group(2))

    payload_words = (len(raw) - end - 1) // 2
    if payload_words != nrow * ncol:
        raise ValueError(f"{path.name}: payload {payload_words} != "
                         f"{nrow}x{ncol}={nrow * ncol}")

    grid = classic_grid()
    flat = np.frombuffer(raw[-payload_words * 2:], dtype="<u2")
    values = flat.reshape(nrow, ncol)

    clutter = (values & _CLUTTER_BIT) != 0
    error = (values & _ERROR_BIT) != 0
    secondary = (values & _SECONDARY_BIT) != 0
    data_bits = values & _DATA_BITS
    # 0 = no echo (valid); values are unsigned, sign bit only used for
    # derived products (e.g. difference fields) we do not consume here.
    valid_src = ~(clutter | error | secondary)

    if (nrow, ncol) == (grid.nrows, grid.ncols):
        data_bits, valid = data_bits, valid_src
    else:
        try:
            j_p, i_p = _GRIDDEFS[(nrow, ncol)]
            j_n, i_n = _GRIDDEFS[(grid.nrows, grid.ncols)]
        except KeyError as e:
            raise ValueError(f"{path.name}: unsupported RADOLAN grid "
                             f"{nrow}x{ncol}") from e
        # Payload (south-first) offsets of the national SW corner.
        dr, dc = i_p - i_n, j_n - j_p
        if dr < 0 or dc < 0 or dr + grid.nrows > nrow or grid.ncols - dc > ncol:
            raise ValueError(f"{path.name}: national window not inside "
                             f"{nrow}x{ncol} payload")
        cropped = np.zeros((grid.nrows, grid.ncols), dtype=data_bits.dtype)
        valid = np.zeros((grid.nrows, grid.ncols), dtype=bool)
        cropped[:, dc:] = data_bits[dr:dr + grid.nrows, :ncol - dc]
        valid[:, dc:] = valid_src[dr:dr + grid.nrows, :ncol - dc]
        data_bits = cropped

    # Payload row 0 = south; flip to north-up (row 0 = north).
    data_bits = data_bits[::-1]
    valid = valid[::-1]

    out = np.where(valid, data_bits, 0).astype(np.float32)

    product = header[:2]
    # INT token gives the intensity scaling; RADKLIM YW/RW use 0.01/0.1 mm.
    gain = 0.01 if product == "YW" else 0.1
    m = re.search(r"PR\s*([A-Z])-?(\d+)", header)
    if m:  # e.g. "PR E-02" -> 1e-2
        try:
            gain = float(f"1{m.group(1)}-{m.group(2)}".replace("E-", "e-"))
        except ValueError:
            pass
    out *= gain
    if product == "YW":
        out *= 12.0  # classic YW is mm/5min; HDF5 RATE is mm/h

    return RadolanFrame(data=out, valid=valid, grid=grid,
                        timestamp=_timestamp_from_name(path), product=product,
                        quantity="RATE" if product == "YW" else "ACRR")


def read_hdf5(path: Path) -> RadolanFrame:
    """Read an ODIM HDF5 RADOLAN composite.

    Realtime files (RADKLIM HDF5 and the live opendata mirror) ship on the
    1200x1100 ``de1200`` grid (WGS84 ellipsoid); the classic products are on
    the 900x900 national ``rx`` grid (sphere R=6370.04 km). The two
    projections differ, so de1200 frames are reprojected to rx through
    lon/lat using :mod:`n8igall_mvp.grid` (validated: validity-mask
    IoU 0.9995 against the classic twin of the same frame). Frames already
    on rx are returned unchanged.
    """
    with h5py.File(path, "r") as f:
        grid = grid_from_odim_where(f["where"].attrs)
        ds = f["dataset1/data1"]
        raw = ds["data"][()]
        what = ds["what"].attrs

        def _f(key: str) -> float:
            v = what[key]
            return float(v.decode() if isinstance(v, bytes) else v)

        gain, offset = _f("gain"), _f("offset")
        nodata = _f("nodata")
        valid = raw != nodata
        out = np.where(valid, raw.astype(np.float32) * gain + offset, np.nan)
        product = str(f["dataset1/what"].attrs["prodname"].decode()
                      if isinstance(f["dataset1/what"].attrs["prodname"], bytes)
                      else f["dataset1/what"].attrs["prodname"])
        quantity = str(what["quantity"].decode()
                       if isinstance(what["quantity"], bytes) else what["quantity"])

    if (grid.nrows, grid.ncols) != (900, 900):
        out, valid = _reproject_to_rx(out, valid, grid)

    return RadolanFrame(data=out.astype(np.float32), valid=valid, grid=grid,
                        timestamp=_timestamp_from_name(path), product=product,
                        quantity=quantity)


_RX_RESAMPLE: dict | None = None


def _reproject_to_rx(data: np.ndarray, valid: np.ndarray,
                     grid) -> tuple[np.ndarray, np.ndarray]:
    """Resample a de1200 frame onto the 900x900 rx grid (cached weights)."""
    global _RX_RESAMPLE
    from scipy.ndimage import map_coordinates
    from .grid import classic_grid

    if _RX_RESAMPLE is None:
        cg = classic_grid()
        rows, cols = np.meshgrid(np.arange(cg.nrows), np.arange(cg.ncols),
                                 indexing="ij")
        x = cg.x_ll + (cols + 0.5) * cg.cell
        y = cg.y_ll + (cg.nrows - 1 - rows + 0.5) * cg.cell
        inv = Transformer.from_crs(CRS.from_proj4(cg.proj4), CRS.from_epsg(4326),
                                   always_xy=True)
        lon, lat = inv.transform(x, y)
        # lonlat_to_pixels returns (col, row)
        _RX_RESAMPLE = grid.lonlat_to_pixels(lon, lat)
    ocol, orow = _RX_RESAMPLE

    out = map_coordinates(np.nan_to_num(data, nan=0.0),
                          [orow, ocol], order=1, mode="nearest")
    v = map_coordinates(valid.astype(np.float32),
                        [orow, ocol], order=0, mode="nearest") > 0.5
    return out.astype(np.float32), v


def read_frame(path: Path | str) -> RadolanFrame:
    """Read any supported RADOLAN file, dispatching on extension."""
    path = Path(path)
    if path.name.endswith(".hdf5"):
        return read_hdf5(path)
    return read_bin(path)