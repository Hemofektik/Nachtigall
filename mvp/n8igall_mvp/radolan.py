"""Readers for DWD RADOLAN composite files.

Supported formats:

* Classic binary (RADKLIM ``bin`` products, ``raa01-xx_10000-...-bin[.gz]``):
  ASCII header terminated by ``0x03``, then 16-bit little-endian values.
  Data bits = value & 0x0FFF; bit 13 = secondary dataset, bit 14 = error
  (2500), bit 15 = sign, bit 16 = clutter (2490) per the DWD spec.
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

from .grid import RadolanGrid, classic_grid, grid_from_odim_where

_DATA_BITS = 0x0FFF
_SECONDARY_BIT = 1 << 12
_ERROR_BIT = 1 << 13
_SIGN_BIT = 1 << 14
_CLUTTER_BIT = 1 << 15

_FILENAME_TS = re.compile(r"-(\d{10})-dwd")


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
    """Read a classic RADOLAN binary file (optionally gzipped)."""
    raw = path.read_bytes() if path.suffix != ".gz" else gzip.decompress(path.read_bytes())
    # ASCII header is terminated by 0x03 (ETX); a duplicate 0x03 may follow.
    end = raw.find(b"\x03")
    if end < 0:
        raise ValueError(f"no ASCII header in {path}")
    header = raw[:end].decode("ascii", errors="replace")

    grid = classic_grid()
    expected = grid.nrows * grid.ncols
    if len(raw) < end + 1 + expected * 2:
        raise ValueError(f"{path.name}: file too small for {expected} cells")
    # The payload fills the file to EOF; taking the tail is robust against
    # header quirks (duplicate ETX, odd header length).
    payload = np.frombuffer(raw[-expected * 2:], dtype="<u2")

    values = payload.astype(np.int32).reshape(grid.nrows, grid.ncols)
    clutter = (values & _CLUTTER_BIT) != 0
    error = (values & _ERROR_BIT) != 0
    secondary = (values & _SECONDARY_BIT) != 0
    data_bits = values & _DATA_BITS

    valid = ~(clutter | error | secondary)
    # 0 = no echo (valid); values are unsigned, sign bit only used for
    # derived products (e.g. difference fields) we do not consume here.
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

    return RadolanFrame(data=out, valid=valid, grid=grid,
                        timestamp=_timestamp_from_name(path), product=product,
                        quantity="RATE" if product == "YW" else "ACRR")


def read_hdf5(path: Path) -> RadolanFrame:
    """Read an ODIM HDF5 RADOLAN composite."""
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

    return RadolanFrame(data=out.astype(np.float32), valid=valid, grid=grid,
                        timestamp=_timestamp_from_name(path), product=product,
                        quantity=quantity)


def read_frame(path: Path | str) -> RadolanFrame:
    """Read any supported RADOLAN file, dispatching on extension."""
    path = Path(path)
    if path.name.endswith(".hdf5"):
        return read_hdf5(path)
    return read_bin(path)