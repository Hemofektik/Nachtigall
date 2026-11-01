"""Load DWD operational nowcast (composite/rv) tars and map them to the rx grid.

Each ``composite_rv_YYYYMMDD_HHMM.tar`` archived by ``radar_sync.sync_rv``
holds 25 ODIM HDF5 files, one per lead ``000..120`` (5-min steps):

* ``lead 000`` — the observed analysis (``simulated=False``), the 5-min
  ACRR interval *ending* at the issue time (same convention as YW).
* ``lead L>0`` — the forecast interval ending at ``issue + L`` minutes
  (``simulated=True``).

Values are ODIM ``ACRR`` in mm per 5 min; ``gain``/``offset`` decode the
uint32 payload, ``nodata`` marks missing. We convert to mm/h (x12), the
same unit as our YW frames.

The files sit on the 1200x1100 ``de1200`` grid (WGS84 ellipsoid polar
stereographic); our model/eval grid is the 900x900 national ``rx`` grid
(sphere R=6370.04 km). The two projections differ, so reprojection goes
through lon/lat using :mod:`n8igall_mvp.grid` — validated by resampling a
realtime YW HDF5 (de1200) onto its classic twin (rx): validity-mask
IoU = 0.9995.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates

from .grid import classic_grid, grid_from_odim_where

LEADS = list(range(0, 125, 5))  # minutes; 0 = observed analysis


@dataclass
class RvFrame:
    data: np.ndarray        # float32 [900,900] mm/h on the rx grid
    valid: np.ndarray       # bool [900,900]
    lead_min: int           # forecast lead (0 = analysis)
    issue: datetime         # issue time of the containing tar
    simulated: bool


def _decode(ds) -> tuple[np.ndarray, np.ndarray]:
    """Decode an ODIM data1 dataset to (values, valid_mask)."""
    raw = ds["data"][()]
    what = ds["what"].attrs

    def _f(k: str) -> float:
        v = what[k]
        return float(v.decode() if isinstance(v, bytes) else v)

    valid = raw != _f("nodata")
    vals = np.where(valid, raw.astype(np.float32) * _f("gain") + _f("offset"), 0.0)
    return np.maximum(vals, 0.0), valid


class _RxResampler:
    """Map de1200 pixel centers to fractional rx pixel coordinates (once)."""

    def __init__(self, odim_grid):
        self._og = odim_grid
        self._cg = classic_grid()
        rows, cols = np.meshgrid(np.arange(self._cg.nrows),
                                 np.arange(self._cg.ncols), indexing="ij")
        x = self._cg.x_ll + (cols + 0.5) * self._cg.cell
        y = self._cg.y_ll + (self._cg.nrows - 1 - rows + 0.5) * self._cg.cell
        inv = Transformer.from_crs(CRS.from_proj4(self._cg.proj4),
                                   CRS.from_epsg(4326), always_xy=True)
        lon, lat = inv.transform(x, y)
        # lonlat_to_pixels returns (col, row)
        self.ocol, self.orow = odim_grid.lonlat_to_pixels(lon, lat)

    def __call__(self, data: np.ndarray, valid: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        out = map_coordinates(np.nan_to_num(data, nan=0.0),
                              [self.orow, self.ocol], order=1, mode="nearest")
        v = map_coordinates(valid.astype(np.float32),
                            [self.orow, self.ocol], order=0, mode="nearest") > 0.5
        return out.astype(np.float32), v


def read_rv_tar(tar_path: Path, leads: list[int] | None = None
                ) -> list[RvFrame]:
    """Read selected leads from a composite/rv tar, reprojected to rx.

    Returns frames in lead order. ACRR (mm/5min) is converted to mm/h.
    """
    leads = leads if leads is not None else LEADS
    issue = datetime.strptime(tar_path.stem[13:21] + tar_path.stem[22:26],
                              "%Y%m%d%H%M")
    frames: list[RvFrame] = []
    resampler: _RxResampler | None = None
    with tarfile.open(tar_path) as tf:
        for lead in leads:
            name = f"{tar_path.stem}_{lead:03d}-hd5"
            try:
                member = tf.getmember(name)
            except KeyError:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            with h5py.File(fh, "r") as f:
                grid = grid_from_odim_where(f["where"].attrs)
                if resampler is None:
                    resampler = _RxResampler(grid)
                vals, valid = _decode(f["dataset1/data1"])
                sim = f["how"].attrs.get("simulated", b"False")
                simulated = bool(sim) if isinstance(sim, (bytes, str)) else bool(sim)
                if isinstance(sim, bytes):
                    simulated = sim.decode().lower() == "true"
            rate = vals * 12.0  # mm/5min -> mm/h
            data, v = resampler(rate, valid)
            frames.append(RvFrame(data=data, valid=v, lead_min=lead,
                                  issue=issue, simulated=simulated))
    return frames


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tar", type=Path)
    p.add_argument("--leads", type=str, default="0,5,30,60")
    args = p.parse_args(argv)
    leads = [int(v) for v in args.leads.split(",") if v]
    for fr in read_rv_tar(args.tar, leads):
        rain = int((fr.data[fr.valid] > 1.0).sum())
        print(f"lead {fr.lead_min:>3} min  simulated={fr.simulated}  "
              f"issue={fr.issue:%Y%m%d %H%M}  "
              f"valid={fr.valid.mean():.3f}  px>1mm/h={rain}  "
              f"max={fr.data[fr.valid].max() if fr.valid.any() else 0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
