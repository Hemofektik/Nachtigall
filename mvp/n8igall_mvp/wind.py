"""NWP/reanalysis wind fields as model input channels.

ICON-D2 (the operational model for live inference) has no historical archive
on opendata.dwd.de (rolling ~1 day), so training wind comes from ERA5
reanalysis via the open-meteo archive API (free, no key). At inference time
the same code path can be pointed at ICON-D2 grib files later.

Wind is fetched on a coarse regular lat/lon grid (~20 km) — wind fields are
smooth, so this is plenty — and sampled onto the 1 km RADOLAN grid with
precomputed bilinear weights.

Channels produced per sample (all float32 [6, 900, 900]):

* 0/1: u/v wind at 10 m
* 2/3: u/v wind at 100 m (steering-level proxy)
* 4: orographic upslope velocity  w = -(u100 dh/dx + v100 dh/dy)
* 5: horizontal divergence  div = du/dx + dv/dy  (negative = convergence)

Usage (fetch once, then reused by the dataset):
    python -m n8igall_mvp.wind --start 2022-06-01 --end 2025-05-31
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .grid import classic_grid

API = "https://archive-api.open-meteo.com/v1/archive"
VARS = "wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m"
BATCH_POINTS = 300
STEP_DEG = 0.25  # ~20-28 km at German latitudes
N_CHANNELS = 6


def coarse_grid() -> tuple[np.ndarray, np.ndarray]:
    """Regular lat/lon grid covering the RADOLAN national composite."""
    g = classic_grid()
    lon, lat = g.pixel_centers_lonlat()
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lats = np.arange(lat_min, lat_max + STEP_DEG, STEP_DEG)
    lons = np.arange(lon_min, lon_max + STEP_DEG, STEP_DEG)
    return lats, lons


def _fetch_batch(lats: np.ndarray, lons: np.ndarray, start: date, end: date) -> dict:
    lat_s = ",".join(f"{v:.4f}" for v in lats)
    lon_s = ",".join(f"{v:.4f}" for v in lons)
    qs = urllib.parse.urlencode({
        "latitude": lat_s, "longitude": lon_s,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": VARS, "wind_speed_unit": "ms", "timezone": "UTC",
    })
    url = f"{API}?{qs}"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # hourly quota exhausted - wait it out
                time.sleep(300)
                continue
            if attempt == 9:
                raise
            time.sleep(2 ** attempt)
        except Exception:  # noqa: BLE001 - retry transient network errors
            if attempt == 9:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("rate limited by open-meteo after 10 attempts")


def _to_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological wind direction (from, degrees) -> u/v components."""
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def fetch(start: date, end: date, out: Path) -> None:
    """Fetch hourly ERA5 wind for the coarse grid; save as .npz.

    Each month is saved as its own chunk file, so an interrupted run resumes
    where it left off; chunks are merged into ``out`` at the end.
    """
    lats, lons = coarse_grid()
    n_lat, n_lon = len(lats), len(lons)
    flat_lats = np.repeat(lats, n_lon)
    flat_lons = np.tile(lons, n_lat)
    n_pts = flat_lats.size
    print(f"coarse grid {n_lat}x{n_lon} = {n_pts} points, {start} -> {end}")

    out.parent.mkdir(parents=True, exist_ok=True)
    day = start
    while day <= end:
        month_end = (day.replace(day=28) + timedelta(days=4)).replace(day=1) \
            - timedelta(days=1)
        month_end = min(month_end, end)
        chunk_path = out.parent / f"{out.stem}_{day:%Y%m}.npz"
        if chunk_path.exists():
            print(f"  {day:%Y-%m}: cached")
            day = month_end + timedelta(days=1)
            continue

        times: list[str] | None = None
        chunks = []
        for i in range(0, n_pts, BATCH_POINTS):
            idx = slice(i, i + BATCH_POINTS)
            data = _fetch_batch(flat_lats[idx], flat_lons[idx], day, month_end)
            if times is None:
                times = data[0]["hourly"]["time"]
            for loc in data:
                h = loc["hourly"]
                u10, v10 = _to_uv(np.array(h["wind_speed_10m"]),
                                  np.array(h["wind_direction_10m"]))
                u100, v100 = _to_uv(np.array(h["wind_speed_100m"]),
                                    np.array(h["wind_direction_100m"]))
                chunks.append(np.stack([u10, v10, u100, v100], axis=-1))
            time.sleep(2.0)  # be polite to the free API
        arr = np.stack(chunks, axis=1)  # [T, n_pts, 4]
        assert times is not None
        np.savez_compressed(
            chunk_path,
            times=np.array(times, dtype="datetime64[ns]"),
            lat=lats, lon=lons, data=arr.astype(np.float32),
        )
        print(f"  {day:%Y-%m}: {arr.shape[0]} hours -> {chunk_path.name}")
        day = month_end + timedelta(days=1)

    # Merge chunks in chronological order.
    chunk_paths = sorted(out.parent.glob(f"{out.stem}_*.npz"))
    times_all, data_all = [], []
    for cp in chunk_paths:
        z = np.load(cp)
        times_all.append(z["times"])
        data_all.append(z["data"])
    merged = np.concatenate(data_all, axis=0)
    np.savez_compressed(out, times=np.concatenate(times_all),
                        lat=lats, lon=lons, data=merged)
    print(f"wrote {out}: {merged.shape} ({out.stat().st_size/1e6:.1f} MB)")


class WindSampler:
    """Samples the coarse wind store onto the RADOLAN grid.

    Bilinear weights from coarse lat/lon cells to RADOLAN pixels are static,
    so they are computed once and reused for every sample.
    """

    def __init__(self, store_path: Path, elevation: np.ndarray | None = None):
        z = np.load(store_path)
        self.times = z["times"]
        self.lat = z["lat"]
        self.lon = z["lon"]
        self.data = z["data"]  # [T, n_pts, 4]
        self._build_weights()

        # Static derivatives for upslope/divergence channels.
        g = classic_grid()
        if elevation is None:
            elevation = np.zeros((g.nrows, g.ncols), dtype=np.float32)
        self.dh_dy, self.dh_dx = np.gradient(elevation.astype(np.float64), g.cell)

    def _build_weights(self) -> None:
        g = classic_grid()
        lon, lat = g.pixel_centers_lonlat()
        n_lat, n_lon = self.lat.size, self.lon.size

        # Fractional indices into the coarse grid (clamped to valid range).
        fy = (lat - self.lat[0]) / (self.lat[1] - self.lat[0])
        fx = (lon - self.lon[0]) / (self.lon[1] - self.lon[0])
        y0 = np.clip(np.floor(fy).astype(np.int32), 0, n_lat - 2)
        x0 = np.clip(np.floor(fx).astype(np.int32), 0, n_lon - 2)
        wy = np.clip(fy - y0, 0.0, 1.0).astype(np.float32)
        wx = np.clip(fx - x0, 0.0, 1.0).astype(np.float32)

        self._idx = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        self._w = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]

    def _interp(self, t_index: int) -> np.ndarray:
        """Coarse field at hour ``t_index`` interpolated to [4, H, W]."""
        n_lat, n_lon = self.lat.size, self.lon.size
        flat = self.data[t_index]  # [n_pts, 4]
        grid = flat.reshape(n_lat, n_lon, 4)
        out = np.zeros((4, self.dh_dx.shape[0], self.dh_dx.shape[1]),
                       dtype=np.float32)
        for (ys, xs), w in zip(self._idx, self._w):
            patch = grid[ys, xs].transpose(2, 0, 1)  # [4, H, W]
            out += patch * w[None]
        return out

    def channels_for(self, t: datetime, lead_min: int = 60) -> np.ndarray:
        """Wind channels averaged over [t, t+lead_min], shape [6, H, W]."""
        t0 = np.datetime64(t.replace(minute=0, second=0, microsecond=0))
        t1 = np.datetime64((t + timedelta(minutes=lead_min))
                           .replace(minute=0, second=0, microsecond=0))
        i0 = int(np.searchsorted(self.times, t0))
        i1 = int(np.searchsorted(self.times, t1))
        i0 = min(max(i0, 0), len(self.times) - 1)
        i1 = min(max(i1, 0), len(self.times) - 1)

        u10, v10, u100, v100 = self._interp(i0)
        if i1 != i0:
            u10b, v10b, u100b, v100b = self._interp(i1)
            u10, v10, u100, v100 = (u10 + u10b) / 2, (v10 + v10b) / 2, \
                (u100 + u100b) / 2, (v100 + v100b) / 2

        # Orographic upslope: w = -(u dh/dx + v dh/dy)
        upslope = -(u100 * self.dh_dx + v100 * self.dh_dy)
        # Horizontal divergence on the RADOLAN grid (1 km spacing), scaled
        # to "per 1000 m" so values stay O(1).
        g = classic_grid()
        div = np.gradient(u10, g.cell, axis=1) + np.gradient(v10, g.cell, axis=0)

        return np.stack([
            u10, v10, u100, v100,
            np.clip(upslope, -0.5, 0.5).astype(np.float32),
            np.clip(div * 1000.0, -1.0, 1.0).astype(np.float32),  # per 1000 m
        ]).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "data" / "wind" / "era5_wind.npz")
    args = p.parse_args(argv)
    fetch(args.start, args.end, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())