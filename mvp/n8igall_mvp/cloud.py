"""ERA5 cloud-layer channels (low/mid/high cloud fraction).

Vertical context for the nowcaster: cloud depth and high-cloud fraction
are coarse proxies for convective depth (true echo-top heights need the
prospective 3D volume collection, which has no historical archive).

Same coarse-grid + monthly-chunk design as wind.py (open-meteo archive
API, resumable, hourly-quota aware).

Channels produced (float32 [3, H, W], fractions 0..1):
    0: low cloud cover    (base ~0-2 km)
    1: mid cloud cover    (~2-7 km)
    2: high cloud cover   (~7-12 km; deep convection / anvil proxy)

Usage:
    python -m n8igall_mvp.cloud --start 2024-06-01 --end 2025-05-31
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

from .wind import BATCH_POINTS, _to_uv, coarse_grid  # reuse grid + helpers

API = "https://archive-api.open-meteo.com/v1/archive"
VARS = "cloud_cover_low,cloud_cover_mid,cloud_cover_high"
N_CHANNELS = 3


def _fetch_batch(lats: np.ndarray, lons: np.ndarray, start: date,
                 end: date) -> dict:
    qs = urllib.parse.urlencode({
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": VARS, "timezone": "UTC",
    })
    url = f"{API}?{qs}"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # hourly/daily quota - wait it out
                time.sleep(300)
                continue
            if attempt == 9:
                raise
            time.sleep(2 ** attempt)
        except Exception:  # noqa: BLE001 - transient network errors
            if attempt == 9:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("rate limited by open-meteo after 10 attempts")


def fetch(start: date, end: date, out: Path) -> None:
    """Fetch hourly cloud fractions for the coarse grid; monthly chunks."""
    lats, lons = coarse_grid()
    n_lat, n_lon = len(lats), len(lons)
    flat_lats = np.repeat(lats, n_lon)
    flat_lons = np.tile(lons, n_lat)
    n_pts = flat_lats.size
    print(f"cloud coarse grid {n_lat}x{n_lon} = {n_pts} points, "
          f"{start} -> {end}")

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
                # percent -> fraction; None -> NaN (land/ocean gaps)
                vals = [np.array([v if v is not None else np.nan
                                  for v in h[k]], dtype=np.float32) / 100.0
                        for k in ("cloud_cover_low", "cloud_cover_mid",
                                  "cloud_cover_high")]
                chunks.append(np.stack(vals, axis=-1))
            time.sleep(2.0)  # be polite to the free API
        arr = np.stack(chunks, axis=1)  # [T, n_pts, 3]
        assert times is not None
        np.savez_compressed(
            chunk_path,
            times=np.array(times, dtype="datetime64[ns]"),
            lat=lats, lon=lons, data=arr)
        print(f"  {day:%Y-%m}: {arr.shape[0]} hours -> {chunk_path.name}")
        day = month_end + timedelta(days=1)

    # merge chunks chronologically
    chunk_paths = sorted(out.parent.glob(f"{out.stem}_*.npz"))
    times_all, data_all = [], []
    for cp in chunk_paths:
        z = np.load(cp)
        times_all.append(z["times"])
        data_all.append(z["data"])
    merged_t = np.concatenate(times_all)
    merged_d = np.concatenate(data_all, axis=0)
    _, idx = np.unique(merged_t, return_index=True)
    idx = np.sort(idx)
    np.savez_compressed(out, times=merged_t[idx], lat=lats, lon=lons,
                        data=merged_d[idx])
    print(f"wrote {out}: {merged_d[idx].shape} "
          f"({out.stat().st_size/1e6:.1f} MB)")


class CloudSampler:
    """Samples the coarse cloud store onto the RADOLAN grid (bilinear)."""

    def __init__(self, store_path: Path):
        z = np.load(store_path)
        self.times = z["times"]
        self.lat = z["lat"]
        self.lon = z["lon"]
        self.data = z["data"]  # [T, n_pts, 3]
        self._build_weights()

    def _build_weights(self) -> None:
        from .grid import classic_grid
        g = classic_grid()
        lon, lat = g.pixel_centers_lonlat()
        n_lat, n_lon = self.lat.size, self.lon.size
        fy = (lat - self.lat[0]) / (self.lat[1] - self.lat[0])
        fx = (lon - self.lon[0]) / (self.lon[1] - self.lon[0])
        y0 = np.clip(np.floor(fy).astype(np.int32), 0, n_lat - 2)
        x0 = np.clip(np.floor(fx).astype(np.int32), 0, n_lon - 2)
        wy = np.clip(fy - y0, 0.0, 1.0).astype(np.float32)
        wx = np.clip(fx - x0, 0.0, 1.0).astype(np.float32)
        self._idx = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
        self._w = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]

    def channels_for(self, t: datetime) -> np.ndarray:
        """[3, H, W] cloud fractions at hour(t), NaN gaps filled with 0."""
        t0 = np.datetime64(t.replace(minute=0, second=0, microsecond=0))
        i = int(np.searchsorted(self.times, t0))
        i = min(max(i, 0), len(self.times) - 1)
        n_lat, n_lon = self.lat.size, self.lon.size
        grid = self.data[i].reshape(n_lat, n_lon, 3)
        out = np.zeros((3,) + self._w[0].shape, dtype=np.float32)
        for (ys, xs), w in zip(self._idx, self._w):
            patch = grid[ys, xs].transpose(2, 0, 1)  # [3,H,W]
            out += np.nan_to_num(patch) * w[None]
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "data" / "wind" / "era5_cloud.npz")
    args = p.parse_args(argv)
    fetch(args.start, args.end, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())