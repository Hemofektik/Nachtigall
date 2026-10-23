"""Build static elevation channels on the RADOLAN 900x900 grid.

Source: Copernicus DEM (GLO-90, 90 m) from the public AWS mirror
``copernicus-dem-90m`` — 90 m is ~11x finer than the 1 km radar grid, which
is plenty after aggregation (GLO-30 works too, just larger downloads).

Outputs ``channels.npz`` with float32 channels [C, 900, 900]:

* 0: elevation [m], normalized (m / 2000)
* 1: slope magnitude [dimensionless], clipped to [0, 2]
* 2: sin(aspect), 3: cos(aspect) — aspect of the downslope direction

Usage:
    python -m n8igall_mvp.dem --out data/static/channels.npz
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .grid import classic_grid

S3 = "https://copernicus-dem-90m.s3.amazonaws.com"
# Note: GLO-90 tiles carry "COG_30" in their name (product generation id,
# not resolution); GLO-30 uses "COG_10".
TILE = "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"

# RADOLAN national grid bbox in lon/lat (spec corners, padded).
LON_MIN, LON_MAX = 3.0, 16.5
LAT_MIN, LAT_MAX = 46.5, 55.5


def needed_tiles() -> list[str]:
    names = []
    for lat in range(int(LAT_MIN), int(LAT_MAX) + 1):
        for lon in range(int(LON_MIN), int(LON_MAX) + 1):
            ns = "N"
            ew = "E" if lon >= 0 else "W"
            names.append(TILE.format(ns=ns, lat=lat, ew=ew, lon=abs(lon)))
    return names


def fetch_tile(name: str, cache: Path) -> Path | None:
    dest = cache / f"{name}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{S3}/{name}/{name}.tif"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        return dest
    except Exception as e:  # noqa: BLE001 - report and continue with other tiles
        print(f"  tile {name} failed: {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return None


def build(out_path: Path, cache: Path) -> None:
    grid = classic_grid()
    dst = np.full((grid.nrows, grid.ncols), np.nan, dtype=np.float32)

    tiles = needed_tiles()
    print(f"{len(tiles)} candidate tiles")
    ok = 0
    for i, name in enumerate(tiles):
        path = fetch_tile(name, cache)
        if path is None:
            continue
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=rasterio.Affine(grid.cell, 0.0,
                                              grid.x_ll + grid.cell / 2,
                                              0.0, -grid.cell,
                                              grid.y_ll + (grid.nrows - 0.5) * grid.cell),
                dst_crs=rasterio.CRS.from_proj4(grid.proj4),
                resampling=Resampling.bilinear,
                # Preserve previously reprojected tiles; the default would
                # reset the whole destination on every call.
                init_dest_nodata=False,
            )
        ok += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tiles)} tiles")

    covered = ~np.isnan(dst)
    print(f"tiles used: {ok}, grid covered: {covered.mean():.1%}")
    if covered.mean() < 0.9:
        raise SystemExit("DEM coverage too low - check tile downloads")

    # Fill remaining holes (small islands of nodata) with 0 m.
    dst = np.nan_to_num(dst, nan=0.0)

    # Slope/aspect from the elevation gradient (cell = 1000 m).
    dz_dy, dz_dx = np.gradient(dst, grid.cell)  # rows point south -> dy negative
    slope = np.hypot(dz_dx, dz_dy)
    # Aspect of the downslope (steepest descent) direction, as sin/cos.
    aspect_x, aspect_y = -dz_dx, -dz_dy  # downslope vector
    norm = np.hypot(aspect_x, aspect_y)
    safe = norm > 1e-9
    sin_a = np.where(safe, aspect_y / np.where(safe, norm, 1.0), 0.0)
    cos_a = np.where(safe, aspect_x / np.where(safe, norm, 1.0), 0.0)

    channels = np.stack([
        np.clip(dst / 2000.0, -1.0, 2.0),          # elevation (normalized)
        np.clip(slope, 0.0, 2.0),                   # slope magnitude
        sin_a.astype(np.float32),
        cos_a.astype(np.float32),
    ]).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, channels=channels, elevation=dst)
    print(f"wrote {out_path}: {channels.shape} "
          f"elev range [{dst.min():.0f}, {dst.max():.0f}] m")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "data" / "static" / "channels.npz")
    p.add_argument("--cache", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "data" / "dem_tiles")
    args = p.parse_args(argv)
    build(args.out, args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())