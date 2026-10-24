"""Prospective collector for 3D radar volume scans.

DWD serves raw volume scans only on a rolling window (~4 days for ``dx``,
~40 days for the ODIM HDF5 ``sweep_vol_z`` product) with no historical
archive, so 3D training data must be accumulated from now on.

This collector polls every 5 minutes and stores the 10 quality-controlled
reflectivity sweeps (``filter_simple/dbzh``) of the latest completed volume
for every German radar site:

    data/volumes/<site>/<YYYYMMDDHHMM>/sweep_<NN>.hd5

Older-than-retention data is pruned each cycle. Run it persistently:

    nohup python3 -m n8igall_mvp.collect_volumes --retention-days 45 \
        > data/volumes/collector.log 2>&1 &

Usage (single cycle, for testing):
    python -m n8igall_mvp.collect_volumes --once
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://opendata.dwd.de/weather/radar/sites/sweep_vol_z"
SITES = ["asb", "boo", "drs", "eis", "ess", "fbg", "fld", "hnr", "isn",
         "mem", "neu", "nhb", "oft", "pro", "ros", "tur", "umd"]
SWEEPS = range(10)
# A volume is complete ~2 min after its nominal start time.
COMPLETE_LAG_MIN = 3
POLL_SECONDS = 300
FILE_RE = re.compile(r"ras07-stqual-vol5minng01_sweeph5onem_dbzh_(\d{2})-"
                     r"(\d{16})-")


def _get(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, \
                open(dest, "wb") as out:
            out.write(resp.read())
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        dest.unlink(missing_ok=True)
        return False


def list_volumes(site: str) -> dict[str, list[str]]:
    """Map timestamp -> available sweep indices for one site."""
    url = f"{BASE}/{site}/hdf5/filter_simple/"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return {}
    out: dict[str, list[str]] = {}
    for sweep, ts in FILE_RE.findall(html):
        out.setdefault(ts, []).append(sweep)
    return out


def collect_once(root: Path, retention_days: int,
                 bootstrap_hours: int = 6) -> int:
    """One polling cycle; returns number of files stored.

    Only volumes newer than the newest already-stored volume per site are
    downloaded, so an hourly cycle transfers just the new data. On the first
    cycle (empty store) only the trailing ``bootstrap_hours`` are fetched
    instead of backfilling the whole ~40-day server window.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) \
        - timedelta(minutes=COMPLETE_LAG_MIN)
    stored = 0
    for site in SITES:
        site_dir = root / site
        newest = None
        if site_dir.is_dir():
            stamps = []
            for vol_dir in site_dir.iterdir():
                try:
                    stamps.append(datetime.strptime(vol_dir.name, "%Y%m%d%H%M%S%f"))
                except ValueError:
                    pass
            newest = max(stamps) if stamps else None
        if newest is None:
            newest = cutoff - timedelta(hours=bootstrap_hours)

        volumes = list_volumes(site)
        for ts, sweeps in volumes.items():
            try:
                t = datetime.strptime(ts, "%Y%m%d%H%M%S%f")
            except ValueError:
                continue
            if t > cutoff or t <= newest:
                continue  # incomplete or already collected
            dest = site_dir / ts
            for sweep in sorted(set(sweeps) & {f"{i:02d}" for i in SWEEPS}):
                target = dest / f"sweep_{sweep}.hd5"
                if target.exists():
                    continue
                url = (f"{BASE}/{site}/hdf5/filter_simple/"
                       f"ras07-stqual-vol5minng01_sweeph5onem_dbzh_{sweep}-"
                       f"{ts}-{site}-{_site_id(site)}-hd5")
                dest.mkdir(parents=True, exist_ok=True)
                if _get(url, target):
                    stored += 1
    return stored


def _site_id(site: str) -> str:
    # WMO station ids verified from the dx product filenames.
    ids = {"asb": "10103", "boo": "10132", "drs": "10488", "eis": "10780",
           "ess": "10410", "fbg": "10908", "fld": "10440", "hnr": "10339",
           "isn": "10873", "mem": "10950", "neu": "10557", "nhb": "10605",
           "oft": "10629", "pro": "10392", "ros": "10169", "tur": "10832",
           "umd": "10356"}
    return ids[site]


def prune(root: Path, retention_days: int) -> int:
    """Delete volume directories older than the retention window."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) \
        - timedelta(days=retention_days)
    removed = 0
    for site_dir in root.glob("*"):
        if not site_dir.is_dir():
            continue
        for vol_dir in site_dir.glob("*"):
            try:
                t = datetime.strptime(vol_dir.name, "%Y%m%d%H%M%S%f")
            except ValueError:
                continue
            if t < cutoff:
                shutil.rmtree(vol_dir, ignore_errors=True)
                removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "data" / "volumes")
    p.add_argument("--retention-days", type=int, default=45)
    p.add_argument("--once", action="store_true", help="single cycle then exit")
    args = p.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    while True:
        t0 = time.time()
        stored = collect_once(args.root, args.retention_days)
        removed = prune(args.root, args.retention_days)
        print(f"{datetime.now():%Y-%m-%d %H:%M} "
              f"+{stored} files, pruned {removed} volumes "
              f"({time.time()-t0:.0f}s)", flush=True)
        if args.once:
            return 0
        time.sleep(max(POLL_SECONDS - (time.time() - t0), 30))


if __name__ == "__main__":
    raise SystemExit(main())