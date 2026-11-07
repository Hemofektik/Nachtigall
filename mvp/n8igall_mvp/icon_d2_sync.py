"""ICON-D2 wind archive (item 6: train/serve wind parity).

The model currently trains on ERA5 *reanalysis* wind, but at inference
time only forecast wind exists. ERA5 is also a hindsight analysis - the
review flagged this train/serve mismatch as fixable only by collecting
ICON-D2 from now on.

ICON-D2 runs every 3 h (00..21); DWD opendata keeps only the *latest*
run per cycle directory (~24 h retention), so the hourly sync daemon
must grab new files promptly. We archive:

* variables: u_10m, v_10m (regular-lat-lon grid - the icosahedral
  native grid is skipped; there is no single-level 100 m wind, so the
  steering-level channel question is deferred),
* forecast steps 000..12 hourly (~13 steps; enough for the +60 min
  nowcast window with margin, ~26 MB/cycle, ~210 MB/day),
* retention: ``KEEP_DAYS`` days per file (default 7, ~1.5 GB).

Layout: ``<dest>/<YYYYMMDDHH>/<var>_<step>.grib2.bz2`` where the
directory is the cycle init time.

Usage (single cycle, used by radar_sync's hourly loop):
    python -m n8igall_mvp.icon_d2_sync
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
CYCLES = ("00", "03", "06", "09", "12", "15", "18", "21")
VARS = ("u_10m", "v_10m")
MAX_STEP = 12          # forecast hours archived per run
KEEP_DAYS = 7
DEFAULT_DEST = Path("/data/n8igall/data/nwp/icon_d2")
_NAME_RE = re.compile(
    r"icon-d2_germany_regular-lat-lon_single-level_(\d{10})_(\d{3})_2d_"
    r"([a-z0-9_]+)\.grib2\.bz2")


def _index(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return []
    return sorted(set(re.findall(r'href="([^"]+\.grib2\.bz2)"', html)))


def sync(dest: Path = DEFAULT_DEST, max_step: int = MAX_STEP,
         keep_days: int = KEEP_DAYS, now: datetime | None = None) -> int:
    """Download missing ICON-D2 u/v 10m fields; prune old init dirs."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    dest.mkdir(parents=True, exist_ok=True)
    stored = 0
    for cycle in CYCLES:
        for var in VARS:
            names = _index(f"{BASE}/{cycle}/{var}/")
            if not names:
                continue
            for name in names:
                m = _NAME_RE.match(name)
                if not m:
                    continue
                init, step_s, var_f = m.groups()
                if var_f != var:
                    continue
                if int(step_s) > max_step:
                    continue
                # only current and previous init per cycle (opendata
                # already prunes, but be explicit)
                t_init = datetime.strptime(init, "%Y%m%d%H")
                if (now - t_init) > timedelta(days=2):
                    continue
                target = dest / init / f"{var}_{int(step_s):03d}.grib2.bz2"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                url = f"{BASE}/{cycle}/{var}/{name}"
                try:
                    with urllib.request.urlopen(url, timeout=120) as resp, \
                            open(target, "wb") as out:
                        out.write(resp.read())
                    stored += 1
                except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                    target.unlink(missing_ok=True)
    # prune: drop init dirs older than keep_days
    cutoff = now - timedelta(days=keep_days)
    for d in dest.iterdir():
        if not d.is_dir():
            continue
        try:
            t = datetime.strptime(d.name, "%Y%m%d%H")
        except ValueError:
            continue
        if t < cutoff:
            for f in d.glob("*"):
                f.unlink(missing_ok=True)
            d.rmdir()
    return stored


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--max-step", type=int, default=MAX_STEP)
    p.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    args = p.parse_args(argv)
    n = sync(args.dest, args.max_step, args.keep_days)
    print(f"icon_d2: +{n} files -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
