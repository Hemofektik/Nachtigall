"""Hourly radar data sync daemon.

Checks DWD opendata for new RADOLAN YW (5-min) and RW (hourly) frames for
the trailing ``--window-days`` days, downloads whatever is missing, and
mirrors the real-time YW/RW HDF5, the operational nowcast (rv), and
ICON-D2 forecast fields.

3D volume-scan collection was removed 2026-11-08 (4.2 GB/day for data
nothing consumes yet; revisit once the 2D pipeline hits its limits).

Designed to run under systemd (see ``systemd/n8igall-radar-sync.service``):

    python3 -m n8igall_mvp.radar_sync

Usage (single cycle):
    python -m n8igall_mvp.radar_sync --once
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .download_radolan import download_rw, download_yw
from .icon_d2_sync import sync as sync_icon_d2

_PKG_ROOT = Path(__file__).resolve().parent.parent
# Data lives on the dedicated /data volume (477 GB ext4); mvp/data is a
# symlink to /data/n8igall/data.
DEFAULT_DEST = Path("/data/n8igall/data/raw")
DEFAULT_NWP = Path("/data/n8igall/data/nwp/icon_d2")
WINDOW_DAYS = 14  # recent YW daily tars cover ~2 weeks; RW flat files 2025+
REALTIME = "https://opendata.dwd.de/weather/radar/radolan"
# DWD's operational nowcast (RADVOR-derived): one tar per issue time with
# 25 HDF5 forecasts at leads 0..120 min in 5-min steps. Realtime only (no
# archive), so it must be grabbed while on the rolling window. This is the
# honest baseline to compare against — not persistence.
RV_URL = "https://opendata.dwd.de/weather/radar/composite/rv"
RV_WINDOW_HOURS = 40
# The CDC archive lags ~2 days for YW; the real-time HDF5 directory keeps a
# rolling ~48 h window, so the daemon mirrors it into data/raw/rt/.
RT_WINDOW_HOURS = 40


def _rt_index(url: str) -> list[str]:
    import re
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return []
    return sorted(set(re.findall(r'raa01-[a-z]{2}_10000-\d{10}-dwd---bin\.hdf5',
                                 html)))


def sync_realtime(dest: Path, window_hours: int = RT_WINDOW_HOURS) -> int:
    """Mirror recent real-time YW/RW HDF5 frames into data/raw/rt/."""
    rt_dir = dest / "rt"
    rt_dir.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) \
        - timedelta(hours=window_hours)
    stored = 0
    for product in ("yw", "rw"):
        for name in _rt_index(f"{REALTIME}/{product}/"):
            stamp = name.split("-")[2]
            try:
                t = datetime.strptime(stamp, "%y%m%d%H%M")
            except ValueError:
                continue
            if t < cutoff:
                continue
            target = rt_dir / name
            if target.exists():
                continue
            url = f"{REALTIME}/{product}/{name}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp, \
                        open(target, "wb") as out:
                    out.write(resp.read())
                stored += 1
            except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                target.unlink(missing_ok=True)
    return stored


def sync_rv(dest: Path, window_hours: int = RV_WINDOW_HOURS) -> int:
    """Archive DWD operational nowcast tars (composite/rv) into data/rv/.

    Each ``composite_rv_YYYYMMDD_HHMM.tar`` holds 25 HDF5 forecasts
    (leads 0..120 min, 5-min steps) on the 1200x1100 stereographic grid.

    Only exact-hour issue times are kept (24/day instead of 288; user
    directive 2026-11-08): the baseline is scored on hourly cadence and
    the 5-min stream alone would cost 141 GB/year. The rolling window
    means a skipped issue time cannot be refetched, so the filter must
    run before the download, not on existing files.
    """
    import re
    rv_dir = dest / "rv"
    rv_dir.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) \
        - timedelta(hours=window_hours)
    try:
        with urllib.request.urlopen(f"{RV_URL}/", timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return 0
    stored = 0
    for name in sorted(set(re.findall(
            r"composite_rv_\d{8}_\d{4}\.tar", html))):
        try:
            t = datetime.strptime(name[13:21] + name[22:26], "%Y%m%d%H%M")
        except ValueError:
            continue
        if t < cutoff or t.minute != 0:
            continue
        target = rv_dir / name
        if target.exists():
            continue
        try:
            with urllib.request.urlopen(f"{RV_URL}/{name}", timeout=120) \
                    as resp, open(target, "wb") as out:
                out.write(resp.read())
            stored += 1
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            target.unlink(missing_ok=True)
    return stored


def _count(day: date, directory: Path, pattern: str) -> int:
    """Count a day's files in the per-day subdir and the legacy flat dir."""
    stamp = day.strftime("%y%m%d")
    n = len(list((directory / stamp).glob(pattern.format(day=stamp)))) \
        if (directory / stamp).is_dir() else 0
    n += len(list(directory.glob(pattern.format(day=stamp))))
    return n


def sync(dest: Path, window_days: int) -> tuple[int, int]:
    """One sync cycle. Returns (yw_files, rw_files) downloaded.

    Days whose files are already complete are skipped with a cheap glob, so
    an hourly cycle only transfers data for today (and any gap left by an
    outage).
    """
    yw_dir = dest / "yw"
    rw_dir = dest / "rw"
    yw_dir.mkdir(parents=True, exist_ok=True)
    rw_dir.mkdir(parents=True, exist_ok=True)

    minutes = {f"{m:02d}" for m in range(0, 60, 5)}
    today = datetime.now(timezone.utc).date()
    yw_total = rw_total = 0
    window_months: set[str] = set()
    for offset in range(window_days):
        day = today - timedelta(days=offset)
        window_months.add(f"{day:%Y%m}")
        # 288 five-minute frames per day; today is always incomplete.
        if offset > 0 and _count(day, yw_dir, "raa01-yw_10000-{day}*") >= 288:
            pass  # YW complete for this day
        else:
            yw_total += download_yw(day, yw_dir, minutes)
        # 24 hourly RW files per day.
        if offset > 0 and _count(day, rw_dir, "raa01-rw_10000-{day}*") >= 24:
            pass  # RW complete for this day
        else:
            rw_total += download_rw(day, rw_dir)

    # Drop transient monthly reproc tars for the window's months (kept on
    # disk while sibling days extract within one cycle). Scoped to the
    # window so concurrent archive jobs on other months are untouched.
    for month in window_months:
        for stale in yw_dir.glob(f"YW2017.002_{month}.tar*"):
            stale.unlink(missing_ok=True)
        for stale in yw_dir.glob(f"YW-{month[2:]}.tar*"):
            stale.unlink(missing_ok=True)
    return yw_total, rw_total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--nwp", type=Path, default=DEFAULT_NWP,
                   help="ICON-D2 wind archive destination")
    p.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    p.add_argument("--once", action="store_true", help="single cycle then exit")
    args = p.parse_args(argv)

    while True:
        t0 = time.time()
        try:
            yw, rw = sync(args.dest, args.window_days)
            rt = sync_realtime(args.dest)
            rv = sync_rv(args.dest)
            icon = sync_icon_d2(args.nwp)
            print(f"{datetime.now():%Y-%m-%d %H:%M} "
                  f"+{yw} YW, +{rw} RW, +{rt} real-time, +{rv} nowcast, "
                  f"+{icon} icon-d2 ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 - a daemon must not die
            print(f"{datetime.now():%Y-%m-%d %H:%M} ERROR: {e}",
                  file=sys.stderr, flush=True)
        if args.once:
            return 0
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())