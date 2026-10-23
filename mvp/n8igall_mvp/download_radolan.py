"""Download DWD RADOLAN data for training.

Sources (all free, no registration):

* YW 5-min (RADKLIM): daily tars in ``recent/`` for the last ~2 weeks,
  monthly tars in ``reproc/2017_002/bin/YYYY/`` for everything older
  (2001 -> present).
* RW hourly (RADKLIM): flat gz files in ``recent/`` from 2025 on,
  monthly tars in ``historical/bin/YYYY/`` for 2005-2024.

Only the :00 and :30 frames are extracted by default (the dataset uses
30-min spacing), which cuts disk usage 3x.

Usage:
    python -m n8igall_mvp.download_radolan --product yw --start 2024-06-01 --end 2025-05-31
    python -m n8igall_mvp.download_radolan --product rw --start 2024-06-01 --end 2025-05-31
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = "https://opendata.dwd.de"
YW_RECENT = BASE + "/climate_environment/CDC/grids_germany/5_minutes/radolan/recent/YW-{yymmdd}.tar.gz"
YW_REPROC = (BASE + "/climate_environment/CDC/grids_germany/5_minutes/radolan/"
             "reproc/2017_002/bin/{year}/YW2017.002_{year}{month:02d}.tar")
RW_FLAT = BASE + "/climate_environment/CDC/grids_germany/hourly/radolan/recent/bin/"
RW_MONTHLY = (BASE + "/climate_environment/CDC/grids_germany/hourly/radolan/"
              "historical/bin/{year}/RW{year}{month:02d}.tar.gz")

# Recent YW daily tars cover roughly the last 2 weeks; older days go to reproc.
RECENT_WINDOW_DAYS = 14


def _fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, \
                    open(part, "wb") as out:
                while True:  # stream to disk; catches stalls mid-transfer
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if part.stat().st_size == 0:
                raise OSError("empty response")
            part.replace(dest)  # atomic: partial downloads never look complete
            return True
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                print(f"  miss {url}", file=sys.stderr)
                part.unlink(missing_ok=True)
                return False
            time.sleep(5 * (attempt + 1))
        except (OSError, TimeoutError) as e:  # stalls, resets, disk errors
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    part.unlink(missing_ok=True)
    return False


def _member_ts(name: str) -> str | None:
    """YYMMDDHHMM timestamp from a RADOLAN member filename, or None."""
    parts = name.split("-")
    if len(parts) < 3 or not parts[2].isdigit() or len(parts[2]) != 10:
        return None
    return parts[2]


def _extract(tar_path: Path, dest: Path, wanted: set[str],
             depth: int = 0) -> int:
    """Extract RADOLAN members whose YYMMDDHHMM timestamp is in ``wanted``.

    Handles nested archives (monthly tars contain daily tars) transparently.
    Reproc files carry a version token (``raa01-yw2017.002_10000-...``);
    names are normalized to the standard product name on extraction.
    """
    n = 0
    with tarfile.open(tar_path, mode="r:*") as tf:  # transparent decompression
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.endswith((".tar", ".tar.gz", ".tgz")) and depth < 2:
                src = tf.extractfile(member)
                if src is None:
                    continue
                data = src.read()
                nested = dest / f".nested_{name}"
                nested.write_bytes(data)
                n += _extract(nested, dest, wanted, depth + 1)
                nested.unlink(missing_ok=True)
                continue
            ts = _member_ts(name)
            if ts is None or ts not in wanted:
                continue
            target = dest / name.replace("yw2017.002_", "yw_")
            if target.exists():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                out.write(src.read())
            n += 1
    return n


def _wanted_times(day: date, minutes: set[str]) -> set[str]:
    yymmdd = day.strftime("%y%m%d")
    return {f"{yymmdd}{h:02d}{m}" for h in range(24) for m in minutes}


def download_yw(day: date, dest: Path, minutes: set[str]) -> int:
    """Download one day of YW frames (only the requested minute marks)."""
    wanted = _wanted_times(day, minutes)
    # Fast path: recent daily tarball (last ~2 weeks only).
    if (datetime.now().date() - day).days <= RECENT_WINDOW_DAYS:
        tar_path = dest / f"YW-{day:%y%m%d}.tar.gz"
        if _fetch(YW_RECENT.format(yymmdd=day.strftime("%y%m%d")), tar_path):
            n = _extract(tar_path, dest, wanted)
            tar_path.unlink(missing_ok=True)
            if n:
                return n

    # Slow path: monthly reproc tar (kept on disk while sibling days extract).
    month_tar = dest / f"YW2017.002_{day:%Y%m}.tar"
    if _fetch(YW_REPROC.format(year=day.year, month=day.month), month_tar):
        return _extract(month_tar, dest, wanted)
    return 0


def download_rw(day: date, dest: Path) -> int:
    """Download one day of RW hourly files (24 files)."""
    if day.year >= 2025:  # flat recent files
        n = 0
        for hour in range(24):
            name = f"raa01-rw_10000-{day:%y%m%d}{hour:02d}00-dwd---bin.gz"
            if _fetch(RW_FLAT + name, dest / name):
                n += 1
        return n

    # Monthly historical tar (plain tar despite the .tar.gz name; contains
    # nested daily tar.gz files)
    month_tar = dest / f"RW{day:%Y%m}.tar"
    if _fetch(RW_MONTHLY.format(year=day.year, month=day.month), month_tar):
        wanted = _wanted_times(day, {"00"})
        n = _extract(month_tar, dest, wanted)
        month_tar.unlink(missing_ok=True)
        return n
    return 0


def _month_days(start: date, end: date):
    """Group days by (year, month)."""
    groups: dict[tuple[int, int], list[date]] = {}
    day = start
    while day <= end:
        groups.setdefault((day.year, day.month), []).append(day)
        day += timedelta(days=1)
    return groups


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", choices=("yw", "rw"), required=True)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True,
                   help="inclusive end date")
    p.add_argument("--dest", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data" / "raw")
    p.add_argument("--minutes", default="00,30",
                   help="YW minute marks to extract (comma separated, or 'all')")
    args = p.parse_args(argv)

    dest = args.dest / args.product
    if args.minutes == "all":
        minutes = {f"{m:02d}" for m in range(0, 60, 5)}
    else:
        minutes = set(args.minutes.split(","))
    total = 0
    for (year, month), days in _month_days(args.start, args.end).items():
        month_total = 0
        for day in days:
            n = download_yw(day, dest, minutes) if args.product == "yw" \
                else download_rw(day, dest)
            month_total += n
        total += month_total
        print(f"{year}-{month:02d} {args.product}: +{month_total} files")
        # Drop monthly tars once the whole month is processed.
        for stale in dest.glob(f"*{year}{month:02d}.tar*"):
            stale.unlink(missing_ok=True)
    print(f"done: {total} new files in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())