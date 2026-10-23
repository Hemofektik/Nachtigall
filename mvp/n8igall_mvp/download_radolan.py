"""Download DWD RADOLAN data for training.

Sources (all free, no registration):

* Real-time composites (rolling ~48 h window):
  https://opendata.dwd.de/weather/radar/radolan/{yw,rw}/
* RADKLIM archive (daily tarballs, 2001 -> present):
  - YW 5-min: .../CDC/grids_germany/5_minutes/radolan/recent/YW-YYMMDD.tar.gz
    (older years under reproc/2017_002/)
  - RW hourly: .../CDC/grids_germany/hourly/radolan/{recent,historical}/bin/

Usage:
    python -m n8igall_mvp.download_radolan --product yw --start 2025-05-01 --end 2025-05-07
    python -m n8igall_mvp.download_radolan --product rw --start 2025-05-01 --end 2025-05-07
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://opendata.dwd.de"
YW_TAR = BASE + "/climate_environment/CDC/grids_germany/5_minutes/radolan/recent/YW-{yymmdd}.tar.gz"
YW_TAR_REPROC = (BASE + "/climate_environment/CDC/grids_germany/5_minutes/radolan/"
                 "reproc/2017_002/bin/{year}/YW2017.002_{year}{month:02d}.tar")
RW_BIN = BASE + "/climate_environment/CDC/grids_germany/hourly/radolan/{bucket}/bin/"


def _fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        return True
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"  miss {url}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def _extract_tar(tar_path: Path, dest_dir: Path, keep=None,
                 mode: str = "r:gz") -> int:
    """Extract RADOLAN frames; ``keep(name)`` filters members. Removes tar."""
    n = 0
    with tarfile.open(tar_path, mode) as tf:
        for member in tf.getmembers():
            if not (member.isfile() and "-dwd---bin" in member.name):
                continue
            if keep is not None and not keep(Path(member.name).name):
                continue
            target = dest_dir / Path(member.name).name
            # Reproc files carry a version token (raa01-yw2017.002_10000-...);
            # normalize to the standard product name (raa01-yw_10000-...).
            target = dest_dir / target.name.replace("yw2017.002_", "yw_")
            if not target.exists():
                src = tf.extractfile(member)
                with open(target, "wb") as out:
                    out.write(src.read())
                n += 1
    return n


def download_yw(day: date, dest: Path) -> int:
    """Download one day of YW 5-min frames (288 files).

    Tries the daily tarball in ``recent/`` first; older days fall back to the
    monthly reproc tarball (extracting only the requested day).
    """
    yymmdd = day.strftime("%y%m%d")
    tar_dest = dest / f"YW-{yymmdd}.tar.gz"
    if _fetch(YW_TAR.format(yymmdd=yymmdd), tar_dest):
        n = _extract_tar(tar_dest, dest)
        tar_dest.unlink(missing_ok=True)
        return n

    month_tar = dest / f"YW2017.002_{day:%Y%m}.tar"
    url = YW_TAR_REPROC.format(year=day.year, month=day.month)
    if _fetch(url, month_tar):
        # Monthly tar holds one nested daily tar.gz per day; keep the monthly
        # tar on disk so sibling days do not re-download it.
        nested_name = f"YW2017.002_{day:%Y%m%d}.tar.gz"
        nested = dest / nested_name
        with tarfile.open(month_tar, "r:") as tf:
            member = tf.getmember(nested_name)
            with tf.extractfile(member) as src, open(nested, "wb") as out:
                out.write(src.read())
        n = _extract_tar(nested, dest)
        nested.unlink(missing_ok=True)
        return n
    return 0


def download_rw(day: date, dest: Path) -> int:
    """Download one day of RW hourly files (24 files, gzipped)."""
    n = 0
    for hour in range(24):
        stamp = day.strftime("%y%m%d") + f"{hour:02d}00"
        name = f"raa01-rw_10000-{stamp}-dwd---bin.gz"
        for bucket in ("recent", "historical"):
            if _fetch(RW_BIN.format(bucket=bucket) + name, dest / name):
                n += 1
                break
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", choices=("yw", "rw"), required=True)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True,
                   help="inclusive end date")
    p.add_argument("--dest", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data" / "raw")
    args = p.parse_args(argv)

    dl = download_yw if args.product == "yw" else download_rw
    total = 0
    day = args.start
    while day <= args.end:
        n = dl(day, args.dest / args.product)
        total += n
        print(f"{day} {args.product}: +{n} files")
        day += timedelta(days=1)
    print(f"done: {total} new files in {args.dest / args.product}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())