"""Matched-window ICON-D2 baseline scoring (plan: first use, no training).

Scores three systems against radar on the overlap window where the ICON
archive exists (2026-11-06 18Z onward) - the frozen test index ends
Nov 1, so this builds its own issue list from the real-time mirror:

* ``icon``  - ICON-D2 ``tot_prec`` hourly accumulation, decoded from
  .grib2.bz2 (eccodes) and reprojected from the regular 0.02 deg lat/lon
  grid (1215x746, GRIB template 0 - NOT the stereographic de1200 grid)
  onto rx, exactly like rv.py maps de1200 -> rx.
* ``rv``    - DWD operational nowcast (composite/rv tars), mean rate
  over the same clock hour x 1 h.
* ``pers``  - persistence: the issue-time radar rate held constant.

**Window convention** (docs/icon-d2-integration-plan.md, design note 3):
for issue t and lead L, the verification window is the clock hour
containing the valid time t+L: ICON gives ``step[k+1] - step[k]`` for
that hour, radar sums its 5-min frames H+5..H+60 over the same hour.
Everything is an hourly accumulation in mm (== mm/h numerically), so
thresholds 0.1/1/5 match the rest of the suite. Availability: the run
must be published before issue - full step range lands init+44..81 min,
so the latest usable init is the last 3-hourly init <= t - 81 min
(valid time is not hindsight; publication is what matters).

Uncertainty is reported per day (10-day window), never per pixel.

Usage:
    python -m n8igall_mvp.icon_baseline --out runs/icon_baseline.json
"""

from __future__ import annotations

import argparse
import bz2
import json
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .blend_leads import THRESHOLDS, TI, _add, _counts, _csi
from .grid import classic_grid
from .radolan import read_frame

ICON_DIR = Path("/data/n8igall/data/nwp/icon_d2")
RV_DIR = Path("/data/n8igall/data/raw/rv")
RT_DIR = Path("/data/n8igall/data/raw/rt")
ARCHIVE_START = datetime(2026, 11, 6, 18)
PUB_LAG_MIN = 81          # full step range observed by init + 81 min
LEADS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
N_HISTORY = 19
SPACING_MIN = 5
HOUR_FRAMES = 12          # frames H+5 .. H+60, each mm/h over 5 min

_ICON_RESAMPLER = None    # per-worker cache


def _icon_grid_params(raw: bytes) -> dict:
    import eccodes
    gid = eccodes.codes_new_from_message(raw)
    try:
        p = {k: eccodes.codes_get(gid, k)
             for k in ("Ni", "Nj", "La1", "La2", "Lo1", "Di", "Dj")}
        vals = np.asarray(eccodes.codes_get_values(gid), dtype=np.float32)
        miss = float(eccodes.codes_get(gid, "missingValue"))
    finally:
        eccodes.codes_release(gid)
    # GRIB2 template 0 stores angles in 1e-6 degrees
    p["la1"], p["la2"] = p["La1"] / 1e6, p["La2"] / 1e6
    p["lo1"] = p["Lo1"] / 1e6
    if p["lo1"] > 180:                   # 356.06 -> -3.94 for pyproj lon
        p["lo1"] -= 360.0
    p["di"], p["dj"] = p["Di"] / 1e6, p["Dj"] / 1e6
    p["miss"] = miss
    p["vals"] = vals
    return p


def decode_icon(path: Path) -> tuple[np.ndarray, dict]:
    """Decode .grib2.bz2 -> (field [Nj,Ni] with NaN outside domain, params)."""
    p = _icon_grid_params(bz2.open(path, "rb").read())
    f = p["vals"].reshape(p["Nj"], p["Ni"])
    f = np.where(f == p["miss"], np.nan, f)
    return f, p


class _IconResampler:
    """Map rx pixel centers to fractional ICON lat/lon coordinates (once)."""

    def __init__(self, params: dict):
        cg = classic_grid()
        rows, cols = np.meshgrid(np.arange(cg.nrows), np.arange(cg.ncols),
                                 indexing="ij")
        x = cg.x_ll + (cols + 0.5) * cg.cell
        y = cg.y_ll + (cg.nrows - 1 - rows + 0.5) * cg.cell
        from pyproj import CRS, Transformer
        inv = Transformer.from_crs(CRS.from_proj4(cg.proj4),
                                   CRS.from_epsg(4326), always_xy=True)
        lon, lat = inv.transform(x, y)
        # regular_ll: col = (lon - Lo1)/Di (W->E). Row direction comes
        # from La1 vs La2 (do not guess from scanningMode): if La2 > La1
        # data rows run S->N (row = (lat-La1)/Dj), else N->S.
        self.ocol = (lon - params["lo1"]) / params["di"]
        if params["la2"] >= params["la1"]:
            self.orow = (lat - params["la1"]) / params["dj"]
        else:
            self.orow = (params["la1"] - lat) / params["dj"]
        self.shape = (cg.nrows, cg.ncols)

    def __call__(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_src = ~np.isnan(field)
        out = map_coordinates(np.nan_to_num(field),
                              [self.orow, self.ocol], order=1,
                              mode="constant", cval=0.0)
        v = map_coordinates(valid_src.astype(np.float32),
                            [self.orow, self.ocol], order=0,
                            mode="constant", cval=0.0) > 0.5
        return out.astype(np.float32), v


def _init_for(t: datetime) -> datetime | None:
    """Latest 3-hourly init published (init + 81 min) before issue t."""
    cand = t - timedelta(minutes=PUB_LAG_MIN)
    init = cand.replace(minute=0, second=0, microsecond=0)
    init -= timedelta(hours=init.hour % 3)
    if init < ARCHIVE_START:
        return None
    return init


def _icon_hour(init: datetime, hour: datetime
               ) -> tuple[np.ndarray, np.ndarray] | None:
    """Hourly accumulation [hour, hour+1h] in mm from consecutive steps.

    Returns (data, valid) on rx. Falls back to earlier inits when the
    latest one's files are missing (sync lag).
    """
    for off in (0, 3, 6):                # try latest init, then -3/-6 h
        ini = init - timedelta(hours=off)
        k = int((hour - ini).total_seconds() // 3600)
        if k < 0 or k + 1 > 7:
            continue
        stem = ICON_DIR / ini.strftime("%Y%m%d%H")
        try:
            a, _ = decode_icon(stem / f"tot_prec_{k:03d}.grib2.bz2")
            b, params = decode_icon(stem / f"tot_prec_{k + 1:03d}.grib2.bz2")
        except FileNotFoundError:
            continue
        global _ICON_RESAMPLER
        if _ICON_RESAMPLER is None:
            _ICON_RESAMPLER = _IconResampler(params)
        out = _ICON_RESAMPLER(b - a)
        return out
    return None


def _rv_tar_for(t: datetime) -> Path | None:
    """Latest rv tar issued at or before t (tars are 5-minly)."""
    best: Path | None = None
    for p in RV_DIR.glob(f"composite_rv_{t:%Y%m%d}_*.tar"):
        hhmm = p.stem[-4:]
        if hhmm <= t.strftime("%H%M") and (best is None or p > best):
            best = p
    return best


def _rv_hour(tar: Path, hour: datetime
             ) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean rv rate over [hour, hour+1h] x 1 h -> mm, on rx.

    Leads are relative to the *tar's* issue time (parsed from the
    name), not the sample issue - a tar issued before t still forecasts
    the same clock hour.
    """
    from .rv import read_rv_tar
    issue = datetime.strptime(tar.stem[13:21] + tar.stem[22:26],
                              "%Y%m%d%H%M")
    lo = int((hour - issue).total_seconds() // 60)
    leads = [L for L in range(0, 121, 5) if lo < L <= lo + 60]
    if not leads:
        return None
    try:
        frames = read_rv_tar(tar, leads=leads)
    except Exception:                     # noqa: BLE001 - skip broken tar
        return None
    if len(frames) != len(leads):
        return None
    stack = np.stack([np.where(f.valid, f.data, 0.0) for f in frames])
    vmask = np.stack([f.valid for f in frames]).all(axis=0)
    return (stack.mean(axis=0).astype(np.float32), vmask)  # mm/h == mm/1h


def _yw(rt_dir: Path, t: datetime) -> Path | None:
    from .dataset import _yw_path
    p = _yw_path(rt_dir, t)
    return p if p.exists() else None


def score_sample(t: datetime) -> dict:
    """One issue: returns {'counts': ..., 'day': ..., 'skip': [...]}."""
    # radar history (19 frames ending at t) + issue frame rate
    hist = [_yw(RT_DIR, t - timedelta(minutes=SPACING_MIN * (N_HISTORY - 1 - i)))
            for i in range(N_HISTORY)]
    if any(p is None for p in hist):
        return {"skip": ["history"]}
    pers_rate = read_frame(hist[-1])
    pers = np.where(pers_rate.valid, np.clip(pers_rate.data, 0, None), 0.0)

    init = _init_for(t)
    tar = _rv_tar_for(t)
    hours = sorted({(t + timedelta(minutes=L)).replace(minute=0, second=0,
                                                        microsecond=0)
                    for L in LEADS})
    obs: dict[datetime, tuple[np.ndarray, np.ndarray]] = {}
    icon: dict[datetime, tuple[np.ndarray, np.ndarray] | None] = {}
    rv: dict[datetime, tuple[np.ndarray, np.ndarray] | None] = {}
    skip = []
    for H in hours:
        frames = [_yw(RT_DIR, H + timedelta(minutes=5 * (i + 1)))
                  for i in range(HOUR_FRAMES)]
        if any(p is None for p in frames):
            skip.append("obs")
            continue
        acc = np.zeros(pers.shape, dtype=np.float32)
        vmask = np.ones(pers.shape, dtype=bool)
        for p_ in frames:
            fr = read_frame(p_)
            acc += np.where(fr.valid, np.clip(fr.data, 0, None), 0.0) \
                * (SPACING_MIN / 60.0)
            vmask &= fr.valid
        obs[H] = (acc, vmask)
        icon[H] = _icon_hour(init, H) if init else None
        rv[H] = _rv_hour(tar, H) if tar else None
    if not obs:
        return {"skip": skip or ["obs"]}

    counts: dict = {}
    day = t.strftime("%y%m%d")
    for L in LEADS:
        H = (t + timedelta(minutes=L)).replace(minute=0, second=0,
                                                microsecond=0)
        if H not in obs:
            continue
        o, om = obs[H]
        preds = {"icon": icon.get(H), "rv": rv.get(H),
                 "pers": (pers, pers_rate.valid)}   # mm/h x 1 h == mm
        for sysname, pr in preds.items():
            if pr is None:
                continue
            pdata, pvalid = pr
            m = om & pvalid
            for ti, thr in enumerate(THRESHOLDS):
                _add(counts.setdefault((sysname, L, ti), {}),
                     _counts(pdata > thr, o > thr, m))
    return {"counts": counts, "day": day, "skip": skip}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # tot_prec exists from init 2026-11-07 18Z, steps 005-007 from
    # 2026-11-08 21Z -> fully-covered issues start Nov 9 00:30
    p.add_argument("--start", default="2026-11-09T00:30")
    p.add_argument("--end", default="2026-11-11T00:00")
    p.add_argument("--stride", type=int, default=5, help="issue stride, min")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("runs/icon_baseline.json"))
    args = p.parse_args(argv)

    t0 = datetime.fromisoformat(args.start)
    t1 = datetime.fromisoformat(args.end)
    issues = []
    t = t0
    while t <= t1:
        issues.append(t)
        t += timedelta(minutes=args.stride)
    print(f"{len(issues)} issues {args.start} -> {args.end} "
          f"stride {args.stride}min, {args.workers} workers", flush=True)

    from multiprocessing import Pool
    acc: dict = {}
    day_acc: dict[str, dict] = {}
    skips: dict[str, int] = {}
    done = 0
    tic = time.time()
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(score_sample, issues, chunksize=4):
            done += 1
            for s in res.get("skip", []):
                skips[s] = skips.get(s, 0) + 1
            for key, st in res.get("counts", {}).items():
                _add(acc.setdefault(key, {}), st)
                sysname, L, ti = key
                if ti == TI:
                    d = day_acc.setdefault(res["day"], {})
                    _add(d.setdefault((sysname, L), {}), st)
            if done % 100 == 0 or done == len(issues):
                el = time.time() - tic
                print(f"{done}/{len(issues)}  {el / done:.1f}s/sample",
                      flush=True)

    # raw counts FIRST - a bug in any later step must never cost a pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "window": [args.start, args.end], "stride_min": args.stride,
        "n_issues": len(issues), "skips": skips,
        "thresholds": list(THRESHOLDS), "leads": LEADS,
        "basis": "hourly accumulation (mm per clock hour; == mm/h)",
        "raw_acc": {f"{s}|{L}|{ti}": st
                    for (s, L, ti), st in acc.items()},
        "raw_days": {d: {f"{s}|{L}": st for (s, L), st in dst.items()}
                     for d, dst in day_acc.items()},
    }, indent=2))

    systems = ("icon", "rv", "pers")
    pooled = {s: {} for s in systems}
    for (sysname, L, ti), st in acc.items():
        pooled[sysname].setdefault(L, {})[ti] = st
    out = {
        "window": [args.start, args.end], "stride_min": args.stride,
        "n_issues": len(issues), "skips": skips,
        "thresholds": list(THRESHOLDS), "leads": LEADS,
        "basis": "hourly accumulation (mm per clock hour; == mm/h)",
        "pooled": {s: {str(L): pooled[s].get(L, {}) for L in LEADS}
                   for s in systems},
        "per_day": {d: {f"{s}_{L}": v for (s, L), v in dst.items()}
                    for d, dst in sorted(day_acc.items())},
    }
    args.out.write_text(json.dumps(out, indent=2))   # raw BEFORE summary

    # summary: pooled CSI per lead/system + per-day spread vs pers
    day_stats = {}
    for L in LEADS:
        for s in systems:
            deltas = []
            cp = _csi(pooled["pers"].get(L, {}).get(TI, {}))
            cs = _csi(pooled[s].get(L, {}).get(TI, {}))
            for d, st in day_acc.items():
                a, b = st.get((s, L)), st.get(("pers", L))
                if a and b:
                    da, db = _csi(a), _csi(b)
                    if np.isfinite(da) and np.isfinite(db):
                        deltas.append(da - db)
            deltas = np.asarray(deltas)
            day_stats[f"{s}_{L}"] = {
                "pooled_csi": cs if np.isfinite(cs) else None,
                "delta_vs_pers": (cs - cp)
                if np.isfinite(cs) and np.isfinite(cp) else None,
                "day_delta_mean": float(deltas.mean()) if len(deltas) else None,
                "day_delta_std": float(deltas.std()) if len(deltas) else None,
                "day_wins": int((deltas > 0).sum()),
                "n_days": int(len(deltas)),
            }
    out["day_stats"] = day_stats
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nskips: {skips}", flush=True)
    print("\nlead |    icon     rv      pers | icon-pers (day win)",
          flush=True)
    for L in LEADS:
        row = []
        for s in systems:
            c = _csi(pooled[s].get(L, {}).get(TI, {}))
            row.append(f"{c:.3f}" if np.isfinite(c) else "  -  ")
        st = day_stats[f"icon_{L}"]
        d = st["delta_vs_pers"]
        dw = (f"{d:+.3f} {st['day_wins']}/{st['n_days']}"
              if d is not None else "   -  ")
        print(f"{L:>4} | {row[0]:>6} {row[1]:>6} {row[2]:>6} | {dw}",
              flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())