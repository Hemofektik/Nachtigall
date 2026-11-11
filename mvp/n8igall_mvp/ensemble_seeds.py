"""Roadmap item 2 (review at 13a87fb): seed ensemble.

Average the predictions of seed 0 and seed 1 (``multi_lk_cloud`` and
``multi_lk_cloud_s1``) in rate space, then score ensemble vs each
single seed on the frozen test set with the full metric suite.

Falsification check: the ensemble must beat the better single seed by
day (per-day paired deltas at CSI@1.0, same test as blend_leads).

No fitting involved - one test pass, two forwards per batch. Raw
results are saved before the summary (summary bugs must never cost a
pass again).

Usage:
    python -m n8igall_mvp.ensemble_seeds \
        --checkpoints runs/multi_lk_cloud/best.pt \
                      runs/multi_lk_cloud_s1/best.pt \
        --out runs/multi_lk_cloud/ensemble_s0s1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .blend_leads import THRESHOLDS, TI, _add, _build_model, _counts, _csi
from .blend_leads import _make_pass, _TS_RE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs=2, required=True,
                   metavar=("S0", "S1"))
    p.add_argument("--test-index", type=Path,
                   default=Path("data/index_test_rt.json"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--static", type=Path,
                   default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path,
                   default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path,
                   default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/multi_lk_cloud/ensemble_s0s1.json"))
    args = p.parse_args(argv)

    m0, ck0, leads, device = _build_model(args.checkpoints[0])
    m1, ck1, leads1, _ = _build_model(args.checkpoints[1])
    assert leads == leads1, f"lead mismatch: {leads} vs {leads1}"
    assert ck0["in_channels"] == ck1["in_channels"]

    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if ck0["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    samples, dl, _ = _make_pass(args.test_index, leads, args, static, wind,
                                cloud)
    radar_ch = len(samples[0].inputs)
    n = len(samples)

    systems = ("ens", "s0", "s1")
    acc = {s: [[{} for _ in THRESHOLDS] for _ in leads] for s in systems}
    day_acc: dict[str, dict[str, list[dict]]] = {}
    idx = 0
    with torch.no_grad():
        for x, y, mask in dl:
            out0 = m0(x.to(device))
            out1 = m1(x.to(device))
            p0 = torch.expm1((out0[0] if isinstance(out0, tuple)
                              else out0).clamp(min=0)).cpu().numpy()
            p1 = torch.expm1((out1[0] if isinstance(out1, tuple)
                              else out1).clamp(min=0)).cpu().numpy()
            ens = 0.5 * (p0 + p1)
            yr = torch.expm1(y).numpy()
            m = mask.numpy().astype(bool)
            B = x.shape[0]
            for b in range(B):
                key = int(_TS_RE.search(samples[idx].inputs[-1]).group(1))
                idx += 1
                day = str(key)[:6]
                d = day_acc.setdefault(
                    day, {s: [{} for _ in leads] for s in systems})
                for sysname, arr in (("ens", ens[b]), ("s0", p0[b]),
                                     ("s1", p1[b])):
                    for i in range(len(leads)):
                        for ti, t in enumerate(THRESHOLDS):
                            _add(acc[sysname][i][ti],
                                 _counts((arr[i] > t) & m[b][i],
                                         (yr[b][i] > t) & m[b][i],
                                         m[b][i]))
                        _add(d[sysname][i],
                             _counts((arr[i] > THRESHOLDS[TI]) & m[b][i],
                                     (yr[b][i] > THRESHOLDS[TI]) & m[b][i],
                                     m[b][i]))
                if idx % 200 == 0 or idx == n:
                    print(f"[test] {idx}/{n}", flush=True)

    pooled = {s: [[acc[s][i][ti] for ti in range(len(THRESHOLDS))]
                  for i in range(len(leads))] for s in systems}
    out = {
        "checkpoints": [str(c) for c in args.checkpoints],
        "index": str(args.test_index), "n": n,
        "thresholds": list(THRESHOLDS), "leads": leads,
        "pooled": pooled,
        "per_day": {d: {s: [day_acc[d][s][i] for i in range(len(leads))]
                        for s in systems} for d in sorted(day_acc)},
        "day_stats": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))   # raw BEFORE summary

    # falsification: ensemble vs the better single seed, per day
    day_stats = {}
    for i, L in enumerate(leads):
        c0, c1 = _csi(acc["s0"][i][TI]), _csi(acc["s1"][i][TI])
        better = "s0" if c0 >= c1 else "s1"
        deltas = []
        for per_sys in day_acc.values():
            ce, co = _csi(per_sys["ens"][i]), _csi(per_sys[better][i])
            if np.isfinite(ce) and np.isfinite(co):
                deltas.append(ce - co)
        deltas = np.asarray(deltas)
        day_stats[str(L)] = {
            "better_single": better,
            "pooled_delta": _csi(acc["ens"][i][TI])
            - _csi(acc[better][i][TI]),
            "day_delta_mean": float(deltas.mean()) if len(deltas) else None,
            "day_delta_std": float(deltas.std()) if len(deltas) else None,
            "day_wins": int((deltas > 0).sum()),
            "n_days": int(len(deltas)),
        }
    out["day_stats"] = day_stats
    args.out.write_text(json.dumps(out, indent=2))

    print("\nlead   ens     s0     s1   delta*  day-win  (CSI@1.0)",
          flush=True)
    for i, L in enumerate(leads):
        ce = _csi(acc["ens"][i][TI])
        c0 = _csi(acc["s0"][i][TI])
        c1 = _csi(acc["s1"][i][TI])
        st = day_stats[str(L)]
        print(f"{L:>4}  {ce:.3f}  {c0:.3f}  {c1:.3f}  "
              f"{st['pooled_delta']:+.3f}  "
              f"{st['day_wins']}/{st['n_days']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())