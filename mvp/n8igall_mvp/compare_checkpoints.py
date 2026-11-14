"""Paired day-level comparison of two checkpoints on the frozen test set.

Roadmap falsification checks ask for separation *beyond day-level
spread*. eval_checkpoint stores only pooled counts, so this runs both
models in one pass over the same samples and accumulates per-day
counts at CSI@1.0 - then reports pooled delta, per-day paired deltas
(mean +/- std, wins) exactly like blend_leads/ensemble_seeds.

Usage:
    python -m n8igall_mvp.compare_checkpoints \
        runs/multi_lk_cloud/best.pt runs/multi_lk_cloud_3s/best.pt \
        --out runs/compare_2s_vs_3s.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .blend_leads import (THRESHOLDS, TI, _add, _build_model, _counts,
                          _csi, _make_pass)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", type=Path, nargs=2, metavar=("A", "B"))
    p.add_argument("--test-index", type=Path,
                   default=Path("data/index_test_rt.json"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--static", type=Path,
                   default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path,
                   default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path,
                   default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    ma, cka, leads, device = _build_model(args.checkpoints[0])
    mb, ckb, leads_b, _ = _build_model(args.checkpoints[1])
    assert leads == leads_b, f"lead mismatch: {leads} vs {leads_b}"

    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if cka["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    samples, dl, _ = _make_pass(args.test_index, leads, args, static, wind,
                                cloud)
    n = len(samples)
    systems = ("a", "b")
    acc = {s: [[{} for _ in THRESHOLDS] for _ in leads] for s in systems}
    day_acc: dict[str, dict[str, list[dict]]] = {}
    idx = 0
    with torch.no_grad():
        for x, y, mask in dl:
            oa = ma(x.to(device))
            ob = mb(x.to(device))
            pa = torch.expm1((oa[0] if isinstance(oa, tuple)
                              else oa).clamp(min=0)).cpu().numpy()
            pb = torch.expm1((ob[0] if isinstance(ob, tuple)
                              else ob).clamp(min=0)).cpu().numpy()
            yr = torch.expm1(y).numpy()
            m = mask.numpy().astype(bool)
            B = x.shape[0]
            for b in range(B):
                from .blend_leads import _TS_RE
                key = int(_TS_RE.search(samples[idx].inputs[-1]).group(1))
                idx += 1
                day = str(key)[:6]
                d = day_acc.setdefault(
                    day, {s: [{} for _ in leads] for s in systems})
                for sysname, arr in (("a", pa[b]), ("b", pb[b])):
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

    day_stats = {}
    for i, L in enumerate(leads):
        deltas = []
        for per_sys in day_acc.values():
            ca, cb = _csi(per_sys["a"][i]), _csi(per_sys["b"][i])
            if np.isfinite(ca) and np.isfinite(cb):
                deltas.append(cb - ca)
        deltas = np.asarray(deltas)
        day_stats[str(L)] = {
            "pooled_delta": _csi(acc["b"][i][TI]) - _csi(acc["a"][i][TI]),
            "day_delta_mean": float(deltas.mean()) if len(deltas) else None,
            "day_delta_std": float(deltas.std()) if len(deltas) else None,
            "day_wins": int((deltas > 0).sum()),
            "n_days": int(len(deltas)),
        }
    out["day_stats"] = day_stats
    args.out.write_text(json.dumps(out, indent=2))

    print("\nlead      a      b  pooled_d  day_mean  day_std  wins",
          flush=True)
    for i, L in enumerate(leads):
        ca = _csi(acc["a"][i][TI])
        cb = _csi(acc["b"][i][TI])
        st = day_stats[str(L)]
        print(f"{L:>4}  {ca:.3f}  {cb:.3f}  {st['pooled_delta']:+.4f}  "
              f"{st['day_delta_mean']:+.4f}  {st['day_delta_std']:.4f}  "
              f"{st['day_wins']}/{st['n_days']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())