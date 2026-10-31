"""Optical-flow advection baseline (pysteps Lucas-Kanade) for Nachtigall.

The honest "strong persistence" baseline: estimate a dense motion field
from the last two radar frames with pysteps, advect the newest frame by
each lead time, and score it with the *same* metrics as the model
(pooled CSI, Brier, FSS at 10/20/50 px) plus persistence, on the same
chronological validation split.

Usage:
    python -m n8igall_mvp.baseline_pysteps --index data/index_B.json \
        --leads 5,10,...,60 --out runs/baseline_pysteps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import Sample, load_sample
from .train import FSS_SIZES, _binary_stats, _merge

LEADS_DEFAULT = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
THRESHOLDS = (0.1, 1.0, 5.0)


def advect(frame: np.ndarray, flow: np.ndarray, steps: float) -> np.ndarray:
    """Warp ``frame`` [H,W] by ``flow`` [H,W,2] (px per frame-step) x ``steps``.

    Uses grid_sample with align_corners=False — the same resampling the
    model's advect path uses, so baseline and model share the warp.
    pysteps returns flow as [2,H,W]: flow[0]=u (x/col), flow[1]=v (y/row).
    """
    H, W = frame.shape
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    # sample source position: current pixel minus displacement
    sx = xs - torch.as_tensor(flow[0] * steps)
    sy = ys - torch.as_tensor(flow[1] * steps)
    gx = (sx / (W - 1)) * 2 - 1
    gy = (sy / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None].float()
    img = torch.as_tensor(frame, dtype=torch.float32)[None, None]
    out = F.grid_sample(img, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[0, 0].numpy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--leads", type=str, default=",".join(map(str, LEADS_DEFAULT)))
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = all validation samples")
    args = p.parse_args(argv)

    from pysteps.motion.lucaskanade import dense_lucaskanade

    leads = [int(v) for v in args.leads.split(",") if v]
    samples = [Sample(**s) for s in json.loads(args.index.read_text())]
    n_val = max(1, int(len(samples) * args.val_fraction))
    val_s = samples[-n_val:]
    if args.max_samples:
        val_s = val_s[: args.max_samples]

    # metric accumulators: (who, lead, thr) -> pooled stats dict
    pooled = {(w, L, t): {} for w in ("flow", "pers")
              for L in leads for t in THRESHOLDS}
    n_used = n_flow_fail = 0

    for s in val_s:
        # radar history only (channels 0..18); wind/static not needed here
        x, y, mask = load_sample(s, leads=leads)
        radar = np.expm1(x[:19])          # [19,H,W] mm/h, newest last
        y_rate = np.expm1(y)              # [L,H,W] mm/h
        m = mask.astype(bool)             # [L,H,W]

        # optical flow from the last two frames (5-min spacing assumed
        # for the multi-lead YW index; matches the model's dt)
        try:
            flow = dense_lucaskanade(radar[-2:].astype(np.float64))
        except Exception:                 # noqa: BLE001 - skip degenerate frames
            n_flow_fail += 1
            continue
        if not np.isfinite(flow).all():
            n_flow_fail += 1
            continue

        now = radar[-1]
        for i, L in enumerate(leads):
            steps = L / 5.0               # flow is per 5-min frame step
            pred = advect(now, flow, steps)
            obs_t = (y_rate[i] > 0) & m[i]
            for t in THRESHOLDS:
                o = (y_rate[i] > t) & m[i]
                _merge(pooled[("flow", L, t)],
                       _binary_stats((pred > t) & m[i], o, m[i]))
                _merge(pooled[("pers", L, t)],
                       _binary_stats((now > t) & m[i], o, m[i]))
        n_used += 1

    out: dict = {"n_val": len(val_s), "n_used": n_used,
                 "n_flow_fail": n_flow_fail, "leads": leads,
                 "thresholds": list(THRESHOLDS)}
    for L in leads:
        for t in THRESHOLDS:
            for who, tag in (("flow", ""), ("pers", "_pers")):
                st = pooled[(who, L, t)]
                den = st["hits"] + st["fa"] + st["miss"]
                out[f"csi_lead{L}_{t}{tag}"] = (st["hits"] / den) if den else None
                if st["n"]:
                    out[f"brier_lead{L}_{t}{tag}"] = st["brier"] / st["n"]
                    for s_ in FSS_SIZES:
                        d = st["fss_den"].get(s_, 0.0)
                        out[f"fss_lead{L}_{t}_{s_}{tag}"] = (
                            1.0 - st["fss_mse"].get(s_, 0.0) / d) if d else None

    text = json.dumps(out, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
