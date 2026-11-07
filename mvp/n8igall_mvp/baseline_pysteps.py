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


def farneback_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Dense Farneback flow [2,H,W] px/step (u=east, v=south)."""
    import cv2
    a = (np.clip(prev, 0, 50) / 50.0 * 255).astype(np.uint8)
    b = (np.clip(curr, 0, 50) / 50.0 * 255).astype(np.uint8)
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.transpose(fl, (2, 0, 1)).astype(np.float32)


def coarse_flow(prev: np.ndarray, curr: np.ndarray, block: int = 16,
                angle_out: float = 60.0, mad_k: float = 4.0) -> np.ndarray:
    """Block-mean optical flow with outlier filtering (the proven design).

    Dense per-pixel flow is noisy; storm motion is mostly coherent, so:

    1. compute dense Farneback flow,
    2. per ``block x block`` tile: drop pixel vectors far from the tile's
       median (robust: convective outliers and dry-area noise don't drag
       the mean), average the inliers,
    3. drop tiles whose direction deviates more than ``angle_out`` degrees
       from the *global* median direction (replace with that direction) -
       outlier directions would otherwise mess with the mean,
    4. bilinear-upsample the coarse field back to full resolution.

    Returns [2,H,W] px/step (u=east, v=south).
    """
    import cv2
    dense = farneback_flow(prev, curr)
    H, W = dense.shape[1:]
    Hp = -(-H // block) * block
    Wp = -(-W // block) * block
    pad = np.pad(dense, ((0, 0), (0, Hp - H), (0, Wp - W)), mode="edge")
    Hb, Wb = Hp // block, Wp // block
    vecs = pad.reshape(2, Hb, block, Wb, block) \
              .transpose(0, 1, 3, 2, 4) \
              .reshape(2, Hb, Wb, block * block) \
              .transpose(1, 2, 3, 0)          # [Hb,Wb,n,2]
    med = np.median(vecs, axis=2)             # [Hb,Wb,2]
    dev = np.linalg.norm(vecs - med[:, :, None, :], axis=3)
    thr = mad_k * np.median(dev, axis=2, keepdims=True)
    thr = np.maximum(thr, 0.05)               # floor: constant fields keep all
    inl = dev <= thr
    cnt = inl.sum(axis=2, keepdims=True)
    has = cnt[:, :, 0] > 0
    sm = np.where(has[:, :, None],
                  (vecs * inl[:, :, :, None]).sum(axis=2)
                  / np.maximum(cnt, 1), med)  # [Hb,Wb,2]

    # global direction filter: tiles disagreeing wildly with the median
    # direction are noise (textureless dry areas) - use the global median
    gmed = np.median(sm.reshape(-1, 2), axis=0)
    gmag = float(np.linalg.norm(gmed))
    if gmag > 0.1:
        dot = sm @ gmed
        cos = dot / (np.linalg.norm(sm, axis=2) * gmag + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        sm = np.where((ang > angle_out)[:, :, None], gmed[None, None, :], sm)

    out = np.empty((2, Hp, Wp), dtype=np.float32)
    for c in range(2):
        out[c] = cv2.resize(sm[:, :, c], (Wp, Hp),
                            interpolation=cv2.INTER_LINEAR)
    return out[:, :H, :W]


def smooth_flow(prev: np.ndarray, curr: np.ndarray, sigma: float = 8.0,
                angle_out: float = 60.0, estimator: str = "farneback"
                ) -> np.ndarray:
    """Gaussian-smoothed flow with per-pixel direction outlier rejection.

    No tiles: every pixel gets a mean direction from its Gaussian
    neighborhood. Individual pixels whose direction deviates more than
    ``angle_out`` degrees from that local mean are outliers (convective
    cells moving against the storm, textureless dry-area noise) and are
    replaced by the local mean before the final smoothing pass, so they
    cannot pollute their neighbours.

    ``estimator``: "farneback" (dense, fast) or "lk" (pysteps Lucas-Kanade,
    best CSI) - isolates whether losses come from the estimator or the
    smoothing.

    Returns [2,H,W] px/step (u=east, v=south).
    """
    from scipy.ndimage import gaussian_filter
    if estimator == "lk":
        from pysteps.motion.lucaskanade import dense_lucaskanade
        dense = dense_lucaskanade(np.stack([prev, curr]).astype(np.float64))
    else:
        dense = farneback_flow(prev, curr)
    # pass 1: local mean direction per pixel
    loc = gaussian_filter(dense, sigma=(0, sigma, sigma))
    # per-pixel direction test against the local mean
    dot = (dense * loc).sum(axis=0)
    n1 = np.linalg.norm(dense, axis=0)
    n2 = np.linalg.norm(loc, axis=0)
    cos = dot / (n1 * n2 + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    outlier = (ang > angle_out) & (n1 > 0.1)   # moving pixels only
    cleaned = np.where(outlier[None], loc, dense)
    # pass 2: final neighborhood mean
    return gaussian_filter(cleaned, sigma=(0, sigma, sigma)).astype(np.float32)


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
    p.add_argument("--stride", type=int, default=1,
                   help="take every Nth sample (for large test sets)")
    p.add_argument("--flow-mode", choices=["lk", "farneback", "coarse",
                                           "smooth", "smooth-lk"],
                   default="lk",
                   help="lk = pysteps Lucas-Kanade (dense); farneback = "
                        "dense Farneback; coarse = block-mean flow with "
                        "outlier filtering (16px blocks); smooth = "
                        "Gaussian neighbourhood mean with per-pixel "
                        "direction outlier rejection (no tiles); "
                        "smooth-lk = same smoothing over LK flow "
                        "(isolates estimator vs smoothing)")
    p.add_argument("--block", type=int, default=16,
                   help="block size for --flow-mode coarse")
    p.add_argument("--sigma", type=float, default=8.0,
                   help="Gaussian sigma (px) for --flow-mode smooth")
    args = p.parse_args(argv)

    from pysteps.motion.lucaskanade import dense_lucaskanade

    leads = [int(v) for v in args.leads.split(",") if v]
    samples = [Sample(**s) for s in json.loads(args.index.read_text())]
    n_val = max(1, int(len(samples) * args.val_fraction))
    val_s = samples[-n_val:]
    if args.stride > 1:
        val_s = val_s[::args.stride]
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
            if args.flow_mode == "coarse":
                flow = coarse_flow(radar[-2], radar[-1], block=args.block)
            elif args.flow_mode == "smooth":
                flow = smooth_flow(radar[-2], radar[-1], sigma=args.sigma)
            elif args.flow_mode == "smooth-lk":
                flow = smooth_flow(radar[-2], radar[-1], sigma=args.sigma,
                                   estimator="lk")
            elif args.flow_mode == "farneback":
                flow = farneback_flow(radar[-2], radar[-1])
            else:
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
