"""Recalibrate the exceedance probability heads (roadmap item 4 follow-up).

Item 4 found the heads badly miscalibrated (negative Brier skill past
~+15, ECE up to 0.17) - the cause is structural: they were trained with
*class-balanced* BCE (pos_weight up to 50), which deliberately distorts
the posterior so rare heavy-rain pixels carry gradient. That distortion
is a monotone affine map of the logit, so **Platt scaling** (a*z + b per
lead x threshold, fit by NLL on the val tail) should undo it.

Eval-only, one model pass: logits are histogrammed (BCE/Brier/ECE all
depend only on the logit value, so the histogram is a sufficient
statistic - the fit and the before/after scores are exact, no pixel
storage). Fit data = val tail (standard for calibration); the frozen
test set is never touched.

Usage:
    python -m n8igall_mvp.calibrate_prob \
        --checkpoint runs/multi_lk_cloud_3s/best.pt \
        --out runs/multi_lk_cloud_3s/prob_calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .blend_leads import _build_model, _make_pass

BIN_LO, BIN_HI, BIN_W = -30.0, 30.0, 0.05
N_BINS = int((BIN_HI - BIN_LO) / BIN_W)
N_ECE_BINS = 10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_affine(pos: np.ndarray, neg: np.ndarray, centers: np.ndarray
               ) -> tuple[float, float]:
    """Platt scale (a, b), a > 0, minimizing NLL from the histogram."""
    from scipy.optimize import minimize
    w = pos + neg
    keep = w > 0
    p, n, z = pos[keep], neg[keep], centers[keep]

    def nll(ab):
        a, b = ab
        if a <= 1e-4:
            return 1e18
        m = a * z + b
        return float((p * np.logaddexp(0, -m)
                      + n * np.logaddexp(0, m)).sum())

    best, best_v = None, np.inf
    for a0 in (0.25, 0.5, 1.0, 2.0, 4.0):
        r = minimize(nll, [a0, 0.0], method="Nelder-Mead")
        if r.fun < best_v:
            best, best_v = r.x, r.fun
    return float(best[0]), float(best[1])


def scores(pos, neg, centers, a=1.0, b=0.0):
    """Brier, 10-bin ECE, base rate, CSI@0.5 from the histogram."""
    keep = (pos + neg) > 0
    p, n, z = pos[keep], neg[keep], centers[keep]
    wt = p + n                      # filtered weights (same length as q)
    total = wt.sum()
    q = _sigmoid(a * z + b)
    brier = float((p * (q - 1) ** 2 + n * q ** 2).sum() / total)
    base = float(p.sum() / total)
    # ECE over 10 equal-width bins of the calibrated probability
    ece = 0.0
    for lo in range(N_ECE_BINS):
        sel = (q >= lo / N_ECE_BINS) & (q < (lo + 1) / N_ECE_BINS)
        if lo == N_ECE_BINS - 1:
            sel = (q >= lo / N_ECE_BINS)
        c = wt[sel].sum()
        if c <= 0:
            continue
        mp = (q[sel] * wt[sel]).sum() / c
        of = p[sel].sum() / c
        ece += abs(mp - of) * c / total
    # CSI at decision threshold 0.5 == calibrated logit > 0
    dec = (a * z + b) > 0
    hits = float(p[dec].sum())
    fa = float(n[dec].sum())
    miss = float(p[~dec].sum())
    csi = hits / (hits + fa + miss) if (hits + fa + miss) else None
    return {"brier": brier, "ece": float(ece), "base_rate": base,
            "brier_skill": 1 - brier / base if base > 0 else None,
            "csi_0.5": csi}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("data/index_multi.json"))
    p.add_argument("--tail", type=float, default=0.2)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--static", type=Path,
                   default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path,
                   default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path,
                   default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    model, ckpt, leads, device = _build_model(args.checkpoint)
    exc = ckpt.get("exceedance")
    if not exc:
        raise SystemExit(f"{args.checkpoint} has no exceedance heads")
    J, L = len(exc), len(leads)

    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if ckpt["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    all_samples = json.loads(args.index.read_text())
    n_tail = max(1, int(len(all_samples) * args.tail))
    subset_s = all_samples[-n_tail:][::args.stride]
    from .dataset import Sample
    subset = [Sample(**s) for s in subset_s]
    print(f"val tail: {len(subset)} samples (tail {n_tail} of "
          f"{len(all_samples)}, stride {args.stride})", flush=True)

    _, dl, _ = _make_pass(args.index, leads, args, static, wind, cloud,
                          subset=subset)
    # histogram: [L, J, N_BINS] counts of (positive, negative) pixels
    hist = np.zeros((L, J, N_BINS, 2), dtype=np.int64)
    edges = np.arange(N_BINS) * BIN_W + BIN_LO
    centers = edges + BIN_W / 2

    idx = 0
    n = len(subset)
    with torch.no_grad():
        for x, y, mask in dl:
            out = model(x.to(device))
            if not isinstance(out, tuple):
                raise SystemExit("model returned no prob logits")
            logits = out[1]
            yr = torch.expm1(y).numpy()
            m = mask.numpy().astype(bool)
            pl = logits.reshape(x.shape[0], L, J, *x.shape[-2:])
            z = pl.cpu().numpy()
            B = x.shape[0]
            for i in range(L):
                obs = (yr[:, i][:, None]
                       > np.asarray(exc)[None, :, None, None]) \
                    & m[:, i][:, None]              # [B,J,H,W]
                for j in range(J):
                    zb = np.clip(z[:, i, j], BIN_LO, BIN_HI - 1e-6)
                    bins = ((zb - BIN_LO) / BIN_W).astype(np.int64)
                    o = obs[:, j].ravel()
                    v = m[:, i].ravel()
                    bb = bins.ravel()[v]
                    oo = o[v]
                    hist[i, j, :, 0] += np.bincount(
                        bb[oo], minlength=N_BINS)
                    hist[i, j, :, 1] += np.bincount(
                        bb[~oo], minlength=N_BINS)
            idx += B
            if idx % 100 == 0 or idx >= n:
                print(f"{idx}/{n}", flush=True)

    # raw histogram FIRST - a summary bug must never cost a pass again
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "index": str(args.index),
        "tail": args.tail, "stride": args.stride, "n": idx,
        "leads": leads, "exceedance": exc,
        "bin_lo": BIN_LO, "bin_w": BIN_W,
        "hist": hist.tolist(), "per_lead": {},
    }, indent=2))

    out = {"checkpoint": str(args.checkpoint), "index": str(args.index),
           "tail": args.tail, "stride": args.stride, "n": idx,
           "leads": leads, "exceedance": exc, "per_lead": {}}
    print(f"\nlead  thr |    brier        ECE     skill | -> a     b "
          f"| brier'   ECE'   skill'", flush=True)
    for i, Lm in enumerate(leads):
        row = {}
        for j, t in enumerate(exc):
            pos, neg = hist[i, j, :, 0], hist[i, j, :, 1]
            before = scores(pos, neg, centers)
            a, b = fit_affine(pos, neg, centers)
            after = scores(pos, neg, centers, a, b)
            row[str(t)] = {"a": a, "b": b, "before": before,
                           "after": after}
            print(f"{Lm:>4}  {t:>3} | {before['brier']:.4f} "
                  f"{before['ece']:.4f} {(before['brier_skill'] or 0):+.3f} "
                  f"| {a:>5.2f} {b:>+6.2f} | {after['brier']:.4f} "
                  f"{after['ece']:.4f} {(after['brier_skill'] or 0):+.3f}",
                  flush=True)
        out["per_lead"][str(Lm)] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())