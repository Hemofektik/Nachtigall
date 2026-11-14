"""Roadmap item 4 (review at 13a87fb): score the probability heads properly.

Hard CSI ranks checkpoints by deterministic regression, but the models
also carry exceedance heads P(rate > t) (0.1 / 1 / 5 mm/h) trained with
class-balanced BCE. This pass scores those heads on the *validation
tail* of the training index (eval-only, never the frozen test):

* **Brier** per lead x threshold (with base rate -> Brier skill),
* **reliability** tables (10 fixed bins) + ECE,
* **soft multi-scale FSS** (fractional fields, 10/20/50 px) of the
  probability field against the observed exceedance fraction,
* the deterministic head's pooled CSI alongside, so prob and hard
  rankings can be compared directly - the roadmap's falsification-
  relevant question: *if prob scores and hard CSI pick different
  checkpoints, pooled CSI alone can't rank models.*

Usage (one JSON per checkpoint, compare across runs):
    python -m n8igall_mvp.prob_reliability \
        --checkpoint runs/multi_lk_cloud/best.pt \
        --out runs/multi_lk_cloud/prob_reliability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .blend_leads import _build_model, _make_pass
from .train import _box_sum

N_BINS = 10
FSS_SIZES = (10, 20, 50)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("data/index_multi.json"))
    p.add_argument("--tail", type=float, default=0.2,
                   help="chronological val tail of the index (train split)")
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
    J = len(exc)

    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if ckpt["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    import json as _json
    all_samples = _json.loads(args.index.read_text())
    n_tail = max(1, int(len(all_samples) * args.tail))
    subset_s = all_samples[-n_tail:][::args.stride]
    from .dataset import Sample
    subset = [Sample(**s) for s in subset_s]
    print(f"val tail: {len(subset)} samples (tail {n_tail} of "
          f"{len(all_samples)}, stride {args.stride})", flush=True)

    _, dl, _ = _make_pass(args.index, leads, args, static, wind, cloud,
                          subset=subset)
    L = len(leads)

    # accumulators
    brier = np.zeros((L, J))
    brier_n = np.zeros((L, J))
    obs_sum = np.zeros((L, J))
    hits_p = np.zeros((L, J))       # prob head at 0.5
    fa_p = np.zeros((L, J))
    miss_p = np.zeros((L, J))
    hits_h = np.zeros((L, J))       # deterministic head
    fa_h = np.zeros((L, J))
    miss_h = np.zeros((L, J))
    # reliability bins: [L, J, bin] -> (sum_p, sum_o, count)
    rb = np.zeros((L, J, N_BINS, 3))
    # soft FSS: [L, J, scale] -> (mse, den)
    fss = np.zeros((L, J, len(FSS_SIZES), 2))

    idx = 0
    n = len(subset)
    with torch.no_grad():
        for x, y, mask in dl:
            out = model(x.to(device))
            pred, logits = out if isinstance(out, tuple) else (out, None)
            if logits is None:
                raise SystemExit("model returned no prob logits")
            pr = torch.expm1(pred.clamp(min=0)).cpu().numpy()   # [B,L,H,W]
            yr = torch.expm1(y).numpy()
            m = mask.numpy().astype(bool)
            pl = logits.reshape(x.shape[0], L, J, *x.shape[-2:])
            pj = torch.sigmoid(pl).cpu().numpy()                # [B,L,J,H,W]
            B = x.shape[0]

            for i in range(L):
                obs = (yr[:, i][:, None]
                       > np.asarray(exc)[None, :, None, None]) \
                    & m[:, i][:, None]                          # [B,J,H,W]
                obs_f = obs.astype(np.float32)
                mf = m[:, i][:, None].astype(np.float32)        # [B,1,H,W]
                for j in range(J):
                    o = obs[:, j]
                    q = pj[:, i, j]
                    v = m[:, i]
                    brier[i, j] += float(((q - o.astype(np.float32)) ** 2
                                          * v).sum())
                    brier_n[i, j] += int(v.sum())
                    obs_sum[i, j] += int((o & v).sum())
                    hp = (q > 0.5) & v
                    hh = (pr[:, i] > exc[j]) & v
                    oo = o & v
                    hits_p[i, j] += int((hp & oo).sum())
                    fa_p[i, j] += int((hp & ~oo).sum())
                    miss_p[i, j] += int((~hp & oo).sum())
                    hits_h[i, j] += int((hh & oo).sum())
                    fa_h[i, j] += int((hh & ~oo).sum())
                    miss_h[i, j] += int((~hh & oo).sum())
                    # reliability bins (flatten batch)
                    qb = q.ravel()
                    ob = o.ravel().astype(np.float32)
                    vb = v.ravel()
                    if vb.any():
                        bins = np.minimum((qb * N_BINS).astype(int),
                                          N_BINS - 1)
                        b = bins[vb]
                        rb[i, j, :, 0] += np.bincount(
                            b, weights=qb[vb], minlength=N_BINS)
                        rb[i, j, :, 1] += np.bincount(
                            b, weights=ob[vb], minlength=N_BINS)
                        rb[i, j, :, 2] += np.bincount(
                            b, minlength=N_BINS)
                    # soft FSS: fractions over B flattened maps (GPU -
                    # the CPU box-sum path costs ~87 min per pass)
                    obs2 = obs_f[:, j]                          # [B,H,W]
                    pred2 = q.astype(np.float32)                # [B,H,W]
                    mb_t = torch.as_tensor(m[:, i], dtype=torch.float32,
                                           device=device)
                    ob_t = torch.as_tensor(obs2, device=device)
                    pb_t = torch.as_tensor(pred2, device=device)
                    for si, s in enumerate(FSS_SIZES):
                        sz = min(s, ob_t.shape[-2], ob_t.shape[-1])
                        nm_s = _box_sum(mb_t, sz).clamp(min=1e-6)
                        fo = _box_sum(ob_t, sz) / nm_s
                        fp = _box_sum(pb_t, sz) / nm_s
                        mse = (((fo - fp) ** 2) * mb_t).sum().item()
                        den = ((fo ** 2 + fp ** 2) * mb_t).sum().item()
                        fss[i, j, si, 0] += mse
                        fss[i, j, si, 1] += den
            idx += B
            if idx % 100 == 0 or idx >= n:
                print(f"{idx}/{n}", flush=True)

    def csi(h, f, m_):
        d = h + f + m_
        return float(h / d) if d else None

    out = {
        "checkpoint": str(args.checkpoint), "index": str(args.index),
        "tail": args.tail, "stride": args.stride, "n": idx,
        "leads": leads, "exceedance": exc,
        "per_lead": {},
    }
    for i, Lm in enumerate(leads):
        row = {}
        for j, t in enumerate(exc):
            br = brier[i, j] / max(brier_n[i, j], 1)
            base = obs_sum[i, j] / max(brier_n[i, j], 1)
            ece = 0.0
            rel = []
            tot = rb[i, j, :, 2].sum()
            for b in range(N_BINS):
                cnt = rb[i, j, b, 2]
                if cnt <= 0:
                    rel.append(None)
                    continue
                mp = rb[i, j, b, 0] / cnt
                of = rb[i, j, b, 1] / cnt
                ece += abs(mp - of) * cnt / max(tot, 1)
                rel.append({"bin": b / N_BINS, "mean_p": float(mp),
                            "obs_freq": float(of), "count": int(cnt)})
            row[str(t)] = {
                "brier": float(br), "base_rate": float(base),
                "brier_skill": float(1 - br / base) if base > 0 else None,
                "ece": float(ece),
                "reliability": rel,
                "csi_prob_0.5": csi(hits_p[i, j], fa_p[i, j], miss_p[i, j]),
                "csi_hard": csi(hits_h[i, j], fa_h[i, j], miss_h[i, j]),
                "fss_soft": {str(FSS_SIZES[s]):
                             float(1 - fss[i, j, s, 0] / fss[i, j, s, 1])
                             if fss[i, j, s, 1] > 0 else None
                             for s in range(len(FSS_SIZES))},
            }
        out["per_lead"][str(Lm)] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nlead  thr   brier  skill    ECE   csi_p  csi_h  fss50",
          flush=True)
    for Lm in leads:
        for t in exc:
            r = out["per_lead"][str(Lm)][str(t)]
            print(f"{Lm:>4}  {t:>3}  {r['brier']:.4f} "
                  f"{(r['brier_skill'] or 0):+.3f}  {r['ece']:.3f}  "
                  f"{(r['csi_prob_0.5'] or 0):.3f} "
                  f"{(r['csi_hard'] or 0):.3f}  "
                  f"{(r['fss_soft']['50'] or 0):.3f}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())