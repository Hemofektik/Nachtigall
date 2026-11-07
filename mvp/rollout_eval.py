"""Evaluate autoregressive rollout (with bias correction) on an index.

Compares three variants on the same samples with pooled CSI / Brier:
  direct  - single forward pass, all leads (current model inference)
  rollout - iterative +5 min steps with per-step debias (render mode)
The flow-only baseline (s=0) is the model's own advect base.

Usage:
    python rollout_eval.py --index data/index_test_rt.json --stride 8 \
        --out runs/multi_prob2/rollout_test.json
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from n8igall_mvp.dataset import Sample, _yw_path, load_sample, optical_flow
from n8igall_mvp.model import UNet, csi
from n8igall_mvp.radolan import read_frame
from n8igall_mvp.wind import WindSampler

THRS = (0.1, 1.0, 5.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("runs/multi_prob2/best.pt"))
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--rollout-bias", type=float, default=1.19)
    p.add_argument("--static", type=Path, default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path, default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    leads = ck["leads"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = UNet(ck["in_channels"], base=ck["base"], depth=ck["depth"],
             out_channels=len(leads), residual_from=ck["residual_from"],
             delta_cap=ck["delta_cap"], advect=ck["advect"],
             exceedance=ck.get("exceedance"))
    m.load_state_dict(ck["model"])
    m.to(dev).eval()

    samples = [Sample(**s) for s in json.loads(args.index.read_text())][::args.stride]
    static = np.load(args.static)
    static_c = static["channels"].astype(np.float32)
    wind = WindSampler(args.wind, elevation=static["elevation"])
    print(f"{len(samples)} samples", flush=True)

    # accumulators: variant -> (lead, thr) -> counts
    acc = {v: {(L, t): {"h": 0, "fa": 0, "m": 0, "bs": 0.0, "n": 0}
               for L in leads for t in THRS}
           for v in ("direct", "rollout", "flow")}

    def add(variant, pred_r, obs, msk, L):
        for t in THRS:
            a = acc[variant][(L, t)]
            p_ = (pred_r > t) & msk
            o_ = (obs > t) & msk
            a["h"] += int((p_ & o_).sum())
            a["fa"] += int((p_ & ~o_).sum())
            a["m"] += int((~p_ & o_).sum())
            a["bs"] += float(((p_.astype(np.float64) - o_.astype(np.float64)) ** 2).sum())
            a["n"] += int(msk.sum())

    t0 = time.time()
    with torch.no_grad():
        for bi, s in enumerate(samples):
            x, y, mask = load_sample(s, wind_sampler=wind, leads=leads, flow=True)
            x = np.concatenate([x, static_c], axis=0)
            xb = torch.from_numpy(x)[None].to(dev)
            # direct
            out = m(xb)
            direct = (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()
            # flow base for each lead
            now = xb[:, 18:19]
            base = torch.cat([m._advect(now, xb[:, 25:26], xb[:, 26:27], L, i)
                              for i, L in enumerate(leads)], dim=1)
            flowb = base[0].cpu().numpy()
            # rollout
            hist = [x[i] for i in range(19)]  # log1p frames (flow chans excluded)
            r5 = []
            for _ in range(len(leads)):
                st = np.stack(hist)
                # training order: radar(19) + wind(6) + flow(2) + static...
                # (flow BEFORE wind put u_idx=25 on upslope/div - review
                # finding 2; the rollout rejection was measured with
                # scrambled channels)
                st = np.concatenate(
                    [st, x[19:25],
                     optical_flow(np.expm1(hist[-2]), np.expm1(hist[-1])),
                     x[27:]], axis=0)
                o = m(torch.from_numpy(st)[None].to(dev))
                pr = (o[0] if isinstance(out, tuple) else o)[0, 0]
                pr = torch.log1p(torch.expm1(pr.clamp(min=0)) / args.rollout_bias)
                r5.append(pr.cpu().numpy())
                hist = hist[1:] + [pr.cpu().numpy()]
            rollout = np.stack(r5)

            # observations from disk
            ts = re.search(r"-(\d{10})-", s.inputs[-1]).group(1)
            t_issue = datetime.strptime(ts, "%y%m%d%H%M")
            rt_dir = Path(s.inputs[0]).parent if Path(s.inputs[0]).suffix else Path(s.inputs[0]).parent
            for i, L in enumerate(leads):
                try:
                    f = read_frame(_yw_path(Path("data/raw/rt"),
                                            t_issue + timedelta(minutes=L)))
                except FileNotFoundError:
                    continue
                msk = f.valid
                add("direct", np.expm1(direct[i].clip(min=0)), f.data, msk, L)
                add("rollout", np.expm1(rollout[i].clip(min=0)), f.data, msk, L)
                add("flow", np.expm1(flowb[i].clip(min=0)), f.data, msk, L)
            if bi % 20 == 0:
                print(f"  {bi}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    for variant in ("direct", "rollout", "flow"):
        for L in leads:
            for t in THRS:
                a = acc[variant][(L, t)]
                den = a["h"] + a["fa"] + a["m"]
                res[f"{variant}_csi_lead{L}_{t}"] = a["h"] / den if den else None
                if a["n"]:
                    res[f"{variant}_brier_lead{L}_{t}"] = a["bs"] / a["n"]
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    # compact table
    print(f"\n{'lead':>5} | " + " | ".join(
        f"{v:>18}" for v in ("direct", "rollout", "flow")))
    for L in leads:
        row = f"{L:>5} | "
        for v in ("direct", "rollout", "flow"):
            row += f"{res[f'{v}_csi_lead{L}_1.0']:>18.3f} "
        print(row)
    print("\nCSI@5.0:")
    for L in leads:
        row = f"{L:>5} | "
        for v in ("direct", "rollout", "flow"):
            cv = res[f"{v}_csi_lead{L}_5.0"]
            row += f"{(cv if cv is not None else float('nan')):>18.3f} "
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
