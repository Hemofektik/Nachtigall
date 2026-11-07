"""Measure per-lead intensity bias and write bias_calib.npz (finding 3).

The headline result (prob2 + bias calibration) depended on a file that
no script produced - this writes it reproducibly:

    bias[L]   = mean(pred_rate) / mean(obs_rate) over valid pixels
    factor[L] = 1 / bias[L]   (render divides predictions by bias)

Usage:
    python -m n8igall_mvp.compute_bias_calib \
        --checkpoint runs/multi_prob2/best.pt \
        --index data/index_test_rt.json --flow --flow-cache data/index_test_rt_flow \
        --out runs/multi_prob2/bias_calib.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from n8igall_mvp.dataset import Sample, _yw_path, load_sample
from n8igall_mvp.model import UNet
from n8igall_mvp.radolan import read_frame
from n8igall_mvp.wind import WindSampler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=12,
                   help="sample every Nth index entry")
    p.add_argument("--max-samples", type=int, default=60)
    p.add_argument("--flow", action="store_true")
    p.add_argument("--flow-cache", type=Path, default=None)
    p.add_argument("--flow-estimator", choices=["farneback", "lk"],
                   default="farneback")
    p.add_argument("--static", type=Path, default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path, default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path, default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--rt-dir", type=Path, default=Path("data/raw/rt"),
                   help="directory holding the index's target frames")
    args = p.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    leads = ck["leads"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = UNet(ck["in_channels"], base=ck["base"], depth=ck["depth"],
             out_channels=len(leads), residual_from=ck["residual_from"],
             delta_cap=ck["delta_cap"], advect=ck["advect"],
             exceedance=ck.get("exceedance"))
    m.load_state_dict(ck["model"])
    m.to(dev).eval()

    samples = [Sample(**s) for s in json.loads(args.index.read_text())]
    samples = samples[::args.stride][:args.max_samples]
    static = np.load(args.static)
    wind = WindSampler(args.wind, elevation=static["elevation"])
    cloud = None
    if ck["in_channels"] >= 32:
        from n8igall_mvp.cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    from n8igall_mvp.train import RadolanDataset
    ds = RadolanDataset(samples, crop=None,
                        static_channels=static["channels"].astype(np.float32),
                        wind_sampler=wind, leads=leads, flow=args.flow,
                        flow_cache=args.flow_cache,
                        flow_estimator=args.flow_estimator)

    sums = {L: [0.0, 0.0, 0] for L in leads}   # pred_sum, obs_sum, n
    with torch.no_grad():
        for i in range(len(samples)):
            x, y, mask = ds[i]
            out = m(x[None].to(dev))
            pred = (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()
            ts = re.search(r"-(\d{10})-", samples[i].inputs[-1]).group(1)
            t = datetime.strptime(ts, "%y%m%d%H%M")
            for j, L in enumerate(leads):
                try:
                    f = read_frame(_yw_path(args.rt_dir,
                                            t + timedelta(minutes=L)))
                except FileNotFoundError:
                    continue
                v = f.valid
                if not v.any():
                    continue
                pr = np.expm1(pred[j].clip(min=0))[v]
                sums[L][0] += float(pr.sum())
                sums[L][1] += float(f.data[v].sum())
                sums[L][2] += 1

    factors = []
    print(f"per-lead bias (pred/obs) on {len(samples)} samples:")
    for L in leads:
        ps, os_, n = sums[L]
        bias = ps / os_ if (os_ > 0 and n) else 1.0
        factors.append(1.0 / bias)
        print(f"  +{L:>2}: bias={bias:.3f} factor={1.0/bias:.3f} (n={n})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, leads=np.array(leads),
             factors=np.array(factors, dtype=np.float64))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
