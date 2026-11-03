"""Evaluate a trained checkpoint on an arbitrary index (e.g. the held-out
real-time test set) with the full metric suite.

Usage:
    python -m n8igall_mvp.eval_checkpoint --checkpoint runs/advect_flow/best.pt \
        --index data/index_test_rt.json --out runs/advect_flow/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import Sample
from .model import UNet
from .train import RadolanDataset, evaluate
from torch.utils.data import DataLoader


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--static", type=Path,
                   default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path,
                   default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path,
                   default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--flow", action="store_true",
                   help="append optical-flow channels (match training)")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    leads = ckpt.get("leads")
    model = UNet(ckpt["in_channels"], base=ckpt["base"], depth=ckpt["depth"],
                 out_channels=len(leads) if leads else 1,
                 residual_from=ckpt["residual_from"],
                 delta_cap=ckpt["delta_cap"],
                 advect=ckpt["advect"],
                 exceedance=ckpt.get("exceedance"))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = [Sample(**s) for s in json.loads(args.index.read_text())]
    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if ckpt["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    ds = RadolanDataset(samples, crop=None, static_channels=static,
                        wind_sampler=wind, leads=leads,
                        cloud_sampler=cloud, flow=args.flow)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4)

    n_radar = len(samples[0].inputs)
    res = evaluate(model, dl, device, radar_channels=n_radar, leads=leads,
                   full_metrics=True)
    res["_meta"] = {"checkpoint": str(args.checkpoint),
                    "index": str(args.index), "n_samples": len(samples),
                    "in_channels": ckpt["in_channels"], "flow": args.flow}
    text = json.dumps(res, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    # compact summary
    print(f"n={len(samples)}")
    for t in ("0.1", "1.0", "5.0"):
        print(f"  CSI@{t}: {res[f'csi_{t}']:.3f} (pers {res[f'csi_persistence_{t}']:.3f})"
              f" pooled={res[f'csi_lead{leads[-1]}_{t}_pooled']:.3f}"
              f" (pers {res[f'csi_lead{leads[-1]}_{t}_pooled_pers']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
