"""Sweep the delta scale s in pred = base + s*delta per lead on the val split.

Answers: where does the network's delta help vs hurt relative to the pure
flow-advection base? (s=0 is the flow baseline, s=1 the model as-is.)
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from n8igall_mvp.dataset import Sample
from n8igall_mvp.model import UNet, csi
from n8igall_mvp.train import RadolanDataset
from n8igall_mvp.wind import WindSampler
from torch.utils.data import DataLoader

ck = torch.load("runs/multi_prob2/best.pt", map_location="cpu", weights_only=False)
leads = ck["leads"]
m = UNet(ck["in_channels"], base=ck["base"], depth=ck["depth"],
         out_channels=len(leads), residual_from=ck["residual_from"],
         delta_cap=ck["delta_cap"], advect=ck["advect"],
         exceedance=ck.get("exceedance"))
m.load_state_dict(ck["model"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m.to(dev)
m.eval()

samples = [Sample(**s) for s in json.load(open("data/index_multi.json"))]
n_val = int(len(samples) * 0.2)
val_s = samples[-n_val:][::4]
static = np.load("data/static/channels.npz")
wind = WindSampler("data/wind/era5_wind.npz", elevation=static["elevation"])
ds = RadolanDataset(val_s, crop=None,
                    static_channels=static["channels"].astype(np.float32),
                    wind_sampler=wind, leads=leads, flow=True)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

scales = (0.0, 0.5, 1.0)
acc = {(s, L): [] for s in scales for L in leads}
t0 = time.time()
with torch.no_grad():
    for bi, (x, y, mask) in enumerate(dl):
        x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
        out = m(x)
        pred = (out[0] if isinstance(out, tuple) else out)
        now = x[:, 18:19]
        base = torch.cat([m._advect(now, x[:, 25:26], x[:, 26:27], L, i)
                          for i, L in enumerate(leads)], dim=1)
        delta = pred - base
        y_rate = torch.expm1(y)
        for s in scales:
            pr = torch.expm1((base + s * delta).clamp(min=0))
            for i, L in enumerate(leads):
                acc[(s, L)].append(csi(pr[:, i], y_rate[:, i], mask[:, i], 1.0))
        if bi % 100 == 0:
            print(f"  {bi}/{len(val_s)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\nCSI@1.0 by delta scale (val, {len(val_s)} samples):")
print(f"{'lead':>5} | {'s=0 (flow)':>10} {'s=0.5':>7} {'s=1 (model)':>11} | best")
for L in leads:
    v = {s: float(np.nanmean(acc[(s, L)])) for s in scales}
    best = max(v, key=v.get)
    print(f"{L:>5} | {v[0.0]:>10.3f} {v[0.5]:>7.3f} {v[1.0]:>11.3f} | s={best}")
