"""Train the nowcasting U-Net and evaluate against a persistence baseline.

Usage:
    python -m n8igall_mvp.train --index data/index.json --out runs/v1

Baselines reported alongside the model:
* persistence: last input frame (rate now) as the +60 min prediction
* CSI at 0.1 / 1 / 5 mm/h on the validation split
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader, Dataset

from .dataset import Sample, load_sample
from .model import (UNet, csi, rate_explosion_penalty, soft_csi_loss,
                    weighted_huber_loss, weighted_mse_loss)


def make_loss(name: str):
    if name == "mse":
        return weighted_mse_loss
    if name == "huber":
        return weighted_huber_loss
    if name == "huber_csi":
        def loss(pred, target, mask):
            return (weighted_huber_loss(pred, target, mask)
                    + 0.5 * soft_csi_loss(pred, target, mask, threshold=1.0))
        return loss
    if name == "huber_csi_rate":
        def loss(pred, target, mask):
            return (weighted_huber_loss(pred, target, mask)
                    + 0.5 * soft_csi_loss(pred, target, mask, threshold=1.0)
                    + rate_explosion_penalty(pred, target, mask))
        return loss
    raise ValueError(f"unknown loss: {name}")


class RadolanDataset(Dataset):
    def __init__(self, samples: list[Sample], crop: int | None = 512,
                 static_channels: np.ndarray | None = None,
                 wind_sampler=None, leads: list[int] | None = None,
                 cloud_sampler=None):
        self.samples = samples
        self.crop = crop
        self.static = static_channels  # [C_s, H, W] float32 or None
        self.wind = wind_sampler       # WindSampler or None
        self.cloud = cloud_sampler     # CloudSampler or None
        self.leads = leads              # multi-lead target minutes or None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, mask = load_sample(self.samples[idx], wind_sampler=self.wind,
                                 leads=self.leads)
        if self.static is not None:
            x = np.concatenate([x, self.static], axis=0)
        if self.cloud is not None:
            # appended LAST so radar/wind/static channel indices (used by
            # residual_from and advect u/v) remain unchanged
            from datetime import datetime as _dt
            import re as _re
            ts = _re.search(r"-(\d{10})-",
                            self.samples[idx].inputs[-1]).group(1)
            t_now = _dt.strptime(ts, "%y%m%d%H%M")
            x = np.concatenate([x, self.cloud.channels_for(t_now)], axis=0)
        if self.crop:
            _, h, w = x.shape
            r = random.randint(0, h - self.crop)
            c = random.randint(0, w - self.crop)
            x = x[:, r:r + self.crop, c:c + self.crop]
            y = y[:, r:r + self.crop, c:c + self.crop]
            mask = mask[:, r:r + self.crop, c:c + self.crop]
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))


FSS_SIZES = (10, 20, 50)  # neighbourhood widths in pixels (~km at 1 km grid)


def _binary_stats(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> dict:
    """Accumulate pooled contingency counts, Brier sum and FSS terms.

    NumPy variant used by offline baselines (``baseline_pysteps``); the
    in-training ``evaluate`` uses the batched GPU version below.
    """
    p, o = pred & mask, obs & mask
    st = {"hits": int((p & o).sum()), "fa": int((p & ~o).sum()),
          "miss": int((~p & o).sum()), "n": int(mask.sum()),
          "brier": 0.0, "fss_mse": {}, "fss_den": {}}
    if st["n"] == 0:
        return st
    st["brier"] = float(((p.astype(np.float64) - o.astype(np.float64)) ** 2).sum())
    mf = mask.astype(np.float32)
    pf, of = p.astype(np.float32), o.astype(np.float32)
    for s in FSS_SIZES:
        size = min(s, min(mf.shape[-2:]))
        nb = np.maximum(uniform_filter(mf, size=size, mode="constant"), 1e-6)
        fp = uniform_filter(pf, size=size, mode="constant") / nb
        fo = uniform_filter(of, size=size, mode="constant") / nb
        st["fss_mse"][s] = float((((fp - fo) ** 2) * mf).sum())
        st["fss_den"][s] = float(((fp ** 2 + fo ** 2) * mf).sum())
    return st


def _merge(acc: dict, st: dict) -> None:
    for k in ("hits", "fa", "miss", "n"):
        acc[k] = acc.get(k, 0) + st[k]
    acc["brier"] = acc.get("brier", 0.0) + st["brier"]
    for s in FSS_SIZES:
        acc.setdefault("fss_mse", {})[s] = acc.get("fss_mse", {}).get(s, 0.0) + st["fss_mse"].get(s, 0.0)
        acc.setdefault("fss_den", {})[s] = acc.get("fss_den", {}).get(s, 0.0) + st["fss_den"].get(s, 0.0)


def _box_sum(x: torch.Tensor, size: int) -> torch.Tensor:
    """Sum over a size x size window with zero padding, preserving shape.

    ``x`` is [N,H,W]; returns [N,H,W]. Equivalent to
    ``scipy.ndimage.uniform_filter(x, size, mode="constant") * size**2``
    but batched on GPU — the CPU path cost ~87 min per validation pass.
    scipy's window for even ``size`` extends ``size//2`` to the left and
    ``size - 1 - size//2`` to the right (origin=0), so we match that.
    """
    pl = size // 2
    pr = size - 1 - pl
    y = F.pad(x[:, None], (pl, pr, pl, pr))
    return F.avg_pool2d(y, size, stride=1)[:, 0] * float(size * size)


def evaluate(model, loader, device, radar_channels: int,
             leads: list[int] | None = None,
             thresholds=(0.1, 1.0, 5.0),
             full_metrics: bool = False) -> dict:
    """Evaluate vs a persistence baseline, per lead time.

    Always reports the legacy nanmean CSI keys (``csi_lead{L}_{t}``,
    ``csi_{t}``, ``csi_persistence_{t}``) — cheap and used for checkpoint
    selection.

    With ``full_metrics=True`` also reports:

    * ``csi_lead{L}_{t}_pooled`` — hits/misses/false alarms pooled over the
      whole set (stable; all-dry samples don't distort it),
    * Brier score (``brier_...``) and fractions skill score at 10/20/50
      km neighbourhoods (``fss_..._{s}``), each with a ``_pers`` twin.

    The FSS box filters cost ~30 GPU passes per batch (~30 min per 585
    sample pass), so the training loop uses ``full_metrics=False`` each
    epoch and computes the full set once on the best checkpoint.

    ``radar_channels`` is the number of leading radar-history channels;
    persistence uses the most recent radar frame.
    """
    leads = leads or [60]
    model.eval()
    n_leads = len(leads)
    csi_model = {(L, t): [] for L in leads for t in thresholds}
    csi_pers = {(L, t): [] for L in leads for t in thresholds}
    # pooled accumulators as python scalars: (who, lead_idx, thr) -> dict
    acc = {}
    for who in ("m", "p"):
        for i in range(n_leads):
            for t in thresholds:
                acc[(who, i, t)] = {"hits": 0, "fa": 0, "miss": 0, "n": 0,
                                    "brier": 0.0,
                                    "fss_mse": {s: 0.0 for s in FSS_SIZES},
                                    "fss_den": {s: 0.0 for s in FSS_SIZES}}
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss_sum += weighted_mse_loss(pred, y, mask).item() * y.shape[0]
            n += y.shape[0]
            last_radar = torch.expm1(x[:, radar_channels - 1])  # [B,H,W]
            y_rate = torch.expm1(y)                             # [B,L,H,W]
            pred_rate = torch.expm1(pred.clamp(min=0))
            m = mask.bool()
            B = x.shape[0]

            # legacy per-sample CSI (cheap, stays on device)
            for i, L in enumerate(leads):
                for t in thresholds:
                    csi_model[(L, t)].append(
                        csi(pred_rate[:, i], y_rate[:, i], mask[:, i], t))
                    csi_pers[(L, t)].append(
                        csi(last_radar, y_rate[:, i], mask[:, i], t))

            # pooled counts + Brier (vectorized over batch and leads)
            pers = last_radar[:, None].expand(-1, n_leads, -1, -1)  # [B,L,H,W]
            if full_metrics:
                for who, fr in (("m", pred_rate), ("p", pers)):
                    for t in thresholds:
                        o = (y_rate > t) & m
                        f = (fr > t) & m
                        for i in range(n_leads):
                            ai = acc[(who, i, t)]
                            oi, fi, mi = o[:, i], f[:, i], m[:, i]
                            ai["hits"] += int((fi & oi).sum())
                            ai["fa"] += int((fi & ~oi).sum())
                            ai["miss"] += int((~fi & oi).sum())
                            ai["n"] += int(mi.sum())
                            ai["brier"] += float(
                                ((fi.float() - oi.float()) ** 2 * mi.float()).sum())
                # FSS neighbourhood fractions (batched GPU box sums).
                # Flatten batch x leads so each (thr, scale, who) is ONE
                # box-filter call; obs fractions and mask denominators are
                # shared between model and persistence. 30 calls/batch.
                yf = y_rate.flatten(0, 1)
                pf_ = pred_rate.flatten(0, 1)
                rf = pers.flatten(0, 1)
                mb_flat = m.flatten(0, 1)          # bool, for masking
                mf_flat = mb_flat.float()          # float, for box sums
                nm_cache = {}
                for s in FSS_SIZES:
                    sz = min(s, yf.shape[-2], yf.shape[-1])
                    nm_cache[s] = _box_sum(mf_flat, sz).clamp(min=1e-6)
                for t in thresholds:
                    o = ((yf > t) & mb_flat).float()
                    fo_all = {}
                    for s in FSS_SIZES:
                        sz = min(s, yf.shape[-2], yf.shape[-1])
                        fo_all[s] = _box_sum(o, sz) / nm_cache[s]
                    for who, fr in (("m", pf_), ("p", rf)):
                        f = ((fr > t) & mb_flat).float()
                        for s in FSS_SIZES:
                            sz = min(s, yf.shape[-2], yf.shape[-1])
                            ff_ = _box_sum(f, sz) / nm_cache[s]
                            fo_ = fo_all[s]
                            mse = (((fo_ - ff_) ** 2) * mf_flat).reshape(B, n_leads, *yf.shape[-2:])
                            den = ((ff_ ** 2 + fo_ ** 2) * mf_flat).reshape(B, n_leads, *yf.shape[-2:])
                            mse_l = mse.sum(dim=0)   # [L,H,W] -> sum over batch
                            den_l = den.sum(dim=0)
                            for i in range(n_leads):
                                a = acc[(who, i, t)]
                                a["fss_mse"][s] += float(mse_l[i].sum())
                                a["fss_den"][s] += float(den_l[i].sum())
    out = {"loss": loss_sum / max(n, 1)}
    for i, L in enumerate(leads):
        for t in thresholds:
            key = f"csi_lead{L}_{t}"
            out[key] = float(np.nanmean(csi_model[(L, t)]))
            out[f"{key}_pers"] = float(np.nanmean(csi_pers[(L, t)]))
            for who, tag in (("m", ""), ("p", "_pers")):
                st = acc[(who, i, t)]
                if not full_metrics:
                    continue
                den = st["hits"] + st["fa"] + st["miss"]
                out[f"{key}_pooled{tag}"] = (st["hits"] / den) if den else float("nan")
                if st["n"]:
                    out[f"brier_lead{L}_{t}{tag}"] = st["brier"] / st["n"]
                    for s in FSS_SIZES:
                        d = st["fss_den"].get(s, 0.0)
                        out[f"fss_lead{L}_{t}_{s}{tag}"] = (
                            1.0 - st["fss_mse"].get(s, 0.0) / d) if d else float("nan")
    # Legacy keys point at the primary lead (last entry) for comparability.
    primary = leads[-1]
    for t in thresholds:
        out[f"csi_{t}"] = out[f"csi_lead{primary}_{t}"]
        out[f"csi_persistence_{t}"] = out[f"csi_lead{primary}_{t}_pers"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=3,
                   help="U-Net encoder levels (3 = original, 4 = ~7M params)")
    p.add_argument("--loss", choices=("mse", "huber", "huber_csi",
                                      "huber_csi_rate"),
                   default="mse")
    p.add_argument("--residual", action="store_true",
                   help="predict a delta on top of the current radar frame "
                        "(conserves input texture by construction)")
    p.add_argument("--delta-cap", type=float, default=None,
                   help="hard bound on the residual delta in log1p units "
                        "(e.g. 3.0 caps prediction at expm1(now+3))")
    p.add_argument("--advect", action="store_true",
                   help="warp the residual base by the 100m wind field over "
                        "each lead time (motion by construction); requires "
                        "--residual, --wind and multi-lead outputs")
    p.add_argument("--static", type=Path, default=None,
                   help=".npz with static channels [C_s, H, W]")
    p.add_argument("--cloud", type=Path, default=None,
                   help=".npz cloud store (era5_cloud.npz) to add cloud channels")
    p.add_argument("--wind", type=Path, default=None,
                   help=".npz wind store (era5_wind.npz) to add wind channels")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--leads", default="",
                   help="comma-separated lead minutes for multi-lead targets "
                        "(e.g. 5,10,15,20,30,45,60); empty = single RW +60")
    args = p.parse_args(argv)

    leads = [int(v) for v in args.leads.split(",") if v] or None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = [Sample(**s) for s in json.loads(args.index.read_text())]
    if not samples:
        raise SystemExit("empty index")
    # chronological split: last val-fraction of days held out
    n_val = max(1, int(len(samples) * args.val_fraction))
    train_s, val_s = samples[:-n_val], samples[-n_val:]

    static = None
    if args.static:
        static = np.load(args.static)["channels"].astype(np.float32)

    wind_sampler = None
    if args.wind:
        from .wind import WindSampler
        elevation = (np.load(args.static)["elevation"] if args.static else None)
        wind_sampler = WindSampler(args.wind, elevation=elevation)

    cloud_sampler = None
    if args.cloud:
        from .cloud import CloudSampler
        cloud_sampler = CloudSampler(args.cloud)

    train_ds = RadolanDataset(train_s, crop=args.crop, static_channels=static,
                              wind_sampler=wind_sampler, leads=leads,
                              cloud_sampler=cloud_sampler)
    val_ds = RadolanDataset(val_s, crop=None, static_channels=static,
                            wind_sampler=wind_sampler, leads=leads,
                              cloud_sampler=cloud_sampler)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    from .wind import N_CHANNELS as WIND_CHANNELS
    in_channels = (len(train_s[0].inputs)
                   + (0 if static is None else static.shape[0])
                   + (0 if wind_sampler is None else WIND_CHANNELS)
                   + (0 if cloud_sampler is None else 3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_channels = len(leads) if leads else 1
    residual_from = (len(train_s[0].inputs) - 1) if args.residual else None
    advect_cfg = None
    if args.advect:
        if not (args.residual and args.wind and leads):
            raise SystemExit("--advect requires --residual --wind and --leads")
        # wind channels sit right after the radar history: u10, v10, u100,
        # v100, upslope, div (see wind.N_CHANNELS / dataset.load_sample)
        n_radar = len(train_s[0].inputs)
        advect_cfg = {"u_idx": n_radar + 2, "v_idx": n_radar + 3,
                      "leads": leads}
    model = UNet(in_channels, base=args.base_channels, depth=args.depth,
                 out_channels=out_channels,
                 residual_from=residual_from,
                 delta_cap=args.delta_cap,
                 advect=advect_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} in_ch={in_channels} "
          f"out_ch={out_channels}" + (f" leads={leads}" if leads else ""))

    loss_fn = make_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    # Per-run config + metrics files (reproducible experiment records).
    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in sorted(vars(args).items())}
    config.update({"n_params": n_params, "in_channels": in_channels,
                   "out_channels": out_channels, "n_train": len(train_ds),
                   "n_val": len(val_ds), "seed": args.seed})
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.write_text("")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (x, y, mask) in enumerate(train_dl):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            loss = loss_fn(model(x), y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if i % 20 == 0:
                print(f"  ep{epoch} it{i} loss={loss.item():.4f}")
        sched.step()

        val = evaluate(model, val_dl, device,
                       radar_channels=len(train_s[0].inputs), leads=leads)
        line = (f"ep{epoch} train_loss={running/max(1,len(train_dl)):.4f} "
                f"({time.time()-t0:.0f}s) val_loss={val['loss']:.4f} "
                + " ".join(f"CSI@{t}={val[f'csi_{t}']:.3f}"
                           f"(pers {val[f'csi_persistence_{t}']:.3f})"
                           for t in ("0.1", "1.0", "5.0")))
        if leads:
            line += " | " + " ".join(
                f"L{L}={val[f'csi_lead{L}_1.0']:.3f}" for L in leads)
        print(line)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({"epoch": epoch, "train_loss":
                                 running / max(1, len(train_dl)), **val}) + "\n")

        # Select on the cross-lead mean: single-lead keys can be 0.0 in
        # every epoch (frozen best.pt) when the primary lead is unlearned.
        lead_scores = [val[f"csi_lead{L}_1.0"] for L in (leads or [60])]
        score = float(np.nanmean([s for s in lead_scores
                                  if not np.isnan(s)] or [np.nan]))
        if not np.isnan(score) and score > best:
            best = score
            torch.save({"model": model.state_dict(), "in_channels": in_channels,
                        "base": args.base_channels, "depth": args.depth,
                        "residual_from": residual_from,
                        "delta_cap": args.delta_cap,
                        "advect": advect_cfg,
                        "leads": leads,
                        "val": val}, args.out / "best.pt")

    torch.save({"model": model.state_dict(), "in_channels": in_channels,
                "base": args.base_channels, "depth": args.depth,
                "residual_from": residual_from,
                "delta_cap": args.delta_cap,
                "advect": advect_cfg,
                "leads": leads}, args.out / "last.pt")
    print(f"done. best CSI@1.0 = {best:.3f}")

    # Final full-metrics pass (pooled CSI, Brier, FSS) on the best
    # checkpoint — too slow for every epoch, run once at the end.
    best_path = args.out / "best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print("computing full metrics (pooled CSI / Brier / FSS) on best.pt ...")
        full = evaluate(model, val_dl, device,
                        radar_channels=len(train_s[0].inputs), leads=leads,
                        full_metrics=True)
        (args.out / "metrics_final.json").write_text(
            json.dumps(full, indent=2) + "\n")
        print("full metrics written to metrics_final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())