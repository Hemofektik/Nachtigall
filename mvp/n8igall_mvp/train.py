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
                 wind_sampler=None, leads: list[int] | None = None):
        self.samples = samples
        self.crop = crop
        self.static = static_channels  # [C_s, H, W] float32 or None
        self.wind = wind_sampler       # WindSampler or None
        self.leads = leads              # multi-lead target minutes or None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, mask = load_sample(self.samples[idx], wind_sampler=self.wind,
                                 leads=self.leads)
        if self.static is not None:
            x = np.concatenate([x, self.static], axis=0)
        if self.crop:
            _, h, w = x.shape
            r = random.randint(0, h - self.crop)
            c = random.randint(0, w - self.crop)
            x = x[:, r:r + self.crop, c:c + self.crop]
            y = y[:, r:r + self.crop, c:c + self.crop]
            mask = mask[:, r:r + self.crop, c:c + self.crop]
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))


def evaluate(model, loader, device, radar_channels: int,
             leads: list[int] | None = None,
             thresholds=(0.1, 1.0, 5.0)) -> dict:
    """Evaluate CSI vs a persistence baseline, per lead time.

    ``radar_channels`` is the number of leading radar-history channels in the
    input tensor; persistence uses the most recent radar frame (not any static
    channels that may follow it).
    """
    leads = leads or [60]
    model.eval()
    csi_model = {(L, t): [] for L in leads for t in thresholds}
    csi_pers = {(L, t): [] for L in leads for t in thresholds}
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss_sum += weighted_mse_loss(pred, y, mask).item() * y.shape[0]
            n += y.shape[0]
            last_radar = torch.expm1(x[:, radar_channels - 1])  # [B,H,W]
            y_rate = torch.expm1(y)
            pred_rate = torch.expm1(pred.clamp(min=0))
            for i, L in enumerate(leads):
                for t in thresholds:
                    csi_model[(L, t)].append(
                        csi(pred_rate[:, i], y_rate[:, i], mask[:, i], t))
                    csi_pers[(L, t)].append(
                        csi(last_radar, y_rate[:, i], mask[:, i], t))
    out = {"loss": loss_sum / max(n, 1)}
    for L in leads:
        for t in thresholds:
            key = f"csi_lead{L}_{t}"
            out[key] = float(np.nanmean(csi_model[(L, t)]))
            out[f"{key}_pers"] = float(np.nanmean(csi_pers[(L, t)]))
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
    p.add_argument("--static", type=Path, default=None,
                   help=".npz with static channels [C_s, H, W]")
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

    train_ds = RadolanDataset(train_s, crop=args.crop, static_channels=static,
                              wind_sampler=wind_sampler, leads=leads)
    val_ds = RadolanDataset(val_s, crop=None, static_channels=static,
                            wind_sampler=wind_sampler, leads=leads)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    from .wind import N_CHANNELS as WIND_CHANNELS
    in_channels = (len(train_s[0].inputs)
                   + (0 if static is None else static.shape[0])
                   + (0 if wind_sampler is None else WIND_CHANNELS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_channels = len(leads) if leads else 1
    residual_from = (len(train_s[0].inputs) - 1) if args.residual else None
    model = UNet(in_channels, base=args.base_channels, depth=args.depth,
                 out_channels=out_channels,
                 residual_from=residual_from,
                 delta_cap=args.delta_cap).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} in_ch={in_channels} "
          f"out_ch={out_channels}" + (f" leads={leads}" if leads else ""))

    loss_fn = make_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
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
                        "leads": leads,
                        "val": val}, args.out / "best.pt")

    torch.save({"model": model.state_dict(), "in_channels": in_channels,
                "base": args.base_channels, "depth": args.depth,
                "residual_from": residual_from,
                "delta_cap": args.delta_cap,
                "leads": leads}, args.out / "last.pt")
    print(f"done. best CSI@1.0 = {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())