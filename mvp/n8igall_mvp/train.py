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
from .model import UNet, csi, weighted_mse_loss


class RadolanDataset(Dataset):
    def __init__(self, samples: list[Sample], crop: int | None = 512,
                 static_channels: np.ndarray | None = None):
        self.samples = samples
        self.crop = crop
        self.static = static_channels  # [C_s, H, W] float32 or None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, mask = load_sample(self.samples[idx])
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
             thresholds=(0.1, 1.0, 5.0)) -> dict:
    """Evaluate CSI vs a persistence baseline.

    ``radar_channels`` is the number of leading radar-history channels in the
    input tensor; persistence uses the most recent radar frame (not any static
    channels that may follow it).
    """
    model.eval()
    csi_model = {t: [] for t in thresholds}
    csi_pers = {t: [] for t in thresholds}
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss_sum += weighted_mse_loss(pred, y, mask).item() * y.shape[0]
            n += y.shape[0]
            pred_rate = torch.expm1(pred.clamp(min=0))
            y_rate = torch.expm1(y)
            last_radar = x[:, radar_channels - 1: radar_channels]
            pers_rate = torch.expm1(last_radar)
            for t in thresholds:
                csi_model[t].append(csi(pred_rate, y_rate, mask, t))
                csi_pers[t].append(csi(pers_rate, y_rate, mask, t))
    out = {"loss": loss_sum / max(n, 1)}
    for t in thresholds:
        key = str(t)
        out[f"csi_{key}"] = float(np.nanmean(csi_model[t]))
        out[f"csi_persistence_{key}"] = float(np.nanmean(csi_pers[t]))
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
    p.add_argument("--static", type=Path, default=None,
                   help=".npz with static channels [C_s, H, W]")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

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

    train_ds = RadolanDataset(train_s, crop=args.crop, static_channels=static)
    val_ds = RadolanDataset(val_s, crop=None, static_channels=static)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    in_channels = len(train_s[0].inputs) + (0 if static is None else static.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels, base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} in_ch={in_channels}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (x, y, mask) in enumerate(train_dl):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            loss = weighted_mse_loss(model(x), y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if i % 20 == 0:
                print(f"  ep{epoch} it{i} loss={loss.item():.4f}")
        sched.step()

        val = evaluate(model, val_dl, device, radar_channels=len(train_s[0].inputs))
        line = (f"ep{epoch} train_loss={running/max(1,len(train_dl)):.4f} "
                f"({time.time()-t0:.0f}s) val_loss={val['loss']:.4f} "
                + " ".join(f"CSI@{t}={val[f'csi_{t}']:.3f}"
                           f"(pers {val[f'csi_persistence_{t}']:.3f})"
                           for t in ("0.1", "1.0", "5.0")))
        print(line)

        score = val["csi_1.0"]
        if not np.isnan(score) and score > best:
            best = score
            torch.save({"model": model.state_dict(), "in_channels": in_channels,
                        "base": args.base_channels, "val": val},
                       args.out / "best.pt")

    torch.save({"model": model.state_dict(), "in_channels": in_channels,
                "base": args.base_channels}, args.out / "last.pt")
    print(f"done. best CSI@1.0 = {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())