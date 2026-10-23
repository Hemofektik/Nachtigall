"""Compact U-Net for precipitation nowcasting.

Sized for a 4 GB GPU (RTX 3050 Ti): ~2 M parameters, 3 encoder levels,
base width 32. Training uses random 512x512 crops; inference runs on the
full 900x900 grid.

Input:  ``[B, C_in, H, W]`` — radar history channels + static channels
        (elevation/slope/aspect, optional) + wind channels (optional).
Output: ``[B, 1, H, W]`` — predicted ``log1p(rate)`` at +lead minutes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 32, depth: int = 3):
        super().__init__()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        chans = [base * (2 ** i) for i in range(depth)]

        c = in_channels
        for ch in chans:
            self.enc.append(_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)

        for i in reversed(range(depth - 1)):
            skip_ch = chans[i]
            up_in = chans[i + 1]
            self.dec.append(nn.ConvTranspose2d(up_in, skip_ch, 2, stride=2))
            self.dec.append(_block(skip_ch * 2, skip_ch))

        self.head = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for i in range(0, len(self.dec), 2):
            x = self.dec[i](x)  # upsample
            x = torch.cat([x, skips.pop()], dim=1)
            x = self.dec[i + 1](x)
        return self.head(x)


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, rain_threshold: float = 0.1,
                      rain_weight: float = 5.0) -> torch.Tensor:
    """MSE on log1p rates, up-weighting rainy pixels to fight class imbalance."""
    err = (pred - target) ** 2
    weight = torch.where(target > rain_threshold,
                         torch.full_like(target, rain_weight),
                         torch.ones_like(target))
    weight = weight * mask
    return (err * weight).sum() / mask.sum().clamp(min=1.0)


def csi(pred_rate: torch.Tensor, target_rate: torch.Tensor,
        mask: torch.Tensor, threshold: float) -> float:
    """Critical Success Index for pixels above ``threshold`` mm/h."""
    p = (pred_rate > threshold) & mask
    t = (target_rate > threshold) & mask
    hits = (p & t).sum().item()
    false_alarm = (p & ~t).sum().item()
    miss = (~p & t).sum().item()
    denom = hits + false_alarm + miss
    return hits / denom if denom else float("nan")