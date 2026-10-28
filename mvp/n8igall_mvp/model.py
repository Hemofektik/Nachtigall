"""Compact U-Net for precipitation nowcasting.

Sized for a 4 GB GPU (RTX 3050 Ti): ~2 M parameters, 3 encoder levels,
base width 32. Training uses random 512x512 crops; inference runs on the
full 900x900 grid.

Input:  ``[B, C_in, H, W]`` — radar history channels + static channels
        (elevation/slope/aspect, optional) + wind channels (optional).
Output: ``[B, 1, H, W]`` — predicted ``log1p(rate)`` at +lead minutes.
"""

from __future__ import annotations

import math

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
    def __init__(self, in_channels: int, base: int = 32, depth: int = 3,
                 out_channels: int = 1):
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

        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pooling floors odd sizes (900 -> 450 -> 225 -> 112), breaking
        # skip alignment on upsample; pad to the network stride and crop back.
        mult = 2 ** (len(self.enc) - 1)
        h, w = x.shape[-2:]
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

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
        out = self.head(x)
        return out[..., :h, :w]


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


def _rain_weight(target: torch.Tensor, rain_threshold: float,
                 rain_weight: float) -> torch.Tensor:
    return torch.where(target > rain_threshold,
                       torch.full_like(target, rain_weight),
                       torch.ones_like(target))


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor, delta: float = 1.0,
                        rain_threshold: float = 0.1,
                        rain_weight: float = 5.0) -> torch.Tensor:
    """Huber loss on log1p rates: linear tails damp the influence of
    clutter spikes and large misses that blur MSE-trained predictions.

    ``delta`` is in log1p units: 1.0 means errors beyond ~2.7x in rate
    grow only linearly. Too small a delta (0.1) made large overshoots
    nearly free and let predictions explode (740 mm/h vs 7 mm/h actual).
    """
    err = pred - target
    abs_err = err.abs()
    huber = torch.where(abs_err <= delta,
                        0.5 * err ** 2,
                        delta * (abs_err - 0.5 * delta))
    weight = _rain_weight(target, rain_threshold, rain_weight) * mask
    return (huber * weight).sum() / mask.sum().clamp(min=1.0)


def soft_csi_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor, threshold: float = 1.0,
                  sharpness: float = 20.0,
                  over_penalty: float = 0.5) -> torch.Tensor:
    """(1 - soft CSI) at ``threshold`` mm/h in log space.

    Smooth surrogate of the CSI metric itself: rewards hits, penalizes
    misses and false alarms symmetrically - directly counters the
    systematic over-wetting of weighted pointwise losses.

    ``over_penalty`` additionally punishes predicting far above the
    threshold (a 740 mm/h prediction earns no more than a 7 mm/h one
    but must not be free either).
    """
    th = math.log1p(threshold)
    p = torch.sigmoid(sharpness * (pred - th)) * mask
    t = torch.sigmoid(sharpness * (target - th)) * mask
    hits = (p * t).sum()
    denom = p.sum() + t.sum() - hits
    over = F.relu(pred - target) * p  # overshoot on predicted-rain pixels
    return (1.0 - hits / denom.clamp(min=1e-6)) \
        + over_penalty * over.sum() / mask.sum().clamp(min=1.0)


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