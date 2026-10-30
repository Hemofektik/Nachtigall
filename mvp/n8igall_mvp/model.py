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
                 out_channels: int = 1, residual_from: int | None = None,
                 delta_cap: float | None = None,
                 advect: dict | None = None):
        """U-Net predicting log1p rain rates.

        ``residual_from``: channel index of the "now" radar frame. When set,
        the network predicts a *delta* on top of that frame
        (``out = base + delta``) instead of generating the field from
        scratch - input texture is conserved by construction and the model
        only has to learn what changes. None = direct prediction.

        ``advect``: {"u_idx", "v_idx", "leads": [...]} - when set (with
        ``residual_from``), the residual base is the now-frame *warped by
        the wind field over each lead time* (semi-Lagrangian sampling with
        a learnable per-lead scale factor). Motion then exists by
        construction; the network only learns growth/decay on top of the
        advected field.
        """
        super().__init__()
        self.residual_from = residual_from
        self.delta_cap = delta_cap
        self.advect_cfg = advect
        if advect is not None:
            assert residual_from is not None, "advect requires residual mode"
            assert out_channels == len(advect["leads"]), \
                "advect leads must match out_channels"
            self.u_idx = advect["u_idx"]
            self.v_idx = advect["v_idx"]
            self.advect_leads = list(advect["leads"])
            # learnable per-lead displacement scale (100 m wind underestimates
            # the steering-level flow that actually moves precipitation)
            self.advect_scale = nn.Parameter(torch.ones(len(self.advect_leads)))
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

    def _advect(self, now: torch.Tensor, u: torch.Tensor,
                v: torch.Tensor, lead_min: float, lead_i: int) -> torch.Tensor:
        """Semi-Lagrangian warp of ``now`` by wind (u,v) over ``lead_min``.

        u: eastward m/s, v: northward m/s; cell = 1000 m; row 0 = north.
        Destination (r,c) samples the source position the parcel came from:
        src_col = c - u*dt/1000*scale, src_row = r + v*dt/1000*scale.
        """
        scale = self.advect_scale[lead_i]
        dt = float(lead_min) * 60.0
        dx = -u * (dt / 1000.0) * scale   # source offset, east positive
        dy = v * (dt / 1000.0) * scale    # source offset, south positive
        _, _, h, w = now.shape
        cs = torch.arange(w, device=now.device, dtype=now.dtype)
        rs = torch.arange(h, device=now.device, dtype=now.dtype)
        rr, cc = torch.meshgrid(rs, cs, indexing="ij")   # [H,W]
        x_n = 2.0 * (cc.unsqueeze(0) + dx.squeeze(1) + 0.5) / w - 1.0
        y_n = 2.0 * (rr.unsqueeze(0) + dy.squeeze(1) + 0.5) / h - 1.0
        grid = torch.stack([x_n, y_n], dim=-1)           # [B,H,W,2]
        return F.grid_sample(now, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pooling floors odd sizes (900 -> 450 -> 225 -> 112), breaking
        # skip alignment on upsample; pad to the network stride and crop back.
        mult = 2 ** (len(self.enc) - 1)
        h, w = x.shape[-2:]
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        x_orig = x  # keep un-padded input for the residual path
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
        out = out[..., :h, :w]
        if self.residual_from is not None:
            # residual base: the now-frame, optionally advected per lead
            now = x_orig[:, self.residual_from:self.residual_from + 1]
            if self.advect_cfg is not None:
                u = x_orig[:, self.u_idx:self.u_idx + 1]
                v = x_orig[:, self.v_idx:self.v_idx + 1]
                base = torch.cat(
                    [self._advect(now, u, v, L, i)
                     for i, L in enumerate(self.advect_leads)], dim=1)
            else:
                base = now
            if self.delta_cap is not None:
                # hard bound on the delta: the prediction can never exceed
                # expm1(base + cap) mm/h, killing hallucinated spikes
                out = self.delta_cap * torch.tanh(out / self.delta_cap)
            out = base + out
        return out


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


def rate_explosion_penalty(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
    """Penalize over-prediction in RATE space (mm/h), not log space.

    In log space a 61 vs 0 mm/h hallucination looks like a modest 4-unit
    error diluted over 810k pixels; in rate space it is a 61-unit error
    on exactly the pixels that matter. Mean over all masked pixels keeps
    the term comparable to the main loss.
    """
    rate_over = F.relu(torch.expm1(pred) - torch.expm1(target))
    return weight * (rate_over * mask).sum() / mask.sum().clamp(min=1.0)


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