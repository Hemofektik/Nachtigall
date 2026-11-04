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
    # heavy-rain threshold for the inference-time clamp: the delta may not
    # subtract from base pixels above this rate (see forward)
    HEAVY_MMH = 5.0

    def __init__(self, in_channels: int, base: int = 32, depth: int = 3,
                 out_channels: int = 1, residual_from: int | None = None,
                 delta_cap: float | None = None,
                 advect: dict | None = None,
                 exceedance: list[float] | None = None,
                 base_loss_weight: float = 0.0):
        """U-Net predicting log1p rain rates.

        ``residual_from``: channel index of the "now" radar frame. When set,
        the network predicts a *delta* on top of that frame
        (``out = base + delta``) instead of generating the field from
        scratch - input texture is conserved by construction and the model
        only has to learn what changes. None = direct prediction.

        ``advect``: {"u_idx", "v_idx", "leads": [...], "mode": "wind"|"flow"} -
        when set (with ``residual_from``), the residual base is the now-frame
        *warped by the steering field over each lead time* (semi-Lagrangian
        sampling with a learnable per-lead scale factor). Motion then exists
        by construction; the network only learns growth/decay on top of the
        advected field. ``mode="wind"`` (default): u/v are m/s at 100 m
        (ERA5). ``mode="flow"``: u/v are dense optical-flow displacements in
        px per 5-min step, computed from the last two radar frames.

        ``exceedance``: mm/h thresholds for optional probability heads
        (item 5). When set, an extra 1x1 head predicts logits for
        P(rate > t) per lead and threshold (channel order lead-major:
        ``i * len(exceedance) + j``), trained with class-balanced BCE so
        rare heavy-rain pixels get direct gradient instead of being
        averaged away by intensity regression.
        """
        super().__init__()
        self.residual_from = residual_from
        self.delta_cap = delta_cap
        self.advect_cfg = advect
        self.exceedance = list(exceedance) if exceedance else None
        # auxiliary base loss: when > 0 (training), forward stashes the
        # advected base in ``last_base`` so the loop can penalize
        # base-vs-target directly. Without it the network learns to shrink
        # advect_scale to ~0 (frozen base) and paints everything with the
        # delta - heavy cores then never move.
        self.base_loss_weight = base_loss_weight
        self.last_base: torch.Tensor | None = None
        if advect is not None:
            assert residual_from is not None, "advect requires residual mode"
            assert out_channels == len(advect["leads"]), \
                "advect leads must match out_channels"
            self.u_idx = advect["u_idx"]
            self.v_idx = advect["v_idx"]
            self.advect_leads = list(advect["leads"])
            self.advect_mode = advect.get("mode", "wind")
            # learnable per-lead displacement scale: for wind it compensates
            # the 100 m flow underestimating steering-level motion; for flow
            # it lets the network correct Farneback's bias.
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
        if self.exceedance:
            self.head_prob = nn.Conv2d(chans[0],
                                       out_channels * len(self.exceedance), 1)

    def _advect(self, now: torch.Tensor, u: torch.Tensor,
                v: torch.Tensor, lead_min: float, lead_i: int) -> torch.Tensor:
        """Semi-Lagrangian warp of ``now`` by (u,v) over ``lead_min``.

        mode "wind": u eastward m/s, v northward m/s at 100 m; cell = 1000 m.
        mode "flow": u eastward px, v southward px per 5-min frame step
        (Farneback from the last two radar frames).
        Destination (r,c) samples the source position the parcel came from.
        """
        scale = self.advect_scale[lead_i]
        if self.advect_mode == "flow":
            steps = (float(lead_min) / 5.0) * scale
            # u eastward, v SOUTHWARD (Farneback row direction): destination
            # (r,c) samples the source the parcel came from, i.e. north of a
            # southward-moving parcel -> dy = -v*steps. (v sign was flipped
            # here, inverting all vertical motion in the base - the reason
            # the base loss shrank advect_scale to ~0.)
            dx = -u * steps   # source offset, east positive
            dy = -v * steps   # source offset, south positive
        else:
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
        dec = x[..., :h, :w]
        out = self.head(dec)
        prob_logits = self.head_prob(dec) if self.exceedance else None
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
            if self.training and self.base_loss_weight > 0:
                self.last_base = base
            if not self.training:
                # inference-time clamp: the delta may not subtract from
                # heavy base pixels (regression-to-the-mean erased ~97% of
                # >5 mm/h pixels the advection base had correctly moved;
                # clamping recovers CSI@5 from 0.001 to the base level at
                # +60 with zero cost at 0.1/1.0 mm/h - ceiling test 2026-11-03)
                heavy = base > math.log1p(self.HEAVY_MMH)
                out = torch.where(heavy, out.clamp(min=0.0), out)
            out = base + out
        if prob_logits is not None:
            return out, prob_logits
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


def exceedance_bce_loss(prob_logits: torch.Tensor, target_log: torch.Tensor,
                        mask: torch.Tensor, thresholds: list[float],
                        max_pos_weight: float = 50.0) -> torch.Tensor:
    """Class-balanced BCE on the P(rate > t) heads (item 5).

    ``prob_logits``: [B, L*J, H, W] lead-major (channel i*J + j).
    ``target_log``: [B, L, H, W] log1p(rate); ``mask``: [B, L, H, W] bool.

    Positives (pixels above t) are up-weighted by neg/pos (clamped) so rare
    heavy-rain pixels carry gradient comparable to the bulk — the fix for
    regression-to-the-mean erasing extremes. Mean over thresholds and leads.
    """
    B, L, H, W = target_log.shape
    J = len(thresholds)
    logits = prob_logits.reshape(B, L, J, H, W)
    terms = []
    for j, t in enumerate(thresholds):
        pos = (target_log > math.log1p(t)) & mask
        n_pos = pos.sum().clamp(min=1.0)
        n_neg = (mask.sum() - pos.sum()).clamp(min=1.0)
        pw = (n_neg / n_pos).clamp(1.0, max_pos_weight)
        bce = F.binary_cross_entropy_with_logits(
            logits[:, :, j], pos.float(), reduction="none")
        w = torch.where(pos, pw, torch.ones_like(pw))
        terms.append((bce * w * mask).sum() / (w * mask).sum().clamp(min=1.0))
    return torch.stack(terms).mean()


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