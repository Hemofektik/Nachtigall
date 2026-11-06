"""Render ONE issue-time example: measured vs predicted evolution, 0→60 min.

Two panels side by side, one short video for a single issue time ``t``:

    LEFT : MEASURED radar from t (0 min) to t+60, playing forward
    RIGHT: PREDICTION the model issued at t for each of those same times

Both panels show the same valid time at every frame, so the eye can
compare them directly as time advances. The model's discrete lead
outputs (5/10/15/20/30/45/60 min) are linearly interpolated onto the
5-minute playback grid; frame 0 uses the input frame itself.

Usage:
    python -m n8igall_mvp.render_video --start 2024-09-09T01:00
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .dataset import _yw_path
from .model import UNet
from .radolan import read_frame
from .wind import WindSampler

_PKG_ROOT = Path(__file__).resolve().parent.parent
ALL_LEADS = [5, 10, 15, 20, 30, 45, 60]  # model output channels

# Precipitation colormap: white (dry) -> blue -> green -> yellow -> red.
PRECIP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "precip",
    [(0.00, "#ffffff"),
     (0.02, "#dbeeff"),
     (0.06, "#a6d4ff"),
     (0.12, "#5aa9f0"),
     (0.20, "#2e7ac0"),
     (0.35, "#2fa060"),
     (0.50, "#d8d93a"),
     (0.65, "#f0922e"),
     (0.80, "#e04030"),
     (1.00, "#8b1030")],
)
NO_DATA = "#dcdcdc"


def assemble_input(yw_root: Path, t: datetime, static: np.ndarray,
                   wind: WindSampler, cloud=None, flow: bool = False) -> np.ndarray:
    """[29/32/34, H, W] exactly as training's RadolanDataset order:
    19 radar + 6 wind + 2 flow (when enabled, right after wind - matching
    train.py's u_idx = n_radar + WIND_CHANNELS) + 4 static + 3 cloud."""
    radar, raw = [], []
    for i in range(19):
        f = read_frame(_yw_path(yw_root, t - timedelta(minutes=5 * (18 - i))))
        r = np.where(f.valid, np.clip(f.data, 0, None), 0.0)
        raw.append(r)
        radar.append(np.log1p(r))
    x = np.concatenate([np.stack(radar).astype(np.float32),
                        wind.channels_for(t)], axis=0)
    if flow:
        from .dataset import optical_flow
        x = np.concatenate([x, optical_flow(raw[-2], raw[-1])], axis=0)
    x = np.concatenate([x, static], axis=0)
    if cloud is not None:
        x = np.concatenate([x, cloud.channels_for(t)], axis=0)
    return x


def _rollout(model, t: datetime, yw_root: Path, static: np.ndarray,
             wind, cloud, use_flow: bool, max_steps: int,
             bias: float = 1.0) -> np.ndarray:
    """Autoregressive +5 min rollout: predict one step, feed it back.

    Returns [max_steps, H, W] log1p predictions at leads 5,10,...,5*max_steps.
    Each step rebuilds the 19-frame history (shifting out the oldest frame),
    recomputes the optical-flow channels from the predicted frames, and takes
    the model's +5 min output as the new now-frame. Every step is short-lead
    where the model has proven skill; the flow warp inside the model
    re-anchors texture each step.
    """
    from .dataset import optical_flow
    # history of log1p frames, oldest first
    hist = []
    for i in range(19):
        f = read_frame(_yw_path(yw_root, t - timedelta(minutes=5 * (18 - i))))
        hist.append(np.where(f.valid, np.log1p(np.clip(f.data, 0, None)), 0.0)
                    .astype(np.float32))
    preds = []
    with torch.no_grad():
        for _ in range(max_steps):
            x = np.stack(hist)
            if use_flow:
                raw_prev = np.expm1(hist[-2])
                raw_now = np.expm1(hist[-1])
                x = np.concatenate(
                    [x, optical_flow(raw_prev, raw_now)], axis=0)
            x = np.concatenate([x, wind.channels_for(t), static], axis=0)
            if cloud is not None:
                x = np.concatenate([x, cloud.channels_for(t)], axis=0)
            out = model(torch.from_numpy(x)[None])
            pred = (out[0] if isinstance(out, tuple) else out)
            step5 = pred[0, 0]  # +5 min channel [H,W]
            if bias != 1.0:
                # debias in rate space: pred_log = log1p(expm1(pred)/bias)
                step5 = torch.log1p(torch.expm1(step5.clamp(min=0)) / bias)
            preds.append(step5.numpy())
            hist = hist[1:] + [step5.numpy()]  # shift history
    return np.stack(preds)


def pred_at_lead(pred_all: np.ndarray, lead: int,
                 leads: list[int] | None = None) -> np.ndarray:
    """Model output at minute ``lead``, linearly interpolated between the
    discrete lead channels (identity at lead 0: the input frame itself)."""
    leads = leads or ALL_LEADS
    if lead <= 0:
        return pred_all[0] * 0.0  # replaced by input frame by caller
    if lead in leads:
        return pred_all[leads.index(lead)]
    # linear interpolation between bracketing leads
    lo = max(l for l in leads if l < lead)
    hi = min(l for l in leads if l > lead)
    w = (lead - lo) / (hi - lo)
    return pred_all[leads.index(lo)] * (1 - w) + \
        pred_all[leads.index(hi)] * w


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=_PKG_ROOT / "runs" / "multilead" / "last.pt")
    p.add_argument("--start", type=datetime.fromisoformat,
                   default=datetime(2024, 9, 9, 1, 0),
                   help="single issue time")
    p.add_argument("--max-lead", type=int, default=60)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--bias-correction", type=float, default=1.0,
                   help="multiply predictions by this factor "
                        "(model over-predicts ~1.8x; use ~0.56)")
    p.add_argument("--calibrate", action="store_true",
                   help="apply quantile-mapping calibration from "
                        "<checkpoint_dir>/calibration.npz (fixes intensity "
                        "explosion without retraining)")
    p.add_argument("--bias-calib", action="store_true",
                   help="apply per-lead multiplicative bias correction from "
                        "<checkpoint_dir>/bias_calib.npz (measured "
                        "pred/obs ratios on the test set)")
    p.add_argument("--sharpen", type=float, default=0.0,
                   help="unsharp-mask amount applied to predictions in rate "
                        "space: pred + a*(pred - gaussian_blur(pred, 1.0)); "
                        "0 = off (e.g. 0.6)")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--baseline", choices=["none", "flow"], default="none",
                   help="render a pure optical-flow advection forecast "
                        "instead of the model (visual A/B)")
    p.add_argument("--rollout", action="store_true",
                   help="autoregressive +5min rollout: predict one step, "
                        "feed it back as the new now-frame, repeat (each "
                        "step re-anchors via flow warp)")
    p.add_argument("--rollout-bias", type=float, default=1.0,
                   help="divide each rollout step by this factor (the +5 "
                        "step over-predicts ~1.19x on the test set; 1.19 "
                        "keeps the rollout unbiased)")
    p.add_argument("--out", type=Path,
                   default=_PKG_ROOT / "output" / "pred_vs_actual.mp4")
    args = p.parse_args(argv)

    yw_root = _PKG_ROOT / "data" / "raw" / "yw"
    t = args.start

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_leads = ckpt.get("leads") or ALL_LEADS
    model = UNet(ckpt["in_channels"], base=ckpt["base"],
                 depth=ckpt.get("depth", 3),
                 out_channels=len(model_leads),
                 residual_from=ckpt.get("residual_from"),
                 delta_cap=ckpt.get("delta_cap"),
                 advect=ckpt.get("advect"),
                 exceedance=ckpt.get("exceedance"))
    model.load_state_dict(ckpt["model"])
    model.eval()

    static_npz = np.load(_PKG_ROOT / "data" / "static" / "channels.npz")
    static = static_npz["channels"].astype(np.float32)
    wind = WindSampler(_PKG_ROOT / "data" / "wind" / "era5_wind.npz",
                       elevation=static_npz["elevation"])
    # Cloud channels (3) are present when in_channels >= 32 (radar19 +
    # wind6 + static4 + cloud3); flow mode comes from the checkpoint itself
    # (a no-cloud flow model has 31 channels, so don't infer from count).
    cloud = None
    if ckpt["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(_PKG_ROOT / "data" / "wind" / "era5_cloud.npz")
    use_flow = bool(ckpt.get("advect")
                    and ckpt["advect"].get("mode") == "flow")

    # One issue time: model outputs for all discrete leads + input frame.
    x = assemble_input(yw_root, t, static, wind, cloud, flow=use_flow)
    if args.baseline == "flow":
        from .baseline_pysteps import advect as _warp
        from .dataset import optical_flow
        now = np.expm1(x[18])
        flow = optical_flow(np.expm1(x[17]), now)
        pred_all = np.stack([
            np.log1p(np.clip(_warp(now, flow, L / 5.0), 0, None))
            for L in model_leads])
        print("rendering PURE OPTICAL-FLOW baseline (no model)")
    elif args.rollout:
        pred_all = _rollout(model, t, yw_root, static, wind, cloud,
                            use_flow, max_steps=len(model_leads),
                            bias=args.rollout_bias)
        print(f"rendering AUTOREGRESSIVE +5min rollout "
              f"({len(model_leads)} steps, bias={args.rollout_bias})")
    else:
        with torch.no_grad():
            out = model(torch.from_numpy(x)[None])
            pred_all = (out[0] if isinstance(out, tuple) else out).numpy()[0]
    input_now = np.expm1(x[18])  # radar frame at issue time t (last of 19)

    cmap = PRECIP_CMAP.copy()
    cmap.set_bad(NO_DATA)
    norm = mcolors.Normalize(0.0, args.vmax)

    fig, (ax_m, ax_p) = plt.subplots(1, 2, figsize=(12, 6.4))
    fig.subplots_adjust(top=0.82, bottom=0.05, wspace=0.04, left=0.02,
                        right=0.90)

    # Playback grid: 0, 5, 10, ..., max_lead
    leads = list(range(0, args.max_lead + 1, args.step_min))

    calib = None
    if args.calibrate:
        calib_path = args.checkpoint.parent / "calibration.npz"
        if calib_path.exists():
            calib = np.load(calib_path)
            print(f"applying calibration from {calib_path}")
        else:
            print(f"WARNING: no calibration at {calib_path}, rendering raw")

    bias_f = None
    if args.bias_calib:
        bpath = args.checkpoint.parent / "bias_calib.npz"
        if bpath.exists():
            b = np.load(bpath)
            bias_f = dict(zip(b["leads"].tolist(), b["factors"].tolist()))
            print(f"applying per-lead bias correction from {bpath}")
        else:
            print(f"WARNING: no bias_calib at {bpath}, rendering raw")

    frames_data = []
    for lead in leads:
        valid_t = t + timedelta(minutes=lead)
        try:
            af = read_frame(_yw_path(yw_root, valid_t))
            measured = np.where(af.valid, af.data, np.nan)
            valid = af.valid
        except FileNotFoundError:
            measured = np.full_like(input_now, np.nan)
            valid = np.zeros_like(input_now, dtype=bool)
        if lead == 0:
            predicted = input_now
        else:
            predicted = np.expm1(np.clip(
                pred_at_lead(pred_all, lead, model_leads), 0, None)) \
                * args.bias_correction
            if bias_f is not None:
                # per-lead multiplicative correction (interpolated between
                # model leads if the playback grid has finer steps)
                if lead in bias_f:
                    predicted = predicted * bias_f[lead]
                else:
                    lo = max((l for l in model_leads if l <= lead),
                             default=model_leads[0])
                    hi = min((l for l in model_leads if l >= lead),
                             default=model_leads[-1])
                    w = 0 if hi == lo else (lead - lo) / (hi - lo)
                    predicted = predicted * (bias_f.get(lo, 1.0) * (1 - w)
                                             + bias_f.get(hi, 1.0) * w)
            if calib is not None:
                # Per-lead quantile mapping in log space. The model only
                # produces discrete leads (5,10,15,20,30,45,60); playback
                # frames at 25/35/40/50/55 are interpolated predictions, so
                # interpolate the calibration curves too - otherwise those
                # frames render raw (exploding) output while neighbours are
                # calibrated, causing visible pops.
                lo = max((l for l in model_leads if l <= lead),
                         default=model_leads[0])
                hi = min((l for l in model_leads if l >= lead),
                         default=model_leads[-1])
                if lo == hi:
                    pq, aq = calib[f"p{lo}"], calib[f"a{lo}"]
                else:
                    w = (lead - lo) / (hi - lo)
                    pq = calib[f"p{lo}"] * (1 - w) + calib[f"p{hi}"] * w
                    aq = calib[f"a{lo}"] * (1 - w) + calib[f"a{hi}"] * w
                predicted = np.expm1(np.interp(np.log1p(predicted), pq, aq))
            if args.sharpen > 0:
                # unsharp mask in rate space: restore fine structure the
                # delta smoothed away (visual only; keep >=0)
                from scipy.ndimage import gaussian_filter
                blur = gaussian_filter(predicted, sigma=1.0)
                predicted = np.clip(
                    predicted + args.sharpen * (predicted - blur), 0, None)
        frames_data.append((lead, valid_t, measured, predicted, valid))

    def draw(i: int) -> None:
        lead, valid_t, measured, predicted, valid = frames_data[i]
        for ax in (ax_m, ax_p):
            ax.clear()
            ax.axis("off")
        ax_m.imshow(np.ma.masked_where(~valid,
                                       np.clip(measured, 0, args.vmax)),
                    cmap=cmap, norm=norm, interpolation="nearest")
        ax_p.imshow(np.ma.masked_where(~valid,
                                       np.clip(predicted, 0, args.vmax)),
                    cmap=cmap, norm=norm, interpolation="nearest")
        tag = " (input)" if lead == 0 else ""
        ax_m.set_title(f"MEASURED  {valid_t:%m-%d %H:%M}Z  "
                       f"(t+{lead} min)", fontsize=12)
        ax_p.set_title(f"PREDICTED{tag}  {valid_t:%m-%d %H:%M}Z  "
                       f"(issued {t:%H:%M}Z)", fontsize=12)
        fig.suptitle(
            f"Nachtigall — one example, issued {t:%Y-%m-%d %H:%M}Z, "
            f"playing 0 → +{args.max_lead} min   "
            f"(white = dry, grey = no data, 0–{args.vmax:g} mm/h)",
            fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.915, 0.15, 0.014, 0.68])
    fig.colorbar(sm, cax=cax, label="mm/h")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ani = anim.FuncAnimation(fig, draw, frames=len(frames_data),
                             init_func=draw(0))
    ani.save(args.out, writer=anim.FFMpegWriter(fps=args.fps, bitrate=2500))
    print(f"wrote {args.out} ({len(frames_data)} frames, issued "
          f"{t:%Y-%m-%d %H:%M}Z, 0 → +{args.max_lead} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())