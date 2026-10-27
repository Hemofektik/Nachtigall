"""Render a prediction-vs-actual comparison video from a trained checkpoint.

One row per lead time (default 5,10,15,20,30,45,60 min):

    row L:   PREDICTION issued at t (valid at t+L)  |  ACTUAL at t+L

All rows share one issue time ``t``; every row's two panels show the same
valid time, so prediction and measurement are directly comparable. Frames
advance at the native 5-minute radar cadence.

Usage:
    python -m n8igall_mvp.render_video \
        --start 2024-09-08T22:00 --hours 6 --leads 5,10,15,30,60
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
ALL_LEADS = [5, 10, 15, 20, 30, 45, 60]

# Precipitation colormap: white (dry) -> blue -> green -> yellow -> red.
PRECIP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "precip",
    [(0.00, "#ffffff"),
     (0.02, "#dbeeff"),   # ~0.2 mm/h
     (0.06, "#a6d4ff"),
     (0.12, "#5aa9f0"),
     (0.20, "#2e7ac0"),
     (0.35, "#2fa060"),   # ~3 mm/h
     (0.50, "#d8d93a"),
     (0.65, "#f0922e"),
     (0.80, "#e04030"),
     (1.00, "#8b1030")],
)
NO_DATA = "#dcdcdc"


def assemble_input(yw_root: Path, t: datetime, static: np.ndarray,
                   wind: WindSampler) -> np.ndarray:
    """[29, H, W] exactly as training: 19 radar + 4 static + 6 wind."""
    radar = []
    for i in range(19):
        f = read_frame(_yw_path(yw_root, t - timedelta(minutes=5 * (18 - i))))
        radar.append(np.where(f.valid, np.log1p(np.clip(f.data, 0, None)), 0.0))
    return np.concatenate([np.stack(radar).astype(np.float32), static,
                           wind.channels_for(t)], axis=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=_PKG_ROOT / "runs" / "multilead" / "last.pt")
    p.add_argument("--start", type=datetime.fromisoformat,
                   default=datetime(2024, 9, 8, 22, 0))
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--step-min", type=int, default=5)
    p.add_argument("--leads", default="5,10,15,20,30,45,60")
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=_PKG_ROOT / "output" / "pred_vs_actual.mp4")
    args = p.parse_args(argv)

    yw_root = _PKG_ROOT / "data" / "raw" / "yw"
    leads = [int(v) for v in args.leads.split(",")]
    lead_idx = {L: ALL_LEADS.index(L) for L in leads}

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = UNet(ckpt["in_channels"], base=ckpt["base"],
                 out_channels=len(ckpt.get("leads") or ALL_LEADS))
    model.load_state_dict(ckpt["model"])
    model.eval()

    static_npz = np.load(_PKG_ROOT / "data" / "static" / "channels.npz")
    static = static_npz["channels"].astype(np.float32)
    wind = WindSampler(_PKG_ROOT / "data" / "wind" / "era5_wind.npz",
                       elevation=static_npz["elevation"])

    cmap = PRECIP_CMAP.copy()
    cmap.set_bad(NO_DATA)
    norm = mcolors.Normalize(0.0, args.vmax)

    n_rows = len(leads)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.5, 2.7 * n_rows + 1.2))
    fig.subplots_adjust(top=1 - 0.55 / (2.7 * n_rows + 1.2), bottom=0.03,
                        wspace=0.02, hspace=0.35, left=0.02, right=0.90)

    def show(ax, arr, valid, title):
        ax.imshow(np.ma.masked_where(~valid, np.clip(arr, 0, args.vmax)),
                  cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    def draw(i: int) -> None:
        t, preds, actuals, vmasks = frames[i]
        for r, L in enumerate(leads):
            valid_t = t + timedelta(minutes=L)
            pred, act, vm = preds[L], actuals[L], vmasks[L]
            if r == 0:
                show(axes[r][0], pred, vm,
                     f"PREDICTED +{L} min (issued {t:%H:%M}Z)")
                show(axes[r][1], act, vm, f"ACTUAL +{L} min ({valid_t:%H:%M}Z)")
            else:
                show(axes[r][0], pred, vm, f"+{L} min prediction")
                show(axes[r][1], act, vm, f"+{L} min actual ({valid_t:%H:%M}Z)")
        fig.suptitle(
            f"Nachtigall nowcast vs radar — issued {t:%Y-%m-%d %H:%M}Z, "
            f"row = lead time   (white = dry, grey = no data, "
            f"0–{args.vmax:g} mm/h)", fontsize=11.5, y=0.985)

    # Frames: one issue time every --step-min; skip if any needed file missing.
    frames = []
    t = args.start
    end = t + timedelta(hours=args.hours)
    while t < end:
        try:
            x = assemble_input(yw_root, t, static, wind)
            with torch.no_grad():
                pred_all = model(torch.from_numpy(x)[None]).numpy()[0]
            preds, actuals, vmasks = {}, {}, {}
            for L in leads:
                af = read_frame(_yw_path(yw_root, t + timedelta(minutes=L)))
                preds[L] = pred_all[lead_idx[L]]
                actuals[L] = np.where(af.valid, af.data, np.nan)
                vmasks[L] = af.valid
            frames.append((t, preds, actuals, vmasks))
        except FileNotFoundError:
            pass
        t += timedelta(minutes=args.step_min)

    if not frames:
        raise SystemExit("no frames rendered (missing data in window?)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.915, 0.15, 0.014, 0.7])
    fig.colorbar(sm, cax=cax, label="mm/h")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ani = anim.FuncAnimation(fig, draw, frames=len(frames), init_func=draw(0))
    ani.save(args.out, writer=anim.FFMpegWriter(fps=args.fps, bitrate=2500))
    print(f"wrote {args.out} ({len(frames)} frames @ {args.step_min} min, "
          f"leads={leads}, {args.start:%Y-%m-%d %H:%M}Z +{args.hours}h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())