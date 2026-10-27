"""Render a prediction-vs-actual comparison video from a trained checkpoint.

For each valid time ``t`` in a window, renders a grid:

    NOW (radar at t) | PREDICTION at t+L | ACTUAL at t+L     for each lead L

using the same input assembly as training (19x5-min radar history +
static elevation channels + ERA5 wind channels).

Usage:
    python -m n8igall_mvp.render_video \
        --checkpoint runs/multilead/last.pt \
        --start 2024-09-08T22:00 --hours 6 --step-min 15 \
        --leads 5,15,30,60 --out output/pred_vs_actual.mp4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .dataset import _yw_path
from .model import UNet
from .radolan import read_frame
from .wind import WindSampler

_PKG_ROOT = Path(__file__).resolve().parent.parent
ALL_LEADS = [5, 10, 15, 20, 30, 45, 60]


def assemble_input(yw_root: Path, t: datetime, static: np.ndarray,
                   wind: WindSampler) -> np.ndarray:
    """[29, H, W] exactly as training: 19 radar + 4 static + 6 wind."""
    radar = []
    for i in range(19):
        f = read_frame(_yw_path(yw_root, t - timedelta(minutes=5 * (18 - i))))
        radar.append(np.where(f.valid, np.log1p(np.clip(f.data, 0, None)), 0.0))
    x = np.concatenate([np.stack(radar).astype(np.float32), static,
                        wind.channels_for(t)], axis=0)
    return x


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=_PKG_ROOT / "runs" / "multilead" / "last.pt")
    p.add_argument("--index", type=Path,
                   default=_PKG_ROOT / "data" / "index_B.json")
    p.add_argument("--start", type=datetime.fromisoformat,
                   default=datetime(2024, 9, 8, 22, 0))
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--step-min", type=int, default=15)
    p.add_argument("--leads", default="5,15,30,60")
    p.add_argument("--vmax", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--out", type=Path,
                   default=_PKG_ROOT / "output" / "pred_vs_actual.mp4")
    args = p.parse_args(argv)

    yw_root = _PKG_ROOT / "data" / "raw" / "yw"
    leads = [int(v) for v in args.leads.split(",")]
    lead_idx = [ALL_LEADS.index(L) for L in leads]  # model output channels

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = UNet(ckpt["in_channels"], base=ckpt["base"],
                 out_channels=len(ckpt.get("leads") or ALL_LEADS))
    model.load_state_dict(ckpt["model"])
    model.eval()

    static_npz = np.load(_PKG_ROOT / "data" / "static" / "channels.npz")
    static = static_npz["channels"].astype(np.float32)
    wind = WindSampler(_PKG_ROOT / "data" / "wind" / "era5_wind.npz",
                       elevation=static_npz["elevation"])

    # Shared color scale; invalid composite pixels get a light-gray color.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e8e8e8")
    norm = matplotlib.colors.Normalize(0.0, args.vmax)

    def show(ax, arr, valid, title):
        disp = np.ma.masked_where(~valid, np.clip(arr, 0, args.vmax))
        ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    n_rows = len(leads)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.2 * n_rows))
    fig.suptitle("Nachtigall nowcast: NOW | predicted | observed "
                 "(mm/h, radar composite)", fontsize=12)

    frames = []
    t = args.start
    end = t + timedelta(hours=args.hours)
    while t < end:
        try:
            x = assemble_input(yw_root, t, static, wind)
            with torch.no_grad():
                pred = model(torch.from_numpy(x)[None]).numpy()[0]  # [7,H,W]
            now_f = read_frame(_yw_path(yw_root, t))
            actuals = {}
            for L in leads:
                af = read_frame(_yw_path(yw_root, t + timedelta(minutes=L)))
                actuals[L] = (np.where(af.valid, af.data, np.nan), af.valid)
        except FileNotFoundError:
            t += timedelta(minutes=args.step_min)
            continue
        frames.append((t, now_f, pred, actuals))
        t += timedelta(minutes=args.step_min)

    if not frames:
        raise SystemExit("no frames rendered (missing data in window?)")

    def draw(i):
        t, now_f, pred, actuals = frames[i]
        for r, L in enumerate(leads):
            row = axes[r]
            if r == 0:
                show(row[0], np.where(now_f.valid, now_f.data, np.nan),
                     now_f.valid, f"NOW  {t:%m-%d %H:%M}Z")
            else:
                row[0].axis("off")
            pv = now_f.valid  # predictions outside coverage are meaningless
            show(row[1], np.where(pv, pred[lead_idx[r]], np.nan), pv,
                 f"pred +{L} min  (max {np.nanmax(np.where(pv, pred[lead_idx[r]], np.nan)):.1f})")
            act, av = actuals[L]
            show(row[2], act, av,
                 f"actual +{L} min  (max {np.nanmax(act[av]):.1f})"
                 if av.any() else f"actual +{L} min (no data)")
        fig.subplots_adjust(top=0.93, wspace=0.04, hspace=0.18)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ani = anim.FuncAnimation(fig, draw, frames=len(frames), init_func=draw(0))
    ani.save(args.out, writer=anim.FFMpegWriter(fps=args.fps, bitrate=1500))
    print(f"wrote {args.out} ({len(frames)} frames, "
          f"leads={leads}, {args.start:%Y-%m-%d %H:%M}Z +{args.hours}h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())