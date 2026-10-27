"""Render a prediction-vs-actual comparison video from a trained checkpoint.

Two synchronized panels for a fixed lead time (default +60 min):

    PREDICTION (issued at t)     |     ACTUAL (observed at t + lead)

Both panels advance at the native 5-minute radar cadence, so motion is
smooth; both titles show the shared valid time for easy comparison.

Usage:
    python -m n8igall_mvp.render_video \
        --start 2024-09-08T22:00 --hours 6 --lead 60
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
# No-data cells are rendered separately in neutral grey by the masked array.
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
     (1.00, "#8b1030")],  # >= vmax
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
    p.add_argument("--step-min", type=int, default=5,
                   help="frame cadence; data is native 5-min")
    p.add_argument("--lead", type=int, default=60)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=_PKG_ROOT / "output" / "pred_vs_actual.mp4")
    args = p.parse_args(argv)

    yw_root = _PKG_ROOT / "data" / "raw" / "yw"
    lead_idx = ALL_LEADS.index(args.lead)

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

    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.subplots_adjust(top=0.84, bottom=0.04, wspace=0.02, left=0.02,
                        right=0.92)

    def draw(i: int) -> None:
        t, pred, actual, valid = frames[i]
        valid_t = t + timedelta(minutes=args.lead)
        for ax in (ax_p, ax_a):
            ax.clear()
            ax.axis("off")
        ax_p.imshow(np.ma.masked_where(~valid, np.clip(pred, 0, args.vmax)),
                    cmap=cmap, norm=norm, interpolation="nearest")
        ax_a.imshow(np.ma.masked_where(~valid, np.clip(actual, 0, args.vmax)),
                    cmap=cmap, norm=norm, interpolation="nearest")
        ax_p.set_title(f"PREDICTED +{args.lead} min  "
                       f"(issued {t:%m-%d %H:%M}Z)", fontsize=12)
        ax_a.set_title(f"ACTUAL  (valid {valid_t:%m-%d %H:%M}Z)",
                       fontsize=12)
        fig.suptitle(
            f"Nachtigall nowcast vs radar observation — valid time "
            f"{valid_t:%Y-%m-%d %H:%M}Z   "
            f"(white = dry, grey = no data, scale 0–{args.vmax:g} mm/h)",
            fontsize=12)

    # Build frames: issue times every --step-min; skip gaps in data.
    frames = []
    t = args.start
    end = t + timedelta(hours=args.hours)
    while t < end:
        valid_t = t + timedelta(minutes=args.lead)
        try:
            x = assemble_input(yw_root, t, static, wind)
            with torch.no_grad():
                pred = model(torch.from_numpy(x)[None]).numpy()[0, lead_idx]
            af = read_frame(_yw_path(yw_root, valid_t))
            actual = np.where(af.valid, af.data, np.nan)
            frames.append((t, pred, actual, af.valid))
        except FileNotFoundError:
            pass  # leave a gap rather than a wrong frame
        t += timedelta(minutes=args.step_min)

    if not frames:
        raise SystemExit("no frames rendered (missing data in window?)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=[ax_p, ax_a], shrink=0.8, pad=0.02,
                 label="mm/h")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ani = anim.FuncAnimation(fig, draw, frames=len(frames), init_func=draw(0))
    ani.save(args.out, writer=anim.FFMpegWriter(fps=args.fps, bitrate=2500))
    print(f"wrote {args.out} ({len(frames)} frames @ {args.step_min} min, "
          f"+{args.lead} lead, {args.start:%Y-%m-%d %H:%M}Z +{args.hours}h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())