"""Training dataset assembly for the +60 min precipitation nowcast.

A sample is:

* inputs: ``n_history`` YW (5-min rate) frames at ``spacing_min`` spacing,
  ending at time ``t`` (so the history spans ``(n_history-1)*spacing_min``),
* target: the RW hourly accumulation valid over ``(t, t + lead_min]``,
  i.e. the RW file stamped ``t + lead_min`` (RW file ``HH00`` covers
  ``(HH-1, HH]``).

Both products are read from RADKLIM classic bin files, so they share the
900x900 national grid and need no reprojection.

The index is a JSON list of samples; frames are loaded lazily by the
PyTorch dataset to keep memory bounded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .radolan import read_frame

TS_FMT = "%Y%m%d%H%M"


@dataclass(frozen=True)
class Sample:
    inputs: list[str]  # YW file paths, oldest first
    target: str  # RW file path


def _yw_path(yw_dir: Path, t: datetime) -> Path:
    name = f"raa01-yw_10000-{t.strftime('%y%m%d%H%M')}-dwd---bin"
    flat = yw_dir / name
    if flat.exists():
        return flat
    return yw_dir / t.strftime("%y%m%d") / name


def _rw_path(rw_dir: Path, t: datetime) -> Path:
    """RW target path; historical months extract uncompressed (no .gz)."""
    stem = f"raa01-rw_10000-{t.strftime('%y%m%d%H%M')}-dwd---bin"
    for base in (rw_dir / t.strftime("%y%m%d"), rw_dir):  # per-day, then flat
        for candidate in (base / f"{stem}.gz", base / stem):
            if candidate.exists():
                return candidate
    return rw_dir / t.strftime("%y%m%d") / f"{stem}.gz"


def build_index(yw_dir: Path, rw_dir: Path, start: datetime, end: datetime,
                n_history: int = 4, spacing_min: int = 30,
                lead_min: int = 60) -> list[Sample]:
    """Enumerate all samples whose inputs and target exist on disk."""
    samples: list[Sample] = []
    t = start
    while t <= end:
        inputs = [_yw_path(yw_dir, t - timedelta(minutes=spacing_min * (n_history - 1 - i)))
                  for i in range(n_history)]
        target = _rw_path(rw_dir, t + timedelta(minutes=lead_min))
        if all(p.exists() for p in inputs) and target.exists():
            samples.append(Sample(inputs=[str(p) for p in inputs], target=str(target)))
        t += timedelta(minutes=spacing_min)
    return samples


def load_sample(sample: Sample,
                wind_sampler=None,
                leads: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one sample.

    With ``leads`` (minutes), the target is the YW rate stack at each lead
    time (shape [len(leads), H, W]); otherwise the single RW +60 min target
    ([1, H, W]).

    Returns:
        x: float32 [C, H, W] log1p radar history (+ optional wind channels),
           invalid pixels = 0
        y: float32 [C_out, H, W] target rate (mm/h), invalid pixels = 0
        mask: bool [C_out, H, W] target validity mask
    """
    frames = [read_frame(p) for p in sample.inputs]
    x = np.stack([np.where(f.valid, np.log1p(np.clip(f.data, 0, None)), 0.0)
                  for f in frames]).astype(np.float32)

    if wind_sampler is not None:
        # Wind valid over the lead-time window, anchored at the last input
        # frame ("now"); channels are appended after the radar history.
        t_now = datetime.strptime(frames[-1].timestamp, "%y%m%d%H%M")
        x = np.concatenate([x, wind_sampler.channels_for(t_now)], axis=0)

    if leads is None:
        tgt = read_frame(sample.target)
        y = np.where(tgt.valid, np.clip(tgt.data, 0, None), 0.0)[None].astype(np.float32)
        mask = tgt.valid[None]
        return x, y, mask

    t0 = datetime.strptime(frames[-1].timestamp, "%y%m%d%H%M")
    # inputs live either in <yw>/<yymmdd>/<file> or legacy <yw>/<file>
    p0 = Path(sample.inputs[0])
    yw_root = p0.parent.parent if (p0.parent.name.isdigit()
                                   and len(p0.parent.name) == 6) else p0.parent
    ys, ms = [], []
    for lead in leads:
        try:
            tgt = read_frame(_yw_path(yw_root, t0 + timedelta(minutes=lead)))
            ys.append(np.where(tgt.valid, np.clip(tgt.data, 0, None), 0.0))
            ms.append(tgt.valid)
        except FileNotFoundError:
            ys.append(np.zeros(x.shape[-2:], dtype=np.float32))
            ms.append(np.zeros(x.shape[-2:], dtype=bool))
    return x, np.stack(ys).astype(np.float32), np.stack(ms)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yw-dir", type=Path, required=True)
    p.add_argument("--rw-dir", type=Path, required=True)
    p.add_argument("--start", type=datetime.fromisoformat, required=True)
    p.add_argument("--end", type=datetime.fromisoformat, required=True)
    p.add_argument("--n-history", type=int, default=4)
    p.add_argument("--spacing-min", type=int, default=30)
    p.add_argument("--lead-min", type=int, default=60)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    samples = build_index(args.yw_dir, args.rw_dir, args.start, args.end,
                          args.n_history, args.spacing_min, args.lead_min)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(s) for s in samples], indent=1))
    print(f"{len(samples)} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())