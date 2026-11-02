"""Build the untouched real-time test index (item 3 of the review plan).

The sync daemon has been mirroring real-time radar since 2026-10-22. That
data is *truly* out-of-sample: it postdates all training data (Jun-Sep 2024),
comes through the real inference pipeline (unadjusted real-time RADKLIM
HDF5 on de1200, reprojected to rx by read_hdf5) instead of the reprocessed
classic archive, and was never seen by any checkpoint or design decision.

Samples mirror the training index format: 19 YW frames at 5-min spacing
(90 min history) ending at t, targets = YW rate at each lead 5..60 min.

Usage:
    python -m n8igall_mvp.build_test_index --rt-dir data/raw/rt \
        --out data/index_test_rt.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from .dataset import _yw_path
from .radolan import read_frame

LEADS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
N_HISTORY = 19          # 90 min at 5-min spacing, same as training (index_B)
SPACING_MIN = 5


def build(rt_dir: Path, start: datetime, end: datetime,
          issue_stride_min: int = 5, rain_score: bool = False) -> list[dict]:
    samples = []
    t = start
    while t <= end:
        inputs = [_yw_path(rt_dir, t - timedelta(minutes=SPACING_MIN * (N_HISTORY - 1 - i)))
                  for i in range(N_HISTORY)]
        if all(p.exists() for p in inputs):
            # every lead target must exist too
            targets = [_yw_path(rt_dir, t + timedelta(minutes=L)) for L in LEADS]
            if all(p.exists() for p in targets):
                rec: dict = {"inputs": [str(p) for p in inputs],
                             "target": str(targets[-1])}
                if rain_score:
                    # rain fraction of the issue-time frame: cheap proxy for
                    # "will this sample contain rain" (sampler weight)
                    f = read_frame(inputs[-1])
                    rec["rain"] = float((f.data[f.valid] > 0.1).mean())
                samples.append(rec)
        t += timedelta(minutes=issue_stride_min)
    return samples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rt-dir", type=Path,
                   default=Path("data/raw/rt"))
    p.add_argument("--start", type=datetime.fromisoformat,
                   default=datetime(2026, 10, 23))
    p.add_argument("--end", type=datetime.fromisoformat,
                   default=datetime(2026, 11, 1, 23, 55))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--issue-stride", type=int, default=5,
                   help="minutes between issue times (5 = every frame)")
    p.add_argument("--rain-score", action="store_true",
                   help="store per-sample rain fraction for biased sampling")
    args = p.parse_args(argv)

    samples = build(args.rt_dir, args.start, args.end,
                    issue_stride_min=args.issue_stride,
                    rain_score=args.rain_score)
    args.out.write_text(json.dumps(samples, indent=1) + "\n")
    print(f"{len(samples)} test samples "
          f"({args.start:%Y-%m-%d} .. {args.end:%Y-%m-%d}) -> {args.out}")
    if samples:
        print(f"  first: {samples[0]['inputs'][-1]}")
        print(f"  last:  {samples[-1]['inputs'][-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
