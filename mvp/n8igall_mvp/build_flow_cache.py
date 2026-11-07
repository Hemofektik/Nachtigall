"""Precompute pysteps Lucas-Kanade flow channels for an index (LK cache).

Live LK costs ~1.8 s/sample (~3.6 h per training epoch) - the cache
makes it a one-time ~20 min build on 16 workers. This is the flow
source of record: on the test set LK beats dense Farneback by
+0.044..0.058 CSI@1.0 at every lead (smooth-lk == LK to 3 decimals,
so LK needs no smoothing), while the model's previous Farneback base
is what loses to plain LK at +5/+10.

Output (ts-keyed, robust against index slicing/reordering):
    <out>.f16     float16 memmap [N, 2, H, W] (u east, v south, px/5min)
    <out>.json    {"ts": [YYYYMMDDHHMM,...sorted], "shape": [2,H,W]}

Degenerate pairs (LK failure / non-finite) are stored as zeros = no
motion (those frames are dry anyway).

Usage:
    python -m n8igall_mvp.build_flow_cache --index data/index_multi.json \
        --out data/index_multi_flow --workers 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_TS_RE = re.compile(r"-(\d{10})-")


def _issue_ts(input_path: str) -> int:
    return int(_TS_RE.search(input_path).group(1))


def _lk_one(task: tuple) -> tuple[int, np.ndarray]:
    """(idx, prev_path, curr_path) -> (idx, flow[2,900,900] float16)."""
    idx, prev_p, curr_p = task
    from pysteps.motion.lucaskanade import dense_lucaskanade
    from .radolan import read_frame
    try:
        a = read_frame(Path(prev_p))
        b = read_frame(Path(curr_p))
        stack = np.stack([np.where(a.valid, np.clip(a.data, 0, None), 0.0),
                          np.where(b.valid, np.clip(b.data, 0, None), 0.0)]
                         ).astype(np.float64)
        flow = dense_lucaskanade(stack)
        if not np.isfinite(flow).all():
            flow = np.zeros((2, *stack.shape[1:]), dtype=np.float32)
    except Exception:                     # noqa: BLE001 - store no-motion
        flow = np.zeros((2, 900, 900), dtype=np.float32)
    return idx, flow.astype(np.float16)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output stem: writes <stem>.f16 + <stem>.json")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)

    samples = json.loads(args.index.read_text())
    ts_list = [_issue_ts(s["inputs"][-1]) for s in samples]
    order = np.argsort(ts_list, kind="stable")   # sorted slot -> sample idx
    n = len(samples)
    ts_sorted = [ts_list[i] for i in order]
    print(f"{n} samples, {args.workers} workers", flush=True)

    # infer grid shape from the first sample
    from .radolan import read_frame
    f0 = read_frame(Path(samples[0]["inputs"][-1]))
    H, W = f0.data.shape

    f16 = args.out.with_suffix(".f16")
    meta = args.out.with_suffix(".json")
    out = np.memmap(f16, dtype=np.float16, mode="w+",
                    shape=(n, 2, H, W))

    # each task carries its *sorted slot* so rows align with ts_sorted
    tasks = [(int(slot), samples[i]["inputs"][-2], samples[i]["inputs"][-1])
             for slot, i in enumerate(order)]
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for slot, flow in pool.imap_unordered(_lk_one, tasks, chunksize=8):
            out[slot] = flow
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"  {done}/{n} ({el:.0f}s, "
                      f"{el/done*(n-done)/60:.0f} min left)", flush=True)
    out.flush()
    del out

    meta.write_text(json.dumps({
        "index": str(args.index),
        "shape": [2, H, W],
        "n": n,
        "ts": ts_sorted,
        "estimator": "pysteps_dense_lucaskanade",
    }) + "\n")
    print(f"done: {f16} ({f16.stat().st_size/1e9:.1f} GB) + {meta} "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    # Pool workers import this module as __mp_main__
    sys.exit(main())
