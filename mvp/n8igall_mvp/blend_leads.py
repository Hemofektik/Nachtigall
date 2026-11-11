"""Roadmap item 1 (review at 13a87fb): fixed lead-wise blend of the
multi_lk_cloud U-Net with pure LK advection.

One scalar weight per lead: ``pred = w * model + (1 - w) * lk`` in rate
space, thresholded afterwards. Weights are fit per lead on the training
index's chronological validation split - never on the frozen test set -
then applied once to ``index_test_rt``.

Falsification check (roadmap): the blend must separate from the better
single system by more than the day-level spread; per-day paired deltas
at CSI@1.0 are reported for exactly that.

The LK field comes from the same ts-keyed flow cache the model's flow
channels use, so both systems see identical motion information.

Usage:
    python -m n8igall_mvp.blend_leads \
        --checkpoint runs/multi_lk_cloud/best.pt \
        --out runs/multi_lk_cloud/blend_lk.json

Resume: if ``--out`` already contains fit results, the fit pass is
skipped and the stored weights are applied to the test set.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Sample
from .model import UNet
from .train import RadolanDataset

THRESHOLDS = (0.1, 1.0, 5.0)
TI = THRESHOLDS.index(1.0)  # selection metric
_TS_RE = re.compile(r"-(\d{10})-")


class FlowCache:
    """ts-keyed LK flow memmap (same layout as build_flow_cache)."""

    def __init__(self, stem: str):
        meta = json.loads(Path(stem + ".json").read_text())
        self.ts = np.asarray(meta["ts"], dtype=np.int64)
        self.shape = (len(self.ts), *meta["shape"])
        self.stem = stem
        self._mm = None

    def get(self, key: int) -> np.ndarray | None:
        i = int(np.searchsorted(self.ts, key))
        if i >= len(self.ts) or self.ts[i] != key:
            return None
        if self._mm is None:
            self._mm = np.memmap(self.stem + ".f16", dtype=np.float16,
                                 mode="r", shape=self.shape)
        return np.asarray(self._mm[i], dtype=np.float32)


def advect_stack(frame: np.ndarray, flow: np.ndarray,
                 steps: list[float]) -> np.ndarray:
    """Warp ``frame`` [H,W] by ``flow`` [2,H,W] for every lead in one
    batched grid_sample (same math as baseline_pysteps.advect, which
    this replaces lead-by-lead for speed). Returns [L,H,W]."""
    import torch.nn.functional as F
    H, W = frame.shape
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W),
                            indexing="ij")
    st = torch.as_tensor(steps, dtype=torch.float32)          # [L]
    fu = torch.as_tensor(flow[0], dtype=torch.float32)        # [H,W]
    fv = torch.as_tensor(flow[1], dtype=torch.float32)
    sx = xs[None].float() - fu[None] * st[:, None, None]      # [L,H,W]
    sy = ys[None].float() - fv[None] * st[:, None, None]
    gx = (sx / (W - 1)) * 2 - 1
    gy = (sy / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                      # [L,H,W,2]
    img = torch.as_tensor(frame, dtype=torch.float32)[None].expand(
        len(steps), -1, -1)
    return F.grid_sample(img[:, None], grid, mode="bilinear",
                         padding_mode="zeros",
                         align_corners=False)[:, 0].numpy()


def _counts(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> dict:
    p, o = pred & mask, obs & mask
    return {"hits": int((p & o).sum()), "fa": int((p & ~o).sum()),
            "miss": int((~p & o).sum()), "n": int(mask.sum())}


def _add(dst: dict, src: dict) -> None:
    for k in ("hits", "fa", "miss", "n"):
        dst[k] = dst.get(k, 0) + src[k]


def _csi(c: dict) -> float:
    """Pooled CSI = hits / (hits + fa + miss) — same formula as
    train.evaluate's ``_pooled`` keys (an earlier version dropped fa
    from the denominator, silently fitting POD instead of CSI)."""
    d = c.get("hits", 0) + c.get("fa", 0) + c.get("miss", 0)
    return c.get("hits", 0) / d if d else float("nan")


def _flow_stem(index: Path) -> str:
    return str(index.with_suffix("")) + "_flow"


def _build_model(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    leads = ckpt.get("leads")
    model = UNet(ckpt["in_channels"], base=ckpt["base"], depth=ckpt["depth"],
                 out_channels=len(leads) if leads else 1,
                 residual_from=ckpt["residual_from"],
                 delta_cap=ckpt["delta_cap"],
                 advect=ckpt["advect"],
                 exceedance=ckpt.get("exceedance"))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, ckpt, leads, device


def _make_pass(index: Path, leads: list[int], args, static, wind, cloud,
               subset: list[Sample] | None = None):
    samples = subset or [Sample(**s) for s in json.loads(index.read_text())]
    stem = _flow_stem(index)
    ds = RadolanDataset(samples, crop=None, static_channels=static,
                        wind_sampler=wind, leads=leads,
                        cloud_sampler=cloud, flow=True,
                        flow_cache=stem, flow_estimator="lk")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4)
    return samples, dl, FlowCache(stem)


def _run_pass(model, dl, samples, flowcache, leads, device, radar_ch: int,
              tag: str,
              weights: np.ndarray | None = None,
              chosen: np.ndarray | None = None,
              collect_days: bool = False):
    """One full pass. ``weights`` = fit grid, ``chosen`` = per-lead w."""
    n = len(samples)
    fit_acc = ([[{} for _ in THRESHOLDS] for _ in leads]
               for _ in (weights if weights is not None else ()))
    fit_acc = list(fit_acc)
    test_acc = {s: [[{} for _ in THRESHOLDS] for _ in leads]
                for s in ("blend", "model", "lk")}
    day_acc: dict[str, dict[str, list[dict]]] = {}
    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y, mask in dl:
            B = x.shape[0]
            out = model(x.to(device))
            pred = out[0] if isinstance(out, tuple) else out
            pr = torch.expm1(pred.clamp(min=0)).cpu().numpy()   # [B,L,H,W]
            yr = torch.expm1(y).numpy()
            m = mask.numpy().astype(bool)
            last = torch.expm1(x[:, radar_ch - 1]).numpy()      # [B,H,W]
            for b in range(B):
                s = samples[idx]
                idx += 1
                key = int(_TS_RE.search(s.inputs[-1]).group(1))
                flow = flowcache.get(key)
                if flow is None:
                    raise RuntimeError(f"flow cache miss at {key}")
                lk = advect_stack(last[b], flow,
                                  [L / 5.0 for L in leads])     # [L,H,W]
                if weights is not None:
                    for wi, w in enumerate(weights):
                        blend = w * pr[b] + (1.0 - w) * lk
                        for i in range(len(leads)):
                            o = (yr[b][i] > THRESHOLDS[TI]) & m[b][i]
                            for ti, t in enumerate(THRESHOLDS):
                                _add(fit_acc[wi][i][ti],
                                     _counts((blend[i] > t) & m[b][i],
                                             (yr[b][i] > t) & m[b][i],
                                             m[b][i]))
                else:
                    w = chosen.astype(np.float32)[:, None, None]
                    blend = w * pr[b] + (1.0 - w) * lk
                    day = str(key)[:6] if collect_days else None
                    for sysname, arr in (("blend", blend),
                                         ("model", pr[b]), ("lk", lk)):
                        for i in range(len(leads)):
                            for ti, t in enumerate(THRESHOLDS):
                                _add(test_acc[sysname][i][ti],
                                     _counts((arr[i] > t) & m[b][i],
                                             (yr[b][i] > t) & m[b][i],
                                             m[b][i]))
                            if collect_days:
                                d = day_acc.setdefault(
                                    day, {s2: [{} for _ in leads]
                                          for s2 in ("blend", "model", "lk")})
                                _add(d[sysname][i],
                                     _counts((arr[i] > THRESHOLDS[TI])
                                             & m[b][i],
                                             (yr[b][i] > THRESHOLDS[TI])
                                             & m[b][i],
                                             m[b][i]))
                if idx % 200 == 0 or idx == n:
                    el = time.time() - t0
                    print(f"[{tag}] {idx}/{n}  {el / idx:.1f}s/sample",
                          flush=True)
    if weights is not None:
        return fit_acc
    return test_acc, day_acc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/multi_lk_cloud/best.pt"))
    p.add_argument("--fit-index", type=Path,
                   default=Path("data/index_multi.json"))
    p.add_argument("--test-index", type=Path,
                   default=Path("data/index_test_rt.json"))
    p.add_argument("--fit-fraction", type=float, default=0.2,
                   help="chronological tail of fit-index used for fitting "
                        "(matches train.py val_fraction)")
    p.add_argument("--fit-stride", type=int, default=2,
                   help="take every Nth fit sample")
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--static", type=Path,
                   default=Path("data/static/channels.npz"))
    p.add_argument("--wind", type=Path,
                   default=Path("data/wind/era5_wind.npz"))
    p.add_argument("--cloud", type=Path,
                   default=Path("data/wind/era5_cloud.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/multi_lk_cloud/blend_lk.json"))
    args = p.parse_args(argv)

    model, ckpt, leads, device = _build_model(args.checkpoint)
    radar_ch = len(json.loads(args.fit_index.read_text())[0]["inputs"])
    static = np.load(args.static)["channels"].astype(np.float32)
    from .wind import WindSampler
    wind = WindSampler(args.wind, elevation=np.load(args.static)["elevation"])
    cloud = None
    if ckpt["in_channels"] >= 32:
        from .cloud import CloudSampler
        cloud = CloudSampler(args.cloud)

    out: dict = {}
    if args.out.exists():
        out = json.loads(args.out.read_text())

    weights = np.arange(0.0, 1.0 + 1e-9, args.weight_step)

    # ---- fit pass (validation tail of the training index, off test) ----
    if "fit" in out:
        chosen = np.asarray(out["fit"]["chosen_w"], dtype=np.float64)
        print(f"reusing fitted weights from {args.out}", flush=True)
    else:
        fit_samples = [Sample(**s) for
                       s in json.loads(args.fit_index.read_text())]
        n_val = max(1, int(len(fit_samples) * args.fit_fraction))
        fit_val = fit_samples[-n_val:][::args.fit_stride]
        print(f"fit pass: {len(fit_val)} samples "
              f"(tail {n_val} of {len(fit_samples)}, "
              f"stride {args.fit_stride})", flush=True)
        _, dl, fc = _make_pass(args.fit_index, leads, args, static, wind,
                               cloud, subset=fit_val)
        fit_acc = _run_pass(model, dl, fit_val, fc, leads, device, radar_ch,
                            tag="fit", weights=weights)
        fit_csi = [[_csi(fit_acc[wi][i][TI]) for i in range(len(leads))]
                   for wi in range(len(weights))]
        chosen = np.array([weights[int(np.nanargmax([fit_csi[wi][i]
                                                      for wi
                                                      in range(len(weights))]))]
                           for i in range(len(leads))])
        out["fit"] = {
            "index": str(args.fit_index), "n": len(fit_val),
            "fraction": args.fit_fraction, "stride": args.fit_stride,
            "weights": [float(w) for w in weights],
            "leads": leads,
            "csi_at_1.0": fit_csi,
            "chosen_w": [float(w) for w in chosen],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print("fit done, weights:", " ".join(f"{w:.2f}" for w in chosen),
              flush=True)

    # ---- test pass (frozen test set, weights fixed) ----
    if "test" in out:
        print("test results already present", flush=True)
    else:
        print(f"test pass: {args.test_index}", flush=True)
        t_samples, dl, fc = _make_pass(args.test_index, leads, args, static,
                                       wind, cloud)
        test_acc, day_acc = _run_pass(model, dl, t_samples, fc, leads,
                                      device, radar_ch, tag="test",
                                      chosen=chosen, collect_days=True)

        def pack(acc):
            return {s: [[acc[s][i][ti] for ti in range(len(THRESHOLDS))]
                        for i in range(len(leads))]
                    for s in acc}

        pooled = pack(test_acc)
        # save raw results BEFORE the summary so a summary bug can't
        # cost another full pass (learned the hard way)
        out["test"] = {
            "index": str(args.test_index), "n": len(t_samples),
            "thresholds": list(THRESHOLDS), "leads": leads,
            "pooled": pooled,
            "per_day": {d: {s: [day_acc[d][s][i] for i in range(len(leads))]
                            for s in day_acc[d]} for d in sorted(day_acc)},
            "day_stats": {},
        }
        args.out.write_text(json.dumps(out, indent=2))
        # falsification stats: per-day paired deltas at CSI@1.0
        day_stats = {}
        better_sys = {}
        for i, L in enumerate(leads):
            cm, cl = _csi(test_acc["model"][i][TI]), _csi(test_acc["lk"][i][TI])
            better = "model" if cm >= cl else "lk"
            better_sys[L] = better
            deltas = []
            for day, per_sys in day_acc.items():
                # day accumulators hold counts at TI only (no thr dim)
                cb = _csi(per_sys["blend"][i])
                co = _csi(per_sys[better][i])
                if np.isfinite(cb) and np.isfinite(co):
                    deltas.append(cb - co)
            deltas = np.asarray(deltas)
            day_stats[str(L)] = {
                "better_single": better,
                "pooled_delta": _csi(test_acc["blend"][i][TI])
                - _csi(test_acc[better][i][TI]),
                "day_delta_mean": float(deltas.mean()) if len(deltas) else None,
                "day_delta_std": float(deltas.std()) if len(deltas) else None,
                "day_wins": int((deltas > 0).sum()),
                "n_days": int(len(deltas)),
            }
        out["test"]["day_stats"] = day_stats
        args.out.write_text(json.dumps(out, indent=2))

    # ---- summary ----
    fit, test = out["fit"], out.get("test")
    print("\nlead   w*   fit_csi", flush=True)
    for i, L in enumerate(leads):
        w = fit["chosen_w"][i]
        wi = int(round(w / args.weight_step))
        c = fit["csi_at_1.0"][wi][i]
        print(f"{L:>4}  {w:.2f}  {c:.3f}", flush=True)
    if test:
        print("\nlead  blend  model    lk   delta*  day-win  (CSI@1.0)",
              flush=True)
        for i, L in enumerate(leads):
            cb = _csi(test["pooled"]["blend"][i][TI])
            cm = _csi(test["pooled"]["model"][i][TI])
            ck = _csi(test["pooled"]["lk"][i][TI])
            st = test["day_stats"][str(L)]
            print(f"{L:>4}  {cb:.3f}  {cm:.3f}  {ck:.3f}  "
                  f"{st['pooled_delta']:+.3f}  "
                  f"{st['day_wins']}/{st['n_days']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())