# Baselines and verification metrics — research findings

Research for the approved action plan (items 1–2): honest baselines
(pysteps, DWD operational nowcast) and better verification metrics
(pooled CSI, FSS, Brier). Every claim below was checked against a
primary source (official docs, source code, or the product itself);
URLs given inline. Compiled 2026-10-31.

## 1. pysteps baseline (optical-flow advection)

**What it is.** pysteps is the community open-source probabilistic
nowcasting library (Pulkkinen et al. 2019, *Geosci. Model Dev.* 12,
4185–4219, doi:10.5194/gmd-12-4185-2019). Source:
<https://github.com/pySTEPS/pysteps>, docs:
<https://pysteps.readthedocs.io/>.

**Motion estimation** lives in `pysteps.motion` with five methods
(verified from the repo tree `pysteps/motion/`):

| module | method |
|---|---|
| `lucaskanade.py` | `dense_lucaskanade` — Lucas–Kanade optical flow |
| `farneback.py` | `farneback` — Farnebäck dense flow |
| `darts.py` | DARTS |
| `proesmans.py`, `vet.py` | variational methods (Cython) |

The deterministic advection baseline we need is: take the last two
radar frames → estimate a dense motion field → extrapolate the newest
frame by `lead` using that flow. That is exactly what RADVOR does
conceptually (optical-flow-based nowcast), so pysteps is the honest
"strong persistence" baseline the review asked for.

**Verification API** (verified from `pysteps/verification/`):

- `detcatscores.det_cat_fct(pred, obs, thr, scores=...)` — categorical
  (contingency-table) scores: CSI, POD, FAR etc. from pooled counts.
  Has the init/accum/compute pattern (`det_cat_fct_init/_accum/_compute`)
  for accumulating over many samples — i.e. pysteps itself pools.
- `spatialscores.fss(X_f, X_o, thr, scale)` — **FSS is built in**;
  `scale` is the neighbourhood window in pixels. Also
  `fss_init/thr, scale`, `fss_accum`, `fss_compute` for streaming.
- `probscores` — CRPS, reliability diagram, ROC for probabilistic
  forecasts (relevant once we add probabilistic heads, item 5).
- `detcontscores.det_cont_fct` — continuous (MSE/correlation) scores.

**Install.** `pip install pysteps`. PyPI reachable from this machine
(verified `HTTP 200` on `pypi.org/simple/pysteps/`). Dependencies are
the usual scientific stack (numpy, scipy, matplotlib, dask, xarray,
scikit-image, pyproj) — nothing exotic; install was started in this
session.

**Minimal baseline sketch** (to be implemented as
`mvp/n8igall_mvp/baseline_pysteps.py`):

```python
import numpy as np
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.motion.interface import get_motion_method

def advect_baseline(frames, leads_min, dt_min=5):
    """Deterministic optical-flow nowcast from the last two frames.

    frames: [T,H,W] mm/h, newest last. Returns [len(leads),H,W].
    """
    flow = dense_lucaskanade(frames[-2:])          # [H,W,2] px/step
    out = []
    for L in leads_min:
        steps = L // dt_min
        # integrate the (constant) flow field by `steps` frames
        pred = _warp(frames[-1], flow * steps)     # grid_sample
        out.append(pred)
    return np.stack(out)
```

The exact warp helper should reuse our existing `grid_sample` advection
code (`model.py` advect path) so baseline and model share the same
resampling, keeping the comparison apples-to-apples.

## 2. DWD operational nowcast: `composite/rv` (RADVOR-derived)

**Verified live on opendata.** Directory:
<https://opendata.dwd.de/weather/radar/composite/rv/>

- One tar per issue time: `composite_rv_YYYYMMDD_HHMM.tar`, issued every
  5 minutes (rolling ~40 h window; **realtime only, no archive** — must
  be grabbed by the sync daemon).
- Each tar contains **25 HDF5 files**, leads `000`…`120` in 5-min steps
  (`..._000-hd5` … `..._120-hd5`) — i.e. a full 0–120 min forecast at
  exactly our lead times (5..60).
- Format: ODIM_H5 2.3, `quantity=ACRR` (mm), grid 1200×1100,
  stereographic `+proj=stere +lat_ts=60 +lat_0=90 +lon_0=10` with
  `xscale=yscale=1000` m — the DWD `de1200` composite grid, **not** the
  RADOLAN `rx` 900×900 grid. Comparing against our model requires a
  reprojection/crop from de1200 → rx (both are polar-stereographic with
  the same lat_ts/lon_0, so a simple affine + nearest/bilinear sample
  suffices; corners are in the `where` attrs).
- Tar size ≈ 0.95 MB; 25 files verified by extraction and h5py read.
- The `radvor/` directory exists but `radvor/re/` and `radvor/rq/` are
  **empty** — RADVOR itself is not downloadable; `composite/rv` is the
  radar-derived forecast product that *is* served.

**Daemon integration (implemented this session):**
`radar_sync.sync_rv()` archives tars into `data/rv/` with the same
40 h window logic as the realtime YW/RW mirror; wired into the hourly
cycle and the status line (`+N nowcast`).

**Why this matters.** This is the review's point 1: our numbers were
only measured against persistence. `composite/rv` is what DWD actually
publishes; beating it (or not) at +30/+60 min is the honest question.

## 3. FSS — fractions skill score (Roberts & Lean 2008)

Canonical definition (Roberts & Lean 2008, *MWR* 136, 781–796;
implemented identically in `pysteps.verification.spatialscores.fss`):

For neighbourhood width `n` (pixels), let `f_o(x)` and `f_f(x)` be the
**fraction of pixels above threshold `t`** in the `n×n` window centred
at `x`, over the verifying domain `N`:

$$\mathrm{FSS}(n) = 1 - \frac{\frac{1}{|N|}\sum_{x\in N}\left(f_f(x)-f_o(x)\right)^2}
{\frac{1}{|N|}\sum_{x\in N}\left(f_f(x)^2 + f_o(x)^2\right)}$$

Details that matter:

- **Threshold first, then average**: binarize at `t` mm/h, then compute
  window fractions (uniform box filter — pysteps uses a moving window;
  Gaussian weighting is a variant, not the canonical form).
- Range 0–1; 1 = perfect. The denominator is the "no-skill" reference
  (forecast and observation fractions uncorrelated).
- **Scales for a 1 km grid**: 10/20/50 km → window widths 10/20/50 px.
  A deterministic forecast that is spatially displaced by ~10 km still
  scores high at n=50 — this is exactly the property CSI lacks at long
  leads.
- Caveat: FSS depends on the rain fraction; compare only at the same
  threshold and period.

**Implementation status:** added to `evaluate()` in
`mvp/n8igall_mvp/train.py` — uniform-filter fractions via
`scipy.ndimage.uniform_filter`, pooled over the validation set
(`fss_lead{L}_{t}_{scale}` and `_pers` twins, scales 10/20/50 px).
Unit-tested (values in [0,1], all key families present).

## 4. Brier score for threshold exceedance

For a binary event `O = 1[rate > t]` and a binary deterministic
forecast `F = 1[pred > t]`:

$$\mathrm{BS} = \frac{1}{N}\sum_{i=1}^{N}(F_i - O_i)^2$$

- With a deterministic (0/1) forecast the Brier score reduces to the
  **error rate** of the exceedance classification — still informative
  and strictly proper when interpreted as a probability forecast of
  0 or 1; it rewards calibration that CSI cannot express.
- Relation to CSI: CSI ignores true negatives; Brier counts every pixel
  (dominated by correct negatives in sparse rain) — so report **both**:
  CSI for the rain pixels, Brier for the whole-field calibration.
- Once the model emits probabilities (item 5, focal/BCE heads), the
  same formula with `F ∈ [0,1]` becomes a proper probabilistic score —
  no code change needed beyond feeding probabilities instead of the
  thresholded field.

**Implementation status:** added to `evaluate()` (`brier_lead{L}_{t}`
and `_pers` twins), pooled over the validation set. Unit-tested.

## 5. Pooled vs nanmean CSI

- Our old `evaluate()` took `nanmean` of **per-sample** CSI. Samples
  with no rain above threshold at that lead yield `den=0 → NaN` and are
  dropped; samples with almost no rain give extremely noisy ratios.
  Measured on our val set: persistence +5 min CSI@1.0 = 0.110 (nanmean)
  vs **0.171 (pooled)**; 30/117 checked samples were all-dry NaN.
- The standard practice (and what pysteps' `det_cat_fct` does with its
  init/accum/compute pattern) is to **accumulate hits, misses and false
  alarms over all verifying samples, then take one ratio**. This is
  also what forecast-verification guidance (WWRP/WGNE) recommends for
  deterministic categorical scores.
- Caveat: pooling weights rainy samples more (each hit/miss pixel
  counts) — that is the intent; report the legacy nanmean key alongside
  for continuity.

**Implementation status:** `csi_lead{L}_{t}_pooled` (+ `_pers`) added;
legacy `csi_lead{L}_{t}` keys preserved because checkpoint selection
reads them (`train.py` selection uses `csi_lead{L}_1.0`).

## 6. RainNet (Ayzel et al.) — pretrained weights: VERIFIED

- Repo: **<https://github.com/hydrogo/rainnet>** (★156) — "RainNet: a
  convolutional neural network for radar-based precipitation
  nowcasting". Files: `rainnet.py`, `examples/`, `misc/`, LICENSE.
- **Pretrained weights exist**: README states the pretrained keras
  model `rainnet.h5` and `rainnet_weights.h5` are on **Zenodo,
  doi:10.5281/zenodo.3630429**. Sample data: doi:10.5281/zenodo.3629951.
- Trained on **RADOLAN** (German 1 km radar, same source as ours) —
  the claim in the review checks out.
- Caveats for our use:
  - It is a **Keras/TensorFlow** model; loading in our PyTorch stack
    means either a TF-free weights-only port (map layer names) or
    running a separate TF process for evaluation only.
  - RainNet predicts a **single lead** (typically +5 h accumulations
    or one step, per the paper setup) — it is not a 12-lead model, so
    it serves as a *comparison point at one lead* or a warm start for
    the encoder, not a drop-in.
  - License: check `LICENSE` in the repo before redistribution.
- A PyTorch reimplementation also exists: `fmidev/rainnet` (★10).

## 7. Persistence sanity check (review point 4)

The review suspected a timestamp/unit offset because persistence CSI at
+5 min looked low (0.087 @1 mm/h). Findings:

- **YW timestamps are interval END** (verified earlier in the reader
  work: classic YW `PR E-02` = 0.01 mm/5 min accumulated over the
  *preceding* 5 min, ×12 → mm/h). Our target at lead k is the frame
  ending `t0+k`; persistence uses the input frame ending `t0`. That
  alignment is correct — no off-by-one found in the reader.
- The low persistence number is explained by **two real bugs already
  fixed** (commit `5a62333` + `c8e8606`): targets were 12× too small
  (units bug) *and* raw-not-log1p (threshold became ln2≈0.69 instead
  of 1.0 while persistence stayed correctly scaled). Both inflated
  persistence misses. **Absolute CSI numbers must be re-measured on
  the fixed pipeline before drawing any conclusion** — the current
  training run (relaunched on fixed code) will produce the first
  trustworthy numbers.
- Remaining legitimate factor: 1 km YW **speckle/noise** — single-pixel
  rain features decorrelate within 5 min, which genuinely depresses
  pixel-level persistence CSI at fine thresholds. This is precisely
  why FSS at 10–50 km (§3) is the right companion metric.

## Recommended implementation order

1. **Done this session**: pooled CSI + Brier + FSS in `evaluate()`
   (unit-tested; legacy keys preserved for checkpoint selection).
2. **Done this session**: `radar_sync.sync_rv()` archives
   `composite/rv` nowcast tars → `data/rv/` (the DWD operational
   baseline).
3. **Next**: `baseline_pysteps.py` — optical-flow advection baseline
   evaluated with the *same* `evaluate()` metrics on the same val
   split; requires `pip install pysteps` (started) and reusing our
   `grid_sample` warp for identical resampling.
4. **Then**: de1200→rx reprojection loader for `composite/rv` so the
   operational forecast can be scored with the same metrics (both
   grids documented in the HDF5 `where` attrs).
5. **Then**: re-run all baselines (persistence, pysteps, composite/rv,
   model) on the **fixed** pipeline; report CSI-pooled/FSS/Brier at
   0.1/1/5 mm/h × leads 5..60.
