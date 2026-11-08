# ICON-D2 integration plan — NWP precipitation & reflectivity as model input

Status: **archiving started 2026-11-08**, training input pending history.
Written for external review (follow-up to the reviews at `0b5cecc` and
`f47e52f`). Repo revision at time of writing: after `b621be8`.

## Why

The current model can only *advect* existing radar echoes — it has no
knowledge of where **new** rain will form. At +5…15 min that is fine
(radar advection dominates and the model beats all baselines there), but
at +30…60 min growth/decay is driven by NWP physics (fronts,
convection, orography) that radar history cannot contain. This is also
where the known weaknesses live: light-rain fade at +60 and no skill
beyond advection.

US alternatives were checked and rejected: **HRRR is CONUS-only** (does
not reach Germany), **GFS is 0.25° ≈ 25 km** (~25× coarser than needed).
DWD's own **ICON-D2 at 2.2 km** is the highest-resolution precipitation
forecast available for this region, free, no registration.

## What is archived (live since 2026-11-08)

`n8igall_mvp/icon_d2_sync.py`, called hourly by the `radar_sync`
daemon:

| Field | Meaning | Size/step | Role |
|---|---|---|---|
| `u_10m`, `v_10m` | 10 m wind (regular-lat-lon) | ~1 MB | future train/serve wind parity (replaces ERA5 reanalysis at inference) |
| `tot_prec` | total precipitation, 2.2 km | ~0.9 MB | future NWP-precip input + scoreable baseline |
| `dbz_cmax` | synthetic composite max-echo (dBZ) | ~3.6 MB | future input in *radar observation space* — directly comparable pixel-for-pixel with our radar channels |

* Delivery: ICON-D2 runs **8× daily** (cycles 00/03/06/09/12/15/18/21
  UTC), 49 forecast steps per cycle; we archive steps **000–04 hourly**
  (runs renew every 3 h, so 3 h age + 60 min lead covers the +60 min
  nowcast window; steps >4 h never read — user directive 2026-11-08).
* Retention: **space-guarded** (`KEEP_DAYS` 365 is a hard backstop;
  normally everything is kept until free space < 40 GB, then oldest
  init dirs go first), ~270 MB/day at 5 steps, ~100 GB/year on
  `/data/n8igall/data/nwp/icon_d2/<init>/`.
* **No historical archive exists** on opendata (only the latest run per
  cycle, ~24 h) — collecting from now is the only route; this is why
  archiving started immediately.

## Timeline and decision gates

**Retention note (review at ba2279c):** the original 7-day `KEEP_DAYS`
would have pruned cycles before the gates below could be met — 7 days ×
48 issues/day = 336 samples, while 1,400 samples need ~29 days and 4,300
need ~90 days. Pruned cycles cannot be refetched (opendata keeps only
the latest run per cycle). **`KEEP_DAYS` is now 90** (~60 GB at
~0.7 GB/day; 236 GB were free), raised 2026-11-08 before any prune ran
(archiving started the same day).

| When | Milestone | Gate |
|---|---|---|
| now → +2 weeks | archive accumulates | **score a matched-window baseline** on real-time data: ICON-D2 `tot_prec` vs our model / LK / persistence, by lead and by regime (wet/dry, heavy/light). Also score `composite/rv` (rv.py already converts mm/5min ×12) whenever local RV tars exist. |
| before any ICON retrain | cheap checks (see below) | **fixed blend or residual feature on a frozen slice** — complementary value, not a standalone win over LK, is the question. Uncertainty on events/days, not overlapping pixels. |
| ~+1 month (Dec 2026) | ~1,400 samples (30-min stride, needs ~29 days) | **ablation experiment**: add `tot_prec`/`dbz_cmax` channels to a fine-tune of `multi_lk`; compare on the real-time test set. Small sample → directional signal only. |
| ~+3 months (Feb 2027) | ~4,300 samples (needs ~90 days) | **proper training run** if the ablation is positive (reference: current index = 8,774 samples over 2 seasons). |
| ~+6 months | winter+spring covered | seasonally representative index; consider replacing ERA5 wind with archived ICON-D2 wind (closes the train/serve mismatch flagged in review 1). |

Interim (unvalidated, needs no training): inference-time blending of
ICON-D2 precip into the output with weight growing by lead — a hack to
test once ~2 weeks of archive exist, not a substitute for the input.

## Cheap checks before an ICON retrain (review at ba2279c)

These do not displace ongoing radar work; a standalone win over
Lucas–Kanade is encouraging but *not* required. What matters is
**complementary value**:

1. **Matched-window baseline**: `tot_prec` is an hourly accumulation in
   kg m⁻², not a 5-minute rate. Score it as *the accumulation over an
   hour* (difference of the field at hour end and start) against the sum
   of 5-minute radar over that same hour — windows must match. Do not
   spread an hourly difference into a 5-minute rate and call it one.
2. **Fixed blend or residual** on a frozen slice (not used to pick the
   rule), split **by lead** and **by regime** (wet vs dry, heavy vs
   light). Read a negative result as "no complementary value in this
   window", not "ICON is useless".
3. **Uncertainty on events or days**, never on overlapping pixels.
4. **Score `composite/rv`** in the same pass — the operational nowcast
   baseline has been archived but never scored.

## Valid time is not hindsight (review at ba2279c)

A forecast whose *valid time* is later than the radar target is a
legitimate predictor **when the run was published at or before issue
time**. Availability is fixed by publication and init as of the issue —
the opposite of an ERA5 analysis, which did not exist at issue time.
Using a later valid time from an already-published run (and interpolating
between steps of that run) is allowed. What must still match is the
*verification window* (see check 1).

## Rules when adding an ICON field as input

* Convert a matched accumulation into the intended units and **state
  whether the result is an hour sum or an hourly-mean rate**; `log1p`
  of a mm/h rate matches the radar channels. Never feed the raw
  accumulated field as if it were a 5-minute rate.
* **Clip or mask the ICON domain edge** — the German radar grid is not
  the ICON domain.
* GRIB decode + de1200→rx reprojection still to be implemented (same
  machinery as `rv.py`).

## Design notes for the reviewer

1. **`dbz_cmax` over `tot_prec` as the primary input candidate**: it is
   in the same observation space as our radar input (reflectivity →
   rain rate), so the network can compare forecast echo against observed
   echo directly; `tot_prec` needs a unit/accumulation convention
   conversion (GRIB step accumulation vs our mm/h rate).
2. **GRIB decoding is not yet implemented** — files are archived as
   `.grib2.bz2`; the decode step (cfgrib/eccodes or a minimal GRIB2
   reader) plus de1200→rx reprojection (same machinery as `rv.py`) is
   part of the Dec experiment, not done yet.
3. **Lead alignment**: forecast step `k` hours from init must be matched
   to issue time `t` (nearest step ≥ lead). Init times are 3-hourly, so
   a 5-min nowcast issue time uses the most recent cycle's step
   `ceil((t - init)/1h)`; document the exact convention when implemented.
4. **Leakage check**: unlike ERA5, ICON-D2 fields are forecasts issued
   *before* the issue time — no hindsight. (ERA5 issue-hour wind remains
   reanalysis; review 1 flagged this as a train/serve mismatch that only
   ICON-D2 wind can close.)
5. **Disk**: full-rate ≈ 226 GB/year for all four variables (measured
   0.62 GB/day); `icon_d2_sync` keeps everything until free space drops
   below 40 GB, then prunes oldest-first (365-day hard backstop).
   (Volume collection was removed 2026-11-08 — it dominated disk
   growth at 4.2 GB/day.)

## Open questions for review

* Should `dbz_cmax` be archived at all steps 0–48 h (3× the disk) to
  enable longer-lead experiments beyond +60 min?
* For the baseline score: compare ICON-D2 precip at native 2.2 km
  reprojected to rx, or aggregate our 1 km fields up to 2.2 km (fairer
  to ICON-D2)?
