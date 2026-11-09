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
| `dbz_cmax` | synthetic *column-max* dBZ | ~3.6 MB | future input in the reflectivity domain — same units family as radar, but a column maximum, **not** pixel-identical to near-surface YW |
| `clcl`, `clcm`, `clch` | low / mid / high cloud fraction (instantaneous) | ~1.1 MB | operational twin of the ERA5 cloud channels `multi_lk_cloud` trained on (ERA5 is reanalysis — unserveable at issue time) |

* Delivery: ICON-D2 runs **8× daily** (cycles 00/03/06/09/12/15/18/21
  UTC), 49 forecast steps per cycle; we archive steps **000–07 hourly**
  (8 steps). Reason: `tot_prec` is an accumulation from model start, so
  the window covering a lead is a difference of consecutive steps and
  the window **end** must be stored — issue 15:50 with a late 15 UTC
  run falls back to the 12 UTC run, whose 16:00–17:00 window is
  step 5 − step 4; publication delay (+44 min observed to step 000)
  makes that fallback real. 0–7 covers the one-cycle-late case with
  margin (review at 13a87fb).
* Retention: **space-guarded** (`KEEP_DAYS` 365 is a hard backstop;
  normally everything is kept until free space < 40 GB, then oldest
  init dirs go first), ~640 MB/day at 7 vars × 8 steps, ~234 GB/year
  on `/data/n8igall/data/nwp/icon_d2/<init>/`.
* **No historical archive exists** on opendata (only the latest run per
  cycle, ~24 h) — collecting from now is the only route; this is why
  archiving started immediately.

## Timeline and decision gates

**Archive status (2026-11-09, review at 13a87fb):** first init
2026-11-06 18Z; 24 cycles / 559 MB so far at steps 000–004. The review
found step 4 one hour short (tot_prec is an accumulation from model
start, so the hour containing a +60 min lead is *step 5 − step 4* of a
3-hour-old fallback run, and publication delay is +44 min to step 000)
— `MAX_STEP` is now **7** and the low/mid/high cloud fractions
**clcl/clcm/clch** are archived too (operational twin of the ERA5
cloud channels; no historical archive exists, so they start accumulating
from now). ~640 MB/day ≈ 234 GB/yr.

**Retention note (review at ba2279c):** the original 7-day `KEEP_DAYS`
would have pruned cycles before the gates below could be met — 7 days ×
48 issues/day = 336 samples, while 1,400 samples need ~29 days and 4,300
need ~90 days. Pruned cycles cannot be refetched (opendata keeps only
the latest run per cycle). **`KEEP_DAYS` is now 365 as a hard backstop
only** — normal pruning is space-guarded (keep everything until <40 GB
free, then oldest first; user directive 2026-11-08). The age rules were
raised 2026-11-08 before any prune ran.

**DWD keeps no historical ICON archive** — the clock cannot be
backfilled, so any ICON-input model must train on issue times
≥ 2026-11-06. Our existing indexes (`index_multi` 2024–25,
`index_test_rt` Oct 23–Nov 1) predate the archive and can never carry
ICON inputs.

| When | Milestone | Gate |
|---|---|---|
| **this week (by ~2026-11-16)** | 3–10 days of archive | **first use — score a matched-window baseline** (no training needed): decode `dbz_cmax`/`tot_prec`, score vs radar by lead and regime; also finally score `composite/rv` (archived, never scored). 3 days = smoke test, 2 weeks = first trustworthy number |
| before any ICON retrain | cheap checks (see below) | **fixed blend or residual feature on a frozen slice** — complementary value, not a standalone win over LK, is the question. Uncertainty on events/days, not overlapping pixels. |
| **~2026-12-07 (+1 month)** | ~1,400 samples (30-min stride, needs ~29 days) | **first use as model input — ablation**: add `tot_prec`/`dbz_cmax` channels to a fine-tune of `multi_lk_cloud` on a Nov-6+ index, vs an ERA5 control on the *same* window (controls for window difference). Small sample → directional signal only |
| **~2027-02-06 (+3 months)** | ~4,300 samples (needs ~90 days) | **proper training run** if the ablation is positive (reference: current index = 8,774 samples over 2 seasons) |
| **~2027-05 (+6 months)** | winter+spring covered | seasonally representative index; **replace ERA5 wind with archived ICON-D2 wind** — closes the train/serve mismatch flagged in review 1. ICON single-levels have no 100 m (steering-level channels change) → full retrain, not fine-tune |
| **~2027-11 (+1 year)** | full year of ICON archive | ICON-native year-long index replacing ERA5 entirely — matches the one-year retention policy; needs the bigger disk if the current one is tight |

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
* GRIB decode + reprojection to the rx grid still to be implemented.
  Note the ICON file is **regular lat/lon** (GRIB template 0, 0.02°,
  1215×746) — *not* the stereographic de1200 grid `rv.py` handles, so
  the grid math differs; only the "reproject to rx" concept is shared.

## Design notes for the reviewer

1. **`dbz_cmax` over `tot_prec` as the primary input candidate**: it is
   in the same observation space as our radar input (reflectivity →
   rain rate), so the network can compare forecast echo against observed
   echo directly; `tot_prec` needs a unit/accumulation convention
   conversion (GRIB step accumulation vs our mm/h rate).
2. **GRIB decoding is not yet implemented** — files are archived as
   `.grib2.bz2`; the decode step (cfgrib/eccodes or a minimal GRIB2
   reader) plus reprojection to rx is part of the Dec experiment, not
   done yet. The grid is regular lat/lon (template 0, 0.02°,
   1215×746), **not** the stereographic de1200 grid `rv.py` decodes —
   `rv.py` shares only the "map to rx" concept, not the grid math.
3. **Lead alignment**: the scored quantity for a lead is an *hourly
   accumulation window* (difference of consecutive `tot_prec` steps),
   so pick the steps by the **verification window that matches the
   radar sum** (the hour containing the lead's valid time), not by
   `ceil`-ing the lead itself. A later valid time from an already-
   published run is allowed (see "valid time is not hindsight"); what
   must match is the window, not the step index. Init times are
   3-hourly and publication runs +44 min behind init — document the
   exact convention when implemented.
4. **Leakage check**: unlike ERA5, ICON-D2 fields are forecasts issued
   *before* the issue time — no hindsight. (ERA5 issue-hour wind remains
   reanalysis; review 1 flagged this as a train/serve mismatch that only
   ICON-D2 wind can close.)
5. **Disk**: 7 vars × 8 steps ≈ 640 MB/day ≈ **234 GB/year**
   (measured 270 MB/day for 4 vars × 5 steps, scaled; cloud fields
   ~1.1 MB/file confirmed on the index); `icon_d2_sync` keeps
   everything until free space drops below 40 GB, then prunes
   oldest-first (365-day hard backstop). (Volume collection was
   removed 2026-11-08 — it dominated disk growth at 4.2 GB/day.)

## Open questions for review

* Should `dbz_cmax` be archived at all steps 0–48 h (3× the disk) to
  enable longer-lead experiments beyond +60 min?
* Grid choice for the baseline score — **answered (review at
  13a87fb): score both.** Primary = our 1 km radar *aggregated up* to
  the ICON 2.2 km grid plus a neighbourhood FSS (the finer radar must
  not get credit for detail ICON cannot have); secondary = a 1 km
  reprojection of ICON (the grid a blend would actually run on).
