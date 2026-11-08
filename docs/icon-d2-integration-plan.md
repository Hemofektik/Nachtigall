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
  UTC), 49 forecast steps per cycle; we archive steps **000–12 hourly**
  (covers the +60 min nowcast window with margin).
* Retention: 7 days per init time (`KEEP_DAYS`), ~680 MB/day,
  ~4.7 GB/week on `/data/n8igall/data/nwp/icon_d2/<init>/`.
* **No historical archive exists** on opendata (only the latest run per
  cycle, ~24 h) — collecting from now is the only route; this is why
  archiving started immediately.

## Timeline and decision gates

| When | Milestone | Gate |
|---|---|---|
| now → +2 weeks | archive accumulates | **score ICON-D2 `tot_prec` as a baseline** on real-time data vs our model / LK / persistence (operational-honest: a real forecast available at issue time, unlike ERA5). Needs ≥2 weeks for a meaningful sample. |
| ~+1 month (Dec 2026) | ~1,400 samples (30-min stride) | **ablation experiment**: add `tot_prec`/`dbz_cmax` channels to a fine-tune of `multi_lk`; compare on the real-time test set. Small sample → directional signal only. |
| ~+3 months (Feb 2027) | ~4,300 samples | **proper training run** if the ablation is positive (reference: current index = 8,774 samples over 2 seasons). |
| ~+6 months | winter+spring covered | seasonally representative index; consider replacing ERA5 wind with archived ICON-D2 wind (closes the train/serve mismatch flagged in review 1). |

Interim (unvalidated, needs no training): inference-time blending of
ICON-D2 precip into the output with weight growing by lead — a hack to
test once ~2 weeks of archive exist, not a substitute for the input.

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
5. **Disk**: 7-day retention ≈ 4.7 GB on the 236 GB-free `/data` volume;
   `radar_sync` already guards volume collection at <25 GB free.

## Open questions for review

* Should `dbz_cmax` be archived at all steps 0–48 h (3× the disk) to
  enable longer-lead experiments beyond +60 min?
* Is 7-day retention the right window, or should it match the radar
  archive's monthly granularity for the Dec ablation?
* For the baseline score: compare ICON-D2 precip at native 2.2 km
  reprojected to rx, or aggregate our 1 km fields up to 2.2 km (fairer
  to ICON-D2)?
