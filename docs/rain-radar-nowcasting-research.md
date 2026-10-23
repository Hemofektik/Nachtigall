# DWD Rain-Radar Nowcasting Research

*Research date: 2026-10-22/23. All facts verified against primary sources (DWD official pages/opendata server, official model repositories, arXiv/publisher records, NVIDIA/PyTorch docs). Uncertain claims are explicitly marked. Repository context: Nachtigall currently correlates DWD CDC station data with a FANN MLP in C++.*

---

## TL;DR / Recommendation

1. **Data is a solved problem.** DWD opendata serves free (CC BY 4.0) 1 km radar composites: RY (5-min), RW (hourly sums, re-run every 10 min), YW (5-min gauge-adjusted) in HDF5/ODIM, plus a full archive (RADKLIM: RW hourly and YW 5-min, **2001 → present**, as bin/netCDF/HDF5). Format specs are official DWD PDFs (see §1).
2. **DWD's own nowcaster (RADVOR-OP)** is a classical optical-flow (OpenCV) advection system, not a neural net. Its calibrated RQ/RE products were **discontinued end of Feb 2026**; the uncalibrated 5-min and hourly nowcasts (RV/RS) continue on opendata. This is the honest baseline to beat.
3. **"Pretrained nowcasting networks" are mostly a myth for this use case.** DGMR (DeepMind) never released code/weights; MetNet-3 is paper-only; NowcastNet's official repo has been removed; GraphCast/Pangu-Weather are global medium-range models at 0.25°–1° (~30–250 km) — wrong scale and wrong task (not 1 km precipitation nowcasting).
4. **Recommendation: option (b).** Train a small custom U-Net in PyTorch on RADKLIM YW→RW pairs (see §4), export to ONNX, and run inference in the C++ app via ONNX Runtime (add TensorRT only if you have an NVIDIA GPU and need low latency). Include a **static elevation channel** (Copernicus DEM GLO-30, 30 m, resampled to the 1 km RADOLAN grid — see §4.3) and **NWP wind forecast fields** (ICON-D2/ECMWF, incl. an orographic upslope-wind feature — see §4.4) as additional inputs. Optical flow (RADVOR/pysteps) is used **only as an evaluation baseline**, not as a model component — advection is handled physically by the wind fields. Keep FANN out of the nowcasting path (MLP, no convolutions).
5. **GPU reality check:** a single consumer GPU (e.g. RTX 3060-class) trains a 1 km Germany U-Net fine. TensorRT's repo is only *partially* open source (plugins/ONNX parser/samples; core runtime ships as binary wheels) — for a solo project, ONNX Runtime C++ is the simpler deployment path; Torch-TensorRT exists if you later need speed.

---

## 1. DWD rain radar data (opendata.dwd.de)

### 1.1 Where the data lives

Root: <https://opendata.dwd.de/weather/radar/> — plain HTTPS file server (FTP-style indexes), no API key, no registration. Relevant subdirectories (verified 2026-10-23):

| Directory | Content | Cadence (observed) | Format |
|---|---|---|---|
| [`radolan/ry/`](https://opendata.dwd.de/weather/radar/radolan/ry/) | RY: quality-corrected, quantitative 5-min radar composite (mm/5min, 1 km) | every 5 min | `raa01-ry_10000-YYMMDDhhmm-dwd---bin.hdf5` |
| [`radolan/rw/`](https://opendata.dwd.de/weather/radar/radolan/rw/) | RW: gauge-adjusted hourly sums (0.1 mm, 1 km), RADOLAN's flagship product | every 10 min (0030, 0040, …) | `raa01-rw_10000-…-bin.hdf5` |
| [`radolan/yw/`](https://opendata.dwd.de/weather/radar/radolan/yw/) | YW: quasi-gauge-adjusted 5-min composite (basis of RADKLIM-YW) | every 5 min | `raa01-yw_10000-…-bin.hdf5` |
| [`radolan/sf/`](https://opendata.dwd.de/weather/radar/radolan/sf/) | SF: daily sums | daily | binary/HDF5 |
| [`composite/rs/`](https://opendata.dwd.de/weather/radar/composite/rs/) | RADVOR nowcast: 1-h sums forecast to +2 h | every 5 min | `composite_rs_YYYYMMDD_HHMM.tar` |
| [`composite/rv/`](https://opendata.dwd.de/weather/radar/composite/rv/) | RADVOR nowcast: 5-min precipitation forecast (mm/5min) to +2 h | every 5 min | `composite_rv_….tar` |
| [`composite/hx/`](https://opendata.dwd.de/weather/radar/composite/hx/) | HX: 250 m max-reflectivity composite (`quantity=DBZH`, product `MAX`) | every 5 min | `composite_hx_…-hd5` (ODIM HDF5) |
| [`composite/dmax/`](https://opendata.dwd.de/weather/radar/composite/dmax/) | daily radar-derived maximum precipitation | every 5 min | HDF5 |
| [`sites/`](https://opendata.dwd.de/weather/radar/sites/) | per-radar-site scan products: `dx/`, `px/`, `px250/`, `pz/`, `pe/`, `pf/`, `pr/`, `lmax/` (volumes/precipitation scans, incl. 250 m) | ~5 min | DWD binary / HDF5 |
| [`radvor/re/`, `radvor/rq/`](https://opendata.dwd.de/weather/radar/radvor/) | **empty** (only `../`) — RQ/RE discontinued | — | — |

Notes:
- Real-time directories keep only a short rolling window: `radolan/ry/` held ~577 files ≈ **~48 h** at 5-min cadence (oldest observed 2026-10-21 01:35). Grab data yourself if you need more.
- The classic **RX** (12-min/5-min raw composite) directory no longer exists under `composite/` (404-ish empty); the format spec still documents RX as 1-byte RVP6-reflectivity (`dBZ = RVP6/2 − 32.5`). Current equivalent for reflectivity is HX (250 m) and for precipitation RY. *(Uncertain: whether RX is still produced elsewhere; it is not on the public server.)*
- Composite dirs `pg/`, `vii/`, `wn/`, `hymecng/` also exist (wn = warning-related tars, VII = HDF5 every 5 min); exact semantics not documented in the sources checked — treat as "see DWD Produktübersicht".

### 1.2 Formats and projection (official spec)

Primary spec: **[RADOLAN-RADVOR-Kompositformat, Version 2.6, 14.01.2025](https://www.dwd.de/DE/leistungen/radolan/radolan_info/radolan_radvor_op_komposit_format_pdf.pdf)** (DWD Hydrometeorologie; supersedes older "Formatbeschreibung RADOLAN"). Key facts:

- **Classic binary format:** ASCII header (starts `By` marker with ddhhmm, site `10000` for composites, product length, `VS` version flag, `GP` grid size, `MS`/`MF`/`INT` intervals…), then a binary block: **900×900** px (national), 1100×900 (extended national), 1500×1400 (Central-European ME). All products except WW/WX/RX/EX use **2-byte little-endian, 12 data bits (0–4095)**; bits 13–16 flag secondary dataset, error code 2500 (bit 14), sign (bit 15), clutter 2490 (bit 16). RX/WX/EX are 1-byte. RW values are hourly sums with 0.1 mm resolution; intensity scaling per product header (`INT`).
- **HDF5 migration:** "Ab 2023" DWD began migrating RADOLAN products to **HDF5 in OPERA ODIM_H5 v2.3** layout ([ODIM spec](https://www.eumetnet.eu/wp-content/uploads/2019/01/ODIM_H5_v23.pdf)); the national quantitative (1 km) products simultaneously switched to the **WGS84 ellipsoid** and a larger composite extent of **1200 km × 1100 km** (verified in live files: `xsize=1100, ysize=1200, xscale=1000 m`, `projdef=+proj=stere +lat_ts=60 +lat_0=90 +lon_0=10 …`, ODIM `H5rad 2.3`).
- **Classic projection (still in spec & binary products):** polar stereographic, 1 km equidistant grid, projection plane at 60°N, λ₀=10°E, composite center 9°E/51°N, sphere R = 6370.04 km; national grid corners ≈ LL 46.9526°N/3.5889°E, UR 54.7405°N/15.7208°E. WGS84 composites (`VS=5`) have slightly shifted corners.
- File naming: `raa01-<product>_10000-YYMMDDhhmm-dwd---bin[.hdf5]`, timestamps UTC. Forecast products use `RV/RS/RQ…YYMMddhhmm_vvv` (vvv = lead time in minutes).
- Also relevant: HDF5 files carry ODIM `what/where/how` attributes (gain/offset/nodata, corner lat/lon, projdef) — much easier to parse from C++ than reinventing the ASCII header.

Supporting docs: [RADOLAN Kurzbeschreibung](https://www.dwd.de/DE/leistungen/radolan/radarniederschlagsprodukte/radolankurzbeschreibung_pdf.pdf); RADOLAN service page: <https://www.dwd.de/DE/leistungen/radolan/radolan.html>.

### 1.3 Coverage, latency, quality

- **17 German radars** (dual-polarization C-band generation, 150 km quantitative radius each), composite merged with radar data of DK, PL, CZ, AT, CH, FR, BE, NL (RADOLAN-ME extension).
- **RW**: best calibrated product; gauge-adjusted (RADOLAN "Aneichung" against ~40–60 ombrometers per 100-km radius + partner networks); available **within ~30 min**; hourly sums recomputed **every 10 min since 2018-04-25** (source: RADOLAN page).
- Known radar caveats (per DWD docs): clutter, bright band, attenuation — handled by quality composites and Z-R refinement; the YW/RW gauge adjustment removes most systematic bias at 1 km.

### 1.4 Archive availability (RADKLIM, CDC)

- **RADKLIM RW (hourly, 1 km, gauge-adjusted):** <https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/> — `recent/` (rolling, asc/bin/hdf5), `historical/` (bin 2005–2024 observed), `reproc/` (**`2016_003`, `2017_002` reprocessings with year dirs `2001/` … `2026/`**, asc/bin/**netCDF**).
- **RADKLIM-YW (5-min, 1 km, quasi-adjusted):** <https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/> — `recent/` as daily tarballs `YW-YYMMDD.tar.gz` (oldest kept: 2025-02-03 at time of check ≈ 20 months), `reproc/2017_002/` with **years 2001 → 2026** (asc/bin/netCDF). This is the richest free 5-min radar archive for training.
- Climate reference/verification: Weusthoff et al. describe RADKLIM (DKRZ landing page is the canonical citation; DWD service page above is the primary product description).

### 1.5 Licensing

- DWD opendata is **free of charge** (statutory mandate since 25.07.2017, DWD-Gesetz amendment) — see <https://www.dwd.de/DE/leistungen/opendata/opendata.html>.
- **Reuse license: Creative Commons BY 4.0 with attribution**, explicitly covering "all freely accessible geodata … including meteorological weather and climate information with location reference" — DWD [Rechtliche Hinweise](https://www.dwd.de/DE/service/rechtliche_hinweise/rechtliche_hinweise.html). (Older DWD material referenced "Datenlizenz Deutschland – Namensnennung 2.0"; the current legal page states CC BY 4.0.)
- No SLA/guarantee of availability for opendata.dwd.de (stated on the opendata page) — cache your own data.

### 1.6 Baseline: DWD's own nowcaster, RADVOR-OP

Sources: [RADVOR service page](https://www.dwd.de/DE/leistungen/radvor/radvor.html), [RADVOR-Kurzbeschreibung (Winterrath, 2020)](https://www.dwd.de/DE/leistungen/radvor/radvor_info/radvor_kurzbeschreibung_pdf.pdf).

- Purely image-based (no NWP input): **optical flow from OpenCV** (within DWD's POLARA environment) recognizes and tracks precipitation structures in the 5-min QPE_common composite (mm/5min, 1 km); advects **24 × 5-min steps to +2 h**; sums to hourly forecasts **RS**; **RAVOQ** quantification maps the forecast distribution onto the RW distribution (stratiform situations only; convective cells flagged via KONRAD). Winter module (solid fraction RE, snow rate FS/FQ) and hail flag.
- Published verification (winter 2013/14, stratiform): multiplicative **bias of quantified RQ ≈ 0.96–1.00** vs RW, ETS ≈ 0.44/0.51/0.35 at +0/+1/+2 h; unquantified RS biased high (1.13–1.36). Beating this on calibrated 1-hour totals is a serious benchmark.
- **Operational status (2026):** the page states RADVOR runs only until **end of February 2026**; `radvor/re/` and `radvor/rq/` are indeed now empty. However **RV and RS are still published every 5 min** under `weather/radar/composite/{rv,rs}/` (verified with files dated 2026-10-23). *(Uncertain: what replaces RADVOR long-term; no successor named on the checked pages.)*

---

## 2. Existing neural nowcasting approaches

### 2.1 Radar/observation-scale nowcasters

| Model | Paper (primary) | Code/weights | License | Verdict for this project |
|---|---|---|---|---|
| **DGMR** (DeepMind) | [arXiv:2104.00954](https://arxiv.org/abs/2104.00954), Nature [10.1038/s41586-021-03854-z](https://doi.org/10.1038/s41586-021-03854-z) (2021) | **No official code or weights** (none on github.com/google-deepmind or deepmind). Community reimplementations exist, e.g. [TQRTQ/DGMR](https://github.com/TQRTQ/DGMR) (~37★) | reimplementations: per-repo | Paper-only; reimplementation + data collection = high effort, risky payoff |
| **MetNet-3** (Google) | [arXiv:2306.06079](https://arxiv.org/abs/2306.06079) ("Deep Learning for Day Forecasts from Sparse Observations"; operational at Google, 1 km, up to 2 min temporal, up to 24 h) | **Paper-only.** Open reimplementation of MetNet-1/2: [openclimatefix/metnet](https://github.com/openclimatefix/metnet) (MIT) with **pretrained weights** on HuggingFace (`openclimatefix/metnet`, `metnet-2`), trained on GOES-16/MRMS (US data) | MIT | Best-documented open lineage; MetNet-3 itself unreleased; MetNet-2 is a plausible architecture template (dilated-conv encoder over radar/satellite mosaics) |
| **NowcastNet** (Tsinghua et al.) | Nature [10.1038/s41586-023-06184-4](https://doi.org/10.1038/s41586-023-06184-4) (2023), "Skilful nowcasting of extreme precipitation with NowcastNet" | Official repo `github.com/OpenWeatherLab/NowcastNet` **now returns 404** (removed; no arXiv preprint found). Third-party reimplementations only (e.g. [Hitlia/NowcastNet](https://github.com/Hitlia/NowcastNet), ChaoXiNet, nowcastnet-rewritten) | n/a | Interesting physics-embedded generative design, but no trustworthy official weights; reimplementation burden |
| **RainMAE / radar transformers** | **Not verifiable**: arXiv search "RainMAE" returns no results (checked 2026-10-23) | — | — | Marked uncertain — possibly misremembered name; no primary source found |
| **pysteps** (classical baseline) | [pySTEPS/pysteps](https://github.com/pySTEPS/pysteps) (official, 593★) + [example data](https://github.com/pySTEPS/pysteps-data); ERAD nowcasting course: [pySTEPS/ERAD-nowcasting-course-2026](https://github.com/pySTEPS/ERAD-nowcasting-course-2026) | full code, pip-installable | **BSD-3-Clause** | The de-facto open optical-flow/stochastic nowcasting framework; same algorithmic family as RADVOR — the baseline every NN must beat |

### 2.2 Global/medium-range AI models (context, not drop-in solutions)

| Model | Paper | Code/weights | Why not the answer for 1 h @ 1 km |
|---|---|---|---|
| **GraphCast** | Science [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) | Code+weights now in [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) (old `graph_cast` repo redirects there): code **Apache-2.0**, materials **CC BY 4.0**, **weights license updated 2026-08-06 to permit commercial use** | 0.25° (~30 km) global, initialized from ERA5/HRES; medium-range (hours–10 days). Useful precipitation skill only at ~30 km — two orders of magnitude coarser than RADOLAN; also now hosts WN2/GenCast |
| **Pangu-Weather** | Nature [10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3), [arXiv:2211.02556](https://arxiv.org/abs/2211.02556) | Official repo [198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather) with pre-trained checkpoints and inference code; **no LICENSE file** in repo (rights unclear — treat cautiously) | 0.25°/0.5° global 3D EarthFormer, 6-hourly steps, predicts upper-air + surface state (no precipitation-nowcast output); wrong scale/task |

*Bottom line:* the only *openly usable* neural assets directly relevant to radar nowcasting are (1) OCF's MetNet-1/2 implementation + weights (US satellite/radar domain) and (2) community DGMR/NowcastNet reimplementations. There is no "pretrained model you can point at DWD RX/RY" — data format, domain (Europe), projection and licensing all require work regardless. A **small, self-trained U-Net is the pragmatic neural choice**.

---

## 3. GPU inference/training stack (CUDA / TensorRT / ONNX)

Facts (primary sources):

- **PyTorch → ONNX is the standard export path.** `torch.onnx.export` (+ the TorchDynamo-based exporter) is documented at [docs.pytorch.org/docs/stable/onnx.html](https://docs.pytorch.org/docs/stable/onnx.html) (also legacy [onnx_export.html](https://docs.pytorch.org/docs/stable/onnx_export.html)). ONNX itself: [onnx.ai](https://onnx.ai) (Apache-2.0, [github.com/onnx/onnx](https://github.com/onnx/onnx)); inference runtime [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) (MIT, CUDA EP for NVIDIA GPUs) — ships a **C++ API**, which integrates naturally into Nachtigall.
- **TensorRT is only partially open source.** [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) (Apache-2.0, ~13.4k★) states: "This repository contains the **Open Source Software (OSS) components** … plugins and ONNX parser, as well as sample applications … **a subset** of the TensorRT General Availability release." The core compiler/runtime ships as proprietary binary (e.g. `pip install tensorrt`). Current release per README: **TensorRT 11.x** (GA 11.4, recommends CUDA 12.9/13.4; strong typing, explicit quantization, IPluginV3 replace 10.x legacy APIs). See the repo's [Import Workflows Guide](https://github.com/NVIDIA/TensorRT/blob/main/documents/import_workflows.md) for ONNX/Torch-TensorRT/Optimum paths.
- **Torch-TensorRT** ([github.com/pytorch/TensorRT](https://github.com/pytorch/TensorRT), BSD-3) compiles PyTorch (FX/TorchScript) modules directly to TensorRT engines — closer to `torch.compile` ergonomics, but ties you to NVIDIA GPUs and adds graph-compiler friction (dynamic shapes, unsupported ops).
- **Practical guidance (judgment, not vendor claims):**
  - For a 1 km Germany-grid U-Net (~5–30 M params), **ONNX Runtime on CPU already gives seconds-level inference**; TensorRT buys little unless you run frequent full-resolution inferences, many lead times, or quantize.
  - `torch.compile` speeds **training/prototyping**; TensorRT targets **deployment inference**. They are complements, not alternatives.
  - Training such a net fits on **one consumer GPU (≥8–12 GB)**; the data pipeline (HDF5 → crops/tiles) usually dominates cost, not the model.
  - Typical export flow: PyTorch → `torch.onnx.export` (opset ≥17) → validate vs PyTorch (`onnxruntime` + numerical tolerances) → (optional) `trtexec --onnx=model.onnx --saveEngine=model.engine` per GPU arch (TensorRT engines are not portable across GPU architectures/versions).

---

## 4. Recommendation for the Nachtigall project

### 4.1 Option comparison (solo project, existing C++/FANN codebase)

| Option | Effort | Payoff | Verdict |
|---|---|---|---|
| **(a) Extend C++/FANN** | High | Low | FANN is an MLP library — no convolutions, no autograd, no GPU. Radar nowcasting is a dense 2-D sequence problem; hand-rolling conv/gradients/optimizer in C++ is months of work with no research upside. Keep C++ for I/O + serving only. |
| **(b) Custom small net: PyTorch → ONNX → C++ runtime** | Medium (≈4–10 weekends to v1) | High | **Recommended.** Modern tooling does the heavy lifting; you learn the full pipeline; model stays small and inspectable; deployment via ONNX Runtime C++ (or TensorRT later). Direct extension path for Nachtigall: replace `FillFANNTrainingData`-style steps with radar-tile datasets. |
| **(c) Pretrained nowcast architecture** | High | Medium-High (if it works) | No official weights exist for DGMR/NowcastNet; MetNet-3 is closed; GraphCast/Pangu are wrong scale/task. Adopt *architectures* (U-Net, MetNet-2-style dilated encoder, or a reimplemented DGMR discriminator/generator) inside option (b), not whole pipelines. |

### 4.2 Pragmatic MVP (U-Net on RADOLAN, +60 min target)

1. **Data:** download RADKLIM-YW 5-min archive (`5_minutes/radolan/reproc/2017_002/netCDF` or `bin`, years 2001–2026) and RADKLIM-RW hourly (`hourly/radolan/reproc/2017_002/{netCDF,bin}`, 2001–2026). Both are 1 km, same RADOLAN grid → trivially alignable. Sizing: ~1.1 M 5-min frames × ~2.4 MB raw ≈ TB-scale uncompressed — subsample (e.g. 30-min input spacing) or store as compressed 16-bit arrays; one year of decimated YW+RW pairs is already GBs.
2. **Task:** input = 4–12 preceding YW mosaics (1–2 h history) + static channels: **elevation (§4.3)**, radar-distance mask, hour-of-day + **NWP wind forecast fields valid over the lead-time window (§4.4)**; target = **RW valid +60 min** (mm/h, 1 km, Germany domain or 900×900 crop). Mask nodata/clutter bits (bit 14/16 markers in binary; `nodata` attr in HDF5/netCDF).
3. **Model:** small U-Net (3–4 down levels, ~5–20 M params), ReLU/GeLU, predict log1p(rate); loss = MSE (or weighted MSE emphasizing >0.1/1/5 mm thresholds) + optional CSI/FSS-style soft-threshold term. Train on one GPU; ~10–20 epochs over a few years is a realistic first run.
4. **Baselines & metrics:** persistence (RW now), **advection baseline via pysteps**, and the published **RADVOR-RS/RQ bias/ETS numbers** as reference; report CSI (hits on ≥0.1/1/5 mm/h) and FSS at +15/+30/+60 min over seasons. Optical flow is used here *only as a baseline to beat*, not as a model component (§4.4). Realistic v1 goal: beat persistence +30–60 min and approach pysteps; matching gauge-adjusted RADVOR at +1 h is a stretch goal.
5. **Deployment:** `torch.onnx.export` → ONNX; call from the C++ app via ONNX Runtime C++ (`Ort::Session`, CUDA EP optional). Fetch real-time RY/RW from `opendata.dwd.de/weather/radar/radolan/` (48 h window is enough for live inference). If latency matters later: TensorRT engine via the ONNX path.
6. **Stretch:** predict all 12 lead times (+5…+60 min) as an output stack (MetNet-2 style) or an encoder–decoder rollout; add HX (250 m reflectivity) as extra input channel once the 1 km pipeline works.

### 4.3 Elevation model as static input channel

Terrain is one of the strongest deterministic predictors of precipitation patterns:
orographic lift enhances windward rainfall, foehn/rain-shadow effects suppress
leeward precipitation, and valley/cold-air pooling affects bright-band and convective
initiation. A static elevation channel lets the network learn these
location-dependent climatological corrections instead of having to infer them from
radar history alone.

**Recommended source: Copernicus DEM GLO-30** (30 m, verified against official sources):

| Property | Value |
|---|---|
| Resolution | **30 m** (GLO-90 at 90 m also available; EEA-10 at 10 m is access-restricted) — both are ≥33× finer than the 1 km radar grid |
| Coverage | Global; full German coverage |
| Type | DSM (TanDEM-X, acquired 2011–2015; includes vegetation/buildings — irrelevant after 1 km resampling) |
| Access | Free for general public with registration at [Copernicus Data Space](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM); **no-registration mirror**: Cloud Optimized GeoTIFFs on AWS S3 (`copernicus-dem-30m` / `copernicus-dem-90m`, see [AWS Open Data registry](https://registry.opendata.aws/copernicus-dem/)) |
| Licence | [ESA User Licence for CCM data](https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf) — free use, attribution required |

Alternatives (if a non-ESA source is preferred):

- **SRTM 30 m** (NASA/USGS) — public domain over the US, free worldwide; older (2000) and lower vertical accuracy in forested terrain, but perfectly adequate at 1 km.
- **Official national DHMs**: DGM25 (Germany, BKG, 25 m), DHM25 (Switzerland), ALTI3D (France), BEV DGM (Austria) — best vertical accuracy, more licensing/coverage friction for a pan-German composite.

**Preprocessing (one-time, static):**

1. Mosaic the GLO-30 tiles covering Germany (COG HTTP range reads — no full download needed with GDAL `/vsicurl/`).
2. Reproject/resample from WGS84 geographic to the RADOLAN polar-stereographic 1 km grid (same `projdef` as the HDF5 files) using **cubic resampling**; average (not nearest) preserves terrain mass.
3. Derive additional static channels from the 1 km elevation field — these help the net far more than raw elevation alone:
   - **slope magnitude** and **aspect** (windward/leeward exposure depends on wind direction; aspect is best encoded as sin/cos pair),
   - **terrain roughness / local variance** (convective initiation over rough terrain),
   - optionally **elevation percentile within a radius** (orographic prominence).
4. Store as `float32` arrays aligned with the RADOLAN grid; broadcast over the time axis as constant input channels.

**Expected value:** modest but real — elevation is a *static* field, so it cannot help with advection (the dominant error source at +15 min), but it should improve skill at +30–60 min over complex terrain (Alps, Mittelgebirge) and reduce systematic windward/leeward bias. Evaluate with per-region CSI/FSS (flat north vs. Alpine south) to confirm the gain rather than relying on the aggregate score.

### 4.4 NWP wind forecast fields as input

Optical-flow advection (the RADVOR/pysteps approach) estimates motion from radar
images alone: it extrapolates *where echoes move* but knows nothing about *why* —
it cannot see convergence lines, orographic lifting, or wind shear, and it degrades
exactly where precipitation is generated or dissipated rather than advected.
Instead of encoding motion implicitly, feed the network the **physical driver
directly: NWP wind forecasts**. The model then learns the mapping
`wind field + terrain + current echoes → precipitation at +60 min` instead of
having to reverse-engineer motion from image sequences.

**Verified sources (both free, official):**

| Source | Product | Resolution / cadence | Access |
|---|---|---|---|
| **DWD ICON-D2** | `u/v` (and `w`) wind components on model levels & pressure levels, plus T2m, humidity | **2.2 km**, hourly steps to +7.5 h (EPS to +78 h) | <https://opendata.dwd.de/weather/nwp/> — free, no registration (GRIB2, `icon-d2/`) |
| **DWD ICON-EU** | same variables | 5 km, to +18 h | same server (`icon-eu/`) |
| **ECMWF IFS open data** | 10 m/100 m wind (`10u/10v`, `100u/10v`) | 0.25° (0.1° from 2026-10-20), 4 cycles/day, rolling ~2–3 day archive | <https://www.ecmwf.int/en/forecasts/datasets/open-data> — free; also mirrored on AWS/Azure/GCP |

ICON-D2 is the natural fit: German domain, 2.2 km (finer than the 1 km radar grid
after conservative regridding of a 2.2 km field — no false precision claimed),
hourly lead times covering the +60 min target, and it is the same model DWD runs
operationally.

**Preprocessing:**

1. Decode GRIB2 (eccodes/grib3 or wgrib2), select `u`/`v` at ~850–925 hPa (steering
   level for precipitation-bearing clouds) and 10 m (surface exchange), regrid to the
   RADOLAN 1 km grid (bilinear).
2. For each training sample, use the wind fields **valid over the lead-time window**
   (t₀ → t₀+60 min), not the analysis at t₀ — this is a forecast-to-forecast task,
   so no look-ahead leakage: at inference time the same ICON-D2 run is available.
3. Derived features that make the physics explicit for the network:
   - **horizontal advection vector** (u, v) per grid cell,
   - **orographic upslope component**: $w_{upslope} = -(u\,\partial_x h + v\,\partial_y h)$
     using the elevation gradient from §4.3 — directly encodes windward lift,
   - **convergence**: $-\nabla\cdot\mathbf{v}$ (frontal/convergence-line lifting),
   - optionally wind speed/direction as sin/cos channels for easier learning.
4. Historical caveat: ICON-D2 archives on opendata are a rolling window (recent
   months); for long training sets either (a) run a daily downloader from day one, or
   (b) start with reanalysis wind (ERA5, ~0.25°, free with registration) as a
   proxy for the training period and switch to ICON-D2 for inference. *(Uncertain:
   exact retention window of `opendata.dwd.de/weather/nwp/` — check before planning
   dataset size.)*

**Expected value:** larger than the elevation channel. Wind advection is the dominant
signal at +15–60 min; giving the net the *forecast* wind (not just observed motion)
should beat pure optical-flow extrapolation especially in advecting convective
systems, and the upslope/convergence features tie precipitation generation to
terrain. Evaluate against the same baselines (§4.2 step 4) — pysteps/RADVOR remain
*evaluation* baselines only; their optical-flow machinery is not part of the model.

### 4.5 Key risks

- **Class imbalance** (most pixels ≈ 0 mm): use threshold-weighted losses and event-based metrics; otherwise the net learns "no rain".
- **Distribution shift**: stratiform vs convective regimes; evaluate by season and by event (RADVOR's own docs note RAVOQ fails in convective situations — a genuine opening for a learned model).
- **Gauge-adjustment asymmetry**: RW/YW are calibrated; live RY is not — a model trained on YW→RW partly learns the gauge correction, which is legitimate but should be acknowledged.
- **License hygiene**: CC BY 4.0 requires attribution ("© Deutscher Wetterdienst" in product visualizations). Pangu repo has no LICENSE file — avoid copying code from it.

---

## Source list

**DWD (primary)**
- DWD Open Data server root: <https://opendata.dwd.de/weather/radar/> ; radar composite index: <https://opendata.dwd.de/weather/radar/composite/> ; RADOLAN products: <https://opendata.dwd.de/weather/radar/radolan/> ; RADVOR: <https://opendata.dwd.de/weather/radar/radvor/> ; single-site scans: <https://opendata.dwd.de/weather/radar/sites/>
- RADKLIM archives: <https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/> and <https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/>
- RADOLAN/RADVOR composite format spec v2.6 (2025-01-14): <https://www.dwd.de/DE/leistungen/radolan/radolan_info/radolan_radvor_op_komposit_format_pdf.pdf>
- RADOLAN Kurzbeschreibung: <https://www.dwd.de/DE/leistungen/radolan/radarniederschlagsprodukte/radolankurzbeschreibung_pdf.pdf>
- RADOLAN service page: <https://www.dwd.de/DE/leistungen/radolan/radolan.html> ; RADVOR service page (incl. discontinuation notice): <https://www.dwd.de/DE/leistungen/radvor/radvor.html>
- RADVOR-Kurzbeschreibung (Winterrath, 2020): <https://www.dwd.de/DE/leistungen/radvor/radvor_info/radvor_kurzbeschreibung_pdf.pdf>
- DWD Open Data access & legal basis: <https://www.dwd.de/DE/leistungen/opendata/opendata.html> ; license (CC BY 4.0): <https://www.dwd.de/DE/service/rechtliche_hinweise/rechtliche_hinweise.html>
- OPERA ODIM_H5 v2.3 format: <https://www.eumetnet.eu/wp-content/uploads/2019/01/ODIM_H5_v23.pdf>

**Models & code**
- DGMR paper: <https://arxiv.org/abs/2104.00954> , <https://doi.org/10.1038/s41586-021-03854-z> (no official code)
- MetNet-3 paper: <https://arxiv.org/abs/2306.06079> ; MetNet-2: <https://arxiv.org/abs/2111.07470> ; MetNet-1: <https://arxiv.org/abs/2003.12140> ; MetNet Global: <https://arxiv.org/abs/2510.13050> ; open reimplementation: <https://github.com/openclimatefix/metnet> (MIT, weights on HuggingFace `openclimatefix/metnet[-2]`)
- NowcastNet paper: <https://doi.org/10.1038/s41586-023-06184-4> (official repo removed — 404 on 2026-10-23; reimplementations: <https://github.com/Hitlia/NowcastNet> et al.)
- GraphCast/GenCast/WeatherNext code+weights: <https://github.com/google-deepmind/weathernext> ; GraphCast paper: <https://doi.org/10.1126/science.adi2336> ; WN2 report: <https://arxiv.org/abs/2506.10772>
- Pangu-Weather: <https://github.com/198808xc/Pangu-Weather> , <https://arxiv.org/abs/2211.02556> , <https://doi.org/10.1038/s41586-023-06185-3>
- pysteps: <https://github.com/pySTEPS/pysteps> (BSD-3), <https://github.com/pySTEPS/ERAD-nowcasting-course-2026>
- Community DGMR reimplementation: <https://github.com/TQRTQ/DGMR>

**NWP wind forecasts**
- DWD NWP open data root (ICON, ICON-D2, ICON-EU, GRIB2): <https://opendata.dwd.de/weather/nwp/>
- ECMWF Open Data (IFS 0.25°/0.1°, wind parameters, cloud mirrors): <https://www.ecmwf.int/en/forecasts/datasets/open-data>

**Elevation data**
- Copernicus DEM official description & access conditions: <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>
- Copernicus DEM on AWS (GLO-30/GLO-90 COG mirror, licence terms): <https://registry.opendata.aws/copernicus-dem/> ; dataset readme: <https://copernicus-dem-30m.s3.amazonaws.com/readme.html>
- ESA User Licence for Copernicus Contributing Missions data: <https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf>

**GPU stack**
- TensorRT OSS repo: <https://github.com/NVIDIA/TensorRT> ; import workflows: <https://github.com/NVIDIA/TensorRT/blob/main/documents/import_workflows.md> ; TensorRT developer guide: <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html>
- Torch-TensorRT: <https://github.com/pytorch/TensorRT> (BSD-3)
- PyTorch ONNX export: <https://docs.pytorch.org/docs/stable/onnx.html> ; ONNX: <https://onnx.ai> / <https://github.com/onnx/onnx> (Apache-2.0) ; ONNX Runtime: <https://github.com/microsoft/onnxruntime> (MIT)

*Uncertainties flagged in text: current status/successor of RADVOR beyond Feb 2026; exact semantics of opendata composite dirs `pg/vii/wn/hymecng`; whether RX is produced anywhere still; "RainMAE" could not be located in any primary source; Pangu-Weather repo lacks a formal license file; expected skill gain from the elevation and wind channels is a hypothesis to be validated per-region (§4.3/§4.4), not a measured result; exact retention window of the ICON-D2 archive on opendata.dwd.de must be checked before sizing the training dataset.*
