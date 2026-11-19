# Delta SPEC: Second data source (NASA POWER soil wetness)

Status: in progress
Parent specs: `SPEC.md`, `SPEC-model-only-output.md` (all nine tasks done). This delta only adds a second public source.

## Why
The CT-378 guidelines ask to "combine at least two public sources". The model uses one dataset (Zenodo/GitHub crop yield and climate). The proposal's other sources were checked on 2026-11-19 and do not fit:

| Proposal source | Finding |
|---|---|
| Punjab district wheat yield (PBS, opendata.com.pk) | Wheat only, Punjab only, years from about 2007, no licence stated. Weather is province-level, so districts would repeat one feature vector with different targets. |
| Punjab irrigated vs total area (opendata.com.pk) | A few mostly rabi crops (wheat, gram, barley, rape/mustard), Punjab only. |
| Pakistan weather observations (GitHub rabia743) | Islamabad, Karachi, Lahore only (no Balochistan). No licence or origin stated. A football-weather project. |
| PMD NDMC soil moisture | A website, no downloadable series found. |

Chosen by the user: **NASA POWER** (monthly, point queries, MERRA-2 based). It covers all 3 provinces and 2001-2022, and adds the proposal's "soil moisture" category.

## Objective
Add root-zone soil wetness (a drought signal) from NASA POWER as new model inputs, merged with the existing table by province and year. Train, evaluate and report honestly. If the result is no better, say so; do not tune toward a better-looking number.

## Source facts (checked 2026-11-19)
- API: `https://power.larc.nasa.gov/api/temporal/monthly/point` (POWER Monthly and Annual API v2.10.1), source MERRA2, community AG.
- Parameters: `GWETROOT` root zone soil wetness (unit 1, range 0-1), `PRECTOTCORR` corrected precipitation (mm/day).
- Licence: the POWER catalog entry says "License not specified". NASA's default for mission data is CC0, but that is not confirmed for POWER. POWER asks users to cite its Reference and Data Reference (<https://power.larc.nasa.gov/docs/referencing/>). We cite it and say the licence is not specified.
- Access date: 2026-11-19.

## Design
- **Points** (a judgment, documented as a limitation): 3 farming-area points per province, equal weight.
  - Punjab: Lahore 31.5 N 74.3 E, Faisalabad 31.4 N 73.1 E, Multan 30.2 N 71.5 E
  - Sindh: Sukkur 27.7 N 68.9 E, Hyderabad 25.4 N 68.4 E, Mirpurkhas 25.5 N 69.0 E
  - Balochistan: Quetta 30.2 N 67.0 E, Jaffarabad 28.5 N 68.0 E, Turbat 26.0 N 63.0 E
- **Data stored in the repo** (no network at train time): raw API JSON per point in `data/raw/nasa_power/`, tidy `data/raw/nasa_power_monthly.csv`, and `data/reference/nasa_power_source.json` (points, versions, access date, citation, sha256).
- **New model inputs** (annual, per province-year): `soil_wetness_mean` (mean of the 12 monthly GWETROOT values, averaged over the points) and `soil_wetness_min` (the driest month). Optional API inputs, filled from the province median when empty, labelled by source like the others.
- **POWER rain** (`PRECTOTCORR`, converted to mm per year) is used only as a cross-check against the dataset's own rain (agreement plot and correlation in the notebook). It is not a model input, so users still enter one rain number.
- Merge key: Province + Year. All 858 groups must match; the merge fails loudly otherwise.
- Everything else unchanged: year split, GroupKFold by year, models, selection on validation, test scored once.

## Commands
- Fetch (needs internet): `python scripts/fetch_nasa_power.py`
- Train: `python scripts/train.py`
- Checks: `python scripts/verify_model.py`, `verify_backend.py`, `verify_frontend.py`, `verify_readme.py`, `verify_guide.py`, and `node scripts/dryrun_demo.js <url>`.

## Boundaries
- Always: keep raw API responses in the repo; say plainly that the licence is not specified; report the new scores as they come out, including if they get worse.
- Ask first: adding more POWER parameters, changing the point choice, dropping the old features.
- Never: pick points or features by looking at test-year scores; hide a worse result; call the data "licensed CC0".

## Risks
| Risk | Mitigation |
|---|---|
| POWER soil wetness is almost flat in dry months (about 0.37) | Report the real signal. Check variation by year in the notebook. |
| Point choice is subjective | Documented, equal weights, listed under Limitations. |
| Scores fall or stay flat | Report as is. Update README, guide and the Metrics page from the files. |
| Licence "not specified" | Cite as requested, state it plainly. |
| API unavailable later | Raw responses are committed, so nothing needs the network after S1. |
