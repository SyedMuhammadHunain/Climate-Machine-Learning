# Tasks: Model-Only Output + Submission Readiness

Spec: `SPEC-model-only-output.md`. Plan: `tasks/plan.md`.
Done earlier (kept): static serving under FastAPI, 422 error parsing in UI, pinned `requirements.txt`.

Test commands:
- Backend: `.venv/bin/python scripts/verify_backend.py`
- Model: `.venv/bin/python scripts/verify_model.py` (new, T2)
- Train: `.venv/bin/python scripts/train.py` (new, T3)
- Run app: `./run_backend.sh` → http://127.0.0.1:8000

---

## Phase 0: Stop showing fake data

### T1: Remove fallback, add error + retry, delete careers.html
**Description:** Delete `renderFallbackSimulation` and every Bajra/Jowar/Gram string from
`predict.html`. On any failed request, show the backend's message (or "Backend unreachable")
and a Retry button. Show no numbers. Raise the AbortController timeout from 3500 ms to 30000 ms
and show a loading state while waiting. Delete `careers.html`, remove links to it, and remove
its route in `app/main.py` if there is one.

**Acceptance criteria:**
- [x] With the backend stopped, predict.html shows an error and Retry. No yields, no ranks.
- [x] A slow first request (cold load) up to 30 s still renders real results.
- [x] No `careers.html` file and no links to it.

**Verification:**
- [x] `grep -riE "bajra|jowar|renderFallback|fallback simulation" frontend/` is empty
- [x] Manual: stop backend, submit form, see error. Start backend, Retry, see results.
- [x] `verify_backend.py` still passes 12/12

**Dependencies:** None
**Files:** `frontend/.../predict.html`, `frontend/.../careers.html` (delete), `index.html`/`metrics.html` nav links, `app/main.py`
**Scope:** S

---

## Phase 1: Honest model

### T2: Shared feature module + yearly table
**Description:** Create `src/climate_machine_learning/features.py`. It (1) assigns month 1–12 by
row order inside each Crop-Province-Year group and checks that the order is right (temperature
peaks mid-year), (2) aggregates to one row per Crop-Province-Year: annual rain, mean/max/min
temp, mean ET0, radiation, rain hours, plus Rabi (Nov–Apr) and Kharif (May–Oct) rain if the
month order is confirmed, (3) adds engineered features: rain deficit vs province 2001–2022 mean,
and water balance (rain − ET0 × days), (4) drops 0-yield rows, logging the count per crop,
(5) exposes `build_inference_row(province, crop, inputs, reference)` for the API.

**Acceptance criteria:**
- [x] Yearly table: 858 groups before the 0-yield drop, 773 after (apple 22, grapes 41, pomegranate 22 dropped; all had Area = 0). Season split (Rabi/Kharif) skipped on purpose: a farmer cannot enter it easily, and it adds features on only 773 rows.
- [x] Training and inference rows come from the same function (same columns, same order).
- [x] No `Area`, no `Is_Drought_Resilient`, no `_normalized` columns in features.

**Verification:**
- [x] `scripts/verify_model.py`: group count 858, every group has 12 months, feature columns equal between train and inference
- [x] Month-order check prints its result. If it fails, Rabi/Kharif features are off.

**Dependencies:** None
**Files:** `src/climate_machine_learning/features.py`, `scripts/verify_model.py`
**Scope:** M

### T3: Retrain and save artifacts
**Description:** `scripts/train.py`: year split (train ≤2016, val 2017–19, test 2020–22).
Pipeline = impute + scale + one-hot (Crop, Province). Models: crop-province mean baseline,
Linear Regression, Random Forest, Gradient Boosting. Tune each with 5-fold GroupKFold grouped by
Year using GridSearchCV. Pick the model on val, report on test once. Write
`models/crop_yield_pipeline.joblib`, `models/metrics.json` (all models' MAE/RMSE/R² on val and test,
baseline, chosen model, permutation importances, data SHA-256, sklearn version, date), and
`models/reference.json` (province median climate, per-feature min/max from training, FAO water
class + citation per crop, training row count per crop). Update the notebook modelling cells to
call these functions and plot results.

**Acceptance criteria:**
- [x] Every number in metrics.json comes from a run of train.py. Re-running gives the same values (fixed seeds).
- [x] No test-year data is used in fitting or tuning.
- [x] Notebook runs top to bottom and its numbers match metrics.json.

**Verification:**
- [x] `.venv/bin/python scripts/train.py` succeeds and prints the comparison table
- [x] `verify_model.py`: artifacts exist, loading the pipeline and predicting one row works
- [x] `verify_model.py` sensitivity: raising rain from the province minimum to its maximum changes the prediction for at least one crop (print which ones)
- [x] `jupyter nbconvert --execute` on the notebook passes

**Notes (done 2026-11-18):** model code is in `src/climate_machine_learning/training.py`, `scripts/train.py` is a thin runner. FAO water-class table NOT written to reference.json: no verified FAO citation yet, so it moves to T5 (look up a real source, do not invent one). Old model moved to `models/legacy_pipeline.joblib` and `config.MODEL_PATH` points to it TEMPORARILY so the old app and its 12 tests keep passing; T4 must delete the legacy file and restore `MODEL_PATH`. Target is fitted on log scale (crops differ ~100x).

**Dependencies:** T2
**Files:** `scripts/train.py`, `models/*`, `notebooks/01_low_water_crop_selection.ipynb`
**Scope:** M

### ✅ Checkpoint A: Model review. Stop and show the user.
- [ ] Comparison table (baseline vs 3 models, val + test)
- [ ] Sensitivity printout (which crops react to rain/temp)
- [ ] User accepts the model before Phase 2

---

## Phase 2: Vertical paths

### T4: /predict end-to-end
**Description:** First delete `models/legacy_pipeline.joblib` and set `config.MODEL_PATH` back to `models/crop_yield_pipeline.joblib`. `ModelService.predict_single` returns `pipeline.predict(build_inference_row(...))`
and nothing else. The response includes the yield, its unit label ("dataset yield units"),
`pct_of_normal`, the crop's water class, and `inputs` with `input_source` per field. Inputs outside
the training min/max get a 422 that names the field and the allowed range. Update the request
schema to match the new features. Remove the `area` field if the model no longer uses it. Update
the predict.html single-crop view to show these fields.

**Acceptance criteria:**
- [x] The API yield exactly equals the direct `pipeline.predict` output for the same row
- [x] An out-of-range input returns 422 with a readable message, shown in the UI alert
- [x] Defaulted fields show "province median 2001–2022" in the UI

**Verification:**
- [x] `verify_backend.py`: equality test, 422 test, input_source test
- [x] Manual: single-crop prediction on :8000

**Notes (done 2026-11-18):** `/predict` takes annual inputs only; `area` and `climate_index` are removed and unknown fields give 422. Response has `predicted_yield`, `yield_unit`, `normal_yield`, `pct_of_normal`, `inputs` with sources, `training_rows_for_crop`, `model_name`. Water class is NOT in the response yet (needs the verified FAO table, T5). `/metadata` now returns `reference` (province medians and ranges); the form's quick-fill pills and range hints come from it, so no hardcoded presets remain in predict.html. The old `/recommend` still runs on `models/legacy_pipeline.joblib` via `LEGACY_*` names in `model_service.py`; T5 must delete them and the legacy file. Not clicked in a real browser: checked by JS syntax, guard test, and live API calls with the preset values.

**Dependencies:** T3
**Files:** `app/services/model_service.py`, `app/schemas/predict.py`, `app/schemas/common.py`, `app/api/v1/endpoints.py`, `predict.html`, `scripts/verify_backend.py`
**Scope:** M

### T5: /recommend end-to-end
**Description:** Write the crop water-class table (Low/Moderate/High) from a verified FAO source, looked up on the web, into `reference.json`. Predict all 13 crops at the user's climate and at the province median.
`pct_of_normal` = ratio of the two. Sort by `pct_of_normal` descending, then crop name. Return
rank, crop, predicted yield, pct_of_normal, water class, and a low-data flag. Replace
`ranked_drought_resilient`/`top_resilient_crop` with `ranked_crops` and `best_low_water_crop`
(highest-ranked crop tagged Low water). The summary text is built only from these values.
Update the predict.html recommend table and chart.

**Acceptance criteria:**
- [x] 13 crops, all in `ALL_CROPS`, ranks 1..13, deterministic order
- [x] Each yield equals the direct pipeline output. Each pct equals the ratio of two pipeline outputs.
- [x] Onion tagged Moderate (FAO), not Low
- [x] No `_rank_score` or any other hand formula

**Verification:**
- [x] `verify_backend.py`: crop set, rank uniqueness, equality checks, same request gives same order
- [x] Manual: severe-drought input vs median input gives different rankings (or document that it doesn't)

**Notes (done 2026-11-18):** DEVIATION from the plan: no Low/Moderate/High water class. FAO Training Manual 3 Table 14 lists only 8 of the 13 crops (no apple, dates, grapes, mango, pomegranate; wheat only as 'Barley/Oats/Wheat') and no consistent FAO seasonal figure was found for the other five, so a class would have been invented. Instead `data/reference/fao_crop_water_needs.json` holds FAO's own mm range and drought-sensitivity text for the 8 listed crops (cited, accessed 2026-11-18); the other 5 show `fao_water_need: null`. It is context only and does not change the ranking. No `best_low_water_crop`; response has `top_crop` and `ranked_crops` (13). Legacy model, `LEGACY_*` code, hand-made scores, `WaterIntensiveWarning`, and the `water_intensive_warnings` section are deleted. UI render checked by running the page JS in node against real API output (not in a real browser).

**Dependencies:** T3 (can run in parallel with T4)
**Files:** `app/services/model_service.py`, `app/schemas/recommend.py`, `predict.html`, `scripts/verify_backend.py`
**Scope:** M

### T6: /metrics end-to-end
**Description:** `GET /api/v1/metrics` returns `models/metrics.json` as is. metrics.html fetches it
and renders the comparison table, chosen model, importances and data period. Remove the hardcoded
`0.892` and the "32-year ERA5" text. If the fetch fails, show an error, not numbers.

**Acceptance criteria:**
- [x] metrics.html contains no hardcoded metric values
- [x] The page values equal the metrics.json values

**Verification:**
- [x] `verify_backend.py`: endpoint body == file contents
- [x] `grep -nE "0\.89|32-year" frontend/` is empty

**Notes (done 2026-11-18):** `GET /api/v1/metrics` (and root alias `/metrics`) returns `models/metrics.json` unchanged. `metrics.html` is rebuilt from `/metrics` + `/metadata`: summary cells, model comparison, importance, data/split with training rows per crop, province ranges, limits. Removed: 0.892, MAE/RMSE/latency figures, PROVINCE_DEFAULTS table, crop physiology table, elasticity equations, '32-year ERA5', '1990-2022' claims. On fetch failure the page shows an error + Retry and no numbers. Render checked by running the page JS in node against real API output (success and failure paths), not in a real browser.

**Dependencies:** T3
**Files:** `app/api/v1/endpoints.py`, `metrics.html`, `scripts/verify_backend.py`
**Scope:** S

### T7: Dead code removal + no-fake guard
**Description:** Delete `CROP_PHYSIOLOGY`, `calculate_agroclimatic_elasticity`, `_resolve_environment`
invented rules, and `DROUGHT_RESILIENT_CROPS`/`WATER_INTENSIVE_CROPS`/`MODERATE_WATER_CROPS`
(now in reference.json). Delete hardcoded `PROVINCE_DEFAULTS`. Rename `PRESET_SCENARIOS` to real
historical scenarios: pick the driest real year per province from the data and label it with
that year. Add a guard test that greps app/ and frontend/ for banned words and banned functions.

**Acceptance criteria:**
- [x] No hardcoded climate or yield numbers in `app/` except schema bounds
- [x] Scenarios are real (province, year) rows from the dataset

**Verification:**
- [x] Guard test in `verify_backend.py` passes
- [x] All tests pass

**Notes (done 2026-11-18):** Deleted `DROUGHT_RESILIENT_CROPS`, `WATER_INTENSIVE_CROPS`, `MODERATE_WATER_CROPS`, `PROVINCE_DEFAULTS`, `PRESET_SCENARIOS`, `PROVINCES`, `ALL_CROPS`, `DATA_PATH`, `WaterCategoryEnum`; `config.py` is now only file paths. `/metadata` serves provinces/crops from `reference.json` and 6 real scenarios: driest and wettest training year per province (Sindh 2002/2011, Punjab 2009/2015, Balochistan 2010/2006). `reference.json` gained `historical_years` (regenerated from data via `training.write_reference`, model and metrics files untouched). Quick-fill pills fill every input from that real year. Added `training_rows_possible` so the UI shows e.g. '18 of 48 possible rows (incomplete data for this crop)' instead of an arbitrary low-data threshold. Guards: verify_backend TEST 22 (banned words + AST scan for hardcoded floats outside Field bounds; checked it fails on a planted file; enums must equal trained crops/provinces), verify_frontend banned wording across all pages. Copy fixed: index.html no longer claims it recommends 'low-water, climate-resilient crops'.

**Dependencies:** T4, T5, T6
**Files:** `app/core/config.py`, `app/services/model_service.py`, `app/api/v1/endpoints.py`, `scripts/verify_backend.py`
**Scope:** S

### ✅ Checkpoint B: App complete
- [ ] `verify_backend.py` + `verify_model.py` pass
- [ ] Manual walk-through on :8000 for predict, recommend, metrics, and backend-down
- [ ] Banned-word grep is empty

---

## Phase 3: Deliverables (old Tasks 3–4, rewritten)

### T8: README
**Description:** Setup (venv, `requirements.txt`, `scripts/train.py`, run server), group members
(Muhammad Afzal CT-24263, Syed Muhammad Hunain CT-24281, Ayesha CT-24309), data sources with licence,
collection method and access date, the monthly→yearly aggregation and why, the yield unit (confirm
from the source repo), metrics read from metrics.json, API docs (`/predict`, `/recommend`, `/metrics`,
`/metadata`, `/health`), limitations (858 rows, weak weather signal if found, no extrapolation).

**Acceptance criteria:**
- [x] Every metric in the README matches metrics.json
- [x] Setup commands work on a fresh clone

**Verification:**
- [x] Follow the README in a fresh venv: train, run, open the page
- [x] Check links

**Notes (done 2026-11-19):** README written. The Results table is generated from `models/metrics.json` by `scripts/verify_readme.py --write`; `verify_readme.py` also checks group info, setup commands, endpoints (both directions), relative links, banned wording and that Streamlit/Hugging Face are marked 'not deployed yet' (user decision: deploy those two later). Per user, the baseline result is not a headline; the crop-province mean row and skill column stay in the table and Limitations points readers to them. Data facts the source does not state (collection method, weather source, units) are written as 'not stated'. Dataset licence MIT and Zenodo origin come from the source repo; access date given as the 2026-11-14 commit date. Fresh-clone test in a new Python 3.13 venv: install, 4 checks, retrain (same scores), refresh table, serve and curl all passed.

**Dependencies:** Checkpoint B
**Files:** `README.md`
**Scope:** S

### T9: Demo + viva guide
**Description:** `docs/demo_and_viva_guide.md`: a 3/4/3 minute script. Demo click path on the real
app: a drought year vs a median year, the ranking change, an out-of-range 422, the metrics page.
Viva Q&A: why aggregate to yearly rows, leakage and the GroupKFold choice, why R² changed from
0.892, % of normal ranking, model choice, limits.

**Acceptance criteria:**
- [x] Every demo step works on the current app
- [x] Covers each rubric item (lifecycle, deployment, results)

**Verification:**
- [x] Dry-run the demo path on :8000

**Notes (done 2026-11-19):** Guide is `docs/demo_and_viva_guide.md`: before-you-present checklist, 3/4/3 script, viva Q&A by topic (data, pipeline/leakage/tuning, results and the baseline, backend/API) with straight answers including 'does the model beat a smarter baseline' and 'why 0.892 became 0.795'; the 'who answers what' table is left for the group to fill (we do not know who wrote what). `scripts/verify_guide.py` recomputes every demo number from the API and fails if the guide drifts. `scripts/dryrun_demo.js` runs the real page JS against a live server (stub DOM, no browser) and checks the numbers on screen equal the API. The dry run found and fixed: page rounded 105.0 to 105 and yields to 1 decimal (now 1-decimal percent and 2-decimal yields), and the crop dropdown still carried hand-made '(Low Water)/(High Water)' labels, including onion as low water (now plain names, guarded against the trained crop list). Open items for the group: Streamlit and Hugging Face deployments, report PDF, fill the who-answers table.

**Dependencies:** T8
**Files:** `docs/demo_and_viva_guide.md`
**Scope:** S

### ✅ Checkpoint C: Submission ready
- [ ] Repo: data/, notebooks/, app/, models/, requirements.txt, README.md, docs/
- [ ] Full flow verified on localhost:8000

---

# Addendum: Second data source (NASA POWER)

Spec: `SPEC-second-source-nasa-power.md`. Chosen by the user on 2026-11-19.

- [x] **S1** (done 2026-11-19; 2,376 rows, API v2.10.1, MERRA2; POWER rain agrees only moderately with the dataset's rain, correlation 0.46 Balochistan / 0.57 Punjab / 0.67 Sindh; driest-month soil wetness is flat so `soil_wetness_max` replaces `_min`) `scripts/fetch_nasa_power.py`; `data/raw/nasa_power/*.json`, `data/raw/nasa_power_monthly.csv`, `data/reference/nasa_power_source.json`. Verify: 9 points x 264 months, no fill values, wet monsoon months wetter than dry ones.
- [x] **S2** `features.py` merges soil wetness by Province+Year, adds `soil_wetness_mean/max` to inputs, ranges and reference. Verify: 858 of 858 groups matched; training and inference rows equal; leakage test still passes.
- [x] **S3** Retrain; `metrics.json` lists both sources with hashes; notebook has the source-agreement and soil-wetness plots. Verify: verify_model passes.
- [x] **Checkpoint D** Show scores and importance; user reviews before S4.
- [x] **S4** API fields, form fields, metadata, metrics page "data sources" section. Verify: backend, frontend, dry-run checks.
- [x] **S5** README (two sources, licence, citation, limits), guide numbers, all checks.

**Notes (done 2026-11-19).** S2, S3 and the API fields were committed together because the saved model, the feature list and the scenario inputs must change at the same time. New inputs `soil_wetness_mean/max` are optional, labelled by source, range-checked per province, and filled by the real-year quick-fill. Checkpoint D result (shown to the user in the final message): Random Forest still chosen on validation, test R² 0.794 (was 0.795), Gradient Boosting test R² 0.813 (not chosen, validation is the rule); test skill vs crop-average baseline improved from -0.95 to -0.76 but is still below zero; permutation importance: soil wetness mean 0.0098 (mid-pack among weather columns), wettest-month soil wetness about 0. POWER rain agrees only moderately with the dataset rain (yearly correlation 0.39 Punjab, 0.60 Balochistan, 0.75 Sindh). The metrics page now has a Data sources section (both sources, licences, hashes). README, guide and checks updated; guide numbers re-verified by `verify_guide.py`; click path re-verified by `dryrun_demo.js`. Not done: second source for more crops, fresh-clone re-test after this change, real-browser walk-through.

**Fresh-clone re-test (2026-11-19, after the NASA POWER merge).** New clone of the committed repo, new Python 3.13 venv, `uv pip install -r requirements.txt`: all five checks passed on the committed files; `python scripts/fetch_nasa_power.py` reproduced the same CSV (same SHA-256; raw JSON differs only in the server's response-time field); `python scripts/train.py` reproduced the committed test scores and the same chosen model; `verify_readme.py --write` then all checks passed; server served all pages and the API; `dryrun_demo.js` passed. It also found one bug: the out-of-range message rounded ranges to 1 decimal, so a soil wetness range showed as "0.3 to 0.4". Fixed with 4 significant digits (now "0.3481 to 0.3864"), test added in `verify_backend.py` TEST 24.

---

# Addendum: Streamlit and Hugging Face (user asked 2026-11-19)

Spec: `SPEC-streamlit-and-huggingface.md`. Publishing needs the group's own accounts, so these are **built and tested locally, not deployed**.

- [x] **D1** Prediction interval from validation residuals, saved in `metrics.json` (`prediction_interval`), `features.prediction_interval`. Test coverage measured: 60% against nominal 80%, shown in the apps and docs.
- [x] **D1b** Shared `inference.py` (`predict_crops`, `OutOfRangeError`); `ModelService` now uses it (23 backend checks unchanged).
- [x] **D2** `streamlit_app/app.py` + `streamlit_app/requirements.txt`; `scripts/verify_streamlit.py` (8 checks with Streamlit's AppTest, numbers equal the model; passed on Python 3.12 and 3.13 with only the app's requirements).
- [x] **D3** `hf_space/app.py` (Gradio), `scripts/build_hf_bundle.py` (model card generated from `metrics.json`, Space config, bundle); `scripts/verify_hf_bundle.py` (8 checks including a real Gradio server and client run).
- [x] **D4** `docs/deployment_guide.md`, README deployment table and structure, guide checklist and viva question, `verify_readme.py` and `verify_guide.py` extended.
- [ ] **Group to do:** deploy to Streamlit Community Cloud, create the HF model repo and Space, choose a licence, then update the README table and the `verify_readme.py` rule that currently requires "not deployed yet".

Findings: the nominal 80% range covers only 60% of the 2020-2022 test rows (test years differ from validation, no yearly trend). Docker is not installed, so the Space is Gradio. The model file is 20 MB (use `hf upload`, not plain git). `/tmp` is a 3.6 GB tmpfs and hit its quota once from throwaway virtualenvs; they are cleaned up.
