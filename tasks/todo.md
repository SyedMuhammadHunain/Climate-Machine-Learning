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
- [ ] The API yield exactly equals the direct `pipeline.predict` output for the same row
- [ ] An out-of-range input returns 422 with a readable message, shown in the UI alert
- [ ] Defaulted fields show "province median 2001–2022" in the UI

**Verification:**
- [ ] `verify_backend.py`: equality test, 422 test, input_source test
- [ ] Manual: single-crop prediction on :8000

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
- [ ] 13 crops, all in `ALL_CROPS`, ranks 1..13, deterministic order
- [ ] Each yield equals the direct pipeline output. Each pct equals the ratio of two pipeline outputs.
- [ ] Onion tagged Moderate (FAO), not Low
- [ ] No `_rank_score` or any other hand formula

**Verification:**
- [ ] `verify_backend.py`: crop set, rank uniqueness, equality checks, same request gives same order
- [ ] Manual: severe-drought input vs median input gives different rankings (or document that it doesn't)

**Dependencies:** T3 (can run in parallel with T4)
**Files:** `app/services/model_service.py`, `app/schemas/recommend.py`, `predict.html`, `scripts/verify_backend.py`
**Scope:** M

### T6: /metrics end-to-end
**Description:** `GET /api/v1/metrics` returns `models/metrics.json` as is. metrics.html fetches it
and renders the comparison table, chosen model, importances and data period. Remove the hardcoded
`0.892` and the "32-year ERA5" text. If the fetch fails, show an error, not numbers.

**Acceptance criteria:**
- [ ] metrics.html contains no hardcoded metric values
- [ ] The page values equal the metrics.json values

**Verification:**
- [ ] `verify_backend.py`: endpoint body == file contents
- [ ] `grep -nE "0\.89|32-year" frontend/` is empty

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
- [ ] No hardcoded climate or yield numbers in `app/` except schema bounds
- [ ] Scenarios are real (province, year) rows from the dataset

**Verification:**
- [ ] Guard test in `verify_backend.py` passes
- [ ] All tests pass

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
- [ ] Every metric in the README matches metrics.json
- [ ] Setup commands work on a fresh clone

**Verification:**
- [ ] Follow the README in a fresh venv: train, run, open the page
- [ ] Check links

**Dependencies:** Checkpoint B
**Files:** `README.md`
**Scope:** S

### T9: Demo + viva guide
**Description:** `docs/demo_and_viva_guide.md`: a 3/4/3 minute script. Demo click path on the real
app: a drought year vs a median year, the ranking change, an out-of-range 422, the metrics page.
Viva Q&A: why aggregate to yearly rows, leakage and the GroupKFold choice, why R² changed from
0.892, % of normal ranking, model choice, limits.

**Acceptance criteria:**
- [ ] Every demo step works on the current app
- [ ] Covers each rubric item (lifecycle, deployment, results)

**Verification:**
- [ ] Dry-run the demo path on :8000

**Dependencies:** T8
**Files:** `docs/demo_and_viva_guide.md`
**Scope:** S

### ✅ Checkpoint C: Submission ready
- [ ] Repo: data/, notebooks/, app/, models/, requirements.txt, README.md, docs/
- [ ] Full flow verified on localhost:8000
