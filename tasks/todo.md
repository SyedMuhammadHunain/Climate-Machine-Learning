# Tasks: CT-378 Local Web App Finalization & Submission Readiness

## Phase 1: Core App Alignment & Error Handling

### Task 1: Unify FastAPI Static Serving & Fix Frontend Error Handling
**Description:** Configure FastAPI in `app/main.py` to mount `frontend/climate-resilient-frontend` as static assets with HTML routing so running `uvicorn app.main:app` serves the entire application (HTML pages and API) on `http://127.0.0.1:8000`. Update `predict.html` (and `careers.html`) error handling logic to parse both `err.message` and `err.details` from the backend's custom exception responses so physical validation rejections display explicit error messages to the user.

**Acceptance criteria:**
- [x] `app/main.py` points `STATIC_DIR` to `frontend/climate-resilient-frontend` and serves `index.html`, `predict.html`, and `metrics.html` cleanly.
- [x] Root route `/` serves `index.html`.
- [x] `predict.html` error parser displays explicit messages from 422 responses (`err.message` or formatted `err.details`) in `#validation-alert` rather than falling back to generic text.
- [x] Backend verification tests pass without regression.

**Verification:**
- [x] `.venv/bin/python scripts/verify_backend.py` passes all 12 tests.
- [x] `curl -s http://127.0.0.1:8000/predict.html` returns 200 OK when server is running.
- [x] Submitting invalid inputs triggers informative text inside the UI `#validation-msg`.

**Dependencies:** None
**Files likely touched:**
- `app/main.py`
- `frontend/climate-resilient-frontend/predict.html`
- `frontend/climate-resilient-frontend/careers.html`
**Estimated scope:** S (3 files)

---

### Task 2: Generate Pinned `requirements.txt` & Environment Verification
**Description:** Generate a fully pinned `requirements.txt` capturing all runtime and modeling dependencies (FastAPI, uvicorn, scikit-learn, joblib, numpy, pandas, pydantic) from the working `.venv` to satisfy CT-378 ML Lifecycle step 7 and deliverables list. Verify clean dependency resolution.

**Acceptance criteria:**
- [ ] `requirements.txt` created at repository root with exact pinned versions.
- [ ] Includes `scikit-learn`, `joblib`, `fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`.
- [ ] No extraneous system build tools or broken platform tags.

**Verification:**
- [ ] `grep -E "scikit-learn|joblib|fastapi" requirements.txt` matches pinned versions.
- [ ] `.venv/bin/python -m pip check` reports no broken dependency trees.

**Dependencies:** None
**Files likely touched:**
- `requirements.txt`
**Estimated scope:** XS (1 file)

---

## Checkpoint: Local Serving & Validation
- [ ] Backend verification test suite passes 10/10 tests.
- [ ] Local web app launches with a single command and serves all HTML routes and API routes.
- [ ] Input validation and custom error messages verified end-to-end.

---

## Phase 2: Submission Deliverables & Viva Prep

### Task 3: Author Comprehensive `README.md`
**Description:** Write an authoritative `README.md` fulfilling all CT-378 deliverable guidelines. Cover project motivation (low-water crop selection under drought in Pakistan), group members and roll numbers, quickstart instructions to launch the Local Web App on localhost, architecture overview, dataset documentation (rainfall, ERA5, NDMC, provincial crop yield statistics), ML pipeline and model evaluation summary ($R^2=0.892$, RMSE, MAE), API documentation for `/predict` and `/recommend`, and deployment status.

**Acceptance criteria:**
- [ ] `README.md` written at repository root.
- [ ] Contains group member names and roll numbers (Muhammad Afzal CT-24263, Syed Muhammad Hunain CT-24281, Ayesha CT-24309).
- [ ] Explicit setup commands for creating venv, installing `requirements.txt`, and launching `uvicorn app.main:app --port 8000`.
- [ ] Lists data sources, license, collection methods, and features per guideline rules.
- [ ] Explains Local Web App platform choice and marks all endpoints (`/predict`, `/recommend`, `/health`, `/metadata`).

**Verification:**
- [ ] Inspect `README.md` for completeness and formatting.
- [ ] Check links and command syntax accuracy.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `README.md`
**Estimated scope:** S (1 file)

---

### Task 4: Create Live Demo Script & Viva Preparation Guide
**Description:** Create `docs/demo_and_viva_guide.md` providing a structured guide for the 10-minute presentation and individual viva. Break down the time budget (3 min Problem & Data, 4 min Live Local App Demo, 3 min Results & Lifecycle) with exact speaking notes, demo click paths (showing severe drought shifts from Onion to Dates, and demonstrating physical error validation), and anticipate individual viva questions for each student.

**Acceptance criteria:**
- [ ] `docs/demo_and_viva_guide.md` created.
- [ ] Timed demo breakdown structured exactly to 3 min / 4 min / 3 min format.
- [ ] Includes step-by-step UI actions to show during the 4-minute demo.
- [ ] Includes targeted viva technical questions and answers covering scikit-learn pipeline, avoiding data leakage, cross-validation tuning, and physical agronomic formulas ($K_w, T_{\text{crit}}$).

**Verification:**
- [ ] Review guide structure against the 10-mark rubric in CT-378 guidelines.

**Dependencies:** Task 1, Task 3
**Files likely touched:**
- `docs/demo_and_viva_guide.md`
**Estimated scope:** S (1 file)

---

## Checkpoint: Final Review & Submission Readiness
- [ ] Repository contains all required components: `data/`, `notebooks/`, `app/`, `models/`, `requirements.txt`, `README.md`, `docs/demo_and_viva_guide.md`.
- [ ] Demo execution verified on localhost:8000.
- [ ] Ready for final project demonstration and individual viva.
