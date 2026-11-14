# Tasks: Interactive Submit Feedback & Result State Management

## Phase 3: Submit Responsiveness, Visual State Transitions & Live Backend

### Task 1: Refactor Results Container to Initial Empty State with Smooth Auto-Scroll
**Description:** Modify `frontend/climate-resilient-frontend/predict.html` and `careers.html` so the `#results-container` is hidden by default (`display: none`) on initial page load. When the user clicks "Run Crop Yield Prediction", enforce a minimum 350ms loading duration so the inference step is visually apparent, reveal `#results-container`, and smoothly scroll into view.

**Acceptance criteria:**
- [x] `#results-container` starts hidden on page load so form is uncluttered.
- [x] Loading spinner displays for at least 350ms to provide perceptible tactile feedback.
- [x] Viewport smoothly scrolls to `#results-container` once results are calculated.

**Verification:**
- [x] On clean page load, results container is not visible.
- [x] On button click, spinner displays, results container appears, and page scrolls.

**Dependencies:** None
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
- `frontend/climate-resilient-frontend/careers.html`
**Estimated scope:** S (2 files)

---

### Task 2: Add Real-Time Inference Badge and Cache-Busting Headers
**Description:** Add an Ashby-style indicator pill above the results list displaying `• Updated at [HH:MM:SS] • [Engine Source]`. Add `<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">` to prevent browser cache from holding stale versions.

**Acceptance criteria:**
- [x] Status badge appears above recommendations showing exact execution timestamp.
- [x] Badge clearly distinguishes `Live ML Model (FastAPI)` from `Agronomic Simulation Engine`.
- [x] Cache-control meta headers present in `<head>`.

**Verification:**
- [x] Submitting form updates badge with new timestamp.
- [x] Cache headers present in HTTP/HTML inspect.

**Dependencies:** Task 1
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
- `frontend/climate-resilient-frontend/careers.html`
**Estimated scope:** S (2 files)

---

### Task 3: Launch & Connect Live FastAPI Backend Service
**Description:** Start the FastAPI application on `http://127.0.0.1:8000` using `.venv/bin/python -m uvicorn app.main:app --port 8000`. Test end-to-end inference from the frontend form to verify live model predictions work alongside the fallback.

**Acceptance criteria:**
- [ ] Backend responds with 200 OK on `/docs` and `/api/v1/recommend`.
- [ ] Form submission successfully queries live API and renders ML pipeline outputs.

**Verification:**
- [ ] Automated CDP test verifies live API submission and badge text `Live ML Model (FastAPI)`.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `tasks/todo.md`
**Estimated scope:** S (0-1 files)

---

## Checkpoint: Phase 3 Verification
- [ ] Button click provides unmistakable visual feedback (spinner -> auto-scroll -> results).
- [ ] Works both with live FastAPI server running and offline fallback.
