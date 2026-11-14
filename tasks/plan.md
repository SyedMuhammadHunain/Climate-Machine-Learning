# Implementation Plan: Interactive Submit Feedback & Result State Management

## Overview
Address the issue where clicking "Run Crop Yield Prediction" provides no perceptible visual feedback to the user ("nothing happens"). This occurs because static results were pre-rendered on page load, the offline fallback resolved instantaneously without scrolling or transition, and no visual timestamp/status badge indicated execution. This phase adds clean initial state hiding, smooth auto-scroll to results, a noticeable 350ms perceptible loading transition, an Ashby-style execution timestamp badge, and live FastAPI server integration.

## Architecture Decisions
- **Empty Initial State**: Hide the results container on initial page load (or display a subtle empty placeholder), ensuring that clicking "Run Crop Yield Prediction" causes an unmistakable visual transition when results appear.
- **Perceptible Transition Window**: Guarantee a minimum 350ms execution window before hiding the spinner so users visibly see the pipeline processing, avoiding 0ms invisible blinks when network calls failover immediately.
- **Smooth Viewport Auto-Scroll**: Upon successful inference or fallback calculation, trigger `resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' })` so user focus immediately tracks to the newly calculated metrics.
- **Execution Badge & Source Indicator**: Display an Ashby-style pill badge above results displaying the timestamp and data source (`Live ML Model (Port 8000)` vs `Agronomic Simulation Mode`).
- **Live Backend Runner**: Provide verified startup for the FastAPI service (`uvicorn app.main:app --port 8000`) so the frontend can query the real Scikit-Learn pipeline directly.

---

## Task List

### Phase 3: Submit Responsiveness, Visual State Transitions & Live Backend

- [ ] **Task 1: Refactor Results Container to Initial Empty State with Smooth Auto-Scroll**
  - Hide `#results-container` on initial load; reveal it dynamically upon form submission.
  - Add smooth auto-scrolling to `#results-container` when prediction calculation completes.
  - Implement a minimum 350ms perceptible loading window to avoid instant 0ms blinks.

- [ ] **Task 2: Add Real-Time Inference Badge and Cache-Busting Headers**
  - Add an Ashby-style status badge showing inference timestamp and execution source (Live Model vs Local Agronomic Engine).
  - Add cache-busting meta headers to prevent browsers from caching stale JavaScript files.
  - Sync updates between `predict.html` and `careers.html`.

- [ ] **Task 3: Launch & Connect Live FastAPI Backend Service**
  - Launch FastAPI application (`app.main:app`) on port 8000 using project Python environment.
  - Verify live API communication between frontend and backend via CDP automated test.

---

## Checkpoint: Phase 3 Verification
- [ ] Initial page load shows form cleanly without pre-baked results cluttering the page.
- [ ] Clicking "Run Crop Yield Prediction" displays loading spinner for at least 350ms, reveals results container, and auto-scrolls down smoothly.
- [ ] Status badge explicitly confirms timestamp and engine source.
- [ ] Automated headless Chrome DevTools test verifies DOM mutation, visibility, and click reaction.

---

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| User browser disk-caching old HTML/JS | High | Add HTML `Cache-Control` meta tags and verify via headless Chrome with cache disabled. |
| Fast failover feels like an unresponsive button | High | Introduce 350ms artificial transition delay so users register the action. |
| Backend port 8000 conflict | Medium | Check port availability before launching uvicorn daemon. |
