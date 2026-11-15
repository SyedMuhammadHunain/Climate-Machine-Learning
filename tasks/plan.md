# Implementation Plan: Dedicated Model Metrics & Agroclimatic Benchmarks Page (`metrics.html`)

## Overview
Create a dedicated "Metrics & Agroclimatic Benchmarks" page (`metrics.html`) linked from the site navigation. This page serves academic evaluators, project reviewers, and agricultural stakeholders by showcasing model performance statistics ($R^2$, RMSE, MAE), historical provincial climate baseline tables (Punjab, Sindh, Balochistan), crop physiological coefficients (vulnerability factor $K_w$, critical heat thresholds $T_{\text{crit}}$), and agroclimatic elasticity equations, keeping `predict.html` clean and focused on prediction.

## Architecture Decisions
- **Page Design & Aesthetic**: Reuse the clean Ashby-inspired aesthetic of `predict.html` (720px centered container, crisp typography with `DenimINK-Regular` and `LD_TechD-Regular`, monochromatic cards, accessible contrast, responsive layout).
- **Offline-First & Zero CDNs**: Use 100% local CSS and vanilla JavaScript; no external CDNs or network scripts.
- **Dynamic Metadata & Fallback**: Integrate live fetch from `/api/v1/metadata` and `/health` with static agronomic fallbacks so the page renders identically whether the backend API is live or offline.
- **Cross-Page Navigation**: Provide an intuitive header navigation bar (`Overview` $\rightarrow$ `index.html`, `Predictor & Advisory` $\rightarrow$ `predict.html`, `Model & Benchmarks` $\rightarrow$ `metrics.html`) across all frontend pages.

## Task List

### Phase 1: Benchmarks Page Creation
- [ ] Task 1: Create `metrics.html` with Model Accuracy, Provincial Climate Baselines, and Crop Physiology Reference Tables
- [ ] Task 2: Implement Cross-Page Header Navigation in `predict.html`, `careers.html`, `index.html`, and `metrics.html`

### Checkpoint: Verification
- [ ] `metrics.html` renders cleanly with 200 OK locally.
- [ ] Navigation links between `index.html`, `predict.html`, and `metrics.html` operate seamlessly.
- [ ] Zero external CDN dependencies in `metrics.html`.
- [ ] All 10 backend verification tests pass.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Backend offline prevents metrics display | Medium | Embed hardcoded reference statistics and provincial baseline tables matching `PROVINCE_DEFAULTS` and model artifacts as fallback defaults. |
| Navigation breaks mobile layout | Low | Use compact flex navigation pills with wrapping and touch-friendly tap targets ($\ge 44\text{px}$). |
