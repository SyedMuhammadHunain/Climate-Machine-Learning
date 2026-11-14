# Implementation Plan: ML Input Form for Climate-Resilient Crop Selection

## Overview
Transform `frontend/climate-resilient-frontend/predict.html` into a dedicated Machine Learning Input Form and Advisory Interface matching the Ashby aesthetic (`jobs.ashbyhq.com/infinitefield/`). The form will expose all essential agricultural and climate features required by the FastAPI/Scikit-Learn backend (`SinglePredictionRequest` and `DroughtScenarioRequest`), organized into primary core fields and a collapsible advanced meteorological section, with one-click presets and live prediction output.

## Architecture Decisions
- **Ashby Visual Identity:** Maintain exact `720px` width container, inline vector `Infinite Field®` wordmark, Whitney SSm typography, `#FF7344` brand accents, and subtle `#d7dae0` bordered inputs.
- **Hierarchical Form Structure:**
  - **Core Grid (Always Visible):** Province, Crop / Recommendation Mode, Cultivated Area (ha), Seasonal Precipitation (mm), and Mean Temperature (°C).
  - **Advanced Section (Collapsible Accordion):** Max Temperature (°C), Min Temperature (°C), FAO Evapotranspiration (mm/day), Solar Shortwave Radiation (MJ/m²), Precipitation Hours, and Climate Dryness Index.
- **One-Click Agroclimatic Presets:** Buttons for "Severe Drought", "Moderate Drought", "Baseline Normal", and "Extreme Heatwave" that auto-populate realistic Pakistan meteorological benchmarks.
- **Backend Connector & Resilient Offline Fallback:** Dispatches asynchronous `fetch()` to `http://localhost:8000/api/v1/predict` or `/api/v1/recommend`. If the backend service is offline, seamlessly computes locally using pre-calculated agronomic weights so the interface always responds.

---

## Task List

### Phase 1: Form Structure & Input Fields
- [ ] **Task 1: Build Core Input Grid and Preset Scenario Selector**
  - Implement 2-column Ashby-styled form fields for Province, Crop, Area, Seasonal Rainfall, and Mean Temperature with preset buttons.
- [ ] **Task 2: Build Collapsible Advanced Meteorological Parameters Section**
  - Implement collapsible accordion containing Max/Min Temp, Evapotranspiration, Solar Radiation, Precipitation Hours, and Climate Index with agronomic defaults.

### Checkpoint: Form Layout & Accessibility
- [ ] All 10 input controls rendered cleanly within 720px container.
- [ ] Responsive on mobile and desktop viewports.
- [ ] Preset buttons correctly update all form fields.

### Phase 2: Action Button & Inference Integration
- [ ] **Task 3: Implement Submit Action, Client Validation, and Backend API Dispatch**
  - Add Ashby-styled "Run Crop Yield Prediction" button. Wire client-side input validation (`max_temp >= min_temp`, valid ranges). Send `POST` requests to `/api/v1/predict` or `/api/v1/recommend`.
- [ ] **Task 4: Build Dynamic Categorized Results View**
  - Render model yield predictions, water footprint classifications, and drought risk warnings in Ashby listing format with fallback simulation.

### Checkpoint: Inference & Verification
- [ ] Single crop prediction renders expected harvest yield and water category.
- [ ] Multi-crop recommendation ranks drought-resilient crops and flags high-risk water-intensive crops.
- [ ] Zero unhandled JavaScript errors in browser console.

---

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Backend API offline during frontend review | Medium | Include comprehensive offline fallback simulation using Pakistani agricultural benchmarks. |
| Incompatible Min/Max temperatures | Low | Client-side validation prevents submission when `max_temp < min_temp` with clear inline feedback. |
