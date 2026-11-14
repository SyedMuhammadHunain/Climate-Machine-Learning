# Tasks: ML Input Form for Climate-Resilient Crop Selection

## Phase 1: Form Structure & Input Fields

### Task 1: Build Core Input Grid and Preset Scenario Selector
**Description:** Implement the primary input controls in `frontend/climate-resilient-frontend/predict.html` using the Ashby minimalist form styling (720px max width, `#d7dae0` border, `8px` radius). Add fields for Province (Punjab, Sindh, Balochistan), Crop Selection (All Crops Recommendation vs. 9 individual crops), Cultivated Area (ha), Seasonal Precipitation (mm), and Mean Temperature (°C). Include 4 quick preset buttons ("Severe Drought", "Moderate Drought", "Baseline Normal", "Extreme Heatwave").

**Acceptance criteria:**
- [x] Core inputs for Province, Crop, Area, Precipitation, and Temperature rendered in responsive 2-column grid.
- [x] One-click scenario preset buttons update inputs with realistic agroclimatic numbers.
- [x] Typography and borders match Ashby design tokens (Whitney SSm, 14px text, 40px input height).

**Verification:**
- [x] All 5 core inputs present with valid min/max constraints matching FastAPI Pydantic schema.
- [x] Clicking preset buttons immediately updates input values.

**Dependencies:** None
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
**Estimated scope:** S (1 file)

---

### Task 2: Build Collapsible Advanced Meteorological Parameters Section
**Description:** Add an expandable/collapsible "Advanced Meteorological Parameters" section below the core grid in `frontend/climate-resilient-frontend/predict.html`. This accordion will contain the secondary features used by the ML pipeline: Maximum Temperature (`temperature_2m_max`), Minimum Temperature (`temperature_2m_min`), FAO Evapotranspiration (`fao_evapotranspiration`), Solar Radiation (`shortwave_radiation`), Precipitation Hours (`precipitation_hours`), and Climate Index (`climate_index`).

**Acceptance criteria:**
- [x] Accordion toggles open and closed with subtle chevron indicator without page reloads.
- [x] Contains 6 advanced meteorological inputs pre-populated with Pakistan regional averages.
- [x] Styled cleanly within the Ashby design system.

**Verification:**
- [x] Accordion toggle works on click.
- [x] Input values read correctly into JavaScript form payload.

**Dependencies:** Task 1
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
**Estimated scope:** S (1 file)

---

## Checkpoint: Form Layout & Accessibility
- [x] Core and advanced inputs render within 720px container.
- [x] Responsive on both mobile and desktop viewports.
- [x] Presets populate both core and advanced fields properly.

---

## Phase 2: Action Button & Inference Integration

### Task 3: Implement Submit Action, Client Validation, and Backend API Dispatch
**Description:** Add a primary Ashby-styled action button ("Run Crop Yield Prediction"). Implement client-side physical validation (ensuring $T_{max} \ge T_{min}$, positive area, and rainfall within 0–600mm). Send asynchronous request to FastAPI endpoints (`POST /api/v1/recommend` or `POST /api/v1/predict`) based on whether "All Crops" or a specific crop is selected.

**Acceptance criteria:**
- [x] Action button styled with Ashby button tokens (rounded, `#232323` background with hover effect).
- [x] Temperature validation prevents submission if $T_{max} < T_{min}$.
- [x] Asynchronous fetch handles API response and displays loading spinner while waiting.

**Verification:**
- [x] Form submission dispatches valid JSON matching `SinglePredictionRequest` or `DroughtScenarioRequest`.
- [x] Invalid temperature trigger displays clean inline warning.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
**Estimated scope:** S (1 file)

---

### Task 4: Build Dynamic Categorized Results View
**Description:** Render the prediction results below the form in the Ashby job-listing typography (crop name in `#FF7344`, metadata line with `•` separator). For "All Crops", group into "Recommended Drought-Resilient Crops" and "Water-Intensive Warnings". For single crop, display expected harvest yield (kg/ha), water consumption rating, and drought vulnerability assessment. Include offline fallback dataset for reliable demonstration.

**Acceptance criteria:**
- [x] Results dynamically populate upon form submission.
- [x] Drought-resilient crops ranked with predicted yield and water needs.
- [x] Water-intensive crops (Sugarcane, Rice) clearly flagged under drought conditions.
- [x] Summary advisory banner displayed with agronomic recommendations.

**Verification:**
- [x] Results section displays cleanly under form.
- [x] Offline fallback operates seamlessly if FastAPI backend is not active.

**Dependencies:** Task 3
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
- `frontend/climate-resilient-frontend/careers.html`
**Estimated scope:** S (1 file)

---

## Checkpoint: Complete Verification
- [x] End-to-end form entry, validation, and prediction flow works.
- [x] Design matches Ashby UI style 100%.
- [x] Tested locally on `http://localhost:3000/predict.html`.
