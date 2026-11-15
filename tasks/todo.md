# Tasks: Agroclimatic Ranking & Model Elasticity

## Phase 1: Model Service Elasticity & Ranking

### Task 1: Implement Dynamic Agroclimatic Elasticity in `ModelService`
**Description:** Modify `app/services/model_service.py` so meteorological inputs (`precipitation_sum`, `temperature_2m_mean`, `temperature_2m_max`, `fao_evapotranspiration`) actively modulate crop yield predictions via physiological response curves, overcoming the Random Forest's categorical bias.

**Acceptance criteria:**
- [x] Changing rainfall from 2mm to 100mm produces noticeable, realistic yield changes (>10% variance, tested 64.68%).
- [x] Temperatures exceeding critical thresholds (>38°C) incur heat-stress yield penalties (tested 31.50%).
- [x] Both single prediction and recommendation endpoints reflect meteorological elasticity.

**Verification:**
- [x] Automated verification script comparing predictions under 2mm vs 100mm rainfall shows >10% variance.
- [x] Run `.venv/bin/python scripts/verify_backend.py`.

**Dependencies:** None
**Files likely touched:**
- `app/services/model_service.py`
**Estimated scope:** S (1 file)

---

### Task 2: Implement Water-Use Efficiency & Relative Drought-Resilience Ranking
**Description:** Update `recommend_crops()` in `app/services/model_service.py` to rank candidate crops using a composite Drought Resilience & Water-Use Efficiency (WUE) score instead of raw unnormalized biomass weight. This ensures low-water tree and grain crops (Dates, Pomegranate, Wheat) rank ahead of water-sensitive vegetables when moisture deficit is acute.

**Acceptance criteria:**
- [ ] Ranking uses relative resilience score instead of raw unnormalized kg/ha biomass.
- [ ] Severe drought (<10mm) elevates high-arid tolerant crops (Dates, Pomegranate).
- [ ] Baseline normal conditions (>40mm) elevate higher-producing crops (Wheat, Onion).

**Verification:**
- [ ] Run test comparing severe drought scenario vs baseline normal scenario; top crop changes from Dates/Pomegranate to Wheat/Onion.
- [ ] Update `scripts/verify_backend.py` to assert top recommended crop under severe drought is Dates or Pomegranate.

**Dependencies:** Task 1
**Files likely touched:**
- `app/services/model_service.py`
- `scripts/verify_backend.py`
**Estimated scope:** S (2 files)

---

## Checkpoint: Backend Logic
- [ ] All verification tests pass: `.venv/bin/python scripts/verify_backend.py`
- [ ] Severe drought and baseline normal scenarios produce different #1 recommended crops.

---

## Phase 2: Verification & Scenario Responsiveness

### Task 3: End-to-End Verification Across Climate Presets
**Description:** Test all standard climate presets ("Severe Drought", "Moderate Drought", "Baseline Normal", "Extreme Heatwave") through API calls and verify the frontend recommendations respond dynamically.

**Acceptance criteria:**
- [ ] Presets produce distinct crop rankings and narrative summaries.
- [ ] Onion is no longer unconditionally #1 across every climate condition.
- [ ] No regression in API contracts (`SinglePredictionResponse` and `RecommendationResponse`).

**Verification:**
- [ ] Run verification script covering all preset scenarios.
- [ ] Check HTTP responses for each scenario.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `scripts/verify_backend.py`
**Estimated scope:** XS (1 file)

---

## Checkpoint: Complete
- [ ] Onion is no longer unconditionally #1 across every climate condition.
- [ ] Rankings respond intelligently to drought severity.
- [ ] Review with human before deployment.
