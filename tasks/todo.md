# Tasks: Agroclimatic Ranking & Model Elasticity

## Phase 4: Agroclimatic Ranking & Model Elasticity

### Task 1: Implement Dynamic Agroclimatic Elasticity in `ModelService`
**Description:** Modify `app/services/model_service.py` so that meteorological inputs (`precipitation_sum`, `temperature_2m_mean`, `temperature_2m_max`, `fao_evapotranspiration`, `climate_index`) actively modulate crop yield predictions via crop-specific physiological response curves, overcoming the Random Forest's 99.7% categorical bias.

**Acceptance criteria:**
- [ ] Changing rainfall from 2mm to 100mm produces noticeable, realistic yield changes.
- [ ] Temperatures exceeding critical thresholds (e.g., >38°C) incur heat-stress yield penalties.
- [ ] Single crop and multi-crop endpoints both reflect meteorological elasticity.

**Verification:**
- [ ] Run automated script comparing predictions under 2mm vs 100mm rainfall; yield difference is greater than 10%.

**Dependencies:** None
**Files likely touched:**
- `app/services/model_service.py`
**Estimated scope:** S (1 file)

---

### Task 2: Implement Water-Use Efficiency & Relative Drought-Resilience Ranking
**Description:** Update `recommend_crops()` in `app/services/model_service.py` to rank candidate crops using a composite Drought Resilience & Water-Use Efficiency (WUE) score instead of raw unnormalized biomass weight. This ensures low-water tree/grain crops (Dates, Pomegranate, Wheat) rank ahead of water-sensitive vegetables when moisture deficit is acute.

**Acceptance criteria:**
- [ ] Ranking uses relative resilience score instead of raw kg/ha biomass.
- [ ] Severe drought (<10mm) elevates high-arid tolerant crops (Dates, Pomegranate).
- [ ] Baseline normal conditions (>40mm) elevate higher-producing crops (Wheat, Onion).

**Verification:**
- [ ] Severe drought scenario test shows different top crop than baseline normal scenario.

**Dependencies:** Task 1
**Files likely touched:**
- `app/services/model_service.py`
**Estimated scope:** S (1 file)

---

### Task 3: End-to-End Verification Across Climate Presets
**Description:** Restart the FastAPI backend daemon with updated logic and run automated headless Chrome DevTools tests verifying that selecting different presets ("Severe Drought", "Moderate Drought", "Baseline Normal", "Extreme Heatwave") visibly re-orders the recommended crop cards in the UI.

**Acceptance criteria:**
- [ ] Presets produce distinct crop rankings and narrative summaries in the UI.
- [ ] Screenshot captured demonstrating dynamic ranking response.

**Verification:**
- [ ] CDP test confirms #1 crop changes across presets.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `tasks/todo.md`
**Estimated scope:** S (0-1 files)

---

## Checkpoint: Dynamic Ranking Verification
- [ ] Onion is no longer unconditionally #1 across every climate condition.
- [ ] Rankings respond intelligently to drought severity.
