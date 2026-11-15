# Implementation Plan: Model Sensitivity & Agroclimatic Ranking

## Overview
Investigate and resolve why "Onion is always #1 and Pomegranate is always #2" across all inputs. Analysis reveals two root causes:
1. **Model Feature Importance Imbalance**: The trained Random Forest regressor in `models/crop_yield_pipeline.joblib` gives 99.7% of split importance to `Crop`, `Area`, and `Province`. Climate variables (`precipitation_sum`, `temperature`, `evapotranspiration`) hold less than 0.3% combined importance. The model outputs essentially static crop-level baseline yields for each province.
2. **Raw Absolute Yield Sorting**: `ModelService.recommend_crops()` sorts drought-resilient crops purely by raw `predicted_yield_kg_ha` descending. Because onion biologically produces higher bulb biomass (~95–136 kg/ha) than grains like wheat (~28 kg/ha) or tree crops like dates (~69 kg/ha), onion will always be #1 when sorting by raw weight.

## Architecture Decisions
- **Drought Resilience Index (DRI) / Water-Use Efficiency Ranking**: Move beyond naive raw yield sorting. Compute an agroclimatic score that accounts for moisture deficit elasticity, crop-specific water requirements, and climate index stress:
  $$\text{Resilience Score} = \frac{\text{Predicted Yield}}{\text{Crop Baseline Yield}} \times \left(1 - \alpha \cdot \text{Moisture Deficit} \cdot \text{Crop Water Vulnerability}\right)$$
- **Agroclimatic Response Elasticity**: In `ModelService`, incorporate physical agronomic penalties for extreme heat (>38°C) and acute drought (<10mm) relative to each crop's biological tolerance thresholds, making recommendations genuinely dynamic to climate changes.
- **Frontend Scenario Responsiveness**: Verify that changing presets (Severe Drought vs Baseline Normal vs Extreme Heatwave) visibly changes crop rankings and yield projections.

---

## Task List

### Phase 4: Agroclimatic Ranking & Model Elasticity

- [ ] **Task 1: Implement Dynamic Agroclimatic Elasticity in `ModelService`**
  - Add crop-specific physiological drought sensitivity and optimal temperature thresholds.
  - Apply moisture-deficit and temperature-stress adjustment factors to model predictions so yields dynamically react to rainfall and heatwave inputs.

- [ ] **Task 2: Implement Water-Use Efficiency & Relative Drought-Resilience Ranking**
  - Update `recommend_crops()` to rank by Water-Use Efficiency (WUE) and drought adaptation index rather than raw unnormalized biomass weight.
  - Ensure Severe Drought prioritizes deepest root / lowest water demand crops (e.g. Dates, Pomegranate), while Moderate Drought or Baseline rainfall elevates Wheat and high-yield vegetables.

- [ ] **Task 3: End-to-End Verification Across Climate Presets**
  - Test preset switching ("Severe Drought" vs "Baseline Normal" vs "Extreme Heatwave") via automated CDP tests.
  - Verify that different scenarios produce different top-ranked crops and dynamic advisory summaries.

---

## Checkpoint: Dynamic Ranking Verification
- [ ] Presets produce distinct top-ranked crops (e.g., Dates/Pomegranate in extreme heat/arid, Wheat/Onion in moderate rainfall).
- [ ] Varying rainfall from 2mm to 200mm visibly scales yield predictions.
- [ ] No regression in API contracts (`SinglePredictionResponse` and `RecommendationResponse`).
