# Implementation Plan: Model Sensitivity & Agroclimatic Ranking

## Overview
Resolve the issue where "Onion is always #1" across all scenario inputs in the climate-resilient crop recommendation system. The solution preserves the pre-trained ML pipeline while incorporating an agronomic Drought Resilience Index (DRI) / Water-Use Efficiency (WUE) scoring function and biological climate response elasticity inside `app/services/model_service.py`.

## Architecture Decisions
- **Drought Resilience Index (DRI) / Water-Use Efficiency Ranking**: Move beyond naive raw yield sorting. Compute an agroclimatic score that accounts for moisture deficit elasticity, crop-specific water requirements, and climate index stress:
  $$\text{Resilience Score} = \frac{\text{Predicted Yield}}{\text{Crop Baseline Yield}} \times \left(1 - \alpha \cdot \text{Moisture Deficit} \cdot \text{Crop Water Vulnerability}\right)$$
  - Severe drought (<10mm rain) prioritizes deepest root / lowest water demand crops (Dates, Pomegranate).
  - Normal or moderate rainfall (>40mm rain) elevates higher-biomass crops (Wheat, Onion).
- **Agroclimatic Response Elasticity**: In `ModelService`, incorporate physical agronomic response factors for extreme heat (>38°C) and acute drought (<10mm) relative to each crop's biological tolerance thresholds, making predictions dynamic to rainfall and temperature.
- **Contract Stability**: Keep all existing API response schemas (`SinglePredictionResponse`, `RecommendationResponse`) intact so that no frontend breaking changes are introduced.

## Task List

### Phase 1: Model Service Elasticity & Ranking
- [x] Task 1: Implement Dynamic Agroclimatic Elasticity in `ModelService`
- [x] Task 2: Implement Water-Use Efficiency & Relative Drought-Resilience Ranking

### Checkpoint: Backend Logic
- [x] Automated verification script demonstrates dynamic yield shifts and proper crop re-ranking across drought levels.
- [x] All backend verification tests pass.

### Phase 2: Verification & Scenario Responsiveness
- [x] Task 3: End-to-End Verification Across Climate Presets

### Checkpoint: Complete
- [x] Presets produce distinct top-ranked crops (Dates/Pomegranate in severe drought/arid, Wheat/Onion in moderate rainfall).
- [x] Varying rainfall from 2mm to 200mm visibly scales yield predictions.
- [x] No regression in API contracts.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Yield adjustments distort baseline predictions too heavily | Medium | Calibrate elasticity factors within realistic agronomic bounds (±15–35% yield elasticity) based on FAO crop water guidelines. |
| API contract mismatch breaks frontend rendering | High | Maintain exact field names and types in `SinglePredictionResponse` and `RecommendationResponse`. |
| Edge cases in user inputs (e.g., zero precipitation or negative ET0) | Low | Utilize existing Pydantic validation and clip minimum yields to >= 0. |

## Open Questions
- None. The approach, thresholds, and ranking criteria are fully aligned.
