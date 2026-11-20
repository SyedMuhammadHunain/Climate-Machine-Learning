# Delta SPEC: Model-only output (no fake, demo, or hand-made numbers)

Parent: `SPEC.md` (Bajra audit). This delta replaces its "Changes (proposed)" section.
Sources read: `CT-378 ML Project Guidelines.pdf`, `CT-378 Project Proposal ... .docx`.

## Objective
Every number and ranking the app shows must come from the trained scikit-learn
pipeline (or from the real dataset / saved evaluation results). No fallback, no demo
mode, no hand-written formulas changing model output.

Proposal contract: *"recommended crops are obtained by ranking the predicted yields of
candidate low-water crops for the given conditions."* Target user: farmer or extension
officer in Punjab, Sindh, Balochistan.

## Audit: what is not model output today (verified 2026-11-18)

| # | Location | Problem |
|---|----------|---------|
| A1 | `predict.html:1001-1063`, `careers.html` same | `renderFallbackSimulation`: Bajra/Jowar/Gram + invented formulas. Fires on any error or >3.5 s (`predict.html:856`). |
| A2 | `model_service.py` `CROP_PHYSIOLOGY` + `calculate_agroclimatic_elasticity` | Hand-made multiplier applied to every prediction. |
| A3 | `model_service.py` `recommend_crops` `_rank_score` | Ranking = blend of hand-made "drought_resilience" and yield. Not yield rank. |
| A4 | `metrics.html:387,416` | Hardcoded `0.892`; claims "32-year ERA5" (data is 2001-2022, 22 years). |
| A5 | `config.py:53` scenarios | "for rapid demonstration & simulation". |
| A6 | `_resolve_environment` | Fills missing inputs with invented rules (`t_mean ± 6`, `precip * 1.5`). |

## Root cause: the saved model learned almost nothing about climate

Measured with raw `pipeline.predict` (no multipliers):

- Rain 0 mm vs 200 mm, Punjab: **identical** output for every crop.
- Dates = grapes = pomegranate = wheat (88.2 Punjab, 124.3 Sindh, 43.8 Balochistan). Exact ties.
- Onion always #1 among resilient crops.
- Feature importance: `Crop_sugarcane` 0.85, `Area` 0.05; all weather features ≈ 0.
- Output swings 4x with `Area` (user field, ha) but training `Area` is a different scale (median 71.7, likely thousand ha).

Why:
1. **Target duplicated over 12 months.** 10,296 rows = 13 crops x 3 provinces x 22 years x 12 months. `Yield` is one yearly value copied onto 12 monthly weather rows. Monthly weather cannot explain a yearly number, so the forest ignores it.
2. **`Area` is a crop/province ID proxy** and a planting decision, not a climate input.
3. **`Is_Drought_Resilient`** is a hand label duplicating `Crop`.
4. **Units mixed.** Yield is not kg/ha (wheat ≈ 28, sugarcane ≈ 537; notebook itself says "100 kg/ha or maunds/acre"). Raw yields across crops are not comparable.
5. **Data gaps.** Grapes 62 % zero yield, apple/pomegranate 33 % zero.
6. **CV leak.** `KFold(shuffle=True)` on monthly rows puts copies of the same year-yield in train and val folds.

So "only model output" with the current model = a ranking that ignores drought.
**Honest output needs a retrained model.** Removing the fakes alone is not enough.

## Changes

### C1. Retrain (notebook + `models/`)
- Aggregate to one row per Crop-Province-Year with season features from monthly data:
  total rain, rain in growing months, mean/max temp, mean ET0, radiation, rain hours.
- Engineered (proposal): rainfall deficit vs province long-term mean; ET0 - rain water balance.
- Drop features: `Area`, `Is_Drought_Resilient`, the `_normalized` duplicates.
- Handle zero-yield rows (drop as missing reports, documented).
- Split by year: train ≤2016, val 2017-2019, test 2020-2022 (keep). CV: `GroupKFold` / `TimeSeriesSplit` by year.
- Baseline (mean per crop) + Linear Regression + Random Forest + Gradient Boosting, tuned with 5-fold CV.
- Save pipeline and `models/metrics.json` (test MAE/RMSE/R², baseline, per model, feature importances, data hash, date).
- Report honestly if weather adds little over the crop-mean baseline. That is a valid finding for the report.

### C2. Backend serves model output only
- Delete `CROP_PHYSIOLOGY`, `calculate_agroclimatic_elasticity`, `_rank_score`.
- `recommend`: predict for each candidate crop, sort by model output (see D1), tie-break by crop name, return rank.
- Inputs the model needs must come from the user or from real dataset values (province historical median), labeled as such in the response (`input_source: "user" | "province_median_2001_2022"`). No invented rules.
- Report yield in the dataset's real unit (see D3).
- New `GET /api/v1/metrics` returns `models/metrics.json`.
- Reject inputs outside training range (422 with message), since the forest cannot extrapolate.

### C3. Frontend
- Delete `renderFallbackSimulation` and all Bajra/Jowar/Gram text.
- On API error: show error message + retry button. No numbers.
- Timeout 3.5 s → 30 s with a loading state.
- `metrics.html` reads `/api/v1/metrics`. Remove "32-year ERA5" and any hardcoded metric.
- Delete `careers.html` (dead copy).
- Remove "demo / simulation" wording in UI and `config.py`.

### C4. Docs
- README and report state: data unit, row aggregation, what the model learned, limits.

## Decisions (approved 2026-11-18)
- D1 → (b) rank by % of normal yield; also show raw predicted yield.
- D2 → rank all 13, tag water class from FAO citation; onion not low-water.
- D3 → label "dataset yield units" until source confirms.
- D4 → monthly table for EDA, yearly aggregate for model; explain in report.
- Retrain approved; R² may drop, report it honestly.

### Original options
- **D1 Rank by what?** (a) raw predicted yield, as proposal says; units differ per crop, so sugarcane/onion always win. (b) predicted yield ÷ that crop's normal yield in that province = "% of normal yield kept under these conditions". Still pure model output, comparable across crops. **Recommend (b), show both columns.**
- **D2 Candidate low-water crops.** Hand list today (includes onion). Options: keep list as cited domain knowledge (FAO crop water need) used only to filter, not as feature; or rank all 13 and just tag water class. **Recommend: rank all 13, tag water class from FAO citation, drop onion from "low-water".**
- **D3 Yield unit.** Need to confirm from source repo (Abdullah-Masood-05/pakistan-crop-yield-vs-climate). Until confirmed, label "dataset yield units", not kg/ha.
- **D4 Row count.** Guideline needs ≥5,000 rows. Honest yearly table = 858 rows (~600 after zero drop). Options: keep 10,296 monthly table for EDA/row count and state the model uses yearly aggregation; or merge district wheat data from proposal sources. **Recommend first, explain in report.**

## Testing
- `recommend` never returns a crop outside `ALL_CROPS`.
- Returned yield == `pipeline.predict` for same row (no post-processing).
- Changing rain changes at least one prediction (sanity: model uses weather). If it fails, report, do not hide.
- No exact ties left unbroken; rank order deterministic.
- Frontend: grep finds no `Bajra|Jowar|Gram|Fallback|Simulation` in `frontend/`.
- `/metrics` values equal `models/metrics.json`.
- Existing 12 backend tests still pass (update ones tied to elasticity).

## Boundaries
- Always: numbers shown = model, dataset, or saved metrics. Say when input came from a default.
- Ask first: dataset merges, changing target, dropping crops.
- Never: fallback numbers, hand multipliers on predictions, hardcoded metrics, "demo" mode.
