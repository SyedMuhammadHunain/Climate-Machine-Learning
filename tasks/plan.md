# Implementation Plan: Model-Only Output + Submission Readiness

Spec: `SPEC-model-only-output.md` (decisions D1–D4 approved 2026-11-18). Parent audit: `SPEC.md`.
Merged from the earlier plan "CT-378 Local Web App Finalization": Phase 1 there (static serving,
error parsing, pinned requirements) is done and stays done. The old Task 3 (README) and Task 4
(viva guide) are now Tasks 8–9, rewritten to use the retrained model's real results.

## Overview
The app must show only what the trained pipeline predicts, real dataset values, and saved
evaluation results. Today it shows a hardcoded Bajra fallback, applies hand-written multipliers
to every prediction, and hardcodes R² = 0.892. The saved model also ignores weather (yearly yield
copied onto 12 monthly rows). So we retrain on one row per crop, province and year, then rebuild the
predict, recommend and metrics paths on top of the new model.

## Architecture Decisions
- **One shared feature module** `src/climate_machine_learning/features.py`, used by training
  and by `ModelService`. Prevents train/serve skew. It holds: monthly→yearly aggregation,
  engineered features (rain deficit vs province mean, ET0 − rain water balance), the input range
  table, and province median climate.
- **Artifacts written by training, read by the app** (no numbers in code):
  `models/crop_yield_pipeline.joblib`, `models/metrics.json` (test scores, baseline, all models,
  feature importances, data hash, train date), `models/reference.json` (province median climate,
  input min/max per feature, water class per crop with its FAO citation).
- **Training as a script** `scripts/train.py` (reproducible, testable). The notebook calls the
  same functions and shows EDA and plots, as the guidelines require.
- **Ranking (D1)**: `pct_of_normal = predict(user climate) / predict(province median climate)`,
  per crop. This ratio uses two model outputs and nothing else. Sort descending, then by crop
  name to break ties. Raw predicted yield is shown next to it.
- **All 13 crops ranked (D2)**, each tagged Low / Moderate / High water from a cited FAO
  table. The tag is only a label. It is not a model feature and does not change ranking.
- **Missing inputs** are filled with the province's 2001–2022 median from `reference.json`. The
  response says where each value came from (`input_source`).
- **Inputs outside the training range** get a 422 with a clear message. A random forest
  cannot extrapolate, so we refuse instead of guessing.
- **Tests without new dependencies**: extend `scripts/verify_backend.py` and add
  `scripts/verify_model.py` (plain asserts, same style). pytest is not installed. Adding it means
  asking first.

## Dependency Graph
```
T1 fallback removal (independent, ships now)

T2 features.py + yearly table ──► T3 train.py + artifacts ──► CHECKPOINT A (model honest?)
                                                              │
                         ┌───────────────┬────────────────────┤
                         ▼               ▼                    ▼
                T4 /predict path   T5 /recommend path   T6 /metrics path
                         └───────────────┴──────┬─────────────┘
                                                ▼
                                   T7 cleanup + no-fake guard ──► CHECKPOINT B
                                                ▼
                                   T8 README ──► T9 viva guide ──► CHECKPOINT C
```

## Task List

### Phase 0: Stop showing fake data (immediate)
- [x] **T1** Remove Bajra fallback, error + retry, 30 s timeout, delete `careers.html` (S)

### Phase 1: Honest model
- [x] **T2** Shared feature module + yearly crop-province-year table (M)
- [x] **T3** Retrain with year-grouped CV, save pipeline + metrics.json + reference.json, update notebook (M)

### Checkpoint A: Model review (stop and show user)
- [ ] Test-set metrics vs crop-mean baseline reported, even if weak
- [ ] Sensitivity check: changing rain/temp changes predictions for at least some crops
- [ ] User accepts the model before the backend is rebuilt on it

### Phase 2: Vertical paths on the new model
- [x] **T4** `/predict` end-to-end: service + schema + predict.html single-crop view (M)
- [x] **T5** `/recommend` end-to-end: % of normal ranking, 13 crops, water tags, UI table (M)
- [x] **T6** `/metrics` end-to-end: endpoint + metrics.html reads it (S)
- [x] **T7** Remove dead code (`CROP_PHYSIOLOGY`, elasticity, `PRESET_SCENARIOS` demo wording, hardcoded `PROVINCE_DEFAULTS`) + no-fake guard test (S)

### Checkpoint B: App complete
- [ ] `verify_backend.py` and `verify_model.py` pass
- [ ] Manual: server on :8000, predict / recommend / metrics pages work; backend stopped shows error and no numbers
- [ ] `grep -riE "bajra|jowar|gram\b|fallback|simulat|demo" frontend/ app/` is empty

### Phase 3: Deliverables (from old plan, rewritten)
- [x] **T8** README with honest metrics from `metrics.json`, data unit, aggregation, limits (S)
- [x] **T9** `docs/demo_and_viva_guide.md` with 3/4/3 minute script on the real app (S)

### Checkpoint C: Submission ready
- [ ] Repo has data/, notebooks/, app/, models/, requirements.txt, README, guide
- [ ] Full flow verified on localhost:8000

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| No month column. Month order is taken from row position inside each group | High | T2 checks that order (temperature should peak in Jun/Jul) before aggregating. If it can't be confirmed, use yearly totals only, no season split. |
| Weather still adds little after retrain (858 rows, 22 years) | High | Checkpoint A. Report honestly against the baseline. Do not add hand rules. |
| 0-yield rows (grapes 62 %) leave grapes with very few rows | Med | Drop them as missing reports and document. Flag crops with < N training rows in the response. |
| Yield unit unknown | Med | Label it "dataset yield units". Check the source repo during T8. Never claim kg/ha unconfirmed. |
| R² drops sharply from 0.892 | Med | Expected. The old score came from duplicated rows. Explain this in README and viva. |
| Under 5,000 rows for the model (guideline) | Med | D4: the 10,296-row monthly table is used for EDA. Explain the aggregation in the report. |
| Frontend breaks when the response schema changes | Med | Each path (T4–T6) changes backend and its page in the same task. |
| `requirements.txt` drift | Low | No new runtime deps planned. Re-pin if one is added. |

## Open Questions
- Confirm the yield unit from the source repo (T8). Not blocking.

---

# Addendum: Second data source (NASA POWER)

Source spec: SPEC-second-source-nasa-power.md. Phases 0-3 above are complete.

| Task | What | Depends on |
|---|---|---|
| S1 | Fetch NASA POWER for 9 points, store raw JSON + tidy CSV + source doc | none |
| S2 | Merge in `features.py` (soil wetness mean/max), reference ranges, tests | S1 |
| S3 | Retrain, metrics.json lists both sources, notebook shows source agreement | S2 |
| Checkpoint D | Show the new scores and whether soil wetness helps | S3 |
| S4 | API inputs, quick-fill, Metrics page sources section, guards | S3 |
| S5 | README, guide, final checks | S4 |
