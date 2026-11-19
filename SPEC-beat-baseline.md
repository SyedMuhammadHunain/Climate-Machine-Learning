# Spec Delta: Beat the crop-province-mean baseline

Status: closed, gate not met (2026-11-19). One honest run; `models/` and both live sites unchanged. See Results.

Parent: `SPEC.md`, with `SPEC-model-only-output.md` and `SPEC-second-source-nasa-power.md` still in force.
Intent confirmed in the interview on 2026-11-19: beat the baseline and reduce error, real data only, same inputs.

## Objective

The saved model must beat the simple baseline (mean training yield of the same crop in the same province) on the
held-out test years, and have lower error than the current model. The app's inputs stay the same and only real data is used.

**This is a bonus, not a rescue.** The submitted proposal's success bar (test R² above 0.7 and a clear margin over a
mean-yield baseline) is already met: Random Forest test R² 0.794 and MAE 40.9, against the global-mean baseline's
R² -0.03 and MAE 100.4. The crop-province-mean baseline (test R² 0.832, MAE 34.7) is a stricter bar this project added
itself. Beating it answers the viva question "does climate add anything beyond crop and province?". If this delta
fails, the project still meets the proposal and nothing live changes.

The proposal sources that are not used (Open Data Pakistan series, rabia743 city weather, Punjab irrigated area,
Punjab district wheat, PMD NDMC) stay unused for the reasons in `README.md` and `SPEC-second-source-nasa-power.md`.
None of them covers 13 crops × 3 provinces × 2001-2022.

## Current Behavior (measured from `models/metrics.json`, trained 2026-11-19)

| Model | val log-MAE | test log-MAE | test skill vs baseline |
|---|---|---|---|
| Baseline: crop-province mean | 0.194 | 0.312 | 0 |
| Random Forest (chosen) | 0.237 | 0.404 | -0.76 |
| Gradient Boosting | 0.239 | 0.413 | -0.68 |
| Ridge | 0.590 | 0.671 | -1.98 |

- 80% likely range covers 0.598 of test rows.
- `Crop` carries 0.93 of permutation importance. Each climate feature carries about 0.02 or less.
- The baseline is fitted on 2001-2016, so it never sees that yields grow over time. The `metrics.json` notes say: "Year/trend is not a feature, so yield growth over time is not modelled."

Root cause: every model learns each crop-province's yield level from scratch out of 565 rows, and does it worse than a
plain average. The climate signal is small next to that level.

## Data Files (unchanged by this delta)

| File | Used by | Role |
|---|---|---|
| `data/raw/pakistan_crop_yield_climate.parquet` | `training.py` | Training data, all 10,296 monthly rows. Offset and trend come from its `Year`, `Crop`, `Province`, `Yield` columns, training years only |
| `data/raw/nasa_power_monthly.csv` | `features.py` | `GWETROOT` gives the soil wetness features; `PRECTOTCORR` is the rain cross-check in the notebook only |
| `data/raw/nasa_power/*.json` (9) | `scripts/fetch_nasa_power.py` | Raw API responses behind the CSV (provenance) |
| `data/reference/nasa_power_source.json` | `training.py` | Source metadata written into `metrics.json` |
| `data/reference/fao_crop_water_needs.json` | all apps | Crop water needs shown next to predictions, not a model input |
| `data/raw/pakistan_crop_yield_climate.csv` | none | Identical copy of the parquet (checked), kept for readers without `pyarrow` |

No new data file, no new rows.

## New Behavior

1. **Offset target.** Each model predicts the *gap* between log yield and a per-row offset, not log yield itself:
   `log(Yield) = offset(Crop, Province, Year) + model(features)`.
   - `offset` = mean training log yield of the crop-province (falling back to crop, then global), plus an optional
     linear year trend fitted on training years only.
   - Trend variant chosen on **validation** years from: `none`, `per_crop` slope, `per_crop_province` slope.
   - The offset lives inside the saved pipeline as a small scikit-learn-compatible regressor class in
     `src/climate_machine_learning/features.py`. That module already ships to every consumer (Streamlit, Vercel bundle,
     Hugging Face bundle), so `joblib.load` works everywhere with no new module. `presentation.py` is still never added to
     the Hugging Face model repo.
2. **Year is not a user input.** At inference the year is pinned to one reference year, stored in `reference.json`
   (default: latest dataset year, see Open Questions).
   - Ranking (scenario yield / median-year yield) uses the same pinned year on both sides. The offset is additive in
     log space, so it cancels exactly and the ranking depends only on the climate model.
   - Absolute predicted yields shown in the apps do change: they move to the pinned year's level.
3. **Same 3 model families** (Ridge, Random Forest, Gradient Boosting), each re-tuned with `GroupKFold` grouped by
   `Year` on training years. Model and trend variant are picked by lowest validation log-MAE. Test years are scored once.
4. **Extra honest baseline reported:** `crop_province_mean_plus_trend` (the offset alone, no climate). It shows how much
   of any win comes from the trend and how much from climate. The README and guide state both numbers.
5. **Likely range** is still fitted from validation residuals. Test coverage is reported, never tuned.

## Success Criteria

- [ ] Chosen model test log-MAE < 0.312 and `chosen.beats_crop_province_mean_on_test == true`, computed by
      `scripts/train.py` with the selection rules above (no test-year peeking, no reruns until it passes).
- [ ] Chosen model test log-MAE lower than current 0.404.
- [ ] `metrics.json` also reports `crop_province_mean_plus_trend` val and test scores, the chosen trend variant, and
      the pinned inference year.
- [ ] 80% range test coverage reported. Target >= 0.70; this is reported, not a gate.
- [ ] No synthetic or augmented rows: `n_train`, `n_val`, `n_test` stay 565 / 106 / 102 and the data sha256 values are unchanged.
- [ ] App inputs unchanged: `BASE_INPUTS`, API request schema, page forms and Streamlit widgets identical.
- [ ] Out-of-range refusal (`OutOfRangeError`) still works, and the ranking stays a ratio of model predictions.
- [ ] All checks pass: `verify_model.py` (including retrain determinism), `verify_backend.py`, `verify_frontend.py`,
      `verify_readme.py`, `verify_guide.py`, `verify_vercel_bundle.py`, `verify_streamlit.py`, `verify_hf_bundle.py`,
      and `node scripts/dryrun_demo.js` against a local server.
- [ ] New checks in `verify_model.py`:
  - [ ] Offset fitted on training years only. Changing val or test yields does not change the offset.
  - [ ] Pinned year read from `reference.json`, not hardcoded in `app/`.
  - [ ] Ratio ranking is identical when the pinned year changes.
- [ ] If it wins: README results block regenerated (`verify_readme.py --write`), and wording that says "does not beat
      the baseline" updated in README, `docs/demo_and_viva_guide.md`, the apps, `CLAUDE.md` and the checkers, all together.
- [ ] If it does not win: the current `models/` artifacts are restored byte for byte, live deploys are untouched, and the
      tried variants with their honest numbers are recorded in this spec.

## Boundaries (deltas only)

- **Always:** select on validation years only; keep test years for a single final score; run `inference.single_thread`
  for bit-exact comparisons.
- **Ask first:**
  - Each live redeploy (Vercel via CLI from `vercel_bundle/`; Streamlit via the `hunain-deploy` branch).
  - Any push.
  - Adding any dependency beyond the current scikit-learn stack.
- **Never:**
  - Synthetic, augmented or imputed-as-new rows.
  - Tuning on test years, or picking a variant because of its test score.
  - Changing app inputs.
  - Touching remote `main`, `Hunain` or `afzal`.

## Out of Scope (Deferred)

- 4th and 5th model families (Extra Trees, HistGradientBoosting). Dropped by the user. Same framing, unlikely to change the outcome.
- Seasonal (rabi/kharif) features. Would change the app inputs.
- New data sources.
- Hugging Face Space deploy. Stays "not deployed yet".
- Refitting the final model on train + val years. That would change what the reported test score means.

## Open Questions

1. **Pinned inference year:** 2022 (latest dataset year, linear trend extrapolated 6 years past training) or 2016
   (last training year, no extrapolation)? Default: 2022, because it reflects current yield levels, and the ranking
   is the same either way.

## Results (2026-11-19): gate not met

Built `OffsetYieldModel` (crop-province offset plus optional year trend) and trained 3 models x 3 trends once into a
scratch folder. Selection on validation years only. Test years scored once. No re-run, no tuning after seeing the test score.
Code is kept on local branch `offset-model-experiment` (commit `20e3142`); `Hunain` has it reverted. `models/` was never touched.

| | val log-MAE | test log-MAE | test R² | test MAE |
|---|---|---|---|---|
| Baseline: crop-province mean | 0.194 | **0.312** | 0.832 | 34.7 |
| Mean + trend, no climate (`per_crop_province`) | 0.167 | 0.345 | 0.875 | 30.3 |
| Ridge | 0.170 | 0.340 | 0.878 | 29.5 |
| Random Forest | 0.175 | 0.360 | 0.873 | 30.6 |
| Gradient Boosting (chosen on validation) | 0.168 | 0.341 | 0.885 | 29.7 |

- Success criterion "test log-MAE < 0.312" failed: 0.341. `beats_crop_province_mean_on_test` would be false.
- The candidate beats the baseline on test R², MAE and RMSE, which weight high-yield crops. It loses on log-MAE, which weights every crop equally.
- The gain over the current model comes from the year trend, not from climate: mean + trend with no climate scores val 0.167, the best model 0.168.
- 80% likely range test coverage 0.578.
- Not used to redefine the gate: the criterion was fixed before the run.

Open Questions resolved: pinned year not needed (spec closed before promotion).

Deferred, if revisited: a trend-only offset is the useful part; a different gate (R² or MAE) would need the user to choose it before any re-run.
