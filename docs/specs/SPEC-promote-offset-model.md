# Spec Delta: Promote the offset model, judged on R² and MAE

Status: implemented (2026-11-19). Plan: `tasks/plan-promote-offset-model.md`. Outcome in the Results section at the end.

Parent: `SPEC.md`. Follows `SPEC-beat-baseline.md` (closed, gate not met). Code from local branch
`offset-model-experiment` (commit `20e3142`).

## Objective

The user chose on 2026-11-19 to promote the candidate from `SPEC-beat-baseline.md` and to judge it against the
crop-province-mean baseline on **test R² and MAE** (the metrics named in the proposal) instead of log-MAE.

**This choice was made after the test results were seen.** Every doc and app that states the result must say so, and
must also state that the model loses on log-MAE. Nothing may present this as a clean pre-registered win.

## Current Behavior

`models/` holds the Random Forest without an offset: test R² 0.794, MAE 40.9, log-MAE 0.404. It does not beat the
crop-province mean (R² 0.832, MAE 34.7, log-MAE 0.312). Docs and apps say so.

## New Behavior

1. Restore `OffsetYieldModel`, the 3 models × 3 trends grid, the `crop_province_mean_plus_trend` baseline and
   `scripts/train.py --out` from `20e3142`. Training and selection rules are unchanged: chosen on validation
   log-MAE (Gradient Boosting, trend `per_crop_province`), test scored once.
2. Retrain into `models/`. Expected from the single run: test R² 0.885, MAE 29.7, log-MAE 0.341. If the retrain
   differs from those numbers beyond the last digit, stop and report.
3. `metrics.json`, `chosen` block:
   - `judged_on: ["r2", "mae"]`, `judged_on_chosen_after_test: true`.
   - `beats_by_metric: {r2, mae, rmse, log_mae}`, all computed against the crop-province mean on test.
   - `beats_crop_province_mean_on_test` = R² better **and** MAE better (true for the candidate).
   - A note stating the choice was made after seeing the test results, and the log-MAE loss.
4. Pinned inference year: 2022 (`INFERENCE_YEAR`, stored in the model and in `reference.json`). The ranking does
   not depend on it.
5. Wording, everywhere the result is stated: "Beats the crop-province mean on test R² (0.885 vs 0.832) and MAE
   (29.7 vs 34.7). Does not beat it on log-MAE (0.341 vs 0.312), where every crop counts equally. These two metrics
   were chosen after seeing the test results. Most of the gain comes from the year trend: the trend alone, with no
   climate, scores R² 0.875." The numbers are generated from `metrics.json`, never typed in.
   - Places: README results block and limits, `docs/demo_and_viva_guide.md`, `metrics.html`, `model_service.py`
     caution text, `streamlit_app/app.py`, `hf_space` and `build_hf_bundle.py` model card, `CLAUDE.md`, and the pins
     in `verify_readme.py`, `verify_guide.py`, `verify_streamlit.py` and `verify_hf_bundle.py`.
   - The README's "No yearly trend" limit is replaced by "Year enters only as a trend in the offset, pinned to 2022
     at prediction; climate adds little beyond it."
   - Likely range: test coverage now 0.578 (docs say 60% today).
6. `inference.single_thread` unwraps `OffsetYieldModel.inner_` (T3 of the old plan).
7. Bundles rebuilt. Vercel and Streamlit redeployed, **each only after asking**.

## Success Criteria

- [x] `models/metrics.json`: chosen = gradient_boosting, test R² > 0.832, test MAE < 34.7,
      `beats_crop_province_mean_on_test == true`, `beats_by_metric.log_mae == false`,
      `judged_on_chosen_after_test == true`.
- [x] Row counts 565 / 106 / 102 and data sha256 values unchanged. No synthetic data.
- [x] App inputs unchanged. Ranking is still a ratio of model predictions, identical across pinned years (test).
- [x] No doc or app claims a win without, in the same place, the log-MAE loss and the after-the-fact metric choice.
      Checked by `verify_readme.py`, `verify_guide.py`, `verify_frontend.py`, `verify_streamlit.py`.
- [x] All checks pass: `verify_model.py` (with retrain determinism), `verify_backend.py`, `verify_frontend.py`,
      `verify_readme.py`, `verify_guide.py`, `verify_vercel_bundle.py`, `verify_streamlit.py`,
      `verify_hf_bundle.py`, plus `node scripts/dryrun_demo.js` against a local server.
- [x] After the deploys: logged-out visitors to Vercel and Streamlit see the new numbers and the caveat. (Vercel: pages, API, caveat in `/recommend` and `dryrun_demo.js` checked. Streamlit: default view checked in a real browser, rice 21.10 and 58% coverage; the About text and the Sindh driest-year click were not re-checked live there.)

## Boundaries (deltas only)

- **Ask first:** each live redeploy; any push (only to `hunain-deploy` and `vercel-deploy`, never remote `Hunain`).
- **Never:**
  - Change the model selection rule or re-run training to improve numbers.
  - Drop the log-MAE result or the after-the-fact note from any place that states the win.
  - Hand-type metric numbers into app code.

## Out of Scope

- New model families or features.
- Hugging Face Space deploy (stays "not deployed yet").
- The report PDF (`tasks/todo-completion.md`). It reads `metrics.json` when built.

## Open Questions

- None. The pinned year is set to 2022, as defaulted earlier. Say so if you want 2016.

## Results (2026-11-19)

Implemented in 9 local commits (`72ef7c3` to `46ad7d5`, then `a0d9f18`, `06e4185`, `1ea700f`, `b3fe100`).

- `models/` holds the Gradient Boosting offset model, trend `per_crop_province`, pinned year 2022. Test R² 0.885 and MAE 29.7 beat the crop-province mean (0.832, 34.7); log-MAE 0.341 does not (0.312). Row counts and data hashes unchanged.
- One shared sentence, `inference.model_caveat`, built from `metrics.json`, states the win with the log-MAE loss and the after-the-fact metric choice in the API, the metrics page, Streamlit, Gradio, the README and the HF model card.
- Vercel redeployed with the CLI (`dpl_A6MBeZRggSgAE5y5JExybSf3MzaE`); `vercel-deploy` branch at `e63da8d`. Streamlit redeployed through `hunain-deploy` (`1ea700f`), which needed a requirements-file touch to restart the cached process.
- Not done: the Hugging Face model repo still holds the earlier model (the card for the new one is in `hf_bundle/model/`); the Space is still not deployed.
- Known weakness, stated in the docs: the demo ranking is mostly tied (eleven crops at 97.7% in Sindh's driest year, all 13 at 100.0% in a typical year), because the model moves almost every crop by the same amount.
