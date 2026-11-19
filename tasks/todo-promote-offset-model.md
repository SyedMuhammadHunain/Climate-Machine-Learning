# Todo: promote the offset model (plan `tasks/plan-promote-offset-model.md`, spec `SPEC-promote-offset-model.md`)

## Phase 1: model

- [x] **T1** Restore the code and add the judged-on fields.
  - Do:
    - `git checkout offset-model-experiment -- src/climate_machine_learning/features.py src/climate_machine_learning/training.py scripts/train.py scripts/verify_model.py`
    - `inference.single_thread` unwraps `OffsetYieldModel.inner_`.
    - `training.py` writes `chosen.beats_by_metric`, `judged_on`, `judged_on_chosen_after_test`, and
      `beats_crop_province_mean_on_test = r2 and mae`, plus the note.
  - Acceptance: new tests (written first) for:
    - The flag logic.
    - `single_thread` reaching the inner regressor.
    - Bit-exact ranking with `single_thread`.
  - Verify: `verify_model.py` minus artifact tests, against a scratch candidate (`scripts/train.py --out`). Candidate
    shows R² 0.885, MAE 29.7, log-MAE 0.341.
  - Files: `features.py`, `training.py`, `inference.py`, `scripts/train.py`, `scripts/verify_model.py`.
- [x] **T2** `python scripts/train.py` into `models/`.
  - Acceptance: the numbers above at 3 decimals; chosen `gradient_boosting`, trend `per_crop_province`;
    `beats_crop_province_mean_on_test` true and `beats_by_metric.log_mae` false; rows 565 / 106 / 102; sha256 unchanged.
  - Verify: full `verify_model.py` (with retrain determinism), `verify_backend.py`.
  - Files: `models/*`.
- [ ] **Checkpoint A** You see the `chosen` block and the table. If anything differs: stop.

## Phase 2: wording (any order, each with its checker)

- [ ] **T3** API and HTML pages.
  - Do:
    - `model_service.py` caution text, driven by `beats_by_metric` (win on R² and MAE, loss on log-MAE, after-the-fact note).
    - `metrics.html`: the "Vs crop-average baseline" cell, a label for `crop_province_mean_plus_trend`, and the note.
  - Verify: `verify_backend.py` (TEST 22: no hardcoded floats), `verify_frontend.py` (caveat present wherever a win is
    stated). Both checkers extended first.
  - Files: `app/services/model_service.py`, `frontend/climate-resilient-frontend/metrics.html`,
    `scripts/verify_backend.py`, `scripts/verify_frontend.py`.
- [ ] **T4** Streamlit and Gradio.
  - Do:
    - One caveat helper in `presentation.py`.
    - `streamlit_app/app.py` "About" text: drop the "no trend over time" sentence, state the win and the caveat.
    - `hf_space/app.py` uses the same helper.
  - Verify: `verify_streamlit.py` (own venv), extended first.
  - Files: `src/climate_machine_learning/presentation.py`, `streamlit_app/app.py`, `hf_space/app.py`,
    `scripts/verify_streamlit.py`.
- [ ] **T5** README.
  - Do:
    - `verify_readme.py` generator gets a label for the new baseline and a results sentence with the caveat.
    - Replace the "No yearly trend" limit.
    - Coverage changes from 60% to 58%.
    - `python scripts/verify_readme.py --write`.
  - Verify: `verify_readme.py`.
  - Files: `README.md`, `scripts/verify_readme.py`.
- [ ] **T6** Guide and project notes.
  - Do:
    - `docs/demo_and_viva_guide.md`:
      - Demo numbers and ranking order (lines 14, 15, 36, 40), read from the local app.
      - The "smarter baseline" answer (line 83): say the metric was chosen after seeing the results.
    - `CLAUDE.md` rule line 53.
    - `tasks/todo-completion.md` Checkpoint 2 numbers.
  - Verify: `verify_guide.py`, extended first.
  - Files: `docs/demo_and_viva_guide.md`, `scripts/verify_guide.py`, `CLAUDE.md`, `tasks/todo-completion.md`.
- [ ] **T7** Bundles.
  - Do:
    - `scripts/build_hf_bundle.py` model card: candidates line 87, trend line 132, caveat, and "HF model repo holds
      the earlier model" unless you choose to republish.
    - `python scripts/build_vercel_bundle.py`, `python scripts/build_hf_bundle.py`.
  - Verify: `verify_vercel_bundle.py`, `verify_hf_bundle.py` (own venv), and
    `node scripts/dryrun_demo.js http://127.0.0.1:8000` with `uvicorn app.main:app --port 8000`.
  - Files: `scripts/build_hf_bundle.py`, `scripts/verify_hf_bundle.py`.
- [ ] **Checkpoint B** Every check green. You review the diff. Local commits on `Hunain`. No push.

## Phase 3: ship (ask before each)

- [ ] **T8a** Vercel: deploy the rebuilt `vercel_bundle/` with the Vercel CLI and update the `vercel-deploy` branch.
  Logged-out check: metrics page caveat, Sindh driest-year rice numbers.
- [ ] **T8b** Streamlit: diff `hunain-deploy` against local `Hunain`, bring over the needed paths, push `hunain-deploy`.
  Logged-out check: "About" caveat, same demo numbers.
- [ ] **Checkpoint C** Both live sites match local.
- [ ] **T9** `SPEC-promote-offset-model.md` set to `Status: implemented`; outcome folded into `SPEC.md`; cross-link
  from `SPEC-beat-baseline.md` Results.
