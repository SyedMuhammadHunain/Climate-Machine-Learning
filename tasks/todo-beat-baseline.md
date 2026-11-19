# Todo: beat the baseline (CLOSED 2026-11-19: gate not met, T3 onward not done, see Results in `SPEC-beat-baseline.md`)
# (plan in `tasks/plan-beat-baseline.md`, spec `SPEC-beat-baseline.md`)

## Phase 1: build and measure (no live change)

- [x] **T1** `OffsetYieldModel` in `features.py`.
  - Acceptance:
    - Offset fitted from training rows only, with crop and global fallbacks.
    - `trend` in {none, per_crop, per_crop_province}.
    - `Year` column used when present, otherwise `inference_year`.
  - Verify: new `verify_model.py` tests:
    - Offset ignores val and test yields.
    - Fallbacks.
    - Ratio identical across pinned years.
  - Files: `src/climate_machine_learning/features.py`, `scripts/verify_model.py`.
- [x] **T2** `training.py`: model × trend grid, `crop_province_mean_plus_trend` baseline, metrics fields
  (`chosen.trend`, `inference_year`, new baseline), notes text driven by the result. `inference_year` also written to
  `reference.json`.
  - Acceptance: `train(out_dir=scratch)` runs, selects on validation, scores test once, leaves `models/` untouched.
  - Verify: run into `$CLAUDE_JOB_DIR/tmp/candidate`; `git status models/` clean.
  - Files: `src/climate_machine_learning/training.py`, `scripts/train.py` (only if an `--out` flag is needed).
- [ ] **T3** `inference.single_thread` unwraps the wrapper; ranking and bit-exact tests pass on the candidate artifact.
  - Verify: `verify_model.py` inference tests pointed at the candidate.
  - Files: `src/climate_machine_learning/inference.py`, `scripts/verify_model.py`.
- [ ] **Checkpoint A** Numbers to user: every model × trend (val, test), both baselines, coverage. Win: go on.
  Loss: record in the spec and stop, with `models/` unchanged.

## Phase 2: promote (only on a win)

- [ ] **T4** Copy the candidate into `models/`. Run `verify_model.py`, `verify_backend.py`, `verify_frontend.py`.
  - Files: `models/*`.
- [ ] **T5** Wording and generated numbers: `verify_readme.py --write`, plus README "No yearly trend",
  `docs/demo_and_viva_guide.md` (demo numbers, "smarter baseline" answer), `streamlit_app/app.py:193`,
  `scripts/build_hf_bundle.py:132`, `CLAUDE.md:53`, `verify_readme.py` and `verify_guide.py` pins, and the
  Checkpoint 2 numbers in `tasks/todo-completion.md`.
  - Verify: `verify_readme.py`, `verify_guide.py`, `verify_streamlit.py` (own venv).
- [ ] **T6** `python scripts/build_vercel_bundle.py`, `python scripts/build_hf_bundle.py`; `verify_vercel_bundle.py`,
  `verify_hf_bundle.py` (own venv); `node scripts/dryrun_demo.js http://127.0.0.1:8000`.
- [ ] **Checkpoint B** User reviews the full diff and approves a commit on local `Hunain`. No push.

## Phase 3: ship (ask before each)

- [ ] **T7a** Vercel: deploy `vercel_bundle/` with the Vercel CLI, update the `vercel-deploy` branch. Check the live
  numbers while logged out.
- [ ] **T7b** Streamlit: bring the change onto `hunain-deploy`, push, and check the live app while logged out.
- [ ] **T8** `SPEC-beat-baseline.md` set to `Status: implemented`; outcome folded into `SPEC.md`.
