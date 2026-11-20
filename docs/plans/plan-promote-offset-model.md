# Plan: Promote the offset model, judged on R² and MAE

Source spec: `SPEC-promote-offset-model.md` (approved 2026-11-19). Tasks: `tasks/todo-promote-offset-model.md`.
Code source: local branch `offset-model-experiment` (commit `20e3142`).

## Dependency graph

```
T1 restore code + single_thread + judged_on fields   (src, scripts/verify_model.py)
  -> T2 retrain into models/, full model + backend checks
     -> Checkpoint A: numbers match the single run (R² 0.885, MAE 29.7, log-MAE 0.341)
        -> T3 API + HTML pages wording        (app/, frontend/)
        -> T4 Streamlit + HF app wording      (streamlit_app/, hf_space/, presentation.py)
        -> T5 README + verify_readme
        -> T6 guide + CLAUDE.md + verify_guide
        -> T7 bundles (Vercel, HF), bundle checks, dryrun_demo
           -> Checkpoint B: every check green, user reviews diff, local commits
              -> T8a Vercel deploy (ask)   T8b Streamlit deploy (ask)
                 -> T9 specs closed
```

T3–T6 all depend on T2's `metrics.json` but not on each other. They run one at a time, but in any order.

## Key decisions

- **Restore, don't rewrite.** `git checkout offset-model-experiment -- <4 files>` brings back `OffsetYieldModel`, the
  3×3 grid, the trend baseline, `train.py --out`, and the 7 tests already written and passed.
- **Flag definition (spec item 3)** goes in `training.py`:
  - `beats_by_metric = {r2, mae, rmse, log_mae}` against the crop-province mean on test.
  - `beats_crop_province_mean_on_test = r2 and mae`.
  - `judged_on = ["r2", "mae"]`, `judged_on_chosen_after_test = true`.
  - A note sentence.
  - Every consumer reads these fields. None hardcodes "true".
- **One caveat sentence, generated.** Each front end builds the caveat from `metrics.json`: the log-MAE pair, the
  after-the-fact note, and the trend-only R². The Python front ends share one helper in `presentation.py`;
  `metrics.html` builds its own in JS from the same fields. Checkers assert the caveat appears wherever "beats" or
  "better than" appears.
- **Retrain is a check, not a tuning step.** If `models/metrics.json` differs from the scratch run's 0.885 / 29.7 /
  0.341 at 3 decimals, stop and report. No re-runs.

## Risks

| Risk | Mitigation |
|---|---|
| Retrain differs from the scratch run | Same versions and seed. Compare at 3 decimals in T2. Stop if different. |
| `verify_backend.py` TEST 22 bans hardcoded floats in `app/` | Caveat numbers come from `metrics.json` at runtime. |
| `single_thread` breaks on the wrapper (no `named_steps`) | T1 unwraps `inner_`. A bit-exact ranking test runs on the saved artifact. |
| `features.py` now imports scikit-learn at module level | Every consumer already needs it to unpickle. Confirmed by the T7 clean-venv bundle checks. |
| Guide pins demo numbers (19.71 / 58.5%, ranking order) | T6 re-reads them from the local app. `verify_guide.py` compares against it. |
| The already-published Hugging Face **model repo** still holds the old model | Out of scope unless you say otherwise (see Open). Docs must not claim the HF repo has the new model. |
| `hunain-deploy` may have drifted from local `Hunain` | T8b diffs first, then brings over only the needed paths. Never push remote `Hunain`. |

## Checkpoints

- **A (after T2):** you see the `chosen` block and the full table from `models/metrics.json`.
- **B (after T7):**
  - All checks pass: `verify_model` (with determinism), `verify_backend`, `verify_frontend`, `verify_readme`,
    `verify_guide`, `verify_vercel_bundle`, `verify_streamlit` (own venv), `verify_hf_bundle` (own venv), and
    `dryrun_demo.js`.
  - You review the diff.
- **C (after T8):** logged-out visitors see the new numbers and the caveat on both live sites.

## Open

- Republish the Hugging Face model repo with the new model? Default: **no** (out of scope). The model card then has to
  say it holds the earlier model.
