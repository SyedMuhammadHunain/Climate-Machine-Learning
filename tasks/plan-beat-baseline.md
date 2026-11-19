# Plan: Beat the crop-province-mean baseline

Spec: `SPEC-beat-baseline.md` (approved 2026-11-19). Tasks: `tasks/todo-beat-baseline.md`.

## Approach

### Saved artifact shape

The new artifact is a wrapper estimator, `OffsetYieldModel`, in `src/climate_machine_learning/features.py`:

- It has a `fit(X, y)` / `predict(X)` interface on the raw feature DataFrame, so every consumer still calls
  `pipeline.predict(batch)` unchanged.
- `fit`:
  - Learns the offset from training rows only: mean log yield per crop-province, with crop and global fallbacks, plus
    an optional slope, `trend in {none, per_crop, per_crop_province}`.
  - Fits the existing inner `Pipeline` (impute, scale, one-hot, regressor) on `log(Yield) - offset`.
- `predict`: `exp(offset(Crop, Province, year) + inner.predict(X))`.
  - `year` is the row's `Year` column when present (training, validation and test rows).
  - Otherwise it is `inference_year`, stored on the estimator and in `reference.json`.
- `Year` is never in `FEATURE_COLS`, so the inner model cannot see it, and app inputs stay the same.

The offset fits inside each `GroupKFold` fold (it is part of the estimator), so CV has no leakage. The trend variant is
a grid parameter, and model plus trend are picked together by lowest **validation** log-MAE.

### Why the ranking is safe

`pct_of_normal = predict(scenario) / predict(median year)`. Both rows have the same crop, province and pinned year, so
the `exp(offset)` cancels and the ranking depends only on the inner model. A new test asserts this.

### Training cost

Today: 3 grid searches. New: 3 models × 3 trend variants = 9, roughly 3x the current ~1 min. The Random Forest grid
(18 combinations) is the main cost. If the total goes above ~3 min, cut `n_estimators` to `[300]` and `max_depth` to
`[None, 15]`; that is a speed choice, not a result choice. `verify_model.py`'s retrain-determinism test grows the same way.

### Decision gate (Checkpoint A)

Train into a scratch directory first, with `train(out_dir=$CLAUDE_JOB_DIR/tmp/candidate)`. `models/` is not touched
until the numbers are in.

- **Win** (test log-MAE < 0.312 and below 0.404): promote the artifacts and carry on with docs, bundles and deploys.
- **Loss:** stop. Record the numbers in the spec, discard the code changes or keep them on a side branch if you ask,
  and `models/` stays byte-identical. One honest run decides: no re-running with tweaks until it passes.

## Components and order

```
T1 OffsetYieldModel + unit checks
  -> T2 training.py: grid, new baseline, metrics/reference fields  (scratch dir)
  -> T3 inference: single_thread through the wrapper, ranking test
  -> Checkpoint A: numbers to user, win or stop
  -> T4 promote artifacts, all verify scripts
  -> T5 wording: README, guide, apps, CLAUDE.md, checkers
  -> T6 rebuild Vercel + HF bundles, bundle checks, Streamlit check, dryrun_demo
  -> Checkpoint B: user reviews diff
  -> T7 deploys (ask before each)
  -> T8 spec status implemented, fold into SPEC.md
```

Everything is sequential: each step consumes the previous step's artifact.

## Risks

| Risk | Mitigation |
|---|---|
| Win comes only from the trend, not from climate | `crop_province_mean_plus_trend` baseline is reported next to the model. Docs state both numbers. |
| Linear trend extrapolated to 2022 overshoots for some crops | Trend variant picked on validation. The pinned year only moves the displayed absolute level, never the ranking. |
| `single_thread` reaches into `named_steps["regressor"]`, which the wrapper doesn't have | T3 updates `single_thread` to unwrap `OffsetYieldModel.inner_`. Bit-exact tests prove it. |
| Pickle needs the class at load time in every deploy | Class lives in `features.py`, which all bundles already ship. `verify_vercel_bundle.py` and `verify_hf_bundle.py` load the pickle in a clean venv. |
| Guide and checkers pin demo numbers (Sindh, driest year, rice: 19.71 / 58.5%) | T5 regenerates them from the new model. `verify_guide.py` compares them to the live local app. |
| Pushing by mistake to remote `Hunain` (malware branch) | Pushes go only to `hunain-deploy` (Streamlit) and `vercel-deploy`, each after asking you. Never `git push origin Hunain`. |

## Verification checkpoints

- After T1–T3: `verify_model.py` passes with the new tests, using a scratch artifact.
- Checkpoint A: user sees val and test numbers for every model × trend combination and both baselines.
- After T6: all 8 verify scripts pass, plus `dryrun_demo.js` against `uvicorn app.main:app --port 8000`.
- After T7: logged-out visitors to Vercel and Streamlit see the same numbers as local.

## Open

- Pinned year: 2022 by default unless you choose 2016.
