# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CT-378 ML course project: predict crop yield for 13 crops in Punjab, Sindh, Balochistan from annual climate and soil wetness. Rank crops by share of typical year's yield each keeps in scenario. Python 3.13 (`.python-version`), venv at `.venv`, `pip install -r requirements.txt` (full dev environment, Jupyter included).

## Commands

```bash
uvicorn app.main:app --port 8000          # API + HTML pages together (./run_app.sh does the same and sets up .venv; run_backend.sh = --reload)
./run_checks.sh [--all]                   # every check with a summary; --all adds Streamlit and HF checks in .venv-streamlit / .venv-hf
./run_streamlit.sh                        # Streamlit app locally on :8501 (own .venv-streamlit)
python scripts/train.py                   # retrain (~1 min): rewrites models/crop_yield_pipeline.joblib, metrics.json, reference.json
python scripts/verify_readme.py --write   # after any retrain: regenerate the README results block from metrics.json
python scripts/fetch_nasa_power.py        # re-download the second data source (needs internet; files are already committed)
```

No pytest. Every check plain-assert script; each prints PASS/FAIL, exits non-zero on failure:

| Script | Needs |
|---|---|
| `scripts/verify_model.py` (features, artifacts, inference, presentation, speed guard; includes ~1 min retrain-determinism test) | main `.venv` |
| `scripts/verify_backend.py` (one long function, API contract + scan of `app/` for banned words and hardcoded floats) | main `.venv` |
| `scripts/verify_frontend.py`, `verify_readme.py`, `verify_guide.py`, `verify_vercel_bundle.py`, `verify_report.py` (report needs `google-chrome`, `pdftotext`, `pdfinfo`) | main `.venv` |
| `scripts/verify_responsive.py` (headless Chrome renders every page and the predictor's states at 9 sizes, 320 to 2560 px: no sideways scroll, tabs, 44 px targets, 16 px form text, contrast, tables as cards on phones, the landing scene, ranking rows and crop pages against the API, word caps; about 3 minutes; `--only`, `--page`, `--size` narrow it) | main `.venv`, `google-chrome` |
| `scripts/verify_responsive.py --streamlit` (the Streamlit app at phone widths) | main `.venv` plus `.venv-streamlit` |
| `node scripts/dryrun_demo.js http://127.0.0.1:8000` (real page JS vs API, stub DOM) | server running |
| `scripts/verify_streamlit.py` | own venv with `streamlit_app/requirements.txt` |
| `scripts/verify_hf_bundle.py` | own venv with `hf_space/requirements.txt` + `pyyaml` |

Run one test (`verify_model.py`, `verify_hf_bundle.py`, `verify_streamlit.py`, `verify_vercel_bundle.py` keep tests in `TESTS` list; `verify_backend.py` cannot be filtered):

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "scripts"); sys.argv = ["x"]
import verify_model as V
[f() for f in V.TESTS if f.__name__ == "engineered_features_match_hand_computed_values"]
EOF
```

## Architecture

**Data to model.** `data/raw/pakistan_crop_yield_climate.parquet` has 12 monthly rows per crop, province, year, but `Yield` is one yearly number repeated 12 times, no month column (month = row order, verified by `features.check_month_order`). `features.build_yearly_table` collapses to one row per crop-province-year (annual sums/means), merges NASA POWER soil wetness (`data/raw/nasa_power_monthly.csv`, province+year, fails loudly on missing year), drops rows with `Area = 0`. `training.py` splits by year (train ≤2016, val 2017-19, test 2020-22), tunes with `GroupKFold` grouped by `Year`, picks model and trend variant on validation, scores test once. Each model learns log-scale gap from crop-province offset plus optional year trend (`features.OffsetYieldModel`); Year never reaches inner model, pinned to `inference_year` (2022) at prediction. Artifacts in `models/`: pipeline, `metrics.json`, `reference.json` (province medians and trained ranges, real driest/wettest years, crop list, training rows per crop).

**One shared package, four front ends.** `src/climate_machine_learning/` only place feature and ranking logic lives:
- `features.py`: inputs list (`BASE_INPUTS`), `engineered()` (works on floats and pandas columns, so training and inference share one formula), `build_inference_row`, `OffsetYieldModel` (saved model class, so every bundle shipping `features.py` can unpickle it).
- `inference.py`: `predict_crops` ranks every crop by predicted yield in scenario divided by predicted yield in province's median year; raises `OutOfRangeError` for inputs outside trained range (tree model cannot extrapolate, so refused, not clamped).
- `presentation.py`: tables and wording shared by Streamlit and Gradio apps.
- Consumers: FastAPI `app/` (`ModelService`, pages in `frontend/climate-resilient-frontend/` served by `StaticFiles` mount at `/`, pages call same-origin `/api/v1/*`), `streamlit_app/` (own `requirements.txt`), `hf_space/` (Gradio), Vercel bundle.
- `app/core/config.py` puts `src/` on `sys.path` (package not installed). Package `__init__` must not import `uvicorn` at module level: Hugging Face model repo and Vercel bundle do not ship it.

**Generated deploy bundles.** `scripts/build_hf_bundle.py` (model repo + Gradio Space) and `scripts/build_vercel_bundle.py` (lean app + exact-pinned requirements) write git-ignored folders. Share `scripts/bundle_util.py`: builder only replaces entries it owns, refuses non-empty folders without marker file or symlinks. Model repo already published on Hugging Face must not receive `presentation.py` (test asserts it).

## Rules this project enforces

- App shows only model output, dataset values or saved results. No fallback or demo numbers, no hand-written yield formulas, no hardcoded metrics. Guards: `verify_backend.py` TEST 22, `verify_frontend.py`, `verify_streamlit.py`.
- Yield unit not confirmed by data source; label it "dataset yield units". Chosen model beats crop-province-mean baseline on test R² and MAE only, not on log-MAE, and user chose those two metrics after seeing test results; most of gain is year trend (trend-only baseline R² 0.875), likely range covers 58% of test rows. Docs and apps say so, generated from `metrics.json` by `inference.model_caveat`. Never state the win without the log-MAE loss and the after-the-fact choice. See `docs/specs/SPEC-beat-baseline.md` and `docs/specs/SPEC-promote-offset-model.md`.
- README numbers generated (`<!-- METRICS:START -->` block), `verify_readme.py` / `verify_guide.py` pin deployment status (Streamlit and Vercel live, Hugging Face model repo live with current model, no Space by group decision). When status changes, update docs and those checkers together.
- Bit-exact comparisons of predictions need `inference.single_thread(pipeline)` (plain function, call before `predict`; not a context manager): saved forest sums trees in parallel, differs in last digit between runs.
- Report: `scripts/build_report_figures.py`, `scripts/capture_screenshots.py` (live apps, headless Chrome), `scripts/build_report.py` write `docs/report/`; `scripts/verify_report.py` checks it. Group roles live only in `scripts/group_roles.py`. After a retrain rebuild figures and report, recapture screenshots once apps are redeployed.

## Repo state worth knowing

- Remote `main`, remote `Hunain` and `afzal` contain malicious loader (commits `2eafc19`, `43ab724`: `frontend/climate-resilient-frontend/init`, `src/climate_machine_learning/utils/session.py`, `import utils.session`). Never merge, check out or run them (no `npm run dev`, no `node init`). Deploys use clean branches: `hunain-deploy` (Streamlit Cloud, main file `streamlit_app/app.py`, requirements `streamlit_app/requirements.txt`) and `vercel-deploy` (orphan branch holding only generated Vercel bundle; live Vercel site deployed with Vercel CLI from same bundle). Do not delete them.
- Pages in `frontend/climate-resilient-frontend/`: `index.html` (landing, one Start button to `predict.html`), `predict.html`, `metrics.html`, and `crop.html` (one crop, scenario in the query string). All four are self-contained files with their own CSS; `crop.html` is guarded in `verify_frontend.py`, `dryrun_demo.js`, `verify_vercel_bundle.py`. The API's `/recommend` also returns `groups` (tied crops share one rank), `headline`, `scenario_facts`, `model_caveat` and per-crop `barely_grown`, all built in `inference.py`.
- Specs and history: `docs/specs/SPEC.md` and `docs/specs/SPEC-*.md` (deltas, latest `SPEC-responsive.md`), plans and task lists in `docs/plans/`, course documents in `docs/course/`, deployment steps in `docs/guides/deployment_guide.md`, demo and viva script in `docs/guides/demo_and_viva_guide.md`.
- `pyarrow` needed only to read training parquet; deployed API does not need it.