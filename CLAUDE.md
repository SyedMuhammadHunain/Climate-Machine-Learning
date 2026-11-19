# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CT-378 ML course project: predict crop yield for 13 crops in Punjab, Sindh, Balochistan from annual climate and soil wetness. Rank crops by how much of typical year's yield each keeps in scenario. Python 3.13 (`.python-version`), venv at `.venv`, `pip install -r requirements.txt` (full dev environment, Jupyter included).

## Commands

```bash
uvicorn app.main:app --port 8000          # API + HTML pages together (scripts/run_backend.sh does the same)
python scripts/train.py                   # retrain (~1 min): rewrites models/crop_yield_pipeline.joblib, metrics.json, reference.json
python scripts/verify_readme.py --write   # after any retrain: regenerate the README results block from metrics.json
python scripts/fetch_nasa_power.py        # re-download the second data source (needs internet; files are already committed)
```

No pytest. Every check is plain-assert script; each prints PASS/FAIL, exits non-zero on failure:

| Script | Needs |
|---|---|
| `scripts/verify_model.py` (features, artifacts, inference, presentation, speed guard; includes ~1 min retrain-determinism test) | main `.venv` |
| `scripts/verify_backend.py` (one long function, API contract + scan of `app/` for banned words and hardcoded floats) | main `.venv` |
| `scripts/verify_frontend.py`, `verify_readme.py`, `verify_guide.py`, `verify_vercel_bundle.py` | main `.venv` |
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

**Data to model.** `data/raw/pakistan_crop_yield_climate.parquet` has 12 monthly rows per crop, province, year, but `Yield` is one yearly number repeated 12 times, no month column (month = row order, verified by `features.check_month_order`). `features.build_yearly_table` collapses to one row per crop-province-year (annual sums/means), merges NASA POWER soil wetness (`data/raw/nasa_power_monthly.csv`, province+year, fails loudly on missing year), drops rows with `Area = 0`. `training.py` splits by year (train ≤2016, val 2017-19, test 2020-22), tunes with `GroupKFold` grouped by `Year`, fits on log of yield, picks model on validation, scores test once. Artifacts in `models/`: pipeline, `metrics.json`, `reference.json` (province medians and trained ranges, real driest/wettest years, crop list, training rows per crop).

**One shared package, four front ends.** `src/climate_machine_learning/` is only place feature and ranking logic lives:
- `features.py`: inputs list (`BASE_INPUTS`), `engineered()` (works on floats and pandas columns, so training and inference share one formula), `build_inference_row`.
- `inference.py`: `predict_crops` ranks every crop by predicted yield in scenario divided by predicted yield in province's median year; raises `OutOfRangeError` for inputs outside trained range (tree model cannot extrapolate, so refused, not clamped).
- `presentation.py`: tables and wording shared by Streamlit and Gradio apps.
- Consumers: FastAPI `app/` (`ModelService`, pages in `frontend/climate-resilient-frontend/` served by `StaticFiles` mount at `/`, pages call same-origin `/api/v1/*`), `streamlit_app/` (own `requirements.txt`), `hf_space/` (Gradio), Vercel bundle.
- `app/core/config.py` puts `src/` on `sys.path` (package not installed). Package `__init__` must not import `uvicorn` at module level: Hugging Face model repo and Vercel bundle do not ship it.

**Generated deploy bundles.** `scripts/build_hf_bundle.py` (model repo + Gradio Space) and `scripts/build_vercel_bundle.py` (lean app + exact-pinned requirements) write git-ignored folders. Share `scripts/bundle_util.py`: builder only replaces entries it owns, refuses non-empty folders without its marker file or symlinks. Model repo already published on Hugging Face must not receive `presentation.py` (test asserts it).

## Rules this project enforces

- App shows only model output, dataset values or saved results. No fallback or demo numbers, no hand-written yield formulas, no hardcoded metrics. Guards: `verify_backend.py` TEST 22, `verify_frontend.py`, `verify_streamlit.py`.
- Yield unit not confirmed by data source; label it "dataset yield units". Best model does not beat crop-province-mean baseline, likely range covers only 60% of test rows; docs and apps say so. Do not claim otherwise.
- README numbers are generated (`<!-- METRICS:START -->` block), `verify_readme.py` / `verify_guide.py` pin deployment status (Streamlit and Vercel live, Hugging Face Space "not deployed yet"). When status changes, update docs and those checkers together.
- Bit-exact comparisons of predictions need `inference.single_thread(pipeline)`: saved forest sums trees in parallel, differs in last digit between runs.

## Repo state worth knowing

- Remote `Hunain` and `main` force-rewritten on GitHub; local `Hunain` diverged. Deploys use separate branches: `hunain-deploy` (Streamlit Cloud, main file `streamlit_app/app.py`) and `vercel-deploy` (orphan branch holding only generated Vercel bundle). Do not delete them.
- Specs and history: `SPEC.md` and `SPEC-*.md` (deltas), plan in `tasks/`, deployment steps in `docs/deployment_guide.md`, demo and viva script in `docs/demo_and_viva_guide.md`.
- `pyarrow` needed only to read training parquet; deployed API does not need it.