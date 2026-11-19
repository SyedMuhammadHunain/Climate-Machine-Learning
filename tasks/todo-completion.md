# Todo: project completion

Plan: `tasks/plan-completion.md`. Spec: `SPEC-project-completion.md`.

## Phase 1: groundwork

- [x] **T0: reconcile old checklists** (XS) (done 2026-11-19: all 17 open boxes ticked with evidence; `grep "\[ \]"` on both files is empty)
  - Acceptance: every open box in `tasks/todo.md` and `tasks/plan.md` is ticked with date and evidence, or points to a task here.
  - Verify: `grep -n "\[ \]" tasks/todo.md tasks/plan.md`; each hit names a task in this file.
  - Files: `tasks/todo.md`, `tasks/plan.md`
  - Depends on: none

- [x] **T1: roles in README and viva guide** (S) (done 2026-11-19: roles live in `scripts/group_roles.py`; both checkers read it; failed before the docs change, pass after; a placeholder put back by hand fails `verify_readme.py`)
  - Acceptance: README group table has a Role column (Hunain: model creation and documentation; Afzal: frontend and deployments; Ayesha: backend). "Who answers what" table in `docs/demo_and_viva_guide.md` filled from those roles, no `(group to fill)` left. Checkers fail if `(group to fill)` returns or a role is missing.
  - Verify: `.venv/bin/python scripts/verify_readme.py` and `scripts/verify_guide.py` pass; put `(group to fill)` back by hand, checker fails.
  - Files: `README.md`, `docs/demo_and_viva_guide.md`, `scripts/verify_readme.py`, `scripts/verify_guide.py`
  - Depends on: none

- [x] **T2: report figures script** (M) (done 2026-11-19: 6 EDA + 2 results figures; `scripts/verify_report.py` 4/4: figures exist, comparison values equal `metrics.json`, saved model reproduces test R² 0.885 and MAE 29.7 on 102 rows, rebuild byte-identical; a tampered value fails)
  - Acceptance: `scripts/build_report_figures.py` writes to `docs/report/figures/`: at least 5 EDA plots (yield distribution, missing values, outliers by crop, correlation heatmap, rows per crop/province balance, rain vs yield) and results plots (model comparison R²/MAE from `metrics.json`, predicted vs actual on test). Uses `features.build_yearly_table`. Deterministic file names.
  - Verify: run it twice, same files; open each PNG; numbers on the comparison plot equal `metrics.json`.
  - Files: `scripts/build_report_figures.py`, `docs/report/figures/*.png`
  - Depends on: none

- [x] **T3: deployment screenshots** (S) (done 2026-11-19: `scripts/capture_screenshots.py` drives a clean headless Chrome over the DevTools protocol; Streamlit, Vercel (all crops) and local (`/predict`, rice) each show rice 40.85 and 89.3% for Sindh, driest real year 2002; `verify_report.py` 5/5, a tampered check file fails)
  - Acceptance: real headless-Chrome captures of Streamlit (live URL), Vercel (live URL, predict page after a Sindh, driest real year request) and local `uvicorn` app, saved as `docs/report/figures/deploy_*.png`. Shown values match the local model.
  - Verify: open each image; Sindh, driest real year, rice shows 40.85 and 89.3% where that scenario is captured.
  - Files: `scripts/capture_screenshots.py` (or documented one-off commands), `docs/report/figures/deploy_*.png`
  - Depends on: none

### Checkpoint A
- [ ] `verify_readme.py`, `verify_guide.py`, `verify_model.py` pass
- [ ] Figures and screenshots reviewed

## Phase 2: report

- [x] **T4: report builder, results section, checker** (M) (done 2026-11-19: `scripts/build_report.py` writes `docs/report/report.html` and prints `report.pdf` with headless Chrome; prose lives in the script next to the code that fills its numbers, no separate template file; `verify_report.py` 8/8; a changed table value and a changed caveat number in the HTML each make it fail)
  - Acceptance: `scripts/build_report.py` renders `docs/report/report.html` and `docs/report/report.pdf` (Chrome `--headless --print-to-pdf --no-pdf-header-footer`). Results section has the comparison table (baselines and three models, validation and test) built from `metrics.json`, the comparison plot, and the caveat text from `inference.model_caveat`. `scripts/verify_report.py` (`TESTS` list) checks PDF exists, every metric in the PDF text equals `metrics.json`, caveat sentences present, "dataset yield units" present.
  - Verify: `.venv/bin/python scripts/build_report.py && .venv/bin/python scripts/verify_report.py` pass; edit one metric in the HTML and re-print, verify fails.
  - Files: `scripts/build_report.py`, `scripts/verify_report.py`, `docs/report/report_template.html`, `docs/report/report.pdf`
  - Depends on: T2

- [x] **T5: all report sections** (M) (done 2026-11-19: 4-page PDF with all 10 sections plus abstract; `verify_report.py` 13/13: headings in order, 3 to 4 pages, contribution rows equal `scripts/group_roles.py`, proposal criterion verdict computed from `metrics.json` (met: R² 0.885 against overall mean -0.030), data sources, live links, screenshots, EDA figures and references present)
  - Acceptance: sections Abstract, Problem and motivation, Data collection (source, method, licence, access date for crop dataset and NASA POWER), EDA (figures from T2), Methodology, Results, Deployment (screenshots from T3, links), Limitations and future work, References (crop dataset, NASA POWER, FAO water-need table, scikit-learn), Individual contribution table (from T1 roles). Short "What we proposed and what we built" part, with proposal success criterion (test R² above 0.7, beats global mean baseline) checked by code. PDF is 3 to 4 pages. `verify_report.py` checks every heading, page count via `pdfinfo`, roles match README.
  - Verify: `verify_report.py` passes; render pages with `pdftoppm` and read them.
  - Files: `scripts/build_report.py`, `scripts/verify_report.py`, `docs/report/report_template.html`, `docs/report/report.pdf`
  - Depends on: T1, T3, T4

### Checkpoint 1
- [x] User reads and accepts `docs/report/report.pdf` (accepted 2026-11-19)

## Phase 3: Hugging Face and docs

- [x] **T6: republish Hugging Face model repo** (S, ask first) (done 2026-11-19, user approved using the saved `hf` CLI login: commit `056ccbd` on the Hub; token-less download is byte-identical to `hf_bundle/model/`, no `presentation.py`, and gives rice in Sindh, driest real year 2002: 40.85 and 89.3%; `verify_hf_bundle.py` 12/12 in its own venv)
  - Acceptance: `python scripts/build_hf_bundle.py` then upload `hf_bundle/model/` to `SMHunain/crop-yield-drought-pakistan` with `HF_TOKEN` from env only. Token-less download predicts 40.85 and 89.3% for Sindh, driest real year, rice. `presentation.py` not uploaded.
  - Verify: fresh download into `$CLAUDE_JOB_DIR/tmp`, run bundled example; `verify_hf_bundle.py` passes.
  - Files: none in repo (generated `hf_bundle/` is git-ignored)
  - Depends on: user supplies write token

- [x] **T7: docs and checkers for final deployment status** (M) (done 2026-11-19: README, deployment guide, demo checklist and `CLAUDE.md` say model repo current, no Space by group decision; report checklist item ticked; `verify_readme.py` and `verify_guide.py` rules changed first and failed, then passed; putting 'not deployed yet' back fails)
  - Acceptance: README deployment table, `docs/deployment_guide.md`, demo checklist in `docs/demo_and_viva_guide.md`, `CLAUDE.md` say: model repo live with the promoted model; no Space, by the group's decision (model repo plus Streamlit plus FastAPI app are the deployments). `verify_readme.py` and `verify_guide.py` rules updated to match (no longer require "not deployed yet" as pending; no longer say "earlier model").
  - Verify: both checkers pass; change the README row back by hand, checker fails.
  - Files: `README.md`, `docs/deployment_guide.md`, `docs/demo_and_viva_guide.md`, `CLAUDE.md`, `scripts/verify_readme.py`, `scripts/verify_guide.py` (6 files, docs only, one consistency change)
  - Depends on: T6

### Checkpoint 2
- [x] Every main-venv check passes; `node scripts/dryrun_demo.js` passes against local server (2026-11-19: model 51/51, backend, frontend, readme, guide, report 13/13, vercel bundle 7/7, dry run; `verify_hf_bundle.py` 12/12 in T6; `verify_streamlit.py` not re-run, `streamlit_app/` unchanged)
- [x] README links open logged out (2026-11-19: Streamlit loads after the push with no error, Vercel captured in T3, model repo downloaded without a token in T6)

## Phase 4: submission

- [ ] **T8: push and fresh-clone test** (S, ask before push)
  - Acceptance: work committed on local `Hunain`, pushed to `origin/hunain-deploy` (fast-forward only; never to remote `Hunain`, `main`, `afzal`). Fresh clone of `hunain-deploy`, new Python 3.13 venv, `pip install -r requirements.txt`, every check in `CLAUDE.md` table passes. `git log -p` has no `hf_` token strings. Streamlit app still loads after the push.
  - Verify: output of the clone run; headless Chrome on Streamlit URL.
  - Files: none new
  - Depends on: Checkpoint 1, Checkpoint 2

- [ ] **T9: rehearsal** (group)
  - 10-minute demo once on live URLs; open Streamlit and Vercel a few minutes early.

### Checkpoint 3: done
- [ ] Report accepted, Streamlit and Vercel live, model repo current, README complete, fresh clone passes, rehearsal done

## Not mine, open for the group
- [ ] Revoke the two Hugging Face tokens pasted in chat earlier, and the T6 token after use.
- [ ] Decide how to clean the malicious commits on GitHub `main`, `Hunain`, `afzal`.
