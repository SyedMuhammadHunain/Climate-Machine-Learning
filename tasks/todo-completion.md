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

- [ ] **T2: report figures script** (M)
  - Acceptance: `scripts/build_report_figures.py` writes to `docs/report/figures/`: at least 5 EDA plots (yield distribution, missing values, outliers by crop, correlation heatmap, rows per crop/province balance, rain vs yield) and results plots (model comparison R²/MAE from `metrics.json`, predicted vs actual on test). Uses `features.build_yearly_table`. Deterministic file names.
  - Verify: run it twice, same files; open each PNG; numbers on the comparison plot equal `metrics.json`.
  - Files: `scripts/build_report_figures.py`, `docs/report/figures/*.png`
  - Depends on: none

- [ ] **T3: deployment screenshots** (S)
  - Acceptance: real headless-Chrome captures of Streamlit (live URL), Vercel (live URL, predict page after a Sindh, driest real year request) and local `uvicorn` app, saved as `docs/report/figures/deploy_*.png`. Shown values match the local model.
  - Verify: open each image; Sindh, driest real year, rice shows 40.85 and 89.3% where that scenario is captured.
  - Files: `scripts/capture_screenshots.py` (or documented one-off commands), `docs/report/figures/deploy_*.png`
  - Depends on: none

### Checkpoint A
- [ ] `verify_readme.py`, `verify_guide.py`, `verify_model.py` pass
- [ ] Figures and screenshots reviewed

## Phase 2: report

- [ ] **T4: report builder, results section, checker** (M)
  - Acceptance: `scripts/build_report.py` renders `docs/report/report.html` and `docs/report/report.pdf` (Chrome `--headless --print-to-pdf --no-pdf-header-footer`). Results section has the comparison table (baselines and three models, validation and test) built from `metrics.json`, the comparison plot, and the caveat text from `inference.model_caveat`. `scripts/verify_report.py` (`TESTS` list) checks PDF exists, every metric in the PDF text equals `metrics.json`, caveat sentences present, "dataset yield units" present.
  - Verify: `.venv/bin/python scripts/build_report.py && .venv/bin/python scripts/verify_report.py` pass; edit one metric in the HTML and re-print, verify fails.
  - Files: `scripts/build_report.py`, `scripts/verify_report.py`, `docs/report/report_template.html`, `docs/report/report.pdf`
  - Depends on: T2

- [ ] **T5: all report sections** (M)
  - Acceptance: sections Abstract, Problem and motivation, Data collection (source, method, licence, access date for crop dataset and NASA POWER), EDA (figures from T2), Methodology, Results, Deployment (screenshots from T3, links), Limitations and future work, References (crop dataset, NASA POWER, FAO water-need table, scikit-learn), Individual contribution table (from T1 roles). Short "What we proposed and what we built" part, with proposal success criterion (test R² above 0.7, beats global mean baseline) checked by code. PDF is 3 to 4 pages. `verify_report.py` checks every heading, page count via `pdfinfo`, roles match README.
  - Verify: `verify_report.py` passes; render pages with `pdftoppm` and read them.
  - Files: `scripts/build_report.py`, `scripts/verify_report.py`, `docs/report/report_template.html`, `docs/report/report.pdf`
  - Depends on: T1, T3, T4

### Checkpoint 1
- [ ] User reads and accepts `docs/report/report.pdf`

## Phase 3: Hugging Face and docs

- [ ] **T6: republish Hugging Face model repo** (S, ask first)
  - Acceptance: `python scripts/build_hf_bundle.py` then upload `hf_bundle/model/` to `SMHunain/crop-yield-drought-pakistan` with `HF_TOKEN` from env only. Token-less download predicts 40.85 and 89.3% for Sindh, driest real year, rice. `presentation.py` not uploaded.
  - Verify: fresh download into `$CLAUDE_JOB_DIR/tmp`, run bundled example; `verify_hf_bundle.py` passes.
  - Files: none in repo (generated `hf_bundle/` is git-ignored)
  - Depends on: user supplies write token

- [ ] **T7: docs and checkers for final deployment status** (M)
  - Acceptance: README deployment table, `docs/deployment_guide.md`, demo checklist in `docs/demo_and_viva_guide.md`, `CLAUDE.md` say: model repo live with the promoted model; no Space, by the group's decision (model repo plus Streamlit plus FastAPI app are the deployments). `verify_readme.py` and `verify_guide.py` rules updated to match (no longer require "not deployed yet" as pending; no longer say "earlier model").
  - Verify: both checkers pass; change the README row back by hand, checker fails.
  - Files: `README.md`, `docs/deployment_guide.md`, `docs/demo_and_viva_guide.md`, `CLAUDE.md`, `scripts/verify_readme.py`, `scripts/verify_guide.py` (6 files, docs only, one consistency change)
  - Depends on: T6

### Checkpoint 2
- [ ] Every main-venv check passes; `node scripts/dryrun_demo.js` passes against local server
- [ ] README links open logged out

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
