# Todo: finish the project (plan in `tasks/plan-completion.md`)

## Phase 1: the report
- [ ] **Task 0** Reconcile stale boxes in `tasks/todo.md` (Checkpoints A, B, C and two Phase 1 items). Verify: no stale box left.
- [ ] **Task 1** Group supplies: what each member built (names and roll numbers are in the submitted proposal), submission branch, instructor question about the Space. (Blocking for Tasks 3, 6, 7.)
- [ ] **Task 2** Figures and screenshots saved in `docs/report/figures/` (5 or more EDA plots, comparison table, Streamlit, Vercel and local screenshots).
- [ ] **Task 2b** "What we proposed and what we built" section (README short version, report full version); success criterion (R² above 0.7, beats global mean baseline) checked from `metrics.json`; dataset documentation promise checked.
- [ ] **Task 3** `scripts/build_report.py` and `scripts/verify_report.py`; PDF of 3 to 4 pages, all sections, numbers from `metrics.json`, limits stated, contribution table filled from Task 1.
- [ ] **Checkpoint 1** User reads and accepts the PDF.

## Phase 2: the third deployment
- [ ] **Task 4** Choose the Space route (instructor answer, Streamlit-SDK Space, static Space, or PRO). Publish only after the user approves; needs a fresh write token.
- [ ] **Task 5** README table, deployment guide, demo checklist, `CLAUDE.md`, `verify_readme.py`, `verify_guide.py` updated together.
- [ ] **Checkpoint 2** All three deployment links open for a logged-out visitor and show 19.71 and 58.5% (Sindh, driest real year, rice).

## Phase 3: submission
- [ ] **Task 6** "Who answers what" table in `docs/demo_and_viva_guide.md`.
- [ ] **Task 7** Final state on the chosen branch; fresh clone, new venv, all checks pass; no secrets in history.
- [ ] **Task 8** Group rehearsal of the 10-minute demo on the live URLs.
- [ ] **Checkpoint 3** Done: report accepted, three URLs live, README complete, fresh clone passes, rehearsal done.

## User housekeeping (not code)
- [ ] Revoke the two Hugging Face tokens pasted in chat ("ML 1.1" and the read-and-write one).
- [ ] Check what `main` and `Hunain` on GitHub now contain after the force rewrite.
