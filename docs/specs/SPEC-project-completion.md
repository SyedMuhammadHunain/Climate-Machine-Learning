# Spec Delta: Project completion against the CT-378 guidelines

Status: approved (2026-11-19)

## Objective

Close every gap between the repository and `CT-378 ML Project Guidelines.pdf` so the group can submit and demo. The detailed task list already exists in `tasks/plan-completion.md` and `tasks/todo-completion.md`; this delta fixes the scope, records the group roles the user supplied, and sets the success criteria for "100% complete".

Group roles, as stated by the user (Syed Muhammad Hunain) on 2026-11-19:

| Member | Roll number | Role |
|---|---|---|
| Syed Muhammad Hunain | CT-24281 | Model creation and documentation |
| Muhammad Afzal | CT-24263 | Frontend and deployments |
| Ayesha | CT-24309 | Backend |

This answers the roles part of Task 1 in `tasks/plan-completion.md`.

## Current Behavior

Done and checked (see `SPEC-promote-offset-model.md`, `docs/deployment_guide.md`):

- Full ML lifecycle: EDA notebook, scikit-learn Pipeline, year split, three baselines plus Ridge, Random Forest, Gradient Boosting, `GroupKFold` tuning, test scored once, `joblib`, pinned requirements.
- Streamlit Community Cloud: live, current model.
- Local FastAPI app with `/predict`, HTML pages, validation and error messages; also live on Vercel.
- Hugging Face model repo: live, but holds the earlier model, not the promoted one.
- README with group table, setup steps, generated results block.

Missing:

1. Report PDF (3 to 4 pages, all sections, screenshots, contribution table). Worth 3 of 10 marks. Nothing exists yet.
2. Hugging Face live Space. The Hub refused a free Gradio Space (needs PRO).
3. Hugging Face model repo still serves the earlier model.
4. Contribution and "who answers what" tables: `(group to fill)` in `docs/demo_and_viva_guide.md`; README group table has no roles.
5. Stale unchecked boxes in `tasks/todo.md`, `tasks/plan.md`.
6. No final submission branch decided; no fresh-clone re-test.

## New Behavior

1. **Roles recorded.** README group table gains a Role column. Report contribution table and the "who answers what" table use the roles above, mapped to files:
   - Hunain: `src/climate_machine_learning/features.py`, `training.py`, `inference.py`, `scripts/train.py`, notebook, README, `docs/`, report, verify scripts for model and docs.
   - Afzal: `frontend/climate-resilient-frontend/`, `streamlit_app/`, `hf_space/`, Vercel and Hugging Face bundles, deployment guide.
   - Ayesha: `app/` (FastAPI routes, schemas, `ModelService`, validation and error messages), `scripts/verify_backend.py`.
2. **Report PDF** at `docs/report/report.pdf`, built by `scripts/build_report.py` (HTML to PDF with the local headless Chrome, no new dependency), checked by `scripts/verify_report.py`. Sections: Abstract, Problem and motivation, Data collection (source, method, licence, date per dataset), EDA, Methodology, Results (comparison table and plots), Deployment (screenshots), Limitations and future work, References, Individual contribution table. Includes a short "What we proposed and what we built" part. Figures in `docs/report/figures/`, generated from the notebook data, `metrics.json` and real screenshots.
3. **Hugging Face model repo republished** from `hf_bundle/model/` with the promoted model (needs a fresh write token from the user, passed by env var only).
4. **Third deployment covered** by the route the user picks (see Open Questions). Docs and checkers (`verify_readme.py`, `verify_guide.py`, `CLAUDE.md`) updated together when the status changes.
5. **Old checklists reconciled**: each open box either ticked with evidence or pointed at a task in `tasks/todo-completion.md`.
6. **Submission state**: final work on the chosen branch, fresh clone plus new venv passes every check, no tokens in history.

## Success Criteria

- `docs/report/report.pdf` exists, 3 to 4 pages, every required section heading present.
- `scripts/verify_report.py` passes and fails when one metric in the report is changed by hand; every metric in the report equals `models/metrics.json`.
- Report states the caveats from `inference.model_caveat`: beats crop-province-mean baseline on test R² and MAE only, loses on log-MAE, metrics chosen after seeing test results, most gain from year trend (trend-only R² 0.875), likely range covers 58% of test rows. Yield labelled "dataset yield units".
- Report references cite the crop dataset, NASA POWER and the FAO water-need table.
- Contribution table in the report, README role column and "who answers what" table all show the three roles above and agree with each other; `verify_guide.py` / `verify_readme.py` check that no `(group to fill)` remains.
- Hugging Face model repo, downloaded without a token, gives the same predictions as `models/crop_yield_pipeline.joblib` (Sindh, driest real year, rice: 40.85 and 89.3%).
- Third deployment: per chosen route, opens for a logged-out visitor and shows 40.85 and 89.3% for that scenario; or the docs state the instructor's answer if no Space is built.
- All existing checks still pass: `verify_model.py`, `verify_backend.py`, `verify_frontend.py`, `verify_readme.py`, `verify_guide.py`, `verify_vercel_bundle.py`, `verify_hf_bundle.py`, `verify_streamlit.py`, `node scripts/dryrun_demo.js`.
- Fresh clone of the submission branch passes the same checks.

Must NOT change: the model, `metrics.json`, the live Streamlit and Vercel apps (no redeploy unless docs inside them change), the "model output only" rule.

## Boundaries

Deltas only; everything else inherits from `SPEC.md` and later deltas.

- Always: generate every number in the report from `metrics.json` or the dataset; keep the caveats next to any claim of beating the baseline.
- Ask first: publishing anything to Hugging Face (needs the user's token), buying PRO, pushing to GitHub, changing which branch Streamlit deploys from.
- Never: invent who wrote which code beyond the roles the user gave; merge or run `main`, remote `Hunain` or `afzal` (malicious loader); put a token in a file or command line.

## Out of Scope (Deferred)

- Retraining or a better model: the current model is accepted.
- Best-crop classification model (proposal said "may").
- Cleaning the malicious commits off `main` on GitHub: group decision, separate task.
- Licence choice for the model card: group decision.

## Decisions (user, 2026-11-19)

1. Space: the Hugging Face model repo plus the two live apps (Streamlit, and the FastAPI app on localhost and Vercel) is enough. No Space is built. Docs say this instead of "not deployed yet (free plan)" as a pending item.
2. Submission branch: `hunain-deploy`.
3. Commit history: no note about it is added anywhere.
4. Report: HTML to PDF with the local headless Chrome, as proposed.

Success criterion for the third deployment therefore becomes: docs and checkers state that no Space is deployed by the group's decision, and the model repo serves the promoted model.

## Open Questions (answered, kept for history)

1. **Hugging Face Space route.** (a) Ask the instructor whether the model repo plus two live apps is enough; (b) try a Streamlit-SDK Space on the free plan (one create call tells us if it is refused too); (c) static Space calling the Vercel API (free, but not Gradio or Streamlit); (d) one month of PRO for the ready Gradio Space. Recommended: try (b) first, fall back to (a).
2. **Submission branch.** `hunain-deploy` (clean, Streamlit deploys from it) or a new clean branch. Remote `main` holds the malicious loader, so the instructor should not be pointed there until it is cleaned.
3. **Commit history vs roles.** Git shows 112 commits by Syed Muhammad Hunain, 4 by Afzal-Iqbal, none by Ayesha. The guidelines say every member must contribute code and the viva asks each member to explain their own code. The report will use the roles you gave; you may want Afzal and Ayesha to own and rehearse their parts (`app/` for Ayesha, `frontend/` and deploy bundles for Afzal) before the viva.
4. **Report format.** HTML-to-PDF via the local Chrome (proposed), or a `.docx` the group edits further?
