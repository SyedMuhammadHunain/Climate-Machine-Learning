# Plan: finish the project against the course guidelines

Written 2026-11-19. This plan is separate from `tasks/plan.md` and `tasks/todo.md` on purpose: those files still have unchecked boxes from earlier work, and the instructions say never to overwrite an unfinished plan silently. Task 0 reconciles them.

Source of requirements: `CT-378 ML Project Guidelines.pdf` (two pages) and `CT-378 Project Proposal Low-Water Crop Selection Under Drought.docx`. **The proposal was already submitted by the group, so it is a commitment: the report and the viva must be consistent with it, and every place where the project differs from it must be stated and explained, not hidden.**

## Proposal against what was built

| Proposal said | What the project has | What to do |
|---|---|---|
| Members: Muhammad Afzal (CT-24263), Syed Muhammad Hunain (CT-24281), Ayesha (CT-24309) | Names known; each member's role is not | Task 1 asks only for roles |
| Success: beat a mean-yield baseline clearly and reach test R² above 0.7 | Test R² 0.794 (Random Forest), 0.813 (Gradient Boosting). The global mean-yield baseline scores R² -0.03 and MAE 100.4 against the forest's 40.9. **Criterion met.** | State it plainly in the report |
| (not in proposal) | A stricter baseline, each crop's mean per province, scores R² 0.832 and MAE 34.7, so the chosen model does not beat it | Keep reporting this too; it is the honest limit |
| Target: yield "in kg/ha" | The dataset does not confirm the unit; rice values around 20 to 40 cannot be kg/ha, so the apps say "dataset yield units" | Explain this deviation in the report |
| Split: 70/15/15, year-ordered | Years 2001-2016 / 2017-2019 / 2020-2022 (565 training rows of 773) | Report the real shares |
| Sources: Open Data Pakistan rainfall and temperature (1901-2016), city weather 2000-2024, PMD drought centre, FAO/ERA5, Punjab irrigated area and district yields, plus the crop dataset | Two sources used: the crop and climate dataset, and NASA POWER soil wetness | State which proposed sources were not used and why (national series stop in 2016 and are not provincial; district yields and irrigated area cover Punjab only; the drought centre has no downloadable series). Do not claim they were used |
| Features: rainfall deficit against the long-term mean, irrigated area share | Rainfall against the long-term mean is built; irrigated area share is not (no irrigation column in either source) | Say so |
| Models: baseline, Linear Regression, Random Forest, Gradient Boosting, 5-fold tuned | Same set (Ridge for the linear model); `GroupKFold` by year instead of plain 5-fold, to stop leakage across crops in one year | Explain the grouping choice |
| User enters drought conditions, receives a ranked list of low-water crops with expected yield | All 13 crops ranked by share of a typical year, with predicted yield, likely range and FAO water need as context for 8 crops | Explain why all 13 are ranked |
| Optional: classification of the best crop | Not built; the proposal said "may" | Mention under future work |

## Where the project stands against the guidelines

| Guideline item | State |
|---|---|
| Problem, data from at least two public sources, 5,000 rows and 15 features | Raw data: 10,296 rows, 19 columns, plus NASA POWER (2,376 rows). The modelling table has 773 yearly rows because the yield is one number per year. The README and the viva guide say this openly. |
| ML lifecycle (EDA with 5 plots or more, Pipeline, year split, baseline plus three models, tuning, test once, joblib, pinned requirements) | Done. |
| Streamlit on Community Cloud with a public URL | Done and checked. |
| Local web app (FastAPI, `/predict`, HTML front end, validation, error messages) | Done. Also hosted on Vercel. |
| Hugging Face: model card and pipeline on the Hub | Done. |
| Hugging Face: a live Space | **Not done.** The Hub refused a Gradio Space on the free plan. |
| GitHub repo with README, setup steps and all three deployment links | README is complete except the Space link. |
| Report, PDF, 3 to 4 pages, with screenshots and an individual contribution table | **Not done.** Worth 3 of the 10 marks. |
| Demo (3 minutes problem and data, 4 minutes live demo, 3 minutes results) and viva | Script exists in `docs/demo_and_viva_guide.md`. The "who answers what" table is missing. |

Two things to know about the guidelines themselves. The heading says "choose any one of the following options", but the marking scheme gives 2 marks for "all three deployments working live", and a deployment that fails live scores zero for that platform. The safe reading is all three. The group should ask the instructor which reading applies, because it decides whether the Space is mandatory.

## Dependency graph

```
Task 0  reconcile old checklists
Task 1  facts the group must give (names, roles, submission branch)  ----+
Task 2  report figures and screenshots (needs live URLs, already live)   |
Task 2b proposal alignment section (needs metrics.json only)             |
   |                                                                      |
Task 3  report PDF built from metrics.json  <---- needs Task 1 + 2 + 2b --+
Task 4  Hugging Face Space (route chosen first)  --> Task 5 docs and checkers
Task 6  demo guide: who answers what  <---- needs Task 1
Task 7  submission branch and fresh-clone re-test  <---- needs Tasks 3, 5, 6
Task 8  rehearsal on the live URLs (group)
```

Tasks 2, 4 and 6 do not depend on each other and can run in any order. Task 3 is the critical path.

## Phase 1: the report (critical path, 3 marks)

### Task 0: reconcile the old checklists
Tick the boxes in `tasks/todo.md` that are already done (Checkpoints A, B and C, the comparison table, the sensitivity printout) with the date and the evidence, and mark the two real leftovers as open. Acceptance: no unchecked box in `tasks/todo.md` is stale. Verify: read the file; every remaining open box points to a task in this plan. Scope: small, docs only.

### Task 1: facts only the group can supply (blocking, ask first)
- What each member built, for the contribution table. The names and roll numbers come from the submitted proposal (Muhammad Afzal CT-24263, Syed Muhammad Hunain CT-24281, Ayesha CT-24309). Nobody's role may be guessed; until the group answers, that column says "to be filled by the group".
- Which GitHub branch the instructor will read. `main` and `Hunain` were force-rewritten on GitHub, and the work lives on `hunain-deploy`.
- Whether the group wants to ask the instructor about the Space (Task 4).

### Task 2: figures and screenshots for the report
One slice: produce every image the report needs, from the real apps and the notebook.
- EDA plots (5 or more) exported from `notebooks/01_low_water_crop_selection.ipynb`.
- Model comparison table and plots from `models/metrics.json`.
- Screenshots of the three deployments: Streamlit, Vercel and the local app (and the Space when it exists).
Acceptance: images saved under `docs/report/figures/`, each taken from the live or local app, none edited by hand. Verify: open each file; compare the numbers shown with `metrics.json`. Scope: medium.

### Task 2b: proposal alignment section
One slice: a "What we proposed and what we built" section for the report (and a short version in the README), generated from the table above. Acceptance: every row of the table is covered; the success criterion is checked by code (test R² above 0.7 and beating the global mean-yield baseline, both read from `metrics.json`); every deviation has a reason. Also confirm every dataset file has source, collection method, licence and access date written down, as the proposal promised. Verify: `scripts/verify_report.py` fails if the proposal criterion text and `metrics.json` disagree; read the README Data section against the promise. Scope: small to medium.

### Task 3: build the report PDF
Add `scripts/build_report.py` (HTML to PDF with the headless Chrome already on this machine, so no new dependency) and `scripts/verify_report.py`. The report has the sections the guidelines list: abstract, problem and motivation, data collection (with licences and dates), EDA, methodology, results (comparison table and plots), deployment (screenshots), limitations and future work, references, individual contribution table. Every metric in the text is read from `metrics.json` by the build script, never typed by hand.
Acceptance:
- PDF has 3 to 4 pages.
- Every required section heading is present.
- Every number in the results section equals `metrics.json`.
- The proposal alignment section (Task 2b) is included and the page limit still holds.
- The limits are stated as in the README: the best model does not beat the crop-province-mean baseline, and the likely range covers 60% of test rows against a nominal 80%.
- The yield unit is called "dataset yield units".
- References include both data sources and the FAO table.
Verify: `scripts/verify_report.py` passes; break one number on purpose and see it fail; read the PDF pages as images.
Scope: large. This is the longest task.

### Checkpoint 1 (stop and show the user)
The user reads the PDF. Nothing is submitted until they accept it.

## Phase 2: the third deployment

### Task 4: Hugging Face Space
Decide the route first; each needs the user's approval before anything is published.
1. Ask the instructor whether the model repo plus two live apps is enough (free, no work).
2. Try a Streamlit-SDK Space on the free plan. The guidelines allow "Gradio or Streamlit", and the earlier refusal was for a Gradio Space. Unknown whether it is also refused; one create call answers it.
3. A static Space that is a copy of the HTML pages calling the Vercel API (CORS already allows any origin). It is free and live, but it is neither Gradio nor Streamlit, so the instructor must accept it.
4. Hugging Face PRO for one month, then the already built Gradio Space (`scripts/build_hf_bundle.py`).
Acceptance for any route: the Space URL opens for a logged-out visitor, and Sindh, driest real year, rice shows 19.71 and 58.5%. Verify: open it in a real browser; run the matching verify script.
Needs: a Hugging Face token with write access, created by the user and revoked afterwards.

### Task 5: documents and checkers for the Space
Update the README table, `docs/deployment_guide.md`, the checklist in `docs/demo_and_viva_guide.md`, `CLAUDE.md`, and the rules in `scripts/verify_readme.py` and `scripts/verify_guide.py` together (the checkers currently require "not deployed yet"). Acceptance: all checkers pass and the docs say what is true. Verify: break the Space row on purpose; the checker fails.

### Checkpoint 2
All three deployments open from the README links.

## Phase 3: submission

### Task 6: demo guide, "who answers what"
Add the table to `docs/demo_and_viva_guide.md` from the real contribution split in Task 1. Acceptance: every part of the pipeline has a named owner who can explain it. Verify: `verify_guide.py` passes.

### Task 7: submission branch and fresh-clone re-test
Put the final state on the branch chosen in Task 1. Clone it fresh, create a new Python 3.13 environment, install `requirements.txt`, and run every check. Acceptance: all scripts pass from the clone; `README` setup steps work as written; no secrets in history (`git log -p` grep for token prefixes). Verify: the output of that run.

### Task 8: rehearsal (group)
Run the 10-minute demo once with the three live URLs open, and open the Vercel and Streamlit sites a few minutes beforehand (both can be slow on the first request).

### Checkpoint 3: done
Report accepted, three URLs live, README complete, fresh clone passes, rehearsal done.

## Risks
- The report must not claim proposal sources or features that were not built (irrigated area share, the national rainfall series, the drought centre data). A mismatch with the submitted proposal is easy for a marker to see.
- Report contribution table cannot be completed without the group's input.
- The instructor may require a Gradio or Streamlit Space; routes 3 and 4 differ in cost and risk.
- Vercel cold starts and Streamlit sleeping are real during a demo; open both first.
- A Hugging Face token pasted in chat has to be revoked afterwards. Two earlier ones should be revoked now.
- The model does not beat the simple baseline. That is honest and documented, and the viva answer should say it plainly.
