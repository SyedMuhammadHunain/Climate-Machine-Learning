# Implementation Plan: project completion

Source spec: `SPEC-project-completion.md` (approved 2026-11-19, decisions recorded there).

Replaces the 2026-11-19 draft of this file. Same work, now with the group's answers: roles known, no Hugging Face Space, submission branch `hunain-deploy`, report as HTML to PDF with local Chrome.

## Overview

Finish what the CT-378 guidelines still need: the report PDF (3 marks), the roles in every document, the Hugging Face model repo serving the promoted model, docs and checkers saying no Space is built by decision, old checklists reconciled, and a fresh-clone test of `hunain-deploy`.

## Architecture Decisions

- **Report built by code, not typed.** `scripts/build_report.py` renders `docs/report/report.html` from a template with every metric read from `models/metrics.json` and caveat text from `inference.model_caveat`, then prints it to `docs/report/report.pdf` with `google-chrome --headless --print-to-pdf`. No new dependency (Chrome at `~/.local/bin/google-chrome`, `pdfinfo`/`pdftotext` from poppler already present).
- **Figures by script.** `scripts/build_report_figures.py` regenerates the EDA and results plots with matplotlib (already pinned) from the parquet, NASA CSV and `metrics.json`. The notebook keeps its own plots; the script reuses `features.build_yearly_table` so both see the same table.
- **Screenshots** of Streamlit, Vercel and the local app are real captures (headless Chrome), saved once in `docs/report/figures/`, never edited.
- **Checker style unchanged.** `scripts/verify_report.py` is a plain-assert PASS/FAIL script like the others, with a `TESTS` list.
- **Roles are user-given facts.** Hunain: model and docs. Afzal: frontend and deployments. Ayesha: backend. Written once in README, report and viva guide; checkers make sure they agree.

## Dependency Graph

```
T0 reconcile old checklists            (independent)
T1 roles in README + viva guide + checks
T2 report figures (EDA + results)  --+
T3 deployment screenshots           --+--> T4 report: results + skeleton --> T5 report: all sections --> Checkpoint 1
T1 ----------------------------------------------------------------------> T5 (contribution table)
T6 republish HF model repo (token)   --> T7 docs + checkers: no Space, model repo current --> Checkpoint 2
everything --> T8 push hunain-deploy + fresh-clone test --> T9 rehearsal (group) --> Checkpoint 3
```

T0, T1, T2, T3, T6 are independent. T4/T5 is the critical path. T6 waits on a token from the user.

## Task List

### Phase 1: groundwork
- [ ] T0: reconcile old checklists
- [ ] T1: roles in README and viva guide, enforced by checkers
- [ ] T2: report figures script
- [ ] T3: deployment screenshots

### Checkpoint A
- [ ] `verify_readme.py`, `verify_guide.py` pass; figures and screenshots open and match `metrics.json`

### Phase 2: report
- [ ] T4: report builder with results section and `verify_report.py`
- [ ] T5: remaining report sections, proposal alignment, contribution table, 3 to 4 pages

### Checkpoint 1
- [ ] User reads and accepts the PDF

### Phase 3: Hugging Face and docs
- [ ] T6: republish model repo with promoted model (ask first, token by env var)
- [ ] T7: docs and checkers for "no Space by decision" and "model repo current"

### Checkpoint 2
- [ ] All checks pass; README links open for a logged-out visitor

### Phase 4: submission
- [ ] T8: push to `hunain-deploy` (ask first), fresh clone, every check passes, no secrets in history
- [ ] T9: group rehearsal (group does this)

### Checkpoint 3: done
- [ ] Report accepted, two apps live, model repo current, README complete, fresh clone passes

Full task details: `tasks/todo-completion.md`.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Report spills past 4 pages | High (guideline) | `verify_report.py` checks `pdfinfo` page count; figures sized small, two per row |
| A number typed by hand drifts from `metrics.json` | High (honesty rule) | Template only takes values from code; verify extracts PDF text and compares every metric; break-one-number test |
| Streamlit asleep or slow during screenshot | Low | Wait up to 60 s with `--virtual-time-budget`; retry; if still blank, capture after a manual wake |
| HF republish needs token; token leak | Med | Token only via `HF_TOKEN` env var, never in a file or argv; remind user to revoke after |
| Push to `hunain-deploy` restarts Streamlit app | Low | Docs-only change; check app afterwards in clean headless Chrome |
| Chrome PDF headers/footers or margins differ | Low | `--no-pdf-header-footer`, CSS `@page` margins |

## Open Questions

- None blocking. T6 needs a Hugging Face write token when we reach it.
