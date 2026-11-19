# Implementation Plan: Responsive pages, clearer ranking and crop detail pages

Source spec: `SPEC-responsive.md` (approved 2026-11-19, Decisions table at the end)
Task list: `tasks/todo-responsive.md`

## Overview

Make the three HTML pages (and Streamlit where it applies) work from 320 px Android phones to
2560 px monitors; replace the two landing buttons with one Start button; show tied crops as
groups instead of an alphabetical list that puts apple first; shorten result rows; add one detail
page per crop; restructure the advisory text; cut overwhelming text without touching the
required caveats. No retrain, no number changes, no new endpoint.

## Architecture Decisions

- **Measure first.** `scripts/verify_responsive.py` is built before any CSS change, runs against
  the current pages and records today's failures. Every later task turns some of those failures
  green. It reuses the CDP client from `scripts/capture_screenshots.py`, moved to
  `scripts/chrome_cdp.py` so one copy serves both.
- **Plain CSS inside each page**, no framework (spec Boundaries). Shared tokens (fluid type scale,
  breakpoints 480 / 600 / 1024 px, darker grey tokens, 44 px targets) defined once per page in
  `:root`, same values on `predict.html`, `metrics.html`, `crop.html`. Copy-paste across three
  pages accepted over adding a shared stylesheet because `verify_frontend.py` and the Vercel
  bundle check each page as a self-contained file; a shared `assets/site.css` is an option if the
  duplication hurts (see Open Questions).
- **Tie groups and "barely grown" live in `src/climate_machine_learning/inference.py`** as pure
  functions (`tie_groups`, `barely_grown`), so FastAPI, Streamlit and the Vercel bundle share one
  rule. Not in `presentation.py`: `app/` does not import it. The Hugging Face model repo is not
  republished in this change.
- **API additions only, no removals.** `RecommendationResponse` gains `groups` (list of
  {rank, pct_of_normal, crops}), `headline` (one sentence), `scenario_facts` (label/value list)
  and per-item `barely_grown: bool`. `recommendation_summary` and `ranked_crops` keep their
  meaning so existing checks and clients keep working.
- **Crop page = one static `crop.html`**, scenario in the query string, data from
  `/api/v1/metadata` and `/api/v1/recommend` (one call for the scenario, three for the real-years
  table). Served by the existing `StaticFiles` mount; Vercel bundle already copies the whole
  `frontend/` folder (`OWNED` in `build_vercel_bundle.py`), so only the checker's expected-file
  list changes.
- **Caveats verbatim.** `inference.model_caveat` and `inference.absolute_yield_caveat` are shown
  in full, always visible (Decision 8). The page splits the advisory into parts; it never edits
  the caveat strings.
- **Landing page** keeps the scene; phone path lowers pixel ratio to 1.5 and wave segments
  (Decision 1), sizes from `visualViewport`, debounces resize.

## Dependency graph

```
T1 chrome_cdp.py + verify_responsive.py (baseline, expected failures listed)
 ├── T2 tokens + tabs (predict, metrics)              [layout]
 │    ├── T3 metrics tables: cards, sticky, readable   [layout]
 │    ├── T4 predict form + action row responsive      [layout]
 │    └── T5 landing: Start button + viewport + phone scene  [landing]
 ├── T6 inference.tie_groups / barely_grown + API fields      [ranking]
 │    ├── T7 predict rows (R8) + tie groups on page (R5)     [ranking]
 │    ├── T8 structured advisory (R10) + median preset fix    [ranking]
 │    └── T9 Streamlit: groups, structured advisory, 360 px   [ranking]
 │         T10 crop.html (needs T2, T6, T7)                   [crop-page]
 │         T11 crop page checks + Vercel bundle + dryrun      [crop-page]
 └──────── T12 text cuts on all pages (needs T7-T10)          [copy]
           T13 docs, run_checks.sh, final full run            [copy]
```

## Task List

### Phase 1: Measure (module `layout`)
- [ ] T1: `scripts/chrome_cdp.py` + `scripts/verify_responsive.py` baseline

### Phase 2: Layout (modules `layout`, `landing`)
- [ ] T2: Fluid tokens, contrast, tabs on `predict.html` and `metrics.html`
- [ ] T3: Metrics tables: cards on phones, sticky header and first column, legends
- [ ] T4: Predictor form, action row, widths on all sizes
- [ ] T5: Landing page: one Start button, visual-viewport sizing, phone scene

### Checkpoint A: after T1-T5
- [ ] `verify_responsive.py` layout tests pass for index, predict, metrics at 9 sizes
- [ ] `./run_checks.sh` all PASS (incl. `dryrun_demo.js`)
- [ ] Human looks at 360 px and 1920 px screenshots before ranking work starts

### Phase 3: Ranking (module `ranking`)
- [ ] T6: `inference.tie_groups`, `inference.barely_grown`, new API fields
- [ ] T7: Predictor result rows (rank, share, yield) and tie groups
- [ ] T8: Structured advisory card and exact median preset
- [ ] T9: Streamlit: tie groups, structured advisory, 360 px check

### Checkpoint B: after T6-T9
- [ ] 9 province x preset cases: API equals direct `predict_crops`, groups correct, no sole best when tied
- [ ] `./run_checks.sh --all` PASS
- [ ] Human review of the new ranking view

### Phase 4: Crop page (module `crop-page`)
- [ ] T10: `crop.html` with all sections, prev / next, error state
- [ ] T11: Crop page checks, `dryrun_demo.js`, Vercel bundle checker

### Checkpoint C: after T10-T11
- [ ] All 13 crops open their page; numbers match API; R4/R6 pass at 9 sizes
- [ ] `./run_checks.sh` PASS; Vercel bundle rebuilt locally and verified

### Phase 5: Text (module `copy`)
- [ ] T12: Text cuts on all pages (R7), word caps, banned dev phrases
- [ ] T13: Docs, `run_checks.sh`, spec status, final full run

### Checkpoint D: complete
- [ ] All 18 success criteria in `SPEC-responsive.md` met
- [ ] `./run_checks.sh --all` PASS
- [ ] Manual check on a real Android phone over LAN (`uvicorn app.main:app --host 0.0.0.0 --port 8000`)
- [ ] Human review; redeploy is a separate step (Decision 3)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Changing result markup breaks `dryrun_demo.js` stub DOM | High | Keep ids (`team-items-1`, `advisory-summary`, `btn-submit-prediction`, ...); update dryrun in the same task that changes markup. |
| `verify_backend.py` TEST 22 flags new floats (e.g. `1.5` pixel ratio, `0.01` cut-off) as hardcoded metrics | Med | Keep numeric layout constants in CSS / named JS constants; cut-off 1% lives in `inference.py` as a named constant; run backend check after each task. |
| Caveat text accidentally altered while restructuring | High | Page renders API strings as-is; `verify_responsive.py` asserts DOM contains `model_caveat` and `absolute_yield_caveat` exactly. |
| Headless Chrome CDP timing flaky (fonts, WebGL, GSAP fade-in) | Med | Wait for `document.fonts.ready` and a page-set `data-ready` attribute; landing checks wait for animation end or use reduced-motion emulation. |
| WebGL unavailable in headless Chrome | Med | Run with `--use-angle=swiftshader`; landing layout checks do not depend on canvas content. |
| Landing scene too slow on low-end phones even after cap | Low | Phone path (Decision 1); `prefers-reduced-motion` skips animation. Manual phone check at Checkpoint D. |
| Retrain-determinism test in `verify_model.py` makes every full run ~1 min | Low | Run single tests during tasks (CLAUDE.md snippet); full run at checkpoints. |
| Remote `main` / `Hunain` / `afzal` malware | High | Work on local `Hunain` branch only; never pull or merge those remotes; never `npm run dev` / `node init`. |

## Open Questions

- Shared `assets/site.css` for the common tokens instead of a copy in each page? Default: copy in
  each page (matches current self-contained pages); revisit at Checkpoint A if duplication is
  large.
