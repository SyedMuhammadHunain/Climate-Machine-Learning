# Todo: Responsive pages, clearer ranking and crop detail pages

Spec: `SPEC-responsive.md` (approved). Plan: `tasks/plan-responsive.md`.
Single test run (CLAUDE.md): `.venv/bin/python - <<'EOF' ... [f() for f in V.TESTS if f.__name__ == "<name>"] EOF`.
Server for browser checks: `uvicorn app.main:app --port 8000` (scripts start their own on a free port).

## Phase 1: Measure

- [x] **T1: CDP helper and responsive check baseline** (module `layout`, M)
  - Description: Move the `Chrome` CDP class out of `scripts/capture_screenshots.py` into
    `scripts/chrome_cdp.py`; add device emulation (size, DPR, touch, reduced motion). New
    `scripts/verify_responsive.py` (TESTS list, PASS/FAIL, non-zero exit) starts uvicorn on a free
    port, loads `index.html`, `predict.html`, `metrics.html` at the 9 sizes and runs the generic
    checks: no sideways page scroll, nothing outside viewport, no clipped text, tap targets
    >= 44 px and form text >= 16 px at phone widths, contrast >= 4.5:1. Screenshots to
    `$CLAUDE_JOB_DIR/tmp` or git-ignored `build/responsive/`.
  - Acceptance:
    - `capture_screenshots.py` uses `chrome_cdp.py` and behaves as before.
    - `verify_responsive.py` runs end to end and prints the current failures (expected, listed in
      the commit body as the baseline).
    - `--only <test>` and `--size <WxH>` flags for quick reruns.
  - Verify: `.venv/bin/python scripts/verify_responsive.py` (expected FAILs only),
    `.venv/bin/python scripts/capture_screenshots.py --only local`, `scripts/verify_report.py`.
  - Files: `scripts/chrome_cdp.py`, `scripts/capture_screenshots.py`, `scripts/verify_responsive.py`, `.gitignore`.
  - Depends on: none.
  - Done 2026-11-19. Baseline (2/9 pass): page 330 px wide at 320 px (predict wordmark SVG, 38 px tall,
    330 px wide); 62 tap targets under 44 px; 16 form fields at 14 px; 16 text colours under 4.5:1
    (`#7d8699` 3.66:1, active preset `#FF7344` 2.55:1); tabs touch the screen edge at phone widths and
    are 33 px tall at 800x360; landing has 2 buttons. Mobile emulation widens `innerWidth` to fit
    overflowing content, so the check measures against the emulated device width.

## Phase 2: Layout

- [x] **T2: Tokens, contrast and tabs** (module `layout`, S)
  - Description: In `predict.html` and `metrics.html` add `:root` fluid tokens (clamp type
    scale, gutters, breakpoints 480/600/1024 px), darken grey tokens to >= 4.5:1, widen columns
    (predict 960 px, metrics 1100 px at >= 1920 px, fluid between). Tabs (`.site-nav`,
    `.nav-pill`) wrap under 480 px, one row above, >= 44 px tall on phones, identical on both
    pages.
  - Acceptance:
    - R2 tab checks pass at 9 sizes on both pages.
    - Contrast check passes for all text on both pages.
    - No sideways page scroll on both pages at 9 sizes.
    - `metrics.html` header logo is an "Infinite Field®" wordmark (`svg#Layer_1`), not Low-Water Crops:
      replace it with the same Low-Water Crops wordmark `predict.html` uses (found in T1 screenshots).
    - Wordmark scales down on phones (fits 320 px with gutter).
  - Verify: `verify_responsive.py --only tabs,contrast,no_hscroll`; `scripts/verify_frontend.py`;
    `node scripts/dryrun_demo.js http://127.0.0.1:8000`.
  - Files: `frontend/climate-resilient-frontend/predict.html`, `frontend/climate-resilient-frontend/metrics.html`.
  - Depends on: T1.
  - Done 2026-11-19. Result tests pass at 9 sizes on both pages (6/6). Remaining baseline failures belong to
    T4 (tap targets, form text on predict) and T5 (landing). Grey `#596173` (6.2:1), orange text
    `--colorAccentText #b93a0a` (5.7:1, `#FF7344` kept for borders and fills only). Tabs stack one per row under 480 px.
    Columns: predict 720 to 960 px, metrics 780 to 1100 px, fluid between. `metrics.html` Infinite Field wordmark replaced.

- [x] **T3: Metrics tables** (module `layout`, S)
  - Description: Render JS sets `data-label` from each column's `<th>`; under 600 px tables 1, 3, 5
    become cards, tables 2 and 4 fit. From 600 px real tables with sticky header and sticky first
    column inside `.data-table-wrapper`, edge shadow when scrollable. Body text >= 14 px, headers
    >= 13 px semibold, chosen row highlighted, importance bar column headed "Relative weight" with
    value beside bar, SHA prefix monospace no-wrap, one-line legend under tables 1 and 2 built from
    existing labels.
  - Acceptance:
    - R3 and R6 table checks pass at 9 sizes; tables 1 and 5 no inner scroll at >= 1440 px.
    - Set of cell texts at every size equals the set at 1280 px.
    - `verify_frontend.py` (no typed metric values) passes.
  - Verify: `verify_responsive.py --only tables`; `scripts/verify_frontend.py`; `scripts/verify_backend.py`.
  - Files: `frontend/climate-resilient-frontend/metrics.html`, `scripts/verify_responsive.py`.
  - Depends on: T2.
  - Done 2026-11-19. 6 new tests in `verify_responsive.py` (cards, fit/scroll/sticky, text size, identical values,
    chosen row, legends) pass at 9 sizes, as do the shared layout tests on `metrics.html`. Also in this task:
    metrics column `clamp(780px, 76vw, 1100px)` (the 8-column tables fit from 1440 px), range cells as median over
    "(min to max)", narrow importance table keeps input names whole, stat grid 1/2/4 columns, section text 14 px,
    prose lines at most 75ch, logo link 44 px tall on both pages. Remaining responsive failures: predictor form (T4),
    landing (T5). Legends duplicate part of `cmp-desc`; T12 trims the description.

- [ ] **T4: Predictor form and widths** (module `layout`, S)
  - Description: Inputs one column under 600 px, two above; selects and inputs 16 px text and
    >= 44 px tall on phones; preset pills >= 44 px tap area, wrap; Predict / Reset stack
    full-width under 400 px; validation alert, loading indicator and result cards wrap with no
    clipped number.
  - Acceptance:
    - Generic checks pass on `predict.html` at 9 sizes before and after a prediction (Sindh driest
      preset, run by the check).
    - Full ranked list readable at 320 and 360 px.
  - Verify: `verify_responsive.py --only predict`; `node scripts/dryrun_demo.js http://127.0.0.1:8000`.
  - Files: `frontend/climate-resilient-frontend/predict.html`, `scripts/verify_responsive.py`.
  - Depends on: T2.

- [ ] **T5: Landing page** (module `landing`, M)
  - Description: Replace the two `careers-button` links with one `<a href="predict.html"
    class="careers-button">Start</a>` (same style, animation, >= 44 px). Canvas and overlay use
    `100dvh` and `visualViewport` size; resize debounced; phone path (under 800 px or touch):
    pixel ratio 1.5, fewer wave segments; `scroll-space` height scaled for phones; header,
    paragraph, button, footer non-overlapping at 320 x 568 and 800 x 360; `prefers-reduced-motion`
    skips Lenis and long animations and shows paragraph and button directly.
  - Acceptance:
    - Exactly one `a.careers-button`, text "Start", relative `href="predict.html"`; following it
      returns 200 for `/predict.html`.
    - Landing checks (fit, no overlap, button visible and tappable) pass at 9 sizes with reduced
      motion and after scroll-in animation without it.
    - Desktop scene unchanged at >= 1024 px (screenshot compare by eye at checkpoint).
  - Verify: `verify_responsive.py --only landing`; `scripts/verify_frontend.py`; `scripts/verify_backend.py` (serves `/`).
  - Files: `frontend/climate-resilient-frontend/index.html`, `scripts/verify_responsive.py`, `scripts/run_app.sh` (only if its printed URLs mention the buttons).
  - Depends on: T1.

### Checkpoint A (after T1-T5)
- [ ] `verify_responsive.py` layout and landing tests PASS at 9 sizes
- [ ] `./run_checks.sh` PASS
- [ ] Human reviews 360 px and 1920 px screenshots
- [ ] Commit per task (`caveman-commit` style messages)

## Phase 3: Ranking

- [ ] **T6: Tie groups, barely grown, API fields** (module `ranking`, M)
  - Description: In `inference.py` add `tie_groups(items)` (groups by share at `TIE_DIGITS`, rank
    = group position) and `barely_grown(items)` (typical predicted yield < `BARELY_GROWN_SHARE`
    = 1% of the largest typical predicted yield in the same result). `ModelService.recommend_crops`
    adds `groups`, `headline`, `scenario_facts`, per-item `barely_grown`; headline never names a
    sole best crop when the top group has more than one crop, and never leads with a barely-grown
    crop. Schemas updated. `recommendation_summary`, `ranked_crops` unchanged.
  - Acceptance:
    - New `verify_model.py` tests: groups for hand-built ties; barely-grown rule; for all 9
      province x preset cases API ranking equals direct `predict_crops` (single_thread).
    - New `verify_backend.py` asserts: `groups` present and consistent with `ranked_crops`; all-tied
      scenario headline names no single crop; Sindh apple `barely_grown` true; old asserts still pass.
  - Verify: single tests via CLAUDE.md snippet; `scripts/verify_backend.py`.
  - Files: `src/climate_machine_learning/inference.py`, `app/services/model_service.py`, `app/schemas/recommend.py`, `scripts/verify_model.py`, `scripts/verify_backend.py`.
  - Depends on: T1.

- [ ] **T7: Result rows and tie groups on the page** (module `ranking`, S)
  - Description: `predict.html` renders `groups`: group heading "Rank N · X% of typical" (with
    "N crops tied" when tied), rows with crop name (link to `crop.html?...`, built in T10, link
    present now) and "Yield Y"; unique crops show rank, share, yield on one row. Unit once in the
    results heading. FAO, training rows, typical yield, tie note removed from rows. Barely-grown
    crops get a small "barely grown here" tag. Whole row tappable, >= 44 px.
  - Acceptance:
    - Row visible text only crop, rank, share, yield (+ tag); unit appears once.
    - `dryrun_demo.js` updated: asserts groups rendered from API, no per-crop rank inside a tie.
    - R4 checks still pass on `predict.html` after prediction.
  - Verify: `node scripts/dryrun_demo.js http://127.0.0.1:8000`; `verify_responsive.py --only predict,rows`; `scripts/verify_frontend.py`.
  - Files: `frontend/climate-resilient-frontend/predict.html`, `scripts/dryrun_demo.js`, `scripts/verify_responsive.py`.
  - Depends on: T4, T6.

- [ ] **T8: Structured advisory and exact median preset** (module `ranking`, S)
  - Description: Advisory card shows `headline` (larger), `scenario_facts` grid (1 column under
    480 px, 2 above), "How crops are ranked" (`ranking_basis`), "About these numbers" with
    `model_caveat` and `absolute_yield_note` as separate paragraphs, always visible. Median preset
    fills exact `ref.median` values (no `toFixed(1)` in the value sent).
  - Acceptance:
    - R10 checks: four blocks present; facts equal request inputs; caveat DOM text equals
      `inference.model_caveat` and `inference.absolute_yield_caveat`.
    - Median preset gives 100.0% for every crop in all 3 provinces (dryrun or responsive check).
  - Verify: `verify_responsive.py --only advisory`; `node scripts/dryrun_demo.js http://127.0.0.1:8000`; `scripts/verify_frontend.py`.
  - Files: `frontend/climate-resilient-frontend/predict.html`, `scripts/dryrun_demo.js`, `scripts/verify_responsive.py`.
  - Depends on: T6, T7.

- [ ] **T9: Streamlit ranking view** (module `ranking`, S)
  - Description: `presentation.py` ranking table grouped by `inference.tie_groups` with
    barely-grown tag; app shows headline, facts, ranking rule, caveats (full, visible); median
    preset exact; dataframes `width="stretch"`; no sideways page scroll at 360 px.
  - Acceptance:
    - `verify_streamlit.py` passes (caveat still present) plus new asserts for groups and headline.
    - `verify_responsive.py --streamlit` passes at 360 x 800.
  - Verify: `.venv-streamlit/bin/python scripts/verify_streamlit.py`; `verify_responsive.py --streamlit`.
  - Files: `src/climate_machine_learning/presentation.py`, `streamlit_app/app.py`, `scripts/verify_streamlit.py`, `scripts/verify_responsive.py`.
  - Depends on: T6.

### Checkpoint B (after T6-T9)
- [ ] 9 province x preset cases verified against direct `predict_crops`
- [ ] `./run_checks.sh --all` PASS
- [ ] Human review of grouped ranking and advisory

## Phase 4: Crop page

- [ ] **T10: `crop.html`** (module `crop-page`, M)
  - Description: New page, same tokens and tabs as `predict.html`. Reads `crop`, `province` and
    inputs from the query string; validates crop and province against `/api/v1/metadata`; calls
    `/api/v1/recommend` for the scenario and for driest, median, wettest real years. Sections:
    header with back link, result (rank or tie group, predicted, typical, share, unit once),
    same crop across real years (small table, R3/R6), water need (FAO or "not listed"), data
    behind it (training rows, incomplete note, barely-grown flag), inputs used (definition list),
    about these numbers (caveats verbatim). Prev / next in ranking order. Error state: message and
    link back, no numbers. Out-of-range: API refusal message. Cards stacked on phones, two columns
    from 1024 px.
  - Acceptance:
    - Each of 13 crops: page `<h1>` names crop; numbers equal `/recommend` for same inputs.
    - Bad crop or province shows error and no numbers.
    - Generic R4/R6 checks pass at 9 sizes.
  - Verify: `verify_responsive.py --only crop_page`; open `http://127.0.0.1:8000/crop.html?crop=rice&province=Sindh&...` by hand.
  - Files: `frontend/climate-resilient-frontend/crop.html`, `scripts/verify_responsive.py`.
  - Depends on: T2, T6, T7.

- [ ] **T11: Crop page in every check and bundle** (module `crop-page`, S)
  - Description: `verify_frontend.py` PAGES gains `crop.html` (no hardcoded floats, no fallback,
    API-only numbers); `verify_backend.py` serves `/crop.html` 200 and scans it;
    `dryrun_demo.js` runs `crop.html` JS against the API for 13 crops including prev / next
    cycle; `verify_vercel_bundle.py` expects `crop.html`.
  - Acceptance:
    - All four checks pass; Vercel bundle rebuilt locally passes `verify_vercel_bundle.py`.
  - Verify: `scripts/verify_frontend.py`; `scripts/verify_backend.py`; `node scripts/dryrun_demo.js http://127.0.0.1:8000`; `.venv/bin/python scripts/build_vercel_bundle.py && .venv/bin/python scripts/verify_vercel_bundle.py`.
  - Files: `scripts/verify_frontend.py`, `scripts/verify_backend.py`, `scripts/dryrun_demo.js`, `scripts/verify_vercel_bundle.py`.
  - Depends on: T10.

### Checkpoint C (after T10-T11)
- [ ] 13 crop pages open from the ranking; numbers match API; R4/R6 at 9 sizes
- [ ] `./run_checks.sh` PASS

## Phase 5: Text

- [ ] **T12: Text cuts** (module `copy`, M)
  - Description: Apply R7 on `index.html`, `predict.html`, `metrics.html`, `crop.html`: plain
    labels ("Predicting…", "Updated", "Server: online/offline", "Climate inputs", "Optional inputs
    (blank = province median)", footer "CT-378 Machine Learning Project"); one-sentence intro and
    section descriptions on metrics; notes list drops items already shown above; caveats
    untouched. Record visible word count per page at 1280 px before and after; set caps in
    `verify_responsive.py` at the after-count. Banned phrases check.
  - Acceptance:
    - Word counts under caps; banned dev phrases absent; caveat strings present unchanged.
    - `verify_frontend.py`, `verify_backend.py` (banned words), `dryrun_demo.js` pass.
  - Verify: `verify_responsive.py --only copy`; `scripts/verify_frontend.py`; `scripts/verify_backend.py`; `node scripts/dryrun_demo.js http://127.0.0.1:8000`.
  - Files: the 4 HTML pages, `scripts/verify_responsive.py`.
  - Depends on: T7, T8, T10.

- [ ] **T13: Docs, run_checks, close-out** (module `copy`, S)
  - Description: Add `verify_responsive.py` to `run_checks.sh` (and `--streamlit` under
    `--all`); add it to the CLAUDE.md checks table; update `docs/demo_and_viva_guide.md` click path
    (Start button, crop page, grouped ranking) and any README / guide text quoting changed
    wording; `verify_readme.py` / `verify_guide.py` pass. Mark `SPEC-responsive.md` `Status:
    implemented` after checkpoint D; fold outcome into `SPEC.md` front-end section.
  - Acceptance:
    - `./run_checks.sh --all` PASS with `verify_responsive` listed.
    - Docs mention Start button and crop page; no doc quotes removed wording.
  - Verify: `./run_checks.sh --all`.
  - Files: `run_checks.sh`, `CLAUDE.md`, `docs/demo_and_viva_guide.md`, `SPEC-responsive.md`, `SPEC.md`.
  - Depends on: T12.

### Checkpoint D (complete)
- [ ] All 18 success criteria in `SPEC-responsive.md` met
- [ ] `./run_checks.sh --all` PASS
- [ ] Manual check on real Android phone over LAN
- [ ] Human review; redeploy Vercel and Streamlit as a separate step (Decision 3); report
      screenshots recaptured only after redeploy
