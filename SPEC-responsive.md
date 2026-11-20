# Spec Delta: Responsive pages, clearer ranking and crop detail pages

Status: approved (2026-11-19, recommended answers to every open question)

## Objective

Every page a visitor can open must be usable and look intended on any screen: small Android
phones (320 px wide) up to 4K desktop monitors (2560 px wide and above), in portrait and
landscape, with touch or mouse. Today the HTML pages were built and checked at desktop width
only (report screenshots are 1280 x 1220). The viva demo may be shown on a phone, and the live
Vercel and Streamlit links are shared with people who open them on Android.

User stories:

- A visitor on a 360 px Android phone opens the landing page, sees the 3D scene and both buttons,
  and can scroll the intro with a finger.
- The same visitor opens the predictor, fills every input, presses Predict and reads the ranked
  crop list without zooming or scrolling sideways.
- A reviewer on a tablet or a 27" monitor opens Model & Benchmarks and reads every table; the
  page does not become a narrow strip in the middle of a huge screen, nor stretch lines past
  comfortable reading length.

- A visitor on the landing page sees one clear **Start** button and lands on the predictor.
- On every page and every screen, the three tabs (Overview, Crop Predictor & Advisory, Model &
  Benchmarks) are fully visible and tappable.
- Every table on Model & Benchmarks reads cleanly on a phone and on a large monitor.

Layout change plus one requested copy change (the landing buttons, item R1). No model, API or
metric change; no other wording change.

## Requested changes (from the user, 2026-11-19)

- **R1. One Start button.** Replace the two landing buttons "Crop Advisory & Predictor" and
  "Model & Benchmarks" with one button labelled "Start" that opens the predictor page.
- **R2. Tabs always visible.** All tabs perfectly visible on small screens, big screens and
  Android.
- **R3. Tables laid out perfectly** on every screen size.
- **R4. Fully responsive** on Android and on smaller and bigger screens (the rest of this spec).
- **R5. "Apple always on top".** On the predictor, apple is first after almost every preset
  click. Verify it is real model output, not fake or demo data, find whether it is a bug, and make
  the page say plainly what is going on.

- **R6. Tables readable.** Table structure and every value in it readable and visible on every
  screen (adds readability rules to the R3 layout rules).
- **R7. Less text.** Remove unnecessary, repeated or overwhelming text from the pages.

- **R8. Short result rows.** Under each crop in the ranking show only its rank and its yield.
  Everything else moves off the row (to the crop page, R9).
- **R9. One page per crop.** Clicking any of the 13 crops opens a detailed page for that crop.
- **R10. Structured ranking description.** The crop ranking description (advisory card) shown in
  a structured layout, not a paragraph.
- R8 to R10 must also meet R4 (Android, small and large screens).

## Build order (capability map)

The requests now span separate pieces that can ship and be checked on their own:

| Module id | Covers | Depends on |
|---|---|---|
| `layout` | R2, R3, R4, R6: responsive CSS, tabs, tables, `verify_responsive.py` | none |
| `landing` | R1: Start button, landing page fit | `layout` |
| `ranking` | R5, R8, R10: tie groups, short rows, structured description (shared wording in `src/`) | `layout` |
| `crop-page` | R9: new `crop.html` | `layout`, `ranking` |
| `copy` | R7: text cuts on all pages and Streamlit | `ranking`, `crop-page` |

Build order: `layout`, then `landing` and `ranking`, then `crop-page`, then `copy`. Each module
ends with all checks green and its own commit. One plan file covers all five
(`tasks/plan-responsive.md`).

## R5 findings (verified 2026-11-19)

Checked with the saved model, through the real API (`/api/v1/recommend` via FastAPI
`TestClient`) and directly through `inference.predict_crops` with `inference.single_thread`:

- **Real model output.** API and direct `predict_crops` agree on every crop and share for
  Sindh driest year (0 mismatches, 13 rows). `predict.html` renders only the live API response
  (`isLiveApi` guard; no fallback, demo or random data in the page).
- **Why apple is first: ties sorted by name.** The model learns each crop's level from crop and
  province, but its climate effect is close to the same multiplier for most crops (permutation
  importance: `Crop` 1.148, `Province` 0.369, best climate input `rain_deficit_pct` 0.010). So
  most crops get the same share of a typical year at one decimal, and
  `inference.rank_key` breaks exact ties by crop name (`TIE_DIGITS = 1`). Apple is first
  alphabetically.

| Province | Preset | Top of list (share of typical year) | Crops tied with apple |
|---|---|---|---|
| Punjab | median | apple 100.0% | 12 others |
| Punjab | driest | apple 99.8% | 12 others |
| Punjab | wettest | apple 98.0% | 11 others |
| Sindh | median | apple 100.0% | 12 others |
| Sindh | driest | pomegranate 99.2%, then apple 97.7% | 10 others |
| Sindh | wettest | cotton 107.2%, then apple 99.3% | 10 others |
| Balochistan | median | apple 99.6% | 12 others |
| Balochistan | driest | pomegranate 102.1%, banana 101.2%, apple 100.5% | 9 others |
| Balochistan | wettest | apple 84.9% | 11 others |

  Sindh driest year has only 3 different shares across 13 crops (89.3, 97.7, 99.2).
- **Not a code bug in the ranking**: the order follows the documented rule, and each tied item
  already says "tied with N other crops (same share, ordered by name only)". It is a
  **presentation problem**: a numbered list (Rank #1, #2, ...) implies apple beats crops it is
  tied with.
- **Two smaller findings:**
  1. Apple in Sindh: predicted yield 0.01 and typical 0.01 dataset yield units (apple is barely
     grown there), yet it sits at the top of the list on share.
  2. "Typical (median) year" preset fills the inputs rounded to 1 decimal (`toFixed(1)`), so
     Balochistan median shows 99.6% instead of 100.0% (temperature 18.73 sent as 18.7).

### R5 new behavior (presentation only, no retrain)

- Tied crops shown as **one group with one shared rank**, e.g. "Rank 2: 10 crops at 97.7% of a
  typical year: apple, banana, ...", not as separate Rank #2 to #11 lines. Crops that differ are
  ranked above or below the group as today.
- When every crop is tied, the advisory says the model sees no difference between crops for this
  scenario, instead of naming apple as best. Text built from model output in
  `src/climate_machine_learning/` (shared with Streamlit, which shows the same ranking), no
  hardcoded numbers.
- Crops whose typical predicted yield is negligible in the province (rule and cut-off decided in
  Open Question 7, computed from saved data only) are marked "barely grown in this province" and
  never lead the advisory text.
- Median preset sends the exact province median (no rounding) so the typical year scores 100.0%.
- Same grouping in Streamlit (`streamlit_app/app.py` through `presentation.py`) and in the API
  response field used for the summary, so every front end says the same thing.

Not in this spec: changing the model so climate affects crops differently (crop x climate
interactions). That needs a retrain and a new model choice, so it is its own delta spec
(`SPEC-crop-climate-interactions.md`) if wanted.

## Current Behavior

Front ends (see `CLAUDE.md`, "One shared package, four front ends"):

| Front end | Files | Today |
|---|---|---|
| Landing page | `frontend/climate-resilient-frontend/index.html` | Viewport meta present. Breakpoints at 800/700/600 px. Full-screen WebGL canvas, Lenis smooth scroll, 30000 px scroll space (20000 px under 800 px). Global `* { pointer-events: none; }`. Renderer size from `window.innerWidth/innerHeight`, which jumps when the Android address bar hides or shows. |
| Predictor | `frontend/climate-resilient-frontend/predict.html` | 720 px column. Input grid goes to 1 column under 600 px. Nav pills in one non-wrapping row (3 pills, longest "Crop Predictor & Advisory"). Inputs 40 px tall, 14 px font: Android Chrome does not zoom on focus at 14 px, but iOS Safari zooms under 16 px. Buttons 42 px, pills about 30 px tall (under 44 px touch target). |
| Metrics | `frontend/climate-resilient-frontend/metrics.html` | 720 px column. Tables in `overflow-x: auto` wrapper, headers `nowrap`. Stat grid 4 columns, 2 under 650 px. |
| Streamlit | `streamlit_app/app.py` | `layout="wide"`. Streamlit stacks `st.columns` under 640 px by itself. Dataframes scroll sideways inside their frame. |
| Vercel bundle | generated by `scripts/build_vercel_bundle.py` from the same three HTML pages | Inherits whatever the HTML pages do. |
| Gradio | `hf_space/app.py` | Not deployed (no Space, group decision). |

No check today renders any page at a phone width.

## New Behavior

### Target screens

Checked widths (CSS px) and the device each stands for:

| Width x height | Stands for |
|---|---|
| 320 x 568 | smallest phones still in use |
| 360 x 800 | most common Android phone |
| 412 x 915 | large Android phone (Pixel, Galaxy S Ultra) |
| 800 x 360 | Android phone in landscape |
| 768 x 1024 | tablet portrait |
| 1024 x 768 | tablet landscape, small laptop |
| 1440 x 900 | laptop |
| 1920 x 1080 | desktop |
| 2560 x 1440 | large desktop |

Phone widths also run with device pixel ratio 2.625 or 3 and touch emulation, so `ontouchstart`
code paths and high-density canvas sizing are exercised.

### R1. Landing page: one Start button

- `index.html` shows exactly one button, text "Start", where the two buttons are today, same
  `careers-button` style and fade-in animation.
- It links to `predict.html` with a **relative** href, so it opens the predictor on whatever host
  serves the pages: `http://localhost:3000/predict.html` when served on port 3000,
  `http://127.0.0.1:8000/predict.html` under uvicorn, and the Vercel URL in production.
  No hardcoded host or port.
- Model & Benchmarks stays reachable from the tabs on the predictor page (no page removed).
- Tap target at least 44 x 44 px; visible and tappable at every target size, including
  800 x 360 landscape.

Note on port 3000: `package.json` `dev` serves the folder with `python3 -m http.server 3000`,
a static server with no `/api/v1/*`, so predictions fail there. `CLAUDE.md` also forbids
`npm run dev` (malicious `package.json` on remote branches). The supported local URL stays
`uvicorn app.main:app --port 8000`; the relative link works on both.

### R2. Tabs (`.site-nav` / `.nav-pill` on `predict.html` and `metrics.html`)

- All three tabs fully visible at every target width: no clipped text, no tab pushed off-screen,
  no sideways page scroll caused by the nav.
- 320 to 479 px: tabs wrap to rows or stack full-width, each label on one line or wrapped inside
  its pill, never cut with an ellipsis.
- 480 px and up: one centred row.
- Large screens: one row, sized with the fluid type scale (not tiny on 2560 px).
- Each tab at least 44 px tall at phone widths; active tab still clearly marked.
- Same markup and class names on both pages so they behave identically.

### R3. Tables (`metrics.html`, Streamlit dataframes)

Tables on `metrics.html`:

| # | Table | Columns |
|---|---|---|
| 1 | Model comparison | 8 (1 text, 7 numeric) |
| 2 | What the chosen model relies on | 3 (text, numeric, bar) |
| 3 | Data sources | 5 (text, includes SHA-256 prefix) |
| 4 | Data and split | 2 |
| 5 | Trained climate and soil ranges | 8 (1 text, 7 numeric "median (min to max)") |

Rules:

- **Phone widths (under 600 px), wide tables 1, 3, 5:** each row becomes a card: the row's first
  cell (model, role, province) as card title, then one "header: value" line per other column,
  header labels taken from the table's own `<th>` (via `data-label` set in the render JS, no
  copied text). No sideways scroll needed to read any value.
- **Phone widths, narrow tables 2, 4:** stay real tables, fit the width with no sideways scroll;
  bar column in table 2 shrinks but stays visible.
- **600 px and up:** real tables. If a table is still wider than its column (tablet, table 1 or
  5), it scrolls inside its wrapper with the first column sticky and an edge shadow showing more
  content; the page itself never scrolls sideways.
- **Large screens:** tables use the wider column (up to about 1100 px for the metrics page at
  1920 px and wider) so tables 1 and 5 fit with no inner scroll at 1440 px and up.
- Numeric columns right-aligned with tabular figures, headers aligned with their column, no
  header wraps mid-word, no value wraps mid-number, consistent row padding, zebra or divider
  lines kept.
- Every value shown exactly as today (same numbers, same rounding); only presentation changes.
- Streamlit: dataframes `width="stretch"`, scroll inside their own frame on phones, page does not
  scroll sideways.

### R6. Table readability (`metrics.html`; Streamlit dataframes where Streamlit allows)

Found today: table text 13 px, section text 13.5 px, metric detail 11.5 px; grey text
`--colorNeutral600` `#7d8699` on white is about 3.7:1 contrast, under the WCAG AA 4.5:1 minimum
for normal text; header labels such as "CV log-MAE" and "Test skill vs crop mean" have no
explanation next to the table; the importance table has an unlabelled bar column.

- Table body text at least 14 px, headers at least 13 px semibold, on every screen.
- All table and label text at least 4.5:1 contrast (darken the grey tokens; same colour family).
- Header row sticky when the table scrolls inside its wrapper; first column sticky (tablet).
- Row dividers or zebra stripes kept; row padding at least 10 px; hover highlight kept for mouse
  only.
- Chosen model row visibly highlighted (bold or tinted), not only "(chosen)" in the text.
- Jargon headers get a short plain-word legend under the table (one line each, built from the
  existing labels, no new numbers), e.g. "MAE: average error in dataset yield units".
- Bar column in the importance table gets a header ("Relative weight") and the bar keeps its
  numeric value beside it.
- SHA-256 column: shown as the existing 16-character prefix in monospace, never wraps the row;
  on phones it moves to the card's last line.
- Every value fully visible: no ellipsis, no clipped cell, no value overlapping another.
- Same rules apply to the cards from R3 on phones.

### R7. Less text, same facts

Found today:

- Predictor advisory card is one block of about 210 words: `recommendation_summary` (120 words:
  top and bottom crop plus the whole model caveat), `ranking_basis` (28), `absolute_yield_note`
  (64).
- Each of the 13 result lines repeats "% of a typical year", "Predicted yield ... (typical ...)"
  and the full unit "dataset yield units (unit not confirmed)".
- Developer-style wording visible to visitors: "Executing machine learning pipeline
  inference...", "Inference completed • Updated just now", "Backend Pipeline: checking...",
  "Climate Intelligence Engine", "Core Agricultural & Climate Parameters:".
- Metrics page: long intro ("Nothing on this page is typed in by hand."), long section
  descriptions (comparison 3 sentences, importance 2, ranges 2), notes list repeats the caveat
  shown in the summary cells.

New behavior:

- **One idea per line, said once.** The unit "dataset yield units (unit not confirmed)" shown
  once in the results heading, not on every line. Result lines show crop, share, predicted yield
  and typical yield only.
- **Advisory card split**: a short headline (top crop or tied group and bottom crop, one
  sentence), then "About these numbers" holding the ranking basis, the model caveat and the
  absolute-yield note.
- **Plain labels**: "Predicting…", "Updated", "Server: online/offline", "Climate inputs",
  "Optional inputs (blank = province median)"; footer just "CT-378 Machine Learning Project".
- **Metrics page**: one-sentence intro, one-sentence section descriptions, legend under tables
  (R6) replaces long description text; notes list keeps only notes not already shown above.
- **Never removed or reworded** (CLAUDE.md rules and existing checks): the generated
  `inference.model_caveat` text (win on R² and MAE, loss on log-MAE, test chosen after seeing
  results, trend-only R² 0.875, likely-range coverage), `inference.absolute_yield_caveat`,
  "dataset yield units" label, the out-of-range refusal messages, data sources and licences. They
  may move, be grouped, or sit behind a disclosure only if the decision in Open Question 8 allows
  it; checks `verify_streamlit.py`, `verify_hf_bundle.py`, `verify_report.py`,
  `verify_backend.py` (summary names top crop and its share) must keep passing.
- Text cuts apply to the HTML pages and Streamlit alike, using the shared wording in
  `presentation.py` / `inference.py` so both front ends say the same thing.
- README, report and demo guide quotes of page text updated where they quote changed wording.

### R8. Result rows: rank and yield only (`predict.html`, all-crops ranking)

Today each row has two metadata lines: rank, share of typical year, predicted and typical yield,
unit, tie note, FAO water need, drought sensitivity, training rows, incomplete-data note.

- Each row: crop name (link to its crop page, R9), then one line "Rank N", "X% of typical" and
  "Yield Y" (predicted yield in the scenario, 2 decimals). Nothing else on the row (Decision 10:
  share kept because the ranking is by share, not yield; apple in Sindh has yield 0.01).
- Tied crops (R5): one rank and one share shown on the group heading; rows inside the group show
  only crop name and yield.
- Unit "dataset yield units (unit not confirmed)" shown once in the results heading (R7).
- Typical yield, tie note, FAO water need, training rows move to the crop page (R9).
- Rows laid out as a two-column list (name left, rank and yield right, numbers right-aligned,
  tabular figures) from 480 px up; stacked under 480 px. Whole row is the tap target (at least
  44 px tall).
- Single-crop mode (one crop selected in the form): same short card, plus a "Details" link to the
  crop page.

### R9. Crop detail page (`frontend/climate-resilient-frontend/crop.html`)

- **One page, one URL per crop**:
  `crop.html?crop=<name>&province=<province>&<input>=<value>...`. The 13 crops each get their
  own address; the scenario inputs travel in the query string so the page is shareable and the
  browser back button returns to the ranking. Relative links only (works on port 3000, 8000,
  Vercel). Unknown crop or province in the URL: plain error message with a link back to the
  predictor, no numbers.
- **Data only from the existing API** (no new endpoint, no typed numbers): one `POST
  /api/v1/recommend` with the same inputs (gives this crop's rank, tie group, predicted and
  typical yield, share, FAO water need, training rows) and `GET /api/v1/metadata` (crop list,
  province real driest and wettest years). Out-of-range inputs show the API's refusal message.
- **Sections** (each a card, stacked on phones, two columns from 1024 px):
  1. Header: crop name, province, back link "All crops", tabs (R2).
  2. Result: rank (or "tied with N crops at X%"), predicted yield, typical-year yield, share of
     typical year, unit once.
  3. Same crop across real years: predicted yield and share for the province's driest real year,
     typical (median) year and wettest real year, from three `/api/v1/recommend` calls with
     those inputs (model output). Small table, R3/R6 rules.
  4. Water need: FAO water need range and drought sensitivity with its source link, or "not
     listed in the FAO table".
  5. Data behind it: training rows for this crop of possible rows, incomplete-data note,
     "barely grown in this province" flag (R5).
  6. Inputs used: each input with value and source (entered by you / province median), as a
     definition list.
  7. About these numbers: `absolute_yield_note` and `inference.model_caveat` text (R7 rules,
     unchanged wording).
- Prev / next crop links in ranking order at the bottom, so all 13 crops can be browsed without
  returning to the list.
- Same look as `predict.html` (fonts, colours, tabs, R4/R6 rules); page title and `<h1>` name the
  crop.
- Served by the existing `StaticFiles` mount and copied into the Vercel bundle
  (`scripts/build_vercel_bundle.py` file list, `verify_vercel_bundle.py` expects it).
- `scripts/verify_frontend.py` and the `app/` scan (`verify_backend.py` TEST 22) cover
  `crop.html`: no hardcoded floats, no fallback data, every number read from the API.

### R10. Structured ranking description (`predict.html` advisory card)

Today: one paragraph (`<strong>scenario:</strong> recommendation_summary`, then `ranking_basis`
and model name, then `absolute_yield_note`), about 210 words.

New layout, top to bottom:

1. **Headline** (one sentence, larger text): best crop or tied group and lowest crop, with their
   share of a typical year.
2. **Scenario facts** as a label-value grid: Province, Annual rain, Mean temperature, Optional
   inputs used (count entered vs province median), Model name. Two columns from 480 px, one
   column under.
3. **How crops are ranked**: `ranking_basis` text, plus tie rule in one line.
4. **About these numbers**: `inference.model_caveat` text and `absolute_yield_note`, each its own
   paragraph, wording unchanged (R7, Open Question 8 decides visible vs disclosure).

- Headline and facts built from fields the API already returns (`top_crop`, `ranked_crops`,
  `inputs`, `province`, `model_name`); if a field is needed that the API does not return as a
  separate value, add it to `RecommendationResponse` from `inference.py` (same text for
  Streamlit through `presentation.py`), never compose numbers in page JS beyond formatting.
- `recommendation_summary` stays in the API response unchanged (checks and other clients rely on
  it); the page shows its parts in structure instead of the one paragraph.
- Streamlit shows the same four parts (headline, facts, ranking rule, caveats); there the pinned-year note stays under the
  predicted yield and the model caveat sits under the ranking, one copy of each (decided in T12).

### All HTML pages (`index.html`, `predict.html`, `metrics.html`, `crop.html`)

1. **No sideways page scroll** at any target width: `document.documentElement.scrollWidth <=
   window.innerWidth`. Wide tables may scroll inside their own wrapper only, and the wrapper shows
   that it scrolls (edge shadow or visible scrollbar).
2. **Fluid type and spacing**: font sizes, paddings and the column width use `clamp()` so text is
   never under 14 px on a phone and the main column grows on large screens (720 px today; up to
   about 960 px at 1920 px and wider for the metrics tables, prose lines kept at or under about
   80 characters).
3. **Tabs**: see R2.
4. **Touch targets**: every link, pill, button, select and input at least 44 x 44 px hit area at
   phone widths (WCAG 2.5.5 target size).
5. **No zoom on focus**: form controls use 16 px text at phone widths.
6. **Predictor**: inputs one column on phones, two from tablet up; the action row (Predict,
   Reset) stacks full-width under 400 px; result items, advisory card and caveat text wrap with no
   clipped number.
7. **Metrics**: stat grid 1 column under 400 px, 2 to tablet, 4 from 1024 px; tables per R3.
8. **Landing page**:
   - canvas and overlay sized from the visual viewport (`100dvh` / `visualViewport`), so hiding
     the Android address bar does not stretch or jump the scene;
   - renderer resize debounced and pixel ratio capped (2 on phones) so mid-range Android phones
     keep a smooth frame rate;
   - header, footer, paragraph and buttons fit at 320 px and in 800 x 360 landscape (no overlap
     between header, paragraph and footer);
   - finger scrolling drives the intro (Lenis touch path works, `scroll-space` height scaled so
     the intro is not a very long swipe on phones);
   - the Start button (R1) reachable and tappable at every width.
9. **Respect `prefers-reduced-motion`**: landing page skips smooth scroll and long animations,
   shows the paragraph and buttons directly.
10. Large screens (1920, 2560): content centred, no element stretched across the full width
    unless it is meant to be full-bleed (the landing canvas).

### Streamlit (`streamlit_app/app.py`)

11. At 360 px: every input reachable, results table readable (scrolls inside its frame), no
    sideways page scroll, metric tiles stack. Only Streamlit's own layout options and a small
    injected CSS block if needed; no new dependency.

### Not changed

- Every number on every page stays as it is (model output, dataset values, saved results only).
  Wording changes only where R1, R5 and R7 say; `inference.model_caveat` and
  `inference.absolute_yield_caveat` text unchanged.
- Models and metrics untouched (no retrain). `app/` and `src/climate_machine_learning/` change
  only for R5 grouping and wording; existing API fields keep their meaning (grouping added, not
  replacing `ranked_crops`).
- Hugging Face model repo untouched.

## Commands

```bash
uvicorn app.main:app --port 8000                     # serve pages locally
.venv/bin/python scripts/verify_responsive.py         # new: renders every page at every target size
./run_checks.sh                                       # gains verify_responsive.py
node scripts/dryrun_demo.js http://127.0.0.1:8000     # must still pass
.venv/bin/python scripts/build_vercel_bundle.py       # rebuild bundle; verify_vercel_bundle.py must pass
```

## Testing Strategy

New plain-assert script `scripts/verify_responsive.py`, same style as the other checks (PASS/FAIL
per test, `TESTS` list, non-zero exit on failure). It reuses the Chrome DevTools client from
`scripts/capture_screenshots.py` (moved to a shared helper, e.g. `scripts/chrome_cdp.py`, so both
scripts use one copy), starts its own local uvicorn, and for each page x target size:

- sets size, DPR and touch with `Emulation.setDeviceMetricsOverride` /
  `Emulation.setTouchEmulationEnabled`;
- asserts no sideways page scroll;
- asserts no visible text element extends past the viewport edge, and no two of header,
  paragraph, buttons, footer overlap on the landing page;
- asserts interactive elements are at least 44 x 44 px at phone widths and form controls at least
  16 px font;
- `index.html`: exactly one `a.careers-button`, text "Start", `href="predict.html"` (relative),
  inside the viewport and at least 44 x 44 px; following it lands on `/predict.html` (status 200);
- `predict.html` and `metrics.html`: all three `.nav-pill` tabs have their full bounding box
  inside the viewport, none overlap, text not clipped (`scrollWidth <= clientWidth`), each at
  least 44 px tall at phone widths;
- `metrics.html` (after data loads from `/api/v1/metrics`): under 600 px, tables 1, 3, 5 render as
  cards with every cell having a non-empty `data-label` equal to its column `<th>` text; tables 2
  and 4 and all tables from 600 px fit or scroll only inside `.data-table-wrapper`; at 1440 px and
  up tables 1 and 5 have no inner scroll; numeric cells right-aligned; no number wrapped across
  lines; the set of cell texts equals the set at 1280 px (nothing lost or changed);
- R5 (new test in `scripts/verify_model.py`, and API test in `scripts/verify_backend.py`): for
  all 9 province x preset cases, API ranking equals direct `predict_crops` (crop order and share);
  tied crops come back as one group with one rank; an all-tied scenario never names a single best
  crop; the median preset gives 100.0% for every crop; `dryrun_demo.js` checks the page renders
  groups, not per-crop ranks inside a tie;
- R6: computed font size of every `td` at least 14 px, `th` at least 13 px; contrast of every
  table, label and description text colour against its background at least 4.5:1 (computed in
  the check from `getComputedStyle`); no cell with `scrollWidth > clientWidth` (clipped); sticky
  header present when wrapper scrolls;
- R7: visible word count per page recorded at 1280 px and asserted under an agreed cap (set in
  the plan from today's count minus the cuts); unit string appears once in the results; banned
  developer phrases ("Executing machine learning pipeline", "Backend Pipeline",
  "Climate Intelligence Engine") absent; `inference.model_caveat` and
  `inference.absolute_yield_caveat` text still present in the page DOM after a prediction;
- R8: after a prediction each result row's visible text is only crop name, "Rank N" (or group
  rank), share and "Yield X" (plus barely-grown tag); no FAO, training-row or share text on rows; unit appears once;
- R9: for each of the 13 crops, clicking its row opens `crop.html?crop=<crop>&...` (status 200),
  the page shows that crop's name in `<h1>`, and its predicted yield, typical yield, share and
  rank equal the `/api/v1/recommend` response for the same inputs; real-years table equals three
  `/recommend` calls; bad crop in URL shows error and no numbers; prev / next links cycle all 13;
  all R4/R6 checks pass on `crop.html` at 9 sizes; `dryrun_demo.js` extended to run
  `crop.html` JS against the API;
- R10: advisory card has headline, facts grid, ranking rule and caveat blocks; facts values
  equal the request inputs; caveat text equals `inference.model_caveat` and
  `inference.absolute_yield_caveat`; grid one column under 480 px, two from 480 px;
- on `predict.html` at 360 px: runs a prediction (driest real Sindh year preset) and asserts the
  ranked list renders with every crop visible inside the viewport width;
- writes screenshots to `$CLAUDE_JOB_DIR/tmp` or a git-ignored folder for human review (not
  committed, not used in the report).

Streamlit check: `verify_responsive.py --streamlit` renders the local Streamlit app at 360 x 800
and asserts no sideways page scroll (needs `.venv-streamlit`; part of `run_checks.sh --all`).

Existing checks must all still pass: `verify_frontend.py`, `verify_backend.py` (banned words and
hardcoded floats scan), `dryrun_demo.js`, `verify_vercel_bundle.py`, `verify_streamlit.py`.

Manual: open the local site on a real Android phone over LAN
(`uvicorn app.main:app --host 0.0.0.0 --port 8000`) once before sign-off.

## Boundaries

- Always: CSS and small layout JS inside the three HTML pages; keep every existing `id`, class
  and element the page JS and `dryrun_demo.js` rely on; run every check before each commit.
- Ask first: adding any library or framework (Tailwind, Bootstrap, etc.); changing visible copy
  beyond R1, R5 and R7; removing or rewording any caveat listed under R7 "Never removed";
  changing the landing 3D scene itself (model, shaders); redeploying Vercel or Streamlit; pushing.
- Never: touch remote `main`, `Hunain`-remote or `afzal` code from the malicious commits; run
  `npm run dev` or `node init`; add fallback or demo numbers to make a layout look full; commit
  screenshots from the check.

## Success Criteria

1. `scripts/verify_responsive.py` passes for all 3 HTML pages x 9 target sizes.
2. Zero sideways page scroll at every target size (only table wrappers scroll).
3. All interactive elements at least 44 x 44 px and form text at least 16 px at phone widths.
4. Predictor completes a prediction at 320 px and 360 px and the full ranked list is readable.
5. Landing page: scene, paragraph and both buttons visible, non-overlapping, at 320 x 568 and
   800 x 360; address-bar show/hide does not resize the canvas visibly (resize uses visual
   viewport, debounced).
6. Streamlit app at 360 px: no sideways page scroll.
7. All existing checks still pass; no number on any page changed; wording changed only per R1,
   R5 and R7.
8. R1: landing page has one "Start" button; clicking it opens `predict.html` on the same host
   (port 3000, port 8000 and Vercel alike).
9. R2: all three tabs fully visible, unclipped, non-overlapping and tappable at all 9 sizes on
   `predict.html` and `metrics.html`.
10. R3: all 5 metrics tables pass the table checks at all 9 sizes; cell values identical to
    1280 px.
11. `docs/demo_and_viva_guide.md` and `scripts/run_app.sh` text still correct after R1 (metrics
    page reached through tabs); docs updated if they mention the landing buttons.
12. R5: page and Streamlit show tied crops as one group; no scenario shows apple (or any crop)
    as sole best when it is tied; Sindh apple flagged as barely grown; median preset 100.0%;
    verified model output equals what the page shows for all 9 cases.
13. R6: all table text at least 14 px body / 13 px header, at least 4.5:1 contrast, no clipped
    or overlapping cell at any of the 9 sizes; jargon headers explained in a legend.
14. R7: advisory headline one sentence; unit shown once; developer wording gone; visible word
    count per page under the agreed cap; every required caveat still present and unchanged.
15. R8: result rows show only crop, rank, share of typical year and yield at all 9 sizes.
16. R9: all 13 crops open their own `crop.html` page; every number on it matches the API; page
    passes R4/R6 at all 9 sizes; included in the Vercel bundle.
17. R10: ranking description shown as headline, facts, ranking rule, caveats; readable at all 9
    sizes; caveat text unchanged.
18. Vercel bundle rebuilt from the changed pages passes `verify_vercel_bundle.py`.

## Out of Scope (Deferred)

- Gradio app (`hf_space/app.py`): not deployed, no Space by group decision.
- Dark mode: not asked; current pages are light only.
- PWA / installable app, offline mode: separate feature.
- Native Android app: request is about the web pages.
- Redesign of look or copy: layout only.
- Recapturing report screenshots at phone size: report screenshots stay 1280 px; can be a later
  delta if the group wants a mobile figure.
- Redeploying Vercel and Streamlit: done after approval of the implemented change, as its own step.

## Decisions (approved 2026-11-19)

The user approved the spec with the recommended answer to every open question. Where a question
had no stated recommendation, the conservative option below was taken.

| # | Question | Decision |
|---|---|---|
| 1 | Lighter landing scene on phones | Yes, only on phone widths (under 800 px or touch): pixel ratio capped at 1.5, fewer wave segments. Desktop scene unchanged. |
| 2 | Column width on large screens | Widen: predictor and crop page up to 960 px, metrics up to 1100 px, at 1920 px and wider (fluid between). |
| 3 | Redeploy Vercel and Streamlit | Separate step after review, not part of this change. |
| 4 | Wide tables on phones | Cards (tables 1, 3, 5) under 600 px. |
| 5 | Port 3000 | uvicorn on 8000 stays the supported local URL; Start link relative. |
| 6 | R5 scope | Presentation only, no retrain. No `SPEC-crop-climate-interactions.md` now. |
| 7 | "Barely grown" rule | Typical predicted yield under 1% of the largest typical predicted yield in that province, from the same request's median-year predictions. |
| 8 | Caveat display | Full caveat always visible under the headline. No disclosure. |
| 9 | Plain labels | Approved as listed in R7. |
| 10 | Result row content | Rank, share of typical year and yield (share added, it is the reason for the rank). |
| 11 | Actual yearly yields on crop page | Not included. No new endpoint. |
| 12 | Crop pages in Streamlit | HTML site only. Streamlit gets R5, R7, R10 only. |
