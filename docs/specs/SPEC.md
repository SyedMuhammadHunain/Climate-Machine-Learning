# SPEC: Fix crop ranking honesty (Bajra bug) and close completeness gaps

## Objective
Recommendations shown to users must come from the real model, or be clearly marked as a fallback that uses only the 13 dataset crops. Target users: CT-378 graders and demo viewers.

## Findings (audit 2026-11-18)
1. **Bajra #1 is hardcoded fake data.** `predict.html:1031` and `careers.html:1031` (`renderFallbackSimulation`) show Bajra/Jowar/Gram. None exist in the dataset, `ALL_CROPS`, or the model. It appears whenever the backend is unreachable or slower than 3.5s (`predict.html` AbortController). The real API never returns Bajra.
2. **Fallback is silent-ish.** Only a small amber badge says fallback. Numbers are invented formulas (`1750 + precip*6 ...`).
3. **Real ranking is weak.** Ties (dates/grapes/pomegranate all 124.26 in Sindh, 88.23 x3 in Punjab) mean order is arbitrary. Model saturates at province level; crop effect comes mostly from the hand-written `CROP_PHYSIOLOGY` elasticity, not learned.
4. **Yield units look wrong.** Wheat predicts ~40-90 kg/ha; real is about 2,800. Fallback shows 1,750 kg/ha for Bajra. Mixed scales.
5. **Resilient set is hardcoded** (`DROUGHT_RESILIENT_CROPS`) and is also a model feature (`Is_Drought_Resilient`). Ranking only among 5 preselected crops. Onion is flagged resilient; it is not in real agronomy.
6. **Dead duplicate page.** `careers.html` is a copy of `predict.html` (template leftover).
7. Backend 12/12 tests pass, but none test ranking quality or ties.

## Commands
- Backend tests: `.venv/bin/python scripts/verify_backend.py`
- Run: `./run_backend.sh`, `./run_frontend.sh`

## Changes (proposed)
- Remove Bajra/Jowar/Gram. Fallback must use only dataset crops and show a loud banner: "Backend offline - illustrative only".
- Better: drop invented fallback numbers; show an error + retry. (Decision needed.)
- Raise timeout from 3.5s to ~15s (cold model load).
- Add deterministic tie-breaker and a test: no ties in top-3, rank order stable.
- Check/normalise yield units against `Yield` column in the CSV.
- Delete `careers.html` or turn into real page.

## Testing
Add tests: recommend never returns crops outside `ALL_CROPS`; ranking monotone with severe drought; no duplicate yields in top 3.

## Boundaries
- Always: use only real dataset crops in UI output.
- Ask first: retrain model, change resilient-crop list.
- Never: show invented numbers as model output.

## Later changes (delta specs)

This file is the original audit. What happened afterwards lives in the deltas; the current model is described by the last two.

- `SPEC-model-only-output.md`: no fallback, demo or hand-made numbers; ranking by share of a typical year.
- `SPEC-second-source-nasa-power.md`: NASA POWER soil wetness as the second source.
- `SPEC-streamlit-and-huggingface.md`, `SPEC-vercel-deploy.md`: the other front ends and deploy bundles.
- `SPEC-beat-baseline.md` (closed, gate not met): an offset model that did not beat the crop-province mean on log-MAE.
- `SPEC-promote-offset-model.md` (implemented 2026-11-19): that model promoted, judged on test R² and MAE after seeing the test results. It beats the crop-province mean on those two and loses on log-MAE; most of the gain is the year trend. Live on Vercel and Streamlit; the Hugging Face model repo still holds the earlier model.

- `SPEC-responsive.md` (implemented 2026-11-20 in code and automated checks; a real-Android check and the redeploy of Vercel and Streamlit are still open): responsive pages from 320 to 2560 px, one Start button, tied crops shown as one group instead of an alphabetical list led by apple (verified: real model output, not a bug in the ranking), a page per crop (`crop.html`), a structured advisory, readable tables, less text. No retrain, no number changed.
