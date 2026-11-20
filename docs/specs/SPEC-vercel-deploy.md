# Delta SPEC: Deploy the FastAPI app and HTML pages on Vercel

Status: approved 2026-11-19 (whole app on Vercel; deployed from here through the connected Vercel account)
Parents: `SPEC.md`, `SPEC-model-only-output.md`, `SPEC-second-source-nasa-power.md`, `SPEC-streamlit-and-huggingface.md`.

## Question asked
"I want to deploy the frontend on Vercel. Is it possible?" (not Hugging Face)

## Finding: possible, but not "frontend only"
- The pages (`index`, `predict`, `metrics`) call `/api/v1/*` on their own origin. Hosted alone on Vercel they would show "Cannot reach the model server" and never a prediction.
- Vercel supports FastAPI natively as a Python Function: an `app` variable in `app/main.py` is a valid entrypoint (Vercel docs, checked 2026-11-19). So the backend and the pages can be one Vercel project on the free Hobby plan. The account is connected and is on Hobby.

## Measured on this machine (lean install: fastapi, pydantic, scikit-learn, pandas, numpy, joblib; Python 3.12)
| | |
|---|---|
| Installed packages | 262 MB (428 MB with `pyarrow`, which the API does not need at run time) |
| App payload | models 20 MB, frontend 2 MB, code 2 MB |
| Total uncompressed | about 286 MB |
| Cold start (import app, load the 20 MB model) | 1.3 s warm cache, 3.3 s first run |
| Peak memory | about 205 MB |
| Backend test suite in that lean environment | passes (all tests) |

Vercel's own docs disagree on the size limit: the "limitations" page quotes 250 MB uncompressed, the Python runtime page says 500 MB. 286 MB fits the second and not the first. The deploy itself will tell; if it fails at 250 MB, trim with `excludeFiles` (tests, unused scipy parts).

## Proposed design (not yet built)
- A generated deploy folder, like the Hugging Face bundle: `python scripts/build_vercel_bundle.py` writes `vercel_bundle/` (git-ignored) with `app/`, `src/climate_machine_learning/`, `models/`, `data/reference/fao_crop_water_needs.json`, `frontend/climate-resilient-frontend/`, a **lean** `requirements.txt` (same pins as the root file, no Jupyter, no `pyarrow`), and a `vercel.json`.
- Why a separate folder: the root `requirements.txt` is the full development environment (123 packages) and would blow the bundle limit. The bundle builder refuses to delete anything that is not its own (same rule as the Hugging Face builder).
- The pages are served by the FastAPI app itself (existing `StaticFiles` mount); no change to application code.
- Python version pinned to one the model was tested on (3.12 and 3.13 both pass).

## Acceptance criteria
- Public `*.vercel.app` URL: `/`, `/predict.html`, `/metrics.html` load; `/health` returns 200.
- `/api/v1/predict` and `/api/v1/recommend` return exactly the numbers the local app returns (Sindh, driest real year, rice: 19.71, 58.5%).
- A real-browser run of the demo click path against the live URL passes, as it did for Streamlit.
- No secrets in the bundle or the repo. Local test suites still pass.

## Boundaries
- Always: lean requirements only; verify the live URL in a real browser; state deployment protection and limits as found.
- Ask first: deploying under the group's Vercel account (public URL), choosing the deploy method, any CORS change.
- Never: put tokens in files or chat; claim the app is live before the live URL was tested.

## Decisions (2026-11-19)
1. Scope: whole app, API plus pages (chosen).
2. Method: deploy from here through the connected account (chosen).
3. Found while planning: the connection's file upload sends files as base64 inside a tool call, which is not practical for the 20 MB model file (about 27 MB of text). So the bundle is committed to its own Git branch (`vercel-deploy`, bundle at the repo root, nothing else) and Vercel deploys that branch through its GitHub connection. If the GitHub connection is not available, fall back to the Vercel CLI run by the group.

## Open decisions (resolved above)
1. Scope: whole app (API plus pages), recommended; or pages only (predictions would not work).
2. Method: deploy from here through the already connected Vercel account, or prepare the bundle and let the group deploy with the Vercel CLI or dashboard.

## Risks
| Risk | Mitigation |
|---|---|
| Bundle 286 MB against a doc-conflicted 250/500 MB limit | Measure on deploy; trim with `excludeFiles` |
| Deployment Protection may put a login in front of the URL (as Streamlit's handshake looked) | Check the production URL logged out; adjust the project setting if needed |
| Cold start of a few seconds on first request | Documented; warm the URL before a demo |
| Pickle tied to scikit-learn 1.9.1 | Pin it in the lean requirements |
