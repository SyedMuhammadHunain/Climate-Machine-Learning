# Delta SPEC: Streamlit app and Hugging Face deployment bundle

Status: built and tested locally; **not deployed** (publishing needs the group's accounts)
Parent: `SPEC.md`, `SPEC-model-only-output.md`, `SPEC-second-source-nasa-power.md`.

## Why
The CT-378 guidelines ask for three deployments. Only the local FastAPI app existed.
- Streamlit: "input form, prediction with confidence/probability, one explanatory chart", on Streamlit Community Cloud.
- Hugging Face: "model card and pipeline on the HF Hub, plus a live Space (Gradio or Streamlit)".

## Decisions (checked against the services' docs on 2026-11-19)
- Streamlit Cloud reads `streamlit_app/requirements.txt` first (entrypoint directory before repo root), default Python 3.12. The app is tested on 3.12 and 3.13 with only that file installed.
- HF Space SDKs are now `gradio`, `docker` or `static`; a Streamlit Space needs Docker. Docker is not installed here, so the Space is a native **Gradio** app that can be run and tested locally.
- Confidence for a regression = a **likely range**: empirical 10% to 90% quantiles of log(actual/predicted) on the validation years (test years stay untouched). Its coverage on the test rows is measured and shown (60% against a nominal 80%, so it is optimistic).
- One shared module, `src/climate_machine_learning/inference.py`, holds prediction and ranking for the API, Streamlit and Gradio, so the three cannot drift.
- Model card has no `license` field: the group has not chosen one. Data licences are stated.

## Out of scope
Actual publishing, tokens and URLs (the group does that, see `docs/deployment_guide.md`); adding the interval to the FastAPI response.

## Boundaries
- Always: show the measured interval coverage; keep numbers generated from `models/metrics.json`; state that the deployments are not live until they are.
- Ask first: choosing a licence, changing pins (the model is a pickle tied to scikit-learn 1.9.1).
- Never: put an access token in the repo or in chat; claim a deployment that was not done.
