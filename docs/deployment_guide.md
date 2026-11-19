# Deployment guide: Streamlit Community Cloud and Hugging Face

The course guidelines ask for three deployments. The local FastAPI app is done and **Streamlit is deployed** (status in section 1). The Hugging Face **model repo is live**, but the **Space is not deployed**: the Hub needs a PRO plan for a free-CPU Gradio Space (status in section 2). This guide lists exactly what to click and type.

## What was tested here, and what was not

Tested on this machine:

- The Streamlit app (`streamlit_app/app.py`) with Streamlit's test runner, on Python 3.13 and on Python 3.12 (the Community Cloud default), installing **only** `streamlit_app/requirements.txt`. Every number on screen equals the trained model's output. The real server also started and answered on its health endpoint.
- The Gradio Space app (`hf_space/app.py`) through a real Gradio server and client, and the model repo's `inference_example.py` with plain Python.
- The pins in both requirements files resolve on Python 3.12 and 3.13.

**Not tested** (needs the services themselves): the Community Cloud build, the Hub upload, and the Space build. A deployment that fails live scores zero for that platform, so after deploying, open each URL and try it (steps below). Docker is not installed here, so the Hugging Face Space uses the native Gradio SDK (Streamlit Spaces now need Docker).

Before you start, push the branch that contains these files to GitHub. The repository is <https://github.com/SyedMuhammadHunain/Climate-Machine-Learning>.

## 1. Streamlit Community Cloud

**Status 2026-11-19: deployed and checked.** Live at <https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/> from branch `hunain-deploy` (the plain `Hunain` branch could not be pushed because the remote `Hunain` and `main` had been force-rewritten). A real headless Chrome with a clean profile (not logged in to Streamlit) loaded it in 19 to 28 seconds. The default view showed rice in Punjab at 18.55 (range 13.88 to 25.90, 100.1% of a typical year), and after real clicks on Sindh and Driest real year it showed 19.71 (14.75 to 27.51, 58.5%), both equal to the local model. The chart drew all 13 crops in the model's order and the page had no exceptions or console errors. If you later merge into `main`, change the app's branch in the Streamlit settings *before* deleting `hunain-deploy`.

From Streamlit's deploy docs:

1. Go to <https://share.streamlit.io> and sign in with GitHub. Click **Create app**, then **Yup, I have an app**.
2. Repository: `SyedMuhammadHunain/Climate-Machine-Learning`. Branch: the branch you pushed (merge to `main` first if you prefer). **Main file path: `streamlit_app/app.py`**.
3. Optional: choose the app URL (subdomain on `streamlit.app`).
4. **Advanced settings**: Python version. The default is 3.12, which works (tested). 3.13 also works.
5. Click **Save** and watch the logs.

Dependencies: Community Cloud looks in the directory of the main file first, then the repo root, and uses the first dependency file it finds. So `streamlit_app/requirements.txt` (5 packages) is used, not the big root `requirements.txt` (Jupyter and more). Do not add another dependency file next to the app.

If the install fails, read the log. Do **not** change the `scikit-learn==1.9.1` pin: the model file is a pickle made with that version.

Check the live app (each of these should work):

- It loads with Punjab, "Typical (median) year", crop rice, no error.
- Province **Sindh**, **Driest real year**, crop rice: predicted yield 19.71, share of a typical year 58.5%, and the same ranking as the demo guide.
- The chart and the table of 13 crops show. The note under the prediction says the likely range covered 60% of the test rows.

Then put the URL in the README deployment table.

## 2. Hugging Face (model repo + Space)

**Status 2026-11-19:** the model repo is **live** at <https://huggingface.co/SMHunain/crop-yield-drought-pakistan> (uploaded and checked with a token-less download). Creating the Space was **refused**: the Hub says hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription (the account is on the free plan; static Spaces are free). The Space bundle is ready, so the steps below for the Space work as soon as that is settled.

You need a Hugging Face account and an access token with **write** permission (Settings, Access Tokens). Keep the token secret: do not paste it into chat, commit it, or put it in a file in the repo.

```bash
pip install huggingface_hub
hf auth login                                   # paste the token when asked
python scripts/build_hf_bundle.py               # writes hf_bundle/model and hf_bundle/space (git-ignored)
```

The builder only replaces `model/` and `space/` inside its output folder, and refuses a folder that is not empty and was not made by it (it never deletes anything else). If you built a bundle with an older version of the script, delete that `hf_bundle` folder once, or pass a new folder with `--out`.

Model repo (model card, pipeline, reference, metrics, inference code):

```bash
hf repos create <your-username>/crop-yield-drought-pakistan
hf upload <your-username>/crop-yield-drought-pakistan hf_bundle/model . --repo-type model
```

Space (the live app, native Gradio SDK):

```bash
hf repos create <your-username>/crop-yield-drought --repo-type space --sdk gradio
hf upload <your-username>/crop-yield-drought hf_bundle/space . --repo-type space
```

(`hf repos create` and `hf upload <repo_id> <local_folder> <path_in_repo> --repo-type ...` are the forms in the Hugging Face CLI docs; run `hf upload --help` if a flag is rejected.) The model file is 20 MB, so upload with `hf upload` rather than a plain `git push`, which would need Git LFS.

The Space builds for a few minutes. Open its **Logs** tab if it fails. Then check it:

- It loads, the form fills itself, and **Predict** shows a prediction, a likely range, a chart and a table.
- Province Sindh, "Driest real year", crop rice gives the same 19.71 and 58.5% as above.
- Typing rain 5 for Sindh shows the "outside the range the model was trained on" message.

The Space config (`hf_bundle/space/README.md`) sets `sdk: gradio`, `sdk_version: 6.32.0` and `python_version: "3.13"`. If the build rejects the Python version, change it in `space_readme()` in `scripts/build_hf_bundle.py` (3.12 also resolves), rebuild and upload again.

**Decide a licence.** The model card has no `license` field on purpose: the group has not chosen one. Add `license: <id>` to the card's front matter in `build_hf_bundle.py` when you decide. The data licences are stated in the card (crop dataset MIT; NASA POWER "License not specified", citation requested).

## 3. When the Hugging Face Space goes live (Streamlit is already done)

- Edit the Hugging Face row of the README deployment table with the Space URL and the date.
- `scripts/verify_readme.py` requires the Streamlit row to show the live URL and not say "not deployed", and the Hugging Face row to link the model repo and say the Space is "not deployed yet". Change that second rule at the same time, or it will fail. `scripts/verify_guide.py` likewise keeps the Hugging Face checklist item open until you tick it.
- Tick the Hugging Face item in the "Before you present" list of `docs/demo_and_viva_guide.md`.
- Rehearse the live demo with the three URLs open.

## Re-checking the local builds

```bash
pip install -r streamlit_app/requirements.txt && python scripts/verify_streamlit.py
pip install -r hf_space/requirements.txt pyyaml && python scripts/verify_hf_bundle.py
```

Use a separate virtual environment for these two (they need Streamlit and Gradio, which the main app does not).
