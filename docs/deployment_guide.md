# Deployment guide: Streamlit Community Cloud, Vercel and Hugging Face

The course guidelines ask for three deployments. The local FastAPI app is done, **Streamlit is deployed** (status in section 1) and the same FastAPI app and pages are **deployed on Vercel** (section 4). The Hugging Face **model repo is live**, but the **Space is not deployed**: the Hub needs a PRO plan for a free-CPU Gradio Space (status in section 2). This guide lists exactly what to click and type.

## What was tested here, and what was not

Tested on this machine:

- The Streamlit app (`streamlit_app/app.py`) with Streamlit's test runner, on Python 3.13 and on Python 3.12 (the Community Cloud default), installing **only** `streamlit_app/requirements.txt`. Every number on screen equals the trained model's output. The real server also started and answered on its health endpoint.
- The Gradio Space app (`hf_space/app.py`) through a real Gradio server and client, and the model repo's `inference_example.py` with plain Python.
- The pins in both requirements files resolve on Python 3.12 and 3.13.

**Not tested** (needs the services themselves): the Community Cloud build, the Hub upload, and the Space build. A deployment that fails live scores zero for that platform, so after deploying, open each URL and try it (steps below). Docker is not installed here, so the Hugging Face Space uses the native Gradio SDK (Streamlit Spaces now need Docker).

Before you start, push the branch that contains these files to GitHub. The repository is <https://github.com/SyedMuhammadHunain/Climate-Machine-Learning>.

## Branches: which one deploys what, and which ones must not be used

| Branch | What it is | Use |
|---|---|---|
| `hunain-deploy` | The project code. Streamlit Community Cloud builds from it: repository `SyedMuhammadHunain/Climate-Machine-Learning`, main file `streamlit_app/app.py`, dependencies from `streamlit_app/requirements.txt` (read first, in the main file's folder). | Clean. Work and deploy from here. |
| `vercel-deploy` | An orphan branch holding only the generated Vercel bundle (`python scripts/build_vercel_bundle.py`). The live Vercel site was deployed with the Vercel CLI from the same bundle, not from this branch. | Clean. Needed only for GitHub-driven Vercel deploys. |
| `Hunain` on GitHub, `main`, `afzal` | **Contain a malicious loader** added on 2026-11-15: `frontend/climate-resilient-frontend/init` (hidden obfuscated JavaScript), `src/climate_machine_learning/utils/session.py` (downloads code from an Ethereum transaction and runs it with `node`), an `import utils.session` line in the package `__init__.py`, and a frontend `dev` script that runs `node init`. | **Do not merge, check out or run these.** Do not run `npm run dev` or `node init` from them. `main` has deliberately not been merged yet; the group decides how to clean it. |

`hunain-deploy`, `vercel-deploy` and the local `Hunain` branch were checked and contain none of those files. The Streamlit and Vercel deployments were built from clean code.

## 1. Streamlit Community Cloud

**Status 2026-11-19: deployed and checked (on the earlier model; the promoted model is not deployed yet, see `tasks/todo-promote-offset-model.md` T8b).** Live at <https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/> from branch `hunain-deploy` (the remote `Hunain` and `main` were rewritten and contain a malicious loader, see the branch table above, so they are not used). A real headless Chrome with a clean profile (not logged in to Streamlit) loaded it in 19 to 28 seconds. The default view showed rice in Punjab at 18.55 (range 13.88 to 25.90, 100.1% of a typical year), and after real clicks on Sindh and Driest real year it showed 19.71 (14.75 to 27.51, 58.5%), both equal to the local model of that day (the earlier one; the promoted model shows 40.85 and 89.3% for the same click). The chart drew all 13 crops in the model's order and the page had no exceptions or console errors. If you later merge into `main`, change the app's branch in the Streamlit settings *before* deleting `hunain-deploy`.

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
- Province **Sindh**, **Driest real year**, crop rice: predicted yield 40.85, share of a typical year 89.3%, and the same ranking as the demo guide. (An app still showing 19.71 and 58.5% is running the earlier model: redeploy it. Vercel was redeployed on 2026-11-19 and shows the new numbers; Streamlit is the one still to update.)
- The chart and the table of 13 crops show. The note under the prediction says the likely range covered 58% of the test rows.

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
- Province Sindh, "Driest real year", crop rice gives the same 40.85 and 89.3% as above.
- Typing rain 5 for Sindh shows the "outside the range the model was trained on" message.

The Space config (`hf_bundle/space/README.md`) sets `sdk: gradio`, `sdk_version: 6.32.0` and `python_version: "3.13"`. If the build rejects the Python version, change it in `space_readme()` in `scripts/build_hf_bundle.py` (3.12 also resolves), rebuild and upload again.

**Decide a licence.** The model card has no `license` field on purpose: the group has not chosen one. Add `license: <id>` to the card's front matter in `build_hf_bundle.py` when you decide. The data licences are stated in the card (crop dataset MIT; NASA POWER "License not specified", citation requested).

## 3. When the Hugging Face Space goes live (Streamlit is already done)

- Edit the Hugging Face row of the README deployment table with the Space URL and the date.
- `scripts/verify_readme.py` requires the Vercel row to show the live URL and not say "not deployed", the Streamlit row to show the live URL and not say "not deployed", and the Hugging Face row to link the model repo and say the Space is "not deployed yet". Change that second rule at the same time, or it will fail. `scripts/verify_guide.py` likewise keeps the Hugging Face checklist item open until you tick it.
- Tick the Hugging Face item in the "Before you present" list of `docs/demo_and_viva_guide.md`.
- Rehearse the live demo with the three URLs open.

## 4. Vercel (FastAPI app and HTML pages)

**Status 2026-11-19: deployed and checked.** Live at <https://climate-machine-learning.vercel.app/>. It is the same app as the local one: the pages are served by FastAPI and call the API on the same address.

What is deployed: the folder that `python scripts/build_vercel_bundle.py` writes (git-ignored, default `vercel_bundle/`). It holds `app/`, three modules of `src/climate_machine_learning/` (not `training.py` and not `presentation.py`), the three files in `models/`, the FAO table, the pages, a `requirements.txt` with 21 exactly pinned packages (taken from the tested root `requirements.txt`, no Jupyter, plotting, `pyarrow` or `uvicorn`) and `.python-version` (3.12). `python scripts/verify_vercel_bundle.py` checks all of this and starts the bundled app on its own.

To deploy again (after a retrain or a code change), with the Vercel CLI logged in (`vercel login`):

```bash
python scripts/build_vercel_bundle.py --out ~/vercel_bundle
rm ~/vercel_bundle/.built-by-build_vercel_bundle
cd ~/vercel_bundle && vercel deploy --prod --yes --name climate-machine-learning --scope hunains-projects-ff0a52fc
```

What was checked on the live site: `/`, `/predict.html`, `/metrics.html`, `/health` and `/api/v1/metrics` answer 200 without a login, `/package.json` answers 404, Sindh in its driest real year gives rice 40.85 and 89.3% of a typical year (the promoted model; the first deployment earlier the same day gave 19.71 and 58.5% with the earlier model, and was redeployed with the same CLI command on 2026-11-19, deployment `dpl_A6MBeZRggSgAE5y5JExybSf3MzaE`), rain of 5 mm is refused with a 422 that states the trained range, `node scripts/dryrun_demo.js https://climate-machine-learning.vercel.app/` passes, and a headless Chrome screenshot of `/predict.html` shows the form filled with the trained ranges from the live API.

Things to know:
- The project has Vercel Authentication switched on, but the production address above is open to everyone. The per-deployment addresses (with a random suffix) probably ask for a Vercel login.
- The first request after the app has been idle can take a few seconds while the model loads (about 4 seconds on the build machine). Open the site before a demo.
- The installed packages are about 264 MB. That was accepted by the Hobby plan on 2026-11-19; if scikit-learn or pandas are upgraded the size may change, so re-check the deploy.
- Branch `vercel-deploy` on GitHub holds the same bundle (an orphan branch), but the deployment above was made with the CLI, not from GitHub. It is only needed if you later connect the repository to Vercel for automatic deploys. A first attempt through the Vercel connection failed because of a GitHub rate limit, not because of the app.

## Re-checking the local builds

```bash
pip install -r streamlit_app/requirements.txt && python scripts/verify_streamlit.py
pip install -r hf_space/requirements.txt pyyaml && python scripts/verify_hf_bundle.py
```

Use a separate virtual environment for these two (they need Streamlit and Gradio, which the main app does not).
