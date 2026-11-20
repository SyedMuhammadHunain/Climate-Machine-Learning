# Climate-Resilient and Low-Water Crop Selection Under Drought

CT-378 Machine Learning semester project. A farmer or extension officer in Punjab, Sindh or Balochistan enters annual climate conditions. The app returns the yield a trained model predicts for each of 13 crops and ranks the crops by how much of a typical year's yield the model expects them to keep in that scenario.

Every number the app shows is model output, a value from the dataset, or a saved evaluation result. The app contains no placeholder results and no hand-written yield formula. If the server cannot be reached, the pages show an error and no numbers.

## Group

| Name | Roll number | Role |
|---|---|---|
| Syed Muhammad Hunain | CT-24281 | Model creation and documentation |
| Muhammad Afzal | CT-24263 | Frontend and deployments |
| Ayesha | CT-24309 | Backend |

### Contributors besides the group

- **Claude** (an AI assistant made by Anthropic, used through Claude Code) wrote and tested parts of the code and docs, among them the responsive layout of the pages, the single Start button, tie groups in the ranking, the crop pages (`crop.html`), `scripts/verify_responsive.py` and the text cuts (`SPEC-responsive.md`). Every commit it took part in ends with a `Co-Authored-By: Claude` line; list them with `git log --grep "Co-Authored-By: Claude"`. The group members are the authors of record and remain responsible for the project.

## Quick start

Needs Python 3.13 (see `.python-version`).

```bash
git clone https://github.com/SyedMuhammadHunain/Climate-Machine-Learning.git
cd Climate-Machine-Learning
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app (the trained model files in `models/` are already in the repo):

```bash
uvicorn app.main:app --port 8000
```

Or use the scripts in the repository root. They create the virtual environment and install the requirements on first run (they use `uv` if it is installed, otherwise `python3.13`):

```bash
./run_app.sh          # API + HTML pages on http://127.0.0.1:8000 (PORT=8080 ./run_app.sh for another port, --reload for development)
./run_streamlit.sh    # the Streamlit app on http://127.0.0.1:8501, in its own .venv-streamlit
./run_checks.sh       # every check that uses .venv, plus the page dry run; --all adds the Streamlit and Hugging Face checks
```

There is no separate frontend server: the pages call the API on their own address, so the backend serves them (`run_frontend.sh` and `run_backend.sh` start the same server).

Open <http://127.0.0.1:8000>. The pages are served by the same server as the API:

| Page | URL |
|---|---|
| Overview (one Start button) | `http://127.0.0.1:8000/` |
| Crop predictor and ranking | `http://127.0.0.1:8000/predict.html` |
| One crop in detail | opened from a crop in the ranking: `http://127.0.0.1:8000/crop.html?crop=rice&province=Sindh&...` |
| Model results | `http://127.0.0.1:8000/metrics.html` |
| API docs (Swagger) | `http://127.0.0.1:8000/docs` |

Re-download the second data source (optional; needs internet; the downloaded files are already in the repo, so training and the app need no network):

```bash
python scripts/fetch_nasa_power.py
```

Retrain (about 1 minute; rewrites `models/crop_yield_pipeline.joblib`, `models/metrics.json`, `models/reference.json`):

```bash
python scripts/train.py
python scripts/verify_readme.py --write   # refresh the Results table below from metrics.json
```

Checks:

```bash
python scripts/verify_model.py      # features, artifacts, sensitivity, deterministic retraining
python scripts/verify_backend.py    # API contract, model equality, no hardcoded numbers
python scripts/verify_frontend.py   # no fake data or demo wording in the pages
python scripts/verify_readme.py     # README matches metrics.json and the real endpoints
python scripts/verify_guide.py      # demo/viva guide numbers match the live app
python scripts/verify_streamlit.py  # Streamlit app: numbers on screen equal the model (needs streamlit_app/requirements.txt, own venv)
python scripts/verify_vercel_bundle.py  # Vercel bundle: what is in it, lean pinned requirements, runs on its own
python scripts/verify_hf_bundle.py  # Hugging Face bundle and Gradio Space (needs hf_space/requirements.txt and pyyaml, own venv)
python scripts/verify_report.py     # report PDF: sections, 3-4 pages, every number equals metrics.json, roles, figures
node scripts/dryrun_demo.js http://127.0.0.1:8000   # with the server running: demo click path, numbers on screen equal the API
```

For the 10-minute presentation see [docs/guides/demo_and_viva_guide.md](docs/guides/demo_and_viva_guide.md).

The notebook `notebooks/01_low_water_crop_selection.ipynb` holds the EDA, the aggregation, and the evaluation plots. It reads `models/metrics.json`, so its numbers match this README.

## Project structure

```
app/                  FastAPI backend (api, schemas, services, core)
frontend/climate-resilient-frontend/   HTML pages served by FastAPI
src/climate_machine_learning/
    features.py       yearly aggregation, merge of both sources, engineered features, inference rows (shared by training and API)
    training.py       split, tuning, evaluation, saving
    inference.py      prediction and ranking from the trained pipeline (shared by the API, Streamlit and Gradio apps)
    presentation.py   tables and wording shown by the Streamlit and Gradio apps (one copy, so both say the same)
streamlit_app/        Streamlit app (app.py) and its own minimal requirements.txt
hf_space/             Gradio app (app.py) and requirements for the Hugging Face Space
data/raw/             crop yield and climate dataset (parquet + csv)
data/raw/nasa_power/  raw NASA POWER API responses (9 points), plus nasa_power_monthly.csv (tidy table)
data/reference/       NASA POWER source facts, FAO crop water needs (cited context)
models/               pipeline, metrics.json, reference.json
notebooks/            EDA and evaluation
scripts/              fetch_nasa_power.py, train.py, build_hf_bundle.py, build_vercel_bundle.py, the verify_*.py checks and dryrun_demo.js
docs/                 demo script and viva preparation, deployment guide
docs/report/          report.pdf (3-4 pages), report.html and figures/; built by scripts/build_report.py, checked by scripts/verify_report.py
SPEC-*.md, tasks/     specs, plan and task list (model-only output; second data source)
```

## Data

Two public sources are merged by province and year.

### Source 1: crop yield and climate

| | |
|---|---|
| Dataset | Pakistan crop yield vs climate, <https://github.com/Abdullah-Masood-05/pakistan-crop-yield-vs-climate> |
| Upstream source named by that repository | "Climate & Agriculture in Pakistan", Zenodo, <https://zenodo.org/records/15486470> |
| Licence | MIT (stated by the repository) |
| Coverage | 13 crops (apple, banana, cotton, dates, grapes, mango, onion, pomegranate, potato, rice, sugarcane, tomato, wheat), 3 provinces, 2001-2022 |
| Size | 10,296 monthly rows, 19 columns (12 months x 22 years x 3 provinces x 13 crops) |
| Added to this repository | 2026-11-14 (date of the commit that added `data/raw/`) |
| Collection method | Not stated by the repository |
| Weather source | Not stated by the repository |
| Units of Yield, Area, Production | Not stated by the repository |
| Personal data | None |

What we checked ourselves:

- The file has no month column. Row order inside each crop-province-year group is calendar order (coldest in Dec-Feb, hottest in May-Aug). `features.py` checks this and stops if it ever fails.
- `Yield` is one yearly number repeated on 12 monthly rows, and the weather is the same for every crop in a province. Training on the monthly rows would show the model each yield 12 times. So the model trains on **one row per crop, province and year** with annual weather: rain and rain hours are summed, the other columns are averaged.
- In the file, `Yield = 25 x Production / Area` exactly. That is consistent with "40 kg units per hectare", but the source does not say so. The app therefore labels yields "dataset yield units".
- 85 crop-province-years have `Area = 0` and `Yield = 0` (apple 22, grapes 41, pomegranate 22). The crop was not grown or not reported, so these rows are dropped as missing data.
- `Production`, `Area` and `Climate_Index` are not model inputs. The first two would leak the target or only identify the crop. `Climate_Index` has no known formula a user could supply.

### Source 2: NASA POWER soil wetness

| | |
|---|---|
| Source | NASA POWER (Prediction Of Worldwide Energy Resources), monthly point API v2.10.1, data from MERRA-2, <https://power.larc.nasa.gov/> |
| What we take | `GWETROOT` root-zone soil wetness (0 to 1) and `PRECTOTCORR` rain, monthly, 2001-2022 |
| Where | 3 farming-area points per province, equal weight: Punjab (Lahore, Faisalabad, Multan), Sindh (Sukkur, Hyderabad, Mirpurkhas), Balochistan (Quetta, Jaffarabad, Turbat). This is a judgment, not a measurement of a whole province. |
| Accessed | 2026-11-19, with `scripts/fetch_nasa_power.py` |
| Licence | **Not specified** in the POWER catalog entry. Free to use. POWER asks users to cite its Reference and Data Reference (see References). We do not call it CC0. |
| Stored in the repo | raw responses in `data/raw/nasa_power/`, tidy table `data/raw/nasa_power_monthly.csv` (SHA-256 `30646a48ac2e35e2...`), source facts in `data/reference/nasa_power_source.json` |

How it is used: two annual model inputs per province-year, `soil_wetness_mean` (mean of the 12 monthly values) and `soil_wetness_max` (wettest month). The driest-month value is about 0.33 in almost every year, so it carries no signal and is not used (decided from the source's own variation, not from model scores). POWER rain is only a cross-check against the dataset's rain (yearly correlation 0.39 to 0.75 depending on the province, see the notebook); it is not a model input.

Other sources listed in the project proposal were checked on 2026-11-19 and not used. The Punjab district wheat yields (Pakistan Bureau of Statistics via opendata.com.pk) cover only wheat and only Punjab, from about 2007, and state no licence. The Punjab irrigated-area tables cover a few mostly rabi crops in Punjab only. The Pakistan weather repository has only Islamabad, Karachi and Lahore, no Balochistan, and states no licence or origin. PMD NDMC soil moisture is a website with no downloadable series that we found. The proposal's links for the two Punjab tables also returned 404 (they have a trailing hyphen).

Crop water needs shown next to the ranking come from FAO Irrigation Water Management Training Manual no. 3 (Brouwer and Heibloem, 1986), Table 14, <https://www.fao.org/4/s2022e/s2022e07.htm>, accessed 2026-11-18. The table lists 8 of the 13 crops. The other 5 show "not listed in the FAO table".

## Method

1. **Aggregate** monthly rows to yearly rows and merge the soil wetness by province and year (773 usable rows). The merge stops with an error if any province-year is missing.
2. **Features**: annual rain, mean/max/min temperature, evapotranspiration, solar radiation, rain hours, soil wetness (mean and wettest month, from NASA POWER), plus three engineered features (rain deficit against the province's training-year mean, water balance, aridity index), and the crop and province.
3. **Pipeline** (scikit-learn): median imputation, standard scaling, one-hot encoding, then a regressor. The regressor does not predict the yield directly (crops differ by about 100 times). It learns the gap, on the log scale, between a year's yield and the crop-province average of the training years, plus a straight-line year trend picked on the validation years (none, one slope per crop, or one per crop and province). Year is never a model input: at prediction it is pinned to the latest dataset year, which cancels out of the crop ranking and only sets the level of the absolute yields. Everything is fitted on training data only.
4. **Split by year**: train 2001-2016, validation 2017-2019, test 2020-2022.
5. **Models**: three baselines (overall mean yield, crop-province mean yield, and crop-province mean plus the year trend with no climate), Ridge, Random Forest, Gradient Boosting.
6. **Tuning**: `GridSearchCV` with 5-fold `GroupKFold` grouped by year, so a year never appears in two folds.
7. **Selection** on the validation years. The test years were scored once and not used to choose anything.
8. **Save** with `joblib`; versions pinned in `requirements.txt`.

## Results

All numbers below are generated from `models/metrics.json` by `scripts/verify_readme.py --write`.

<!-- METRICS:START -->
Train 2001-2016 (565 rows), validation 2017-2019 (106 rows), test 2020-2022 (102 rows). Yields are in dataset yield units.

| Model | Val R² | Val MAE | Test R² | Test MAE | Test RMSE | Test skill vs crop-province mean |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: overall mean yield | -0.002 | 91.7 | -0.030 | 100.4 | 173.6 | -6.96 |
| Baseline: crop-province mean yield | 0.877 | 20.9 | 0.832 | 34.7 | 70.1 | +0.00 |
| Baseline: crop-province mean + year trend (no climate) | 0.919 | 14.8 | 0.875 | 30.3 | 60.5 | -0.43 |
| Ridge (regularised linear regression) | 0.924 | 15.2 | 0.878 | 29.5 | 59.7 | -0.39 |
| Random Forest | 0.917 | 15.2 | 0.873 | 30.6 | 60.9 | -0.54 |
| **Gradient Boosting (chosen)** | 0.924 | 14.9 | 0.885 | 29.7 | 58.0 | -0.38 |

On the 2020-2022 test years, compared with using each crop's average yield per province, the model was better on R² 0.885 against 0.832, MAE 29.7 against 34.7 and RMSE 58.0 against 70.1; it was not better on log-MAE 0.341 against 0.312. The pass test (R² and MAE) was chosen after seeing the test results. A year trend alone, with no climate inputs, scores R² 0.875. Small differences between crops are weak evidence.

Absolute yields are rough: the model is pinned to 2022, and its typical-year prediction, set against the mean actual yield of 2020-2022, was 0.80 to 1.15 times the actual for the middle half of the 34 crop-province pairs, but 0.21 times for mango in Balochistan and 2.19 times for cotton in Sindh. Trust the ends of the ranking more than a single predicted number.

Likely range shown in the Streamlit and Hugging Face apps: nominal 80% (x0.82 to x1.34 of the prediction), from the validation-year errors. It covered 58% of the 2020-2022 test rows, so it is optimistic.

Chosen on the validation years: **Gradient Boosting**. Trained 2026-11-19T09:24:29+00:00 with scikit-learn 1.9.1, data SHA-256 `998ac2e23c39581a...`.
<!-- METRICS:END -->

"Skill" compares the model to the crop-province mean baseline on a log scale. Above 0 means better than that baseline. The model is judged against that baseline on test R² and MAE, the metrics the proposal names. That choice was made after the test results had been seen; on log-MAE, where every crop counts equally, the model is not better, and most of the gain over the plain crop-province mean comes from the year trend (see the trend-only baseline row). The proposal's success target was R² above 0.7 on the test set and a clear margin over a mean-yield baseline.

## How the ranking works

For each crop the API makes two predictions with the trained pipeline: one for your scenario and one for a typical (median) year in the same province. The crop's score is the first divided by the second, as a percentage. Crops are sorted by that score, and exact ties are ordered by crop name. Crops that show the same share at one decimal are marked as tied: `tied_with` in the `/api/v1/recommend` items gives the number of other crops with that share, the HTML page shows "tied with N other crops", the Streamlit and Gradio tables have a "Tied with" column, and the response summary says so. Tied crops are in name order only, so that order means nothing. Nothing is added or multiplied by hand. Inputs you leave empty are filled from the province's training-year medians, and the response labels the source of every input.

The API refuses inputs outside the range seen in training (it returns 422 with the allowed range), because the model cannot extrapolate.

## API

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | server and model status |
| GET | `/api/v1/metadata` | provinces, crops, real driest/wettest training years, trained climate ranges |
| GET | `/api/v1/metrics` | `models/metrics.json` as saved by training |
| POST | `/api/v1/predict` | yield for one crop |
| POST | `/api/v1/recommend` | all 13 crops ranked |

The same routes also work without the `/api/v1` prefix (for example `/health` and `/predict`).

Inputs for `/predict` and `/recommend` (all annual values; unknown fields are rejected):

| Field | Required | Meaning |
|---|---|---|
| `province` | yes | `Punjab`, `Sindh` or `Balochistan` |
| `crop` | `/predict` only | one of the 13 crops |
| `precipitation_sum` | yes | total rain over the year (mm) |
| `temperature_2m_mean` | yes | mean temperature over the year (deg C) |
| `temperature_2m_max`, `temperature_2m_min` | no | mean of the monthly max / min (deg C) |
| `fao_evapotranspiration` | no | mean evapotranspiration (mm/day) |
| `shortwave_radiation` | no | mean solar radiation (MJ/m²) |
| `precipitation_hours` | no | total hours with rain over the year |
| `soil_wetness_mean`, `soil_wetness_max` | no | NASA POWER root-zone soil wetness (0 to 1): mean of the 12 months / wettest month |
| `scenario_name` | no, `/recommend` only | label |

Example (response values rounded; the Sindh 2002 training year had 14.7 mm of rain):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict -H 'Content-Type: application/json' \
  -d '{"province":"Sindh","crop":"rice","precipitation_sum":14.7,"temperature_2m_mean":27.8}'
```

```json
{
  "crop": "rice", "province": "Sindh",
  "predicted_yield": 18.88,
  "yield_unit": "dataset yield units (unit not confirmed)",
  "normal_yield": 33.86,
  "pct_of_normal": 55.8,
  "training_rows_for_crop": 48, "training_rows_possible": 48,
  "inputs": {
    "precipitation_sum": {"value": 14.7, "source": "user"},
    "temperature_2m_max": {"value": 34.2, "source": "user_mean_plus_province_median_offset"},
    "fao_evapotranspiration": {"value": 6.1, "source": "province_median_training_years"}
  },
  "model_name": "Random Forest"
}
```

Out-of-range input:

```json
{"error": "AgronomicValidationError", "parameter": "precipitation_sum", "status_code": 422,
 "message": "precipitation_sum = 5 is outside the range the model was trained on for Sindh (14.7 to 398.6, annual values 2001-2016). The model cannot extrapolate, so no prediction is made."}
```

`/recommend` returns `ranked_crops` (13 items with `rank`, `predicted_yield`, `normal_yield`, `pct_of_normal`, row counts, `barely_grown` and the optional `fao_water_need`), `groups` (crops that show the same share at one decimal share one rank), `headline`, `scenario_facts`, `top_crop`, `ranking_basis`, `model_caveat` and a `recommendation_summary` built from those values.

## Deployment status

The course guidelines ask for three deployments. Status:

| Platform | Status |
|---|---|
| Local web app (FastAPI + HTML, run on localhost) | Working. See Quick start. |
| Streamlit Community Cloud | **Live:** <https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/> (deployed from branch `hunain-deploy`; app code in `streamlit_app/app.py`, own `streamlit_app/requirements.txt`). Input form, prediction with a likely range, one chart. Tested in a real browser on 2026-11-19: it loads for a logged-out visitor in about 20 to 30 seconds, its numbers equal the local model, and the chart draws all 13 crops. Free apps may sleep when idle, so open it before a demo. |
| Vercel (the same FastAPI app and HTML pages, hosted) | **Live:** <https://climate-machine-learning.vercel.app/> (deployed on 2026-11-19 with the Vercel CLI from the generated bundle, `python scripts/build_vercel_bundle.py`). Checked on the live site: the pages load for a logged-out visitor, `/api/v1/recommend` for Sindh in its driest real year gives rice 40.85 and 89.3% of a typical year (the same as the local app; redeployed with the promoted model the same day, after an earlier deployment of the previous model), a rain value outside the trained range is refused with a 422 that states the range, and `scripts/dryrun_demo.js` passes against it. The first request after an idle period can take a few seconds while the model loads. |
| Hugging Face (model card + pipeline) | **Model repo is live:** <https://huggingface.co/SMHunain/crop-yield-drought-pakistan> (model card, pipeline, reference, metrics, inference code). Republished with the current model on 2026-11-19 (Hub commit `056ccbd`); a token-less download is identical to `hf_bundle/model/` and gives rice in Sindh, driest real year, 40.85 and 89.3%. **No Space, by group decision (2026-11-19):** the model repo plus the Streamlit app and the FastAPI app (local and Vercel) are the deployments. A Gradio Space app is built and tested (`hf_space/app.py`, `python scripts/build_hf_bundle.py`), but the Hub refuses Gradio Spaces on a free account; steps to publish it later are in the deployment guide. |

Streamlit and Vercel are deployed (both from clean code; branches `main`, remote `Hunain` and `afzal` contain a malicious loader and must not be merged or run, see the branch table in [docs/guides/deployment_guide.md](docs/guides/deployment_guide.md)). The Hugging Face model repo is live; no Space is deployed, by the group's decision (see the table). Exact steps, what was tested here and what was not: [docs/guides/deployment_guide.md](docs/guides/deployment_guide.md).

The Streamlit and Gradio apps use the same trained model and the same shared code (`src/climate_machine_learning/inference.py`) as the API. Besides the prediction they show a **likely range**: the empirical 10% to 90% spread of the model's errors on the validation years. It is optimistic, because it covered less than its nominal 80% of the test rows (the exact figure is in the Results block above).

## Limitations

- **Few rows.** After aggregating to yearly rows there are 773 usable rows (565 for training), from 22 years and 3 provinces.
- **The second source adds little.** Soil wetness (NASA POWER) is merged as the guidelines ask. In the model's permutation importance the soil wetness mean ranks about level with the other weather columns and the wettest-month value adds nothing (see the Metrics page). It is a public source with unspecified licence, from 3 hand-picked points per province, and its rain agrees only moderately with the dataset's rain.
- **Crop identity dominates.** Most of the model's skill comes from knowing which crop it is. Read the Results table, including the baseline rows and the skill column, before treating small differences between crops as meaningful. Weather changes the predictions only a little for most crops.
- **Narrow ranges.** Annual mean temperature varies by only about 1.5 deg C within a province across the 16 training years, and rain by a few hundred mm. Inputs outside the training range are rejected. The Metrics page lists the ranges.
- **Incomplete crops.** Some crops are missing years in the data (grapes, pomegranate, apple). The API reports how many training rows each crop has against the possible 48.
- **The win is narrower than it looks.** The model beats the crop-province mean on test R² and MAE, but not on log-MAE, and R² and MAE were chosen as the test after seeing the test results. A year trend alone, with no climate inputs, scores almost the same R² (see the trend-only baseline row), so the weather inputs add little beyond it. The year trend is a straight line fitted on 2001-2016 and pinned to the latest dataset year at prediction.
- **Province-level weather and soil.** Weather in the file, and the soil wetness we merged, are the same for every crop in a province. There is no district or farm level data.
- **Yield unit not confirmed.** The source does not state it.
- **FAO table** covers 8 of the 13 crops and is context only. It does not change the ranking.

## References and licences

- Pakistan crop yield vs climate dataset, MIT licence: <https://github.com/Abdullah-Masood-05/pakistan-crop-yield-vs-climate>. Upstream: Climate & Agriculture in Pakistan, Zenodo, <https://zenodo.org/records/15486470>.
- NASA Langley Research Center (LaRC) POWER Project. Data accessed 2026-11-19 through the POWER Monthly and Annual API v2.10.1 (MERRA-2 based): <https://power.larc.nasa.gov/>. Licence not specified in the catalog entry. POWER asks users to cite its Reference and Data Reference: <https://power.larc.nasa.gov/docs/referencing/>.
- Brouwer, C. and Heibloem, M. (1986). *Irrigation Water Management: Irrigation Water Needs*. FAO Training manual no. 3. <https://www.fao.org/4/s2022e/s2022e07.htm>.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825-2830.
