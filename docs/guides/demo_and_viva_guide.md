# Demo and viva guide (CT-378)

Group: Muhammad Afzal (CT-24263), Syed Muhammad Hunain (CT-24281), Ayesha (CT-24309).

The marks (10 in total): demonstration 4 (ML lifecycle 2, all three deployments live 2), report 3, individual viva 3. The demo is 10 minutes: 3 min problem and data, 4 min live demo, 3 min results. Then the viva. Every member must be able to answer on any part.

Every number in this guide was recomputed from the running app by `python scripts/verify_guide.py`, and `node scripts/dryrun_demo.js` checks that the page shows the same numbers. If you retrain, run both again and fix what they report.

## Before you present

- [ ] Python 3.13 venv active, `pip install -r requirements.txt` done (see the README).
- [ ] `uvicorn app.main:app --port 8000` running, `http://127.0.0.1:8000/` (landing page, one **Start** button), `/predict.html` and `/metrics.html` open in the browser. The model loads at startup, so the first request is not slow.
- [ ] The live Streamlit and Vercel apps show the pages of the last redeploy: no responsive layout, Start button, tie groups or crop pages yet. Redeploy both (see the [deployment guide](deployment_guide.md)) before showing them, or demo the local app.
- [ ] Notebook `notebooks/01_low_water_crop_selection.ipynb` open with outputs visible (5 or more EDA plots, evaluation plots).
- [x] Streamlit Community Cloud deployment: **live** at <https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/> (tested in a real browser on 2026-11-19: Sindh, Driest real year, rice shows 19.71 and 58.5%, the same as the local app **at that time**. Redeployed with the promoted model on 2026-11-19, and again later that day with the rename, performance and tie-order fixes (`hunain-deploy` at `63616ce`), each time checked in a real browser: the default view (Punjab, typical year, rice) shows 21.10, range 17.34 to 28.23, 100.0%, and the range note says 58%, all equal to the local app. The Sindh driest-year click (40.85 and 89.3%) was checked on the local app and on Vercel, not re-clicked on Streamlit). Open it a few minutes before presenting: free apps may sleep when idle and take 20 to 30 seconds to wake. It deploys from branch `hunain-deploy`, so do not delete or force-push that branch before the demo.
- [x] Vercel deployment of the full app: **live** at <https://climate-machine-learning.vercel.app/> (redeployed with the promoted model on 2026-11-19, and again with the rename, performance and tie-order fixes, `dpl_B7hJbTXGYi7DDomQQomgTquFoHZF`: Sindh, Driest real year, rice shows 40.85 and 89.3%, the same as the local app; `scripts/dryrun_demo.js` passes against it). Open it a few minutes before presenting: the first request after an idle period is slower.
- [x] Hugging Face: the **model repo is live** at <https://huggingface.co/SMHunain/crop-yield-drought-pakistan> with the current model (republished 2026-11-19; a token-less download gives rice in Sindh, driest real year, 40.85 and 89.3%). No Space, by group decision: the model repo plus Streamlit and the FastAPI app are the deployments. Show the model card page in the demo (see the [deployment guide](deployment_guide.md)).
- [x] Report PDF: `docs/report/report.pdf`, 4 pages with the contribution table, accepted by the group on 2026-11-19. `python scripts/verify_report.py` checks that every number equals `models/metrics.json`; rebuild with `python scripts/build_report.py` after any retrain.
- [ ] Dry-run the click path with the server up: `node scripts/dryrun_demo.js http://127.0.0.1:8000` (runs the real page code against the server and checks the numbers on screen equal the API's). Then rehearse once with a timer.

## Part 1: Problem and data (3 min)

Say, in this order:

1. **Problem (45 s).** Less rain means rice and sugarcane cannot be sustained in drought years in Punjab, Sindh and Balochistan. The user is a farmer or extension officer. They enter the year's climate and get every crop ranked by how well the model expects it to hold up.
2. **Data (60 s).** Two public sources merged by province and year. (1) Pakistan crop yield vs climate (MIT licence, origin Zenodo "Climate & Agriculture in Pakistan"): 13 crops, 3 provinces, 2001-2022, 10,296 monthly rows and 19 columns. It does not state the collection method, weather source or yield unit. (2) NASA POWER (MERRA-2 based): root-zone soil wetness at 3 farming-area points per province, accessed 2026-11-19, licence not specified (free to use, citation requested). Say plainly that we checked the proposal's other sources and they did not fit (wheat only or Punjab only, or no Balochistan, or no licence).
3. **The key finding (60 s).** The yield is one yearly number repeated on 12 monthly rows, and the weather is the same for every crop. Training on monthly rows would show the model each yield 12 times. We aggregated to one row per crop, province and year: 773 usable rows (85 rows with Area = 0 dropped as not grown or not reported). We also checked that row order is calendar order.
4. **Target and success metric (15 s).** Target: yield (regression). Metric: MAE, RMSE and R² on held-out test years 2020-2022. Proposal target: R² above 0.7 and a clear margin over a mean-yield baseline.

## Part 2: Live demo (4 min)

Browser on `http://127.0.0.1:8000/predict.html`. Numbers below are what you will see.

| Time | Do | Say / point at |
|---|---|---|
| 0:00 | Open the landing page and click the one **Start** button: it opens the predictor. Province **Sindh**. Click **Driest real year (2002)**. Crop **All Crops (Ranked)**. Click Run. | The form filled itself from a real year in the training data: 14.7 mm of rain and that year's soil wetness from the second source (open "Optional inputs" to show it), nothing typed by us. The hints under the fields show the ranges the model was trained on. |
| 0:40 | Read the headline, then the ranking. | The headline: "POMEGRANATE keeps the largest share of a typical year's yield (99.2%), RICE the smallest (89.3%)." Rank 1 is pomegranate 99.2%. Rank 2 is one group, "97.7% of a typical year · 11 crops tied": apple, banana and nine more, listed by name only, and apple carries "barely grown here" (its typical yield is 0.01). Rank 3 is rice 89.3%. Say it plainly: crops in a group share one rank, so only the two ends of this ranking mean something. "% of a typical year" = the model's yield in this scenario divided by its yield in a typical Sindh year. Both numbers come straight from the model. |
| 1:20 | Click **rice** in the ranking to open its crop page. Point at "Water need". | Rice: FAO lists high drought sensitivity. Dates, grapes and pomegranate are not in the FAO table, so their crop page says so. We never invented a class. The "Across real years" table shows rice in the driest, typical and wettest year, each a model run. |
| 1:40 | Point at "Data behind it" (click grapes in the ranking). | Grapes has fewer than 48 possible training rows, and its crop page says "incomplete data for this crop". Treat its number with care. |
| 2:00 | Back on the predictor, click **Typical (median) year**. Run. | The preset fills all nine inputs with the province medians. The headline says "The model sees no difference between crops for this scenario" and there is one group, "100.0% of a typical year · 13 crops tied". That is expected: a typical year compared with a typical year. The ranking only separates crops when the weather differs from typical, and even then the model moves most crops by the same amount. Say it plainly. |
| 2:40 | Crop **rice**, click **Driest real year (2002)** again. Run. | Predicted 40.85 against a typical 45.76 dataset yield units, which is 89.3%. "Details for rice" opens its crop page, whose "Inputs used" list shows which values were ours and which came from the province median. |
| 3:15 | Type rain **5** (below the trained range). Run. | A 422 error, shown in the alert: "precipitation_sum = 5 is outside the range the model was trained on for Sindh (14.7 to 398.6, annual values 2001-2016). The model cannot extrapolate, so no prediction is made." This is the input validation and error message the guidelines require. |
| 3:40 | Open `/metrics.html`. | Everything on this page is read from `models/metrics.json`. Nothing is typed in. |

Optional, if time allows: stop the server, click Run, show the error with a Retry button and no numbers, restart, click Retry. It shows the app never makes up a result.

If something goes wrong live: `http://127.0.0.1:8000/docs` lets you send the same requests by hand.

## Part 3: Results and lifecycle (3 min)

1. **Lifecycle (60 s).** EDA (5 or more plots in the notebook). Pipeline: imputation, scaling, one-hot encoding, fitted on training data only. Split by year: train 2001-2016, validation 2017-2019, test 2020-2022. Models: three baselines (overall mean, crop-province mean, and crop-province mean plus a year trend with no climate) plus Ridge, Random Forest and Gradient Boosting, each learning the gap from the crop-province average. Tuning: `GridSearchCV` with 5-fold `GroupKFold` grouped by year. Model chosen on validation. Test scored once. Saved with `joblib`, versions pinned.
2. **Results (60 s).** Show the table on `/metrics.html`. The chosen model, Gradient Boosting, scores test R² 0.885 and MAE 29.7. That is above the proposal's 0.7 target and far above the overall-mean baseline. It also beats the stricter crop-province mean baseline (R² 0.832, MAE 34.7) on R² and MAE, but not on log-MAE (0.341 against 0.312), where every crop counts equally. We chose R² and MAE as the test after seeing the test results, so say that yourself before anyone asks. A year trend with no climate scores R² 0.875, so most of the gain is the trend and weather adds little.
3. **Limits and next steps (60 s).** One dataset and 773 yearly rows. Narrow temperature range, so the API refuses values outside it. Year enters only as a trend inside the model and is pinned to 2022 when predicting. The ranking is mostly tied (see the demo), so it separates few crops, and the point number is rough: the predicted yield is pinned to 2022 and was 0.21 times the recent actual for mango in Balochistan and 2.19 times for cotton in Sindh, with the middle half of the 34 crop-province pairs between 0.80 and 1.15 times. The yield unit is not confirmed. Next: add district-level data (for example the Punjab district yields from the proposal), seasonal weather inputs, and deploy the Hugging Face Space.

## Viva preparation

Each member may be asked about any part. Short answers below; know the reasoning, not the wording.

### Data and EDA

- **Where is the data from, and what is its licence?** The GitHub repo Abdullah-Masood-05/pakistan-crop-yield-vs-climate (MIT), which names Zenodo record 15486470 as its origin. It does not state the collection method, the weather source or the yield unit.
- **Why did you aggregate to yearly rows?** Yield is one value per crop-province-year repeated on 12 monthly rows, and weather is province-level. With monthly rows the model sees each yield 12 times, and cross-validation puts copies of the same yield in train and validation folds. That is leakage and it inflates scores.
- **How do you know the months are in order?** There is no month column. `features.check_month_order` checks that the coldest rows are Dec-Feb and the hottest May-Aug in every province, and stops if not. A test confirms it rejects shuffled data.
- **Why were rows dropped?** 85 crop-province-years have Area = 0 and Yield = 0 (apple, grapes, pomegranate). The crop was not grown or not reported, so they are missing data, not real zero yields.
- **What is the yield unit?** Not stated. In the file Yield = 25 x Production / Area exactly, which fits 40 kg units per hectare, but we do not claim it. The app says "dataset yield units".
- **Why not use Area or Production?** Production gives the target away (Yield = 25 x Production / Area). Area only identifies the crop and province.
- **The guidelines want two sources and 5,000 rows. Do you meet that?** Two sources: yes, the crop dataset and NASA POWER soil wetness, merged by province and year (the merge stops if any province-year is missing). Rows: the source has 10,296 monthly rows (above 5,000), but the model trains on 773 yearly rows, and you must say why (yield repeated 12 times). Say what else you would add.
- **Why NASA POWER, and why not the sources in the proposal?** We checked them: the Punjab district wheat yields cover only wheat and only Punjab from about 2007 and state no licence, the Punjab irrigated-area tables cover a few mostly rabi crops in Punjab only, and the weather repository has only Islamabad, Karachi and Lahore with no licence. NASA POWER covers all 3 provinces for 2001-2022 and gives soil moisture, the proposal's third data category.
- **What is the licence of NASA POWER?** The catalog entry says "License not specified". It is free to use and the project asks for citation, which the README does. We do not call it CC0.
- **How did you choose the points, and why not soil_wetness_min?** 3 farming-area points per province, equal weight. That is a judgment, so it is listed as a limit. The driest-month soil wetness is about 0.33 in almost every year, so it carries no signal. We dropped it from the source's own variation, not from model scores.

### Pipeline, leakage and tuning

- **How do you prevent leakage?** Split by year, so the model never trains on a later year than it is tested on. Imputer, scaler and encoder sit inside one scikit-learn `Pipeline` fitted on training data only. The province statistics used for engineered features come from training years only (a test proves test-year data cannot change them). Production is not a feature. The test years are scored once.
- **Why `GroupKFold` grouped by year?** All crops and provinces in one year share the same weather. Ordinary k-fold would put some of them in train and some in validation. Grouping keeps a whole year in one fold.
- **Why fit on the log of the yield?** Crops differ by about 100 times (sugarcane against cotton). On the raw scale the biggest crop would dominate the error. We still report MAE, RMSE and R² on the original scale.
- **Which engineered features, and why?** Rain deficit against the province's training-year mean, water balance (rain minus yearly evapotranspiration) and aridity index. They were in the proposal and express drought directly.
- **What was tuned?** Ridge: alpha. Random Forest: trees, depth, minimum leaf size. Gradient Boosting: trees, learning rate, depth. All with 5-fold `GroupKFold`.
- **Why Gradient Boosting, and why not whichever scores best on test?** The model is chosen on the validation years only. On validation (log-MAE) Gradient Boosting, Ridge and the trend-only baseline are almost tied; Gradient Boosting was slightly best. Choosing by test score would use the test set to pick the model, which is exactly the leakage we avoid.

### Results and the baseline

- **Did you meet the success target?** R² above 0.7: yes, 0.885 on the test years. Clear margin over a mean-yield baseline: yes for the overall mean (R² about zero), and also over the crop-province mean on R² and MAE.
- **Does the model beat a smarter baseline?** Answer straight. Against a baseline that predicts each crop's average yield in each province (test R² 0.832, MAE 34.7): yes on R² 0.885 and MAE 29.7, no on log-MAE (0.341 against 0.312). We judged on R² and MAE, the metrics the proposal names, and we chose that after seeing the test results; we say so on every page that states the win. A year trend alone, with no climate, scores R² 0.875, so the model has not shown that weather adds predictive skill beyond crop, province and trend. The skill column in the table is still below zero for every model.
- **Why did you pick R² and MAE after seeing the test results?** The first plan judged on log-MAE, which the model fails. We then chose the two metrics the proposal names. Say it is a post-hoc choice, point at the log-MAE row, and let the trend-only baseline show where the gain comes from. Do not present it as a pre-registered win.
- **How reliable is the predicted yield number?** Rough. The model is pinned to 2022, six years past the training data, and the year trend is extrapolated. Against the mean actual yield of 2020-2022 the typical-year prediction was between 0.80 and 1.15 times the actual for the middle half of the 34 crop-province pairs, but 0.21 times for mango in Balochistan and 2.19 times for cotton in Sindh. Every page shows this sentence next to the number (`absolute_yield_note`), and the ends of the ranking are more trustworthy than a single point number.
- **Is the ranking useful?** Honest answer: only at the ends. In the demo the best crop is pomegranate at 99.2%, eleven more crops are tied at 97.7% (one group, rank 2) and rice is last at 89.3%. The model moves almost every crop by the same amount when the weather changes, because most of what it knows is the crop and the year trend. The page groups tied crops under one rank ("N crops tied", from `groups` in the API), the Streamlit and Gradio tables have a "Tied with" column, the headline says when a tie is at an end of the ranking or the whole ranking, and the apps say small differences are weak evidence.
- **Why did R² change from 0.892 to 0.885?** The earlier 0.892 came from a model trained on monthly rows with repeated yields and ordinary k-fold, and it ignored weather: changing rain from 0 to 200 mm gave the same answer. The two numbers are not comparable. 0.885 is on unseen test years, from a pipeline without duplicated rows, and part of it comes from the year trend.
- **How confident is a prediction? What is the "likely range"?** The Streamlit and Hugging Face apps show a likely range next to each prediction. It is empirical: the 10% to 90% spread of the model's errors on the validation years (2017-2019), not a model-based probability. Say the check too: on the 2020-2022 test rows it covered 58% of the actual yields, below the nominal 80%, so it is optimistic. Our test years differ from the validation years.
- **What did the second source add?** Little. The soil wetness mean ranks about level with the other weather columns in permutation importance, and the wettest-month value adds nothing. Say it straight: we merged it as the guidelines ask and report that it helps only slightly.
- **What does the model actually rely on?** Permutation importance on the validation years: the crop column by far, weather columns very little. It is on `/metrics.html`.
- **Why rank by "% of a typical year" and not by raw yield?** Raw yields differ by about 100 times between crops, so sugarcane or onion would always win. The ratio compares each crop with itself. Both parts are model predictions, and exact ties are ordered by crop name.
- **Why do some results look odd (a crop doing better in a dry year)?** The model learned weak weather effects, and crops with few training rows (grapes, pomegranate, apple) are noisy. The page shows row counts for this reason. Do not defend a single number. Explain the limit.
- **Why does the API refuse some inputs?** A tree model cannot extrapolate: outside the training range it repeats its edge value. Refusing is more honest than returning that.
- **What is the FAO table for?** Context only (8 of 13 crops, FAO Training Manual 3, Table 14). It is not a model input and does not change the ranking.

### Backend and API

- **Does the site work on a phone and on a big monitor?** Yes. The three pages and the crop page are checked in headless Chrome at nine sizes, from a 320 px phone to a 2560 px monitor, with touch and reduced motion (`scripts/verify_responsive.py`, part of `./run_checks.sh`): no sideways page scroll, tabs always visible, tap targets at least 44 px, form text at least 16 px, tables as cards on phones, and the landing scene survives the Android address bar hiding. The Streamlit app is checked at phone width with `verify_responsive.py --streamlit`. A person should still try it once on a real Android phone.

- **Why FastAPI?** Typed request schemas (pydantic), automatic `/docs`, and the same server also serves the HTML pages.
- **What validation do you have?** Schema checks (types, physical bounds, max not below min or mean, unknown fields rejected) and a model check (inside the trained range for that province). Both return 422 with a message and the field name, and the page shows it.
- **What does `/predict` return?** The raw `pipeline.predict` value, the typical-year value, their ratio, the source of every input (user or province median), the crop's training row count and the model name. A test proves the API value equals a direct model call.
- **How are empty inputs handled?** Filled from the province's training-year median. Max and min temperature follow your mean through the province's median offsets. The response labels each source.
- **What happens if the server is down?** The pages show an error and a Retry button, with no numbers. There is no fallback data anywhere (a test scans `app/` and `frontend/` for that).

### Who answers what

Roles: Syed Muhammad Hunain, model creation and documentation; Muhammad Afzal, frontend and deployments; Ayesha, backend. Each topic below belongs to the member whose role covers it. The viva marks are individual and every member must explain their own part, and be ready for questions on any other part.

| Topic | Owner (by role) | Who explains it first |
|---|---|---|
| Data, EDA, aggregation (`features.py`, notebook) | Syed Muhammad Hunain | Syed Muhammad Hunain |
| Pipeline, tuning, evaluation (`training.py`, `scripts/train.py`) | Syed Muhammad Hunain | Syed Muhammad Hunain |
| Ranking and inference (`inference.py`) | Syed Muhammad Hunain | Syed Muhammad Hunain |
| API and validation (`app/`) | Ayesha | Ayesha |
| Frontend pages (`frontend/`) | Muhammad Afzal | Muhammad Afzal |
| Deployments (Streamlit, Vercel, Hugging Face model repo; `streamlit_app/`, `hf_space/`, bundle scripts) | Muhammad Afzal | Muhammad Afzal |
| README, checks, report (`docs/`, `scripts/verify_*.py`) | Syed Muhammad Hunain | Syed Muhammad Hunain |
