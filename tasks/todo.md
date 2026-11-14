# Tasks: Real-World Climate & Crop Yield Machine Learning Pipeline

## Phase 1: Real-World Ingestion & Exploratory Analysis

### Task 1: Real-World Data Ingestion, Schema Cleanup & Validation
**Description:** Download `merged_agri_climate_data.csv` directly from `Abdullah-Masood-05/pakistan-crop-yield-vs-climate`, delete previous synthetic data files in `data/raw/`, store the real dataset in CSV/Parquet format, and inspect schema, data distributions, and missing values.

**Acceptance criteria:**
- [x] Remove old synthetic mock data in `data/raw/`.
- [x] Save genuine 10,296-row dataset to `data/raw/pakistan_crop_yield_climate.csv` and parquet.
- [x] Verify 0 missing values across all 19 columns and confirm 13 distinct crops across 2001–2022.

**Verification:**
- [x] Run notebook ingestion cells; assert `df.shape == (10296, 19)` and `df.isnull().sum().sum() == 0`. (Passed: 10,296 rows, 19 columns, 0 nulls across all features).

**Dependencies:** None
**Files likely touched:**
- `data/raw/`
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (2 files)

---

### Task 2: Exploratory Data Analysis (EDA) with 5 Rubric Visualizations
**Description:** Generate 5 publication-quality seaborn/matplotlib charts exploring real-world relationships between crop yield, drought stress, and climatic factors across Punjab, Sindh, and Balochistan.

**Acceptance criteria:**
- [ ] Plot 1: Yield distribution across all 13 crops (boxplot / violinplot showing real yield variability).
- [ ] Plot 2: Relationship between precipitation / drought stress (`precipitation_sum`, `Climate_Index`) and crop yield.
- [ ] Plot 3: Temperature and evapotranspiration profiles across provinces.
- [ ] Plot 4: Correlation heatmap of climate metrics and crop yield.
- [ ] Plot 5: Temporal yield trend over time (2001–2022) illustrating real historical fluctuations.

**Verification:**
- [ ] Notebook renders all 5 figures cleanly without warnings or clipping.

**Dependencies:** Task 1
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Foundation
- [ ] Real dataset verified on disk and in notebook (10,296 rows, 19 columns).
- [ ] All 5 EDA plots render and reflect genuine historical agricultural dynamics.

---

## Phase 2: Leakage-Free Preprocessing & Model Evaluation

### Task 3: Time-Ordered Train/Val/Test Split & Scikit-Learn Preprocessing Pipeline
**Description:** Split data temporally by `Year` to prevent temporal leakage. Drop `Production` to avoid mathematical target leakage ($\text{Yield} = \text{Production} / \text{Area}$). Build a scikit-learn `ColumnTransformer` with `StandardScaler` for continuous climate features and `OneHotEncoder` for `Crop` and `Province`.

**Acceptance criteria:**
- [ ] Train set: `Year <= 2016` (72.7%, 7,488 rows).
- [ ] Validation set: `2017 <= Year <= 2019` (13.6%, 1,404 rows).
- [ ] Test set: `Year >= 2020` (13.6%, 1,404 rows).
- [ ] `Production` column excluded from feature matrix $X$.
- [ ] Preprocessor pipeline fits strictly on training split.

**Verification:**
- [ ] Assert split row counts and verify transformed feature dimensions on train, val, and test partitions.

**Dependencies:** Task 2
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 4: Baseline + 3 Model Training (Dummy, Ridge, Random Forest, HistGradientBoosting)
**Description:** Construct unified pipelines combining the preprocessor with a `DummyRegressor` baseline, `Ridge`, `RandomForestRegressor`, and `HistGradientBoostingRegressor`. Train on the training partition and log validation metrics (MAE, RMSE, $R^2$).

**Acceptance criteria:**
- [ ] Baseline DummyRegressor trained (predicts train mean).
- [ ] Ridge, Random Forest, and HistGradientBoosting trained on train split.
- [ ] Validation performance table and comparison barplot generated.
- [ ] Non-linear models demonstrate clear improvement over baseline with realistic $R^2$.

**Verification:**
- [ ] Validation summary table printed; ML models beat baseline by a wide margin.

**Dependencies:** Task 3
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 5: 5-Fold Cross-Validation Tuning & Held-Out Test Evaluation
**Description:** Perform 5-fold cross-validated hyperparameter search using `GridSearchCV` on the top-performing model family. Evaluate the optimal model once on the held-out test set (2020–2022). Generate residual plots.

**Acceptance criteria:**
- [ ] 5-fold CV run on training partition to tune tree hyperparameters (e.g. `max_depth`, `min_samples_split`, `n_estimators`).
- [ ] Best hyperparameters logged alongside CV score.
- [ ] Held-out test set evaluation: final MAE, RMSE, and $R^2$ logged.
- [ ] Predicted vs. Actual and residual distribution plots rendered.

**Verification:**
- [ ] Test metrics table printed; residual plots verified.

**Dependencies:** Task 4
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Model Performance
- [ ] Tuned model demonstrates robust generalization on held-out test data without artificial 1.00 overfitting.

---

## Phase 3: Export & Drought Recommendation Engine

### Task 6: Export Pipeline with `joblib` & Implement Crop Ranking Function
**Description:** Export final tuned pipeline to `models/crop_yield_pipeline.joblib`. Build an interactive ranking function `recommend_low_water_crops(...)` that accepts user drought conditions, predicts expected yield for candidate crops, and highlights drought-resilient crops (dates, grapes, wheat) over water-demanding crops (sugarcane, rice, banana).

**Acceptance criteria:**
- [ ] Pipeline serialized via `joblib.dump()` to `models/crop_yield_pipeline.joblib`.
- [ ] Pipeline reloaded and predictions verified with `np.testing.assert_allclose`.
- [ ] `recommend_low_water_crops(...)` simulates drought scenario and prints ranked crops with expected yields.
- [ ] Comparison visualization rendered showing recommended vs. non-recommended crops.

**Verification:**
- [ ] Export file exists and loads cleanly; recommendation function produces valid ranked dataframe and plot.

**Dependencies:** Task 5
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
- `models/crop_yield_pipeline.joblib`
**Estimated scope:** S (2 files)

---

## Checkpoint: Complete Deliverable
- [ ] Notebook runs cleanly end-to-end via `nbconvert`.
- [ ] Git commit and push to branch `Hunain`.
