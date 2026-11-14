# Tasks: Climate-Resilient Low-Water Crop Selection

## Phase 1: Foundation & Data Pipeline

### Task 1: Data Ingestion, Merging & Validation
**Description:** Load the three raw parquet/CSV datasets (`crop_production_and_acreage`, `rainfall_and_drought_indices`, `water_resource_availability`), merge them on `['District', 'Province', 'Crop_Year', 'Season']`, validate schema, shapes, and null counts.

**Acceptance criteria:**
- [x] Loads all 3 raw sources cleanly.
- [x] Produces a unified DataFrame with 11,088 rows and 24 columns.
- [x] Confirms the 5 drought-resilient crops (`Is_Drought_Resilient == True`) and their water demands.

**Verification:**
- [x] Run notebook cell; assert `df.shape == (11088, 24)` and verify null counts. (Passed: executed via nbconvert with zero missing values).

**Dependencies:** None
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 2: Exploratory Data Analysis (EDA) with 5 Rubric Visualizations
**Description:** Build 5 required publication-quality seaborn/matplotlib charts exploring data distributions, correlations, and relationships relevant to drought and crop yield.

**Acceptance criteria:**
- [ ] Plot 1: Yield distribution by crop (`Base_Yield_kg_ha` across 10 crops).
- [ ] Plot 2: Relationship between rainfall deficit (`Rainfall_Deficit_Pct`) and yield.
- [ ] Plot 3: Drought severity distribution (`Drought_Severity_Class` frequency and crop impact).
- [ ] Plot 4: Correlation heatmap across numerical weather/soil/irrigation indices and yield.
- [ ] Plot 5: Water regime comparison (Canal vs. Tubewell vs. Rainfed areas across provinces).

**Verification:**
- [ ] Notebook renders all 5 figures without warnings or clipping.

**Dependencies:** Task 1
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Foundation
- [ ] Clean merged dataset verified.
- [ ] All 5 EDA plots render and reflect key relationships.

---

## Phase 2: Preprocessing & Modeling

### Task 3: Time-Ordered Train/Val/Test Split & Scikit-Learn Preprocessing Pipeline
**Description:** Implement temporal dataset split by year to avoid temporal leakage. Create a scikit-learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` fitted strictly on the training partition.

**Acceptance criteria:**
- [ ] Train set: `Crop_Year <= 2016` (70.8%, 7,854 rows).
- [ ] Val set: `2017 <= Crop_Year <= 2019` (12.5%, 1,386 rows).
- [ ] Test set: `Crop_Year >= 2020` (16.7%, 1,848 rows).
- [ ] ColumnTransformer handles numeric features, categorical features, and prevents leakage.

**Verification:**
- [ ] Assert split row counts and verify that preprocessor transforms train/val/test without error.

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 4: Baseline + 3 Model Training (Dummy, Ridge, Random Forest, HistGradientBoosting)
**Description:** Construct complete scikit-learn pipelines pairing the preprocessor with a DummyRegressor baseline, Ridge Regression, RandomForestRegressor, and HistGradientBoostingRegressor. Train each and record validation metrics.

**Acceptance criteria:**
- [ ] Baseline DummyRegressor trained (predicts training mean).
- [ ] Ridge, Random Forest, and HistGradientBoosting trained on train split.
- [ ] Validation MAE, RMSE, and $R^2$ computed and compared in a summary table.

**Verification:**
- [ ] Models beat the baseline by a clear margin on validation data.

**Dependencies:** Task 3
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 5: 5-Fold Cross-Validation Tuning & Test Set Evaluation
**Description:** Perform 5-fold cross-validated hyperparameter tuning using `GridSearchCV` / `RandomizedSearchCV` on the best model family. Evaluate the optimal model once on the held-out test set (2020–2023).

**Acceptance criteria:**
- [ ] 5-fold CV run on training data for hyperparameter search.
- [ ] Optimal hyperparameters identified.
- [ ] Final evaluation on held-out test set: MAE, RMSE, and $R^2$ logged.
- [ ] Confirmed $R^2 > 0.70$ test success threshold.
- [ ] Residual and error analysis plots generated (predicted vs actual, residuals).

**Verification:**
- [ ] Metrics table printed; residual plot rendered.

**Dependencies:** Task 4
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Model Performance
- [ ] Model beats baseline, achieves $R^2 > 0.70$ on 2020–2023 test set.

---

## Phase 3: Model Export & Low-Water Crop Recommendation Engine

### Task 6: Export Pipeline with `joblib` & Implement Crop Ranking Function
**Description:** Export final tuned pipeline to `models/crop_yield_pipeline.joblib`. Create a high-level Python function `recommend_low_water_crops(...)` that simulates farmer drought inputs, evaluates candidate low-water crops, and returns a sorted recommendation table with a comparison bar chart.

**Acceptance criteria:**
- [ ] `models/` directory created and `joblib.dump()` writes pipeline.
- [ ] Pipeline reloaded with `joblib.load()` and output verified.
- [ ] `recommend_low_water_crops(...)` accepts farmer scenario parameters and returns ranked crops (Bajra, Sorghum, Chickpea, Canola, Barley).
- [ ] Example drought scenario demonstrated with visualization.

**Verification:**
- [ ] File exists at `models/crop_yield_pipeline.joblib`.
- [ ] Recommendation function successfully runs on sample input and prints ranked table.

**Dependencies:** Task 5
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
- `models/crop_yield_pipeline.joblib`
**Estimated scope:** S (1 file)

---

## Checkpoint: Complete Deliverable
- [ ] Complete notebook runs sequentially without errors.
- [ ] Final model exported for teammate's downstream script/app conversion.
