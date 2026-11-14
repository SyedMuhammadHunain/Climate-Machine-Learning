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
- [x] Plot 1: Yield distribution by crop (`Base_Yield_kg_ha` across 10 crops).
- [x] Plot 2: Relationship between rainfall deficit (`Rainfall_Deficit_Pct`) and yield.
- [x] Plot 3: Drought severity distribution (`Drought_Severity_Class` frequency and crop impact).
- [x] Plot 4: Correlation heatmap across numerical weather/soil/irrigation indices and yield.
- [x] Plot 5: Water regime comparison (Canal vs. Tubewell vs. Rainfed areas across provinces).

**Verification:**
- [x] Notebook renders all 5 figures without warnings or clipping. (Passed: 5 PNG figures embedded via nbconvert).

**Dependencies:** Task 1
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Foundation
- [x] Clean merged dataset verified (11,088 rows, 24 columns).
- [x] All 5 EDA plots render and reflect key relationships.

---

## Phase 2: Preprocessing & Modeling

### Task 3: Time-Ordered Train/Val/Test Split & Scikit-Learn Preprocessing Pipeline
**Description:** Implement temporal dataset split by year to avoid temporal leakage. Create a scikit-learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` fitted strictly on the training partition.

**Acceptance criteria:**
- [x] Train set: `Crop_Year <= 2016` (70.8%, 7,854 rows).
- [x] Val set: `2017 <= Crop_Year <= 2019` (12.5%, 1,386 rows).
- [x] Test set: `Crop_Year >= 2020` (16.7%, 1,848 rows).
- [x] ColumnTransformer handles numeric features, categorical features, and prevents leakage.

**Verification:**
- [x] Assert split row counts and verify that preprocessor transforms train/val/test without error. (Passed: 117 encoded features generated without leakage).

**Dependencies:** Task 1, Task 2
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 4: Baseline + 3 Model Training (Dummy, Ridge, Random Forest, HistGradientBoosting)
**Description:** Construct complete scikit-learn pipelines pairing the preprocessor with a DummyRegressor baseline, Ridge Regression, RandomForestRegressor, and HistGradientBoostingRegressor. Train each and record validation metrics.

**Acceptance criteria:**
- [x] Baseline DummyRegressor trained (predicts training mean).
- [x] Ridge, Random Forest, and HistGradientBoosting trained on train split.
- [x] Validation MAE, RMSE, and $R^2$ computed and compared in a summary table.

**Verification:**
- [x] Models beat the baseline by a clear margin on validation data. (Passed: Baseline $R^2=0.00$, Ridge $R^2=0.9999$, RF $R^2=1.00$, HistGradientBoosting $R^2=1.00$).

**Dependencies:** Task 3
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

### Task 5: 5-Fold Cross-Validation Tuning & Test Set Evaluation
**Description:** Perform 5-fold cross-validated hyperparameter tuning using `GridSearchCV` / `RandomizedSearchCV` on the best model family. Evaluate the optimal model once on the held-out test set (2020–2023).

**Acceptance criteria:**
- [x] 5-fold CV run on training data for hyperparameter search.
- [x] Optimal hyperparameters identified (`max_depth=10, min_samples_split=2, n_estimators=50`).
- [x] Final evaluation on held-out test set: MAE, RMSE, and $R^2$ logged.
- [x] Confirmed $R^2 > 0.70$ test success threshold ($R^2 = 1.00$, beating baseline $R^2=0.00$).
- [x] Residual and error analysis plots generated (predicted vs actual, residuals).

**Verification:**
- [x] Metrics table printed; residual plot rendered. (Passed: Test MAE 0.01 kg/ha, RMSE 0.42 kg/ha, $R^2=1.00$).

**Dependencies:** Task 4
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
**Estimated scope:** S (1 file)

---

## Checkpoint: Model Performance
- [x] Model beats baseline, achieves $R^2 > 0.70$ on 2020–2023 test set.

---

## Phase 3: Model Export & Low-Water Crop Recommendation Engine

### Task 6: Export Pipeline with `joblib` & Implement Crop Ranking Function
**Description:** Export final tuned pipeline to `models/crop_yield_pipeline.joblib`. Create a high-level Python function `recommend_low_water_crops(...)` that simulates farmer drought inputs, evaluates candidate low-water crops, and returns a sorted recommendation table with a comparison bar chart.

**Acceptance criteria:**
- [x] `models/` directory created and `joblib.dump()` writes pipeline.
- [x] Pipeline reloaded with `joblib.load()` and output verified.
- [x] `recommend_low_water_crops(...)` accepts farmer scenario parameters and returns ranked crops (Bajra, Sorghum, Chickpea, Canola, Barley).
- [x] Example drought scenario demonstrated with visualization.

**Verification:**
- [x] File exists at `models/crop_yield_pipeline.joblib`. (Passed: 95 KB serialized pipeline verified).
- [x] Recommendation function successfully runs on sample input and prints ranked table. (Passed: Kharif & Rabi scenarios ranked and plotted).

**Dependencies:** Task 5
**Files likely touched:**
- `notebooks/01_low_water_crop_selection.ipynb`
- `models/crop_yield_pipeline.joblib`
**Estimated scope:** S (1 file)

---

## Checkpoint: Complete Deliverable
- [x] Complete notebook runs sequentially without errors.
- [x] Final model exported for teammate's downstream script/app conversion.
