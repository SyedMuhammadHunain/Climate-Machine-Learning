# Implementation Plan: Real-World Climate & Crop Yield ML Pipeline

## Overview
Rebuild the end-to-end Machine Learning Jupyter Notebook (`notebooks/01_low_water_crop_selection.ipynb`) using the genuine, real-world provincial agroclimatology dataset from Pakistan (2001–2022) referenced in the CT-378 project proposal (`Abdullah-Masood-05/pakistan-crop-yield-vs-climate`). This replaces the synthetic zero-variance mock data with 10,296 historical observations across 13 crops in Punjab, Sindh, and Balochistan. The model predicts crop yield from climate and agro-environmental features under real historical variance, and ranks drought-tolerant crops when simulated drought conditions occur.

## Architecture Decisions
- **Dataset Source:** Direct ingestion of the real 10,296-row `merged_agri_climate_data.csv` covering 13 crops across 22 years (2001–2022) and 3 provinces.
- **Preventing Target Leakage:** `Production` is strictly excluded from features because $\text{Yield} = \text{Production} / \text{Area}$. Features consist of `Crop`, `Province`, `Area`, and the 12 meteorological / agroclimatological indices (`temperature_2m_mean`, `precipitation_sum`, `fao_evapotranspiration`, `Climate_Index`, etc.).
- **Time-Ordered Partition:** Split strictly by year to prevent temporal leakage:
  - **Train:** 2001–2016 (~72.7%, 7,488 rows)
  - **Validation:** 2017–2019 (~13.6%, 1,404 rows)
  - **Test (Held-Out):** 2020–2022 (~13.6%, 1,404 rows)
- **Pipeline-Centric Preprocessing:** Scikit-learn `ColumnTransformer` with `StandardScaler` for numeric climate variables and `OneHotEncoder(handle_unknown="ignore")` for categorical features (`Crop`, `Province`), fitted strictly on the training partition.
- **Algorithms Evaluated:** `DummyRegressor` (mean baseline), `Ridge`, `RandomForestRegressor`, and `HistGradientBoostingRegressor`.
- **5-Fold Cross-Validation Tuning:** Hyperparameter optimization via `GridSearchCV` with 5-fold cross-validation on the training set.
- **Model Persistence:** Serialized export to `models/crop_yield_pipeline.joblib`.
- **Drought Recommendation Engine:** User inputs environmental / drought scenario (low precipitation, high temperature and evapotranspiration); the engine simulates candidate crops and ranks them by expected yield and water efficiency.

## Task List

### Phase 1: Real-World Ingestion & Exploratory Analysis
- [ ] Task 1: Real-World Data Ingestion, Schema Cleanup & Validation
- [ ] Task 2: Exploratory Data Analysis (EDA) with 5 Rubric Visualizations

### Checkpoint: Foundation
- [ ] Old synthetic data removed; real 10,296-row dataset loaded, validated with 0 critical nulls, and 5 distinct EDA plots rendered.

### Phase 2: Leakage-Free Preprocessing & Model Evaluation
- [ ] Task 3: Time-Ordered Train/Val/Test Split & Scikit-Learn Preprocessing Pipeline
- [ ] Task 4: Baseline + 3 Model Training (Dummy, Ridge, Random Forest, HistGradientBoosting)
- [ ] Task 5: 5-Fold Cross-Validation Tuning & Held-Out Test Evaluation

### Checkpoint: Model Performance
- [ ] Tuned model evaluated on held-out 2020–2022 test set with genuine, realistic $R^2$ and error metrics clearly beating baseline.

### Phase 3: Export & Drought Recommendation Engine
- [ ] Task 6: Export Pipeline with `joblib` & Implement Crop Ranking Function

### Checkpoint: Complete Deliverable
- [ ] Pipeline saved at `models/crop_yield_pipeline.joblib`; notebook executes end-to-end via `nbconvert` without warnings.

## Risks and Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Arithmetic target leakage via `Production` | High | Drop `Production` column entirely prior to splitting |
| Temporal leakage across years | High | Strict time-based year split (≤2016 train, 2017–2019 val, ≥2020 test) |
| Feature scale discrepancy | Medium | Standardize all numeric climate features within Pipeline |
| Crop scale variance (e.g. sugarcane yield >> cotton yield) | Medium | Tree ensembles (RandomForest/HistGradientBoosting) naturally handle multi-scale outputs with one-hot crop indicator |
