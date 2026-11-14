# Implementation Plan: Climate-Resilient Low-Water Crop Selection

## Overview
Build an end-to-end Machine Learning Jupyter Notebook (`notebooks/01_low_water_crop_selection.ipynb`) satisfying the CT-378 Machine Learning semester project guidelines. The project trains a unified regression model on historical climate, drought, and agricultural data across Pakistan districts (2000–2023) to predict crop yield (`Base_Yield_kg_ha`), then evaluates candidate low-water crops under drought conditions to recommend the highest-yielding crop.

## Architecture Decisions
- **Unified Regression Architecture:** Train a single regression pipeline using `Crop_Name` as a feature on all 11,088 dataset rows to maximize sample size and climate-interaction learning across all 10 crops.
- **Time-Ordered Split:** Split by `Crop_Year` into Train (2000–2016, ~70%), Validation (2017–2019, ~15%), and Test (2020–2023, ~15%) to prevent temporal data leakage.
- **Pipeline-Centric Preprocessing:** Fit all encoders and scalers strictly on the training partition inside a `ColumnTransformer` / `Pipeline` to eliminate leakage.
- **Algorithms Evaluated:** DummyRegressor (baseline), Ridge Regression, RandomForestRegressor, and HistGradientBoostingRegressor.
- **5-Fold Cross-Validation Tuning:** Perform hyperparameter optimization with `GridSearchCV` / `RandomizedSearchCV` on the training set.
- **Model Persistence:** Export final tuned pipeline via `joblib.dump()` to `models/crop_yield_pipeline.joblib`.
- **Ranking Engine:** Inference function takes user environmental conditions, constructs 5 candidate instances (Pearl Millet, Sorghum, Chickpea, Canola, Barley), predicts yields, and outputs ranked recommendations.

## Task List

### Phase 1: Foundation & Data Pipeline
- [ ] Task 1: Data Ingestion, Merging & Validation
- [ ] Task 2: Exploratory Data Analysis (EDA) with 5 Rubric Visualizations

### Checkpoint: Foundation
- [ ] Merged dataset has 11,088 rows, 24 columns, 0 critical nulls, and 5 clear EDA plots rendered.

### Phase 2: Preprocessing & Modeling
- [ ] Task 3: Time-Ordered Train/Val/Test Split & Scikit-Learn Preprocessing Pipeline
- [ ] Task 4: Baseline + 3 Model Training (Dummy, Ridge, Random Forest, HistGradientBoosting)
- [ ] Task 5: 5-Fold Cross-Validation Tuning & Test Set Evaluation (Target $R^2 > 0.70$)

### Checkpoint: Model Performance
- [ ] Tuned model evaluated on held-out 2020–2023 test set; MAE, RMSE, $R^2$ logged and compared against baseline.

### Phase 3: Model Export & Low-Water Crop Recommendation Engine
- [ ] Task 6: Export Pipeline with `joblib` & Implement Crop Ranking Function

### Checkpoint: Deliverable Ready
- [ ] Final pipeline saved at `models/crop_yield_pipeline.joblib` and recommendation function tested with dry-condition scenario.

## Risks and Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Temporal leakage across years | High | Strict year-based split (Train: <=2016, Val: 2017-2019, Test: >=2020) |
| Feature preprocessing leakage | High | Put all imputers, scalers, and encoders inside scikit-learn Pipeline |
| Underperformance on drought extremes | Medium | Include drought indices (`SPI_3Month`, `TVDI`, `Rainfall_Deficit_Pct`) and non-linear tree models |
| Slow execution during CV | Low | Use HistGradientBoosting and parallelized jobs (`n_jobs=-1`) |

## Open Questions
- None. Confirmation received to produce the complete Jupyter Notebook first.
