"""Training, tuning and evaluation. Used by scripts/train.py and the notebook.

Design (see SPEC-model-only-output.md):
- one row per Crop x Province x Year (features.build_yearly_table)
- year split: train 2001-2016, val 2017-2019, test 2020-2022
- 5-fold GroupKFold grouped by Year inside the training years (no year is in two folds)
- target is Yield in dataset units, fitted on the log scale because crops differ by
  ~100x (sugarcane vs cotton); metrics are reported on the original scale too
- model chosen on the validation years; test years are scored once, never used to choose
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features as F

SEED = 42
TRAIN_YEARS = (2001, 2016)
VAL_YEARS = (2017, 2019)
TEST_YEARS = (2020, 2022)

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "pakistan_crop_yield_climate.parquet"


# --------------------------------------------------------------------------- data
def load_tables() -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Returns (full table with engineered features, dropped counts, reference, splits)."""
    raw = pd.read_parquet(RAW_PATH)
    table, dropped = F.build_yearly_table(raw)
    train_mask = table["Year"].between(*TRAIN_YEARS)
    reference = F.build_reference(table[train_mask])
    table = F.add_engineered(table, reference)
    splits = {
        "train": table[train_mask].reset_index(drop=True),
        "val": table[table["Year"].between(*VAL_YEARS)].reset_index(drop=True),
        "test": table[table["Year"].between(*TEST_YEARS)].reset_index(drop=True),
    }
    return table, dropped, reference, splits


# --------------------------------------------------------------------------- model
def make_pipeline(estimator) -> Pipeline:
    """Impute + scale + one-hot, then a log-target regressor. All fitted on training data only."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
         F.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), F.CATEGORICAL_FEATURES),
    ])
    reg = TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)
    return Pipeline([("preprocessor", pre), ("regressor", reg)])


def _log_mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(np.log(y_true), np.log(np.clip(y_pred, 1e-6, None))))


LOG_MAE_SCORER = make_scorer(_log_mae, greater_is_better=False)

CANDIDATES = {
    "ridge": (
        Ridge(random_state=SEED),
        {"regressor__regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
        "Ridge (regularised linear regression)",
    ),
    "random_forest": (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {
            "regressor__regressor__n_estimators": [200, 400],
            "regressor__regressor__max_depth": [None, 8, 15],
            "regressor__regressor__min_samples_leaf": [1, 3, 5],
        },
        "Random Forest",
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=SEED),
        {
            "regressor__regressor__n_estimators": [100, 300],
            "regressor__regressor__learning_rate": [0.03, 0.1],
            "regressor__regressor__max_depth": [2, 3],
        },
        "Gradient Boosting",
    ),
}


# --------------------------------------------------------------------------- baselines + metrics
class CropProvinceMean:
    """Baseline: mean training yield of the same crop in the same province."""

    def __init__(self, train: pd.DataFrame):
        self.cp = train.groupby(["Crop", "Province"])["Yield"].mean()
        self.crop = train.groupby("Crop")["Yield"].mean()
        self.global_mean = float(train["Yield"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = []
        for crop, prov in zip(df["Crop"], df["Province"]):
            out.append(self.cp.get((crop, prov), self.crop.get(crop, self.global_mean)))
        return np.asarray(out, dtype=float)


def score(y_true, y_pred, baseline_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    log_t, log_p, log_b = np.log(y_true), np.log(np.clip(y_pred, 1e-6, None)), np.log(baseline_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "log_mae": float(mean_absolute_error(log_t, log_p)),
        # >0: better than the crop-province mean (i.e. weather/engineered features add something)
        "skill_vs_crop_province_mean_log": float(1.0 - np.sum((log_t - log_p) ** 2) / np.sum((log_t - log_b) ** 2)),
    }


# --------------------------------------------------------------------------- train
def train(out_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    table, dropped, reference, splits = load_tables()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    log(f"rows: train={len(tr)} val={len(va)} test={len(te)} (dropped zero-yield: {dropped})")

    X_tr, y_tr = tr[F.FEATURE_COLS], tr["Yield"]
    cv = GroupKFold(n_splits=5)

    base = CropProvinceMean(tr)
    base_preds = {s: base.predict(d) for s, d in splits.items()}
    baselines = {
        "global_mean": {
            s: score(d["Yield"], np.full(len(d), base.global_mean), base_preds[s]) for s, d in
            {"val": va, "test": te}.items()
        },
        "crop_province_mean": {
            s: score(d["Yield"], base_preds[s], base_preds[s]) for s, d in {"val": va, "test": te}.items()
        },
    }

    models: Dict[str, Any] = {}
    fitted: Dict[str, Pipeline] = {}
    for key, (est, grid, label) in CANDIDATES.items():
        gs = GridSearchCV(
            make_pipeline(est), grid, scoring=LOG_MAE_SCORER, cv=cv, n_jobs=1, refit=True,
        )
        gs.fit(X_tr, y_tr, groups=tr["Year"])
        best = gs.best_estimator_
        fitted[key] = best
        models[key] = {
            "label": label,
            "best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
            "cv_folds": 5,
            "cv_grouped_by": "Year",
            "cv_log_mae": float(-gs.best_score_),
            "val": score(va["Yield"], best.predict(va[F.FEATURE_COLS]), base_preds["val"]),
            "test": score(te["Yield"], best.predict(te[F.FEATURE_COLS]), base_preds["test"]),
        }
        log(f"{label:38s} cv_logMAE={models[key]['cv_log_mae']:.3f} "
            f"val_logMAE={models[key]['val']['log_mae']:.3f} test_R2={models[key]['test']['r2']:.3f}")

    chosen = min(models, key=lambda k: models[k]["val"]["log_mae"])
    best_pipe = fitted[chosen]
    beats = models[chosen]["test"]["skill_vs_crop_province_mean_log"] > 0
    log(f"chosen on validation years: {models[chosen]['label']} | beats crop-province mean on test: {beats}")

    imp = permutation_importance(
        best_pipe, va[F.FEATURE_COLS], va["Yield"], scoring=LOG_MAE_SCORER,
        n_repeats=30, random_state=SEED,
    )
    importances = {
        col: {"mean": float(m), "std": float(s)}
        for col, m, s in sorted(zip(F.FEATURE_COLS, imp.importances_mean, imp.importances_std),
                                key=lambda t: -t[1])
    }

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "seed": SEED,
        "target": "Yield in dataset yield units (unit not confirmed); fitted on log scale",
        "split": {"train_years": list(TRAIN_YEARS), "val_years": list(VAL_YEARS), "test_years": list(TEST_YEARS)},
        "data": {
            "file": str(RAW_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
            "raw_monthly_rows": 10296,
            "yearly_rows_after_drop": int(len(table)),
            "dropped_zero_yield_rows": dropped,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        },
        "features": F.FEATURE_COLS,
        "baselines": baselines,
        "models": models,
        "chosen": {
            "name": chosen,
            "selected_on": "lowest validation log-MAE (relative error, comparable across crops)",
            "beats_crop_province_mean_on_test": bool(beats),
        },
        "permutation_importance_val_log_mae": importances,
        "notes": [
            "Metrics on the test years are reported once and were not used to choose the model.",
            "skill_vs_crop_province_mean_log > 0 means the model beats the crop-province mean baseline.",
            "Year/trend is not a feature, so yield growth over time is not modelled.",
        ],
    }

    joblib.dump(best_pipe, out_dir / "crop_yield_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "reference.json").write_text(json.dumps(reference_document(reference, tr), indent=2))
    return metrics


def reference_document(reference: Dict[str, Any], train: pd.DataFrame) -> Dict[str, Any]:
    """The content of models/reference.json (derived from the training years only)."""
    doc = dict(reference)
    doc["train_rows_per_crop"] = {k: int(v) for k, v in train.groupby("Crop").size().items()}
    doc["train_years"] = list(TRAIN_YEARS)
    return doc


def write_reference(out_dir: Path) -> Dict[str, Any]:
    """Rebuild models/reference.json from the data alone (no model fitting)."""
    _, _, reference, splits = load_tables()
    doc = reference_document(reference, splits["train"])
    (Path(out_dir) / "reference.json").write_text(json.dumps(doc, indent=2))
    return doc
