"""Shared feature code. Used by training AND by the API so they never drift apart.

Raw data: one row per Crop x Province x Year x Month (12 rows per group).
Yield is one yearly number copied onto all 12 monthly rows, and the weather is
province-level (same for every crop). The model therefore trains on ONE row per
Crop x Province x Year, with weather aggregated over the year.

All inputs are annual aggregates:
    precipitation_sum       total rain over the year (mm)
    temperature_2m_*        mean over the 12 monthly values (deg C)
    fao_evapotranspiration  mean over the 12 monthly values (mm/day)
    shortwave_radiation     mean over the 12 monthly values (MJ/m2)
    precipitation_hours     total over the year (hours)
    soil_wetness_mean       NASA POWER root-zone soil wetness (0-1): mean of the 12 monthly values
    soil_wetness_max        NASA POWER root-zone soil wetness (0-1): wettest month

The two soil wetness inputs come from a second public source (NASA POWER, monthly, 3 farming-area
points per province, equal weight). They are province-level like the weather and are merged by
Province + Year.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

ROOT = Path(__file__).resolve().parent.parent.parent
SOIL_CSV = ROOT / "data" / "raw" / "nasa_power_monthly.csv"

GROUP_KEYS = ["Crop", "Province", "Year"]

BASE_INPUTS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "fao_evapotranspiration",
    "shortwave_radiation",
    "precipitation_hours",
    "soil_wetness_mean",
    "soil_wetness_max",
]
REQUIRED_INPUTS = ["precipitation_sum", "temperature_2m_mean"]
ENGINEERED = ["rain_deficit_pct", "water_balance_mm", "aridity_index"]
NUMERIC_FEATURES = BASE_INPUTS + ENGINEERED
CATEGORICAL_FEATURES = ["Crop", "Province"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_AGG = {
    "precipitation_sum": "sum",
    "temperature_2m_mean": "mean",
    "temperature_2m_max": "mean",
    "temperature_2m_min": "mean",
    "fao_evapotranspiration": "mean",
    "shortwave_radiation": "mean",
    "precipitation_hours": "sum",
    "Yield": "first",
}

SRC_USER = "user"
SRC_MEDIAN = "province_median_training_years"
SRC_OFFSET = "user_mean_plus_province_median_offset"


# --------------------------------------------------------------------------- months
def add_month(raw: pd.DataFrame) -> pd.DataFrame:
    """Month 1..12 from row order inside each Crop-Province-Year group.

    The source file has no month column. Row order is checked by check_month_order().
    """
    out = raw.copy()
    grouped = out.groupby(GROUP_KEYS, sort=False)
    sizes = grouped.size()
    if not (sizes == 12).all():
        raise ValueError("every Crop-Province-Year group must have exactly 12 rows")
    out["Month"] = grouped.cumcount() + 1
    return out


def check_month_order(with_month: pd.DataFrame) -> None:
    """Fail loudly if row order is not calendar order (coldest in Dec-Feb, hottest in May-Aug)."""
    for province, grp in with_month.groupby("Province"):
        by_month = grp.groupby("Month")["temperature_2m_mean"].mean()
        coldest, hottest = int(by_month.idxmin()), int(by_month.idxmax())
        if coldest not in (12, 1, 2) or hottest not in (5, 6, 7, 8):
            raise ValueError(
                f"row order is not calendar order for {province}: "
                f"coldest row={coldest}, hottest row={hottest}"
            )


# --------------------------------------------------------------------------- second source
def load_soil_monthly(path: Optional[Path] = None) -> pd.DataFrame:
    """Monthly NASA POWER table: Province, Point, Year, Month, GWETROOT, PRECTOTCORR."""
    return pd.read_csv(path or SOIL_CSV)


def soil_annual(soil_monthly: pd.DataFrame) -> pd.DataFrame:
    """Province-Year soil wetness: average the points first, then take the mean and max over 12 months."""
    monthly = soil_monthly.groupby(["Province", "Year", "Month"])["GWETROOT"].mean().reset_index()
    counts = monthly.groupby(["Province", "Year"]).size()
    if not (counts == 12).all():
        bad = counts[counts != 12].index.tolist()
        raise ValueError(f"soil data must have 12 months for every Province-Year; incomplete: {bad}")
    return (
        monthly.groupby(["Province", "Year"])["GWETROOT"]
        .agg(soil_wetness_mean="mean", soil_wetness_max="max")
        .reset_index()
    )


# --------------------------------------------------------------------------- yearly table
def build_yearly_table(
    raw: pd.DataFrame, soil_monthly: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """One row per Crop x Province x Year, with soil wetness merged in. Zero-yield rows are dropped.

    In the source, every zero-yield row also has Area = 0, i.e. the crop was not
    grown or not reported. It is missing data, not a real yield of zero.
    Returns (table, {crop: rows_dropped}).
    """
    with_month = add_month(raw)
    check_month_order(with_month)
    table = with_month.groupby(GROUP_KEYS, sort=True).agg(_AGG).reset_index()

    # second source: soil wetness by Province + Year (fails loudly if any year is missing)
    soil = soil_annual(load_soil_monthly() if soil_monthly is None else soil_monthly)
    table = table.merge(soil, on=["Province", "Year"], how="left", validate="many_to_one")
    missing = table.loc[table["soil_wetness_mean"].isna(), ["Province", "Year"]].drop_duplicates()
    if len(missing):
        pairs = [f"{p} {y}" for p, y in missing.itertuples(index=False)]
        raise ValueError(f"no soil wetness data for: {', '.join(pairs)}")

    zero = table["Yield"] <= 0
    dropped = table.loc[zero, "Crop"].value_counts().sort_index().to_dict()
    table = table.loc[~zero].reset_index(drop=True)
    return table, {k: int(v) for k, v in dropped.items()}


# --------------------------------------------------------------------------- reference
def build_reference(train: pd.DataFrame) -> Dict[str, Any]:
    """Province climate statistics from TRAINING years only (no test-year leakage).

    Weather is the same for every crop, so statistics use one row per Province-Year.
    """
    climate = train.drop_duplicates(["Province", "Year"])
    provinces: Dict[str, Any] = {}
    for province, grp in climate.groupby("Province"):
        provinces[province] = {
            "median": {k: float(grp[k].median()) for k in BASE_INPUTS},
            "mean_annual_rain": float(grp["precipitation_sum"].mean()),
            "temp_offsets": {
                "max_minus_mean": float((grp["temperature_2m_max"] - grp["temperature_2m_mean"]).median()),
                "min_minus_mean": float((grp["temperature_2m_min"] - grp["temperature_2m_mean"]).median()),
            },
            "range": {k: [float(grp[k].min()), float(grp[k].max())] for k in BASE_INPUTS},
            "years": [int(grp["Year"].min()), int(grp["Year"].max())],
        }
    # Real observed years (training years only) for quick-fill scenarios: driest and wettest by annual rain
    historical: Dict[str, Any] = {}
    for province, grp in climate.groupby("Province"):
        historical[province] = {}
        for kind, idx in (("driest", grp["precipitation_sum"].idxmin()), ("wettest", grp["precipitation_sum"].idxmax())):
            row = grp.loc[idx]
            historical[province][kind] = {
                "year": int(row["Year"]),
                "inputs": {k: float(row[k]) for k in BASE_INPUTS},
            }
    return {
        "provinces": provinces,
        "historical_years": historical,
        "crops": sorted(train["Crop"].unique().tolist()),
        "feature_cols": FEATURE_COLS,
    }


# --------------------------------------------------------------------------- engineered
def engineered(rain, et0, mean_annual_rain) -> Dict[str, Any]:
    """The three drought features, written once. Works on plain floats (one API request) and on
    pandas columns (the training table), so both paths use the same formulas."""
    et_year = et0 * 365.0
    return {
        "rain_deficit_pct": (mean_annual_rain - rain) / mean_annual_rain * 100.0,
        "water_balance_mm": rain - et_year,
        "aridity_index": rain / et_year,
    }


def add_engineered(table: pd.DataFrame, reference: Dict[str, Any]) -> pd.DataFrame:
    """Add engineered columns to a yearly table (training path)."""
    out = table.copy()
    mean_rain = out["Province"].map(lambda p: reference["provinces"][p]["mean_annual_rain"])
    for name, column in engineered(out["precipitation_sum"], out["fao_evapotranspiration"], mean_rain).items():
        out[name] = column
    return out


# --------------------------------------------------------------------------- inference
def build_inference_row(
    province: str,
    crop: str,
    inputs: Dict[str, float],
    reference: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """One model-ready row from user inputs. Returns (row, {input: source}).

    Missing optional inputs come from the training-year province median (data, not rules).
    Max/min temperature follow the user's mean through the province's median offsets.
    """
    if province not in reference["provinces"]:
        raise ValueError(f"unknown province: {province!r}")
    if crop not in reference["crops"]:
        raise ValueError(f"unknown crop: {crop!r}")
    unknown = set(inputs) - set(BASE_INPUTS)
    if unknown:
        raise ValueError(f"unknown inputs: {sorted(unknown)}")
    for name in REQUIRED_INPUTS:
        if inputs.get(name) is None:
            raise ValueError(f"missing required input: {name}")
    for name, val in inputs.items():
        if val is not None and not math.isfinite(float(val)):
            raise ValueError(f"input {name} is not a finite number")

    prov = reference["provinces"][province]
    values: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    for name in BASE_INPUTS:
        given = inputs.get(name)
        if given is not None:
            values[name], sources[name] = float(given), SRC_USER
        elif name in ("temperature_2m_max", "temperature_2m_min"):
            key = "max_minus_mean" if name.endswith("max") else "min_minus_mean"
            values[name] = float(inputs["temperature_2m_mean"]) + prov["temp_offsets"][key]
            sources[name] = SRC_OFFSET
        else:
            values[name], sources[name] = prov["median"][name], SRC_MEDIAN

    values.update(engineered(values["precipitation_sum"], values["fao_evapotranspiration"], prov["mean_annual_rain"]))
    values["Crop"], values["Province"] = crop, province
    return pd.DataFrame([values])[FEATURE_COLS], sources


# --------------------------------------------------------------------------- prediction interval
def prediction_interval(pred, log_quantiles):
    """Likely range for a prediction: pred x exp(low), pred x exp(high).

    The two numbers are quantiles of log(actual / predicted) on the validation years,
    saved in metrics.json. It is an empirical range, not a model-based probability.
    """
    lo_q, hi_q = log_quantiles
    pred = np.asarray(pred, dtype=float)
    return pred * np.exp(lo_q), pred * np.exp(hi_q)


# --------------------------------------------------------------------------- offset model
TRENDS = ("none", "per_crop", "per_crop_province")
MIN_ROWS_FOR_OWN_SLOPE = 3


def _within_slope(dx: np.ndarray, dy: np.ndarray) -> float:
    """Least-squares slope through already-centred data (0 when there is no spread in x)."""
    denom = float(np.sum(dx * dx))
    return float(np.sum(dx * dy) / denom) if denom > 0 else 0.0


class OffsetYieldModel(RegressorMixin, BaseEstimator):
    """Predict yield as exp(offset + gap), where the inner model only learns the gap.

    offset(Crop, Province, Year) is learned from the training rows passed to fit():
        level  = mean log yield of the crop in the province (falls back to the crop, then to everything)
        trend  = optional slope per year, in log yield: none | per_crop | per_crop_province
    The inner model is fitted on log(yield) - offset, so it only has to explain how a year differs
    from that crop-province's normal (SPEC-beat-baseline.md).

    X must hold Crop, Province and Year when fitting. At prediction time the Year column is used
    when present; otherwise the pinned `inference_year` is used. The inner model never sees Year.
    Lives in this module so every consumer that already ships it can unpickle the saved pipeline.
    """

    def __init__(self, inner, trend: str = "none", inference_year: Optional[int] = None):
        self.inner = inner
        self.trend = trend
        self.inference_year = inference_year

    # -- fitting
    def fit(self, X: pd.DataFrame, y):
        if self.trend not in TRENDS:
            raise ValueError(f"trend must be one of {TRENDS}, got {self.trend!r}")
        y = np.asarray(y, dtype=float)
        if not np.all(y > 0):
            raise ValueError("yields must be positive: the offset works on log yield")
        d = pd.DataFrame({
            "Crop": X["Crop"].to_numpy(), "Province": X["Province"].to_numpy(),
            "Year": X["Year"].to_numpy(dtype=float), "log_y": np.log(y),
        })
        d["cp_year"] = d.groupby(["Crop", "Province"])["Year"].transform("mean")
        d["cp_log"] = d.groupby(["Crop", "Province"])["log_y"].transform("mean")
        d["dx"], d["dy"] = d["Year"] - d["cp_year"], d["log_y"] - d["cp_log"]

        self.global_ = (float(d["log_y"].mean()), float(d["Year"].mean()))
        self.level_c_ = {c: (float(g["log_y"].mean()), float(g["Year"].mean())) for c, g in d.groupby("Crop")}
        self.level_cp_ = {k: (float(g["log_y"].mean()), float(g["Year"].mean()))
                          for k, g in d.groupby(["Crop", "Province"])}
        self.slope_c_, self.slope_cp_ = {}, {}
        if self.trend != "none":
            self.slope_c_ = {c: _within_slope(g["dx"].to_numpy(), g["dy"].to_numpy()) for c, g in d.groupby("Crop")}
        if self.trend == "per_crop_province":
            for k, g in d.groupby(["Crop", "Province"]):
                if len(g) >= MIN_ROWS_FOR_OWN_SLOPE:
                    self.slope_cp_[k] = _within_slope(g["dx"].to_numpy(), g["dy"].to_numpy())

        offset = self._offset(X["Crop"], X["Province"], d["Year"].to_numpy())
        self.inner_ = clone(self.inner).fit(self._inner_view(X), d["log_y"].to_numpy() - offset)
        return self

    # -- offset
    def _offset(self, crops, provinces, years) -> np.ndarray:
        out = np.empty(len(years), dtype=float)
        for i, (c, p, t) in enumerate(zip(crops, provinces, years)):
            if (c, p) in self.level_cp_:
                level, mean_year = self.level_cp_[(c, p)]
                slope = self.slope_cp_.get((c, p), self.slope_c_.get(c, 0.0))
            elif c in self.level_c_:
                level, mean_year = self.level_c_[c]
                slope = self.slope_c_.get(c, 0.0)
            else:
                (level, mean_year), slope = self.global_, 0.0
            out[i] = level + slope * (t - mean_year)
        return out

    def _years(self, X: pd.DataFrame) -> np.ndarray:
        if "Year" in X.columns:
            return X["Year"].to_numpy(dtype=float)
        if self.inference_year is None:
            raise ValueError("X has no Year column and inference_year is not set")
        return np.full(len(X), float(self.inference_year))

    @staticmethod
    def _inner_view(X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns="Year", errors="ignore")

    def log_offset(self, X: pd.DataFrame) -> np.ndarray:
        """The learned offset (log yield units) for each row of X."""
        return self._offset(X["Crop"], X["Province"], self._years(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.log_offset(X) + self.inner_.predict(self._inner_view(X)))
