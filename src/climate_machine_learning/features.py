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
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import pandas as pd

GROUP_KEYS = ["Crop", "Province", "Year"]

BASE_INPUTS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "fao_evapotranspiration",
    "shortwave_radiation",
    "precipitation_hours",
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


# --------------------------------------------------------------------------- yearly table
def build_yearly_table(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """One row per Crop x Province x Year. Zero-yield rows are dropped.

    In the source, every zero-yield row also has Area = 0, i.e. the crop was not
    grown or not reported. It is missing data, not a real yield of zero.
    Returns (table, {crop: rows_dropped}).
    """
    with_month = add_month(raw)
    check_month_order(with_month)
    table = with_month.groupby(GROUP_KEYS, sort=True).agg(_AGG).reset_index()

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
def _engineer(values: Dict[str, float], mean_annual_rain: float) -> Dict[str, float]:
    rain = values["precipitation_sum"]
    et_year = values["fao_evapotranspiration"] * 365.0
    return {
        "rain_deficit_pct": (mean_annual_rain - rain) / mean_annual_rain * 100.0,
        "water_balance_mm": rain - et_year,
        "aridity_index": rain / et_year,
    }


def add_engineered(table: pd.DataFrame, reference: Dict[str, Any]) -> pd.DataFrame:
    """Add engineered columns to a yearly table (training path)."""
    out = table.copy()
    mean_rain = out["Province"].map(lambda p: reference["provinces"][p]["mean_annual_rain"])
    et_year = out["fao_evapotranspiration"] * 365.0
    out["rain_deficit_pct"] = (mean_rain - out["precipitation_sum"]) / mean_rain * 100.0
    out["water_balance_mm"] = out["precipitation_sum"] - et_year
    out["aridity_index"] = out["precipitation_sum"] / et_year
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

    values.update(_engineer(values, prov["mean_annual_rain"]))
    values["Crop"], values["Province"] = crop, province
    return pd.DataFrame([values])[FEATURE_COLS], sources
