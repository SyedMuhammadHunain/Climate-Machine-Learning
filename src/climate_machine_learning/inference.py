"""Prediction and ranking from a trained pipeline. Shared by the FastAPI app and the Streamlit app.

Every number comes from `pipeline.predict`. Nothing is adjusted by hand.
For each crop: yield in the user's scenario, yield in a typical (median) year of the province,
and their ratio in percent. Crops are sorted by that ratio, exact ties by crop name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import features as F


class OutOfRangeError(ValueError):
    """An input is outside the range the model was trained on (a tree model cannot extrapolate)."""

    def __init__(self, message: str, parameter: str):
        super().__init__(message)
        self.parameter = parameter


@dataclass
class CropResult:
    crop: str
    predicted_yield: float
    normal_yield: float
    pct_of_normal: float
    low: Optional[float] = None    # likely range, only when log_quantiles are given
    high: Optional[float] = None


@dataclass
class Prediction:
    items: List[CropResult]        # sorted: highest pct_of_normal first, ties by crop name
    inputs: Dict[str, float]       # every model input actually used (after filling empty ones)
    sources: Dict[str, str]        # where each input came from (see features.SRC_*)


def single_thread(pipeline) -> None:
    """One thread at inference: avoids joblib overhead for small predictions."""
    inner = getattr(pipeline, "inner_", pipeline)  # OffsetYieldModel keeps its fitted pipeline in inner_
    reg = inner.named_steps.get("regressor")
    reg = getattr(reg, "regressor_", reg)
    if reg is not None and hasattr(reg, "n_jobs"):
        reg.n_jobs = 1


def tied_with(items: Sequence[CropResult], digits: int = 1) -> Dict[str, int]:
    """For each crop, how many OTHER crops show the same share of a typical year at the displayed precision.

    Crops with equal shares are ordered by name only, so their order carries no information; every
    front end marks them with this count (0 = a unique share).
    """
    shown = [round(i.pct_of_normal, digits) for i in items]
    return {i.crop: shown.count(s) - 1 for i, s in zip(items, shown)}


_METRIC_NAMES = {"r2": "R²", "mae": "MAE", "rmse": "RMSE", "log_mae": "log-MAE"}
_METRIC_FORMATS = {"r2": ".3f", "mae": ".1f", "rmse": ".1f", "log_mae": ".3f"}


def model_caveat(metrics: Dict[str, Any]) -> str:
    """How the chosen model compares with the crop-province mean on the test years, wins and losses together.

    Built only from metrics.json, so every front end (API, Streamlit, Gradio) says the same thing and none
    types a number. Always states the metrics where the model is not better, and that the pass test was
    chosen after the test results were seen, when metrics.json says so.
    """
    chosen = metrics["chosen"]
    first, last = metrics["split"]["test_years"]
    model = metrics["models"][chosen["name"]]["test"]
    base = metrics["baselines"]["crop_province_mean"]["test"]
    by = chosen["beats_by_metric"]

    def pairs(keys):
        items = [f"{_METRIC_NAMES[k]} {model[k]:{_METRIC_FORMATS[k]}} against {base[k]:{_METRIC_FORMATS[k]}}" for k in keys]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    better = [k for k in _METRIC_NAMES if by[k]]
    worse = [k for k in _METRIC_NAMES if not by[k]]
    parts = ([f"was better on {pairs(better)}"] if better else []) + ([f"was not better on {pairs(worse)}"] if worse else [])
    text = (f"On the {first}-{last} test years, compared with using each crop's average yield per province, "
            f"the model {'; it '.join(parts)}.")
    if chosen.get("judged_on_chosen_after_test"):
        judged = " and ".join(_METRIC_NAMES[k] for k in chosen["judged_on"])
        text += f" The pass test ({judged}) was chosen after seeing the test results."
    trend = metrics["baselines"].get("crop_province_mean_plus_trend")
    if trend is not None:
        text += f" A year trend alone, with no climate inputs, scores R² {trend['test']['r2']:.3f}."
    if worse:
        text += " Small differences between crops are weak evidence."
    return text


def check_training_range(reference: Dict[str, Any], province: str, inputs: Dict[str, float]) -> None:
    """Refuse inputs the model never saw."""
    ranges = reference["provinces"][province]["range"]
    first, last = reference["train_years"]
    for name, value in inputs.items():
        lo, hi = ranges[name]
        if value < lo or value > hi:
            raise OutOfRangeError(
                f"{name} = {value:g} is outside the range the model was trained on for {province} "
                f"({lo:.4g} to {hi:.4g}, annual values {first}-{last}). "
                "The model cannot extrapolate, so no prediction is made.",
                parameter=name,
            )


def predict_crops(
    pipeline,
    reference: Dict[str, Any],
    province: str,
    supplied: Dict[str, float],
    crops: Optional[Sequence[str]] = None,
    log_quantiles: Optional[Sequence[float]] = None,
) -> Prediction:
    """Predict the chosen crops (default: all) for a scenario and rank them."""
    supplied = {k: float(v) for k, v in supplied.items() if v is not None}
    check_training_range(reference, province, supplied)
    crops = list(crops) if crops is not None else list(reference["crops"])
    median = dict(reference["provinces"][province]["median"])

    scenario_rows, normal_rows = [], []
    sources: Dict[str, str] = {}
    for crop in crops:
        row, sources = F.build_inference_row(province, crop, supplied, reference)
        scenario_rows.append(row)
        normal_row, _ = F.build_inference_row(province, crop, median, reference)
        normal_rows.append(normal_row)

    batch = pd.concat(scenario_rows + normal_rows, ignore_index=True)
    preds = pipeline.predict(batch)
    scenario_preds, normal_preds = preds[: len(crops)], preds[len(crops):]

    items = []
    for crop, sp, npred in zip(crops, scenario_preds, normal_preds):
        low = high = None
        if log_quantiles is not None:
            low, high = (float(x) for x in F.prediction_interval(sp, log_quantiles))
        items.append(CropResult(crop, float(sp), float(npred), float(sp) / float(npred) * 100.0, low, high))
    items.sort(key=lambda r: (-r.pct_of_normal, r.crop))

    used = {k: float(scenario_rows[0].iloc[0][k]) for k in F.BASE_INPUTS}
    return Prediction(items=items, inputs=used, sources=sources)
