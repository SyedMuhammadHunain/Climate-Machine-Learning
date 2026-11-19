"""Text and tables shown by the Streamlit and Gradio apps, in one place so both say the same thing.

Pure functions of the model output and the saved reference/metrics files. No UI code here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from . import features as F
from . import inference as I

YIELD_UNIT = "dataset yield units (the source does not state the unit)"
TYPICAL, DRIEST, WETTEST = "Typical (median) year", "Driest real year", "Wettest real year"
PRESETS = [TYPICAL, DRIEST, WETTEST]

_LABELS = {
    "precipitation_sum": "Annual rain (mm)",
    "temperature_2m_mean": "Mean annual temperature (°C)",
    "temperature_2m_max": "Mean monthly maximum temperature (°C)",
    "temperature_2m_min": "Mean monthly minimum temperature (°C)",
    "fao_evapotranspiration": "Evapotranspiration, yearly mean (mm/day)",
    "shortwave_radiation": "Solar radiation, yearly mean (MJ/m²)",
    "precipitation_hours": "Hours with rain, yearly total",
    "soil_wetness_mean": "Root-zone soil wetness, yearly mean (0-1, NASA POWER)",
    "soil_wetness_max": "Root-zone soil wetness, wettest month (0-1, NASA POWER)",
}
# (key, label, optional): the model inputs in order. Only rain and mean temperature are required.
FIELDS: List[Tuple[str, str, bool]] = [(k, _LABELS[k], k not in F.REQUIRED_INPUTS) for k in F.BASE_INPUTS]


def preset_inputs(reference: Dict[str, Any], province: str, preset: str) -> Dict[str, float]:
    """Form values for a preset: a typical year (rain and temperature only) or every input of a real year."""
    if preset == TYPICAL:
        median = reference["provinces"][province]["median"]
        return {k: round(median[k], 1) for k in F.REQUIRED_INPUTS}
    kind = "driest" if preset == DRIEST else "wettest"
    return dict(reference["historical_years"][province][kind]["inputs"])


def fao_text(fao: Dict[str, Any], crop: str) -> str:
    entry = fao["crops"].get(crop)
    if entry is None:
        return "not listed in the FAO table"
    return f"{entry['mm_min']}-{entry['mm_max']} mm per growing period, drought sensitivity {entry['drought_sensitivity']}"


def ranking_table(result, reference: Dict[str, Any], fao: Dict[str, Any]) -> pd.DataFrame:
    """One row per crop, in ranked order."""
    first, last = reference["train_years"]
    possible = len(reference["provinces"]) * (last - first + 1)
    items = result.items
    ties = I.tied_with(items)
    return pd.DataFrame(
        {
            "Rank": range(1, len(items) + 1),
            "Crop": [i.crop for i in items],
            "% of typical year": [round(i.pct_of_normal, 1) for i in items],
            "Tied with": [ties[i.crop] for i in items],
            "Predicted yield": [round(i.predicted_yield, 2) for i in items],
            "Likely range": [f"{i.low:.2f} to {i.high:.2f}" for i in items],
            "Training rows": [f"{reference['train_rows_per_crop'][i.crop]} of {possible}" for i in items],
            "FAO water need": [fao_text(fao, i.crop) for i in items],
        }
    )


def interval_note(metrics: Dict[str, Any]) -> str:
    """How the likely range was made and how well it did on the test years (always shown with the range)."""
    pi = metrics["prediction_interval"]
    val, test = metrics["split"]["val_years"], metrics["split"]["test_years"]
    verdict = "optimistic" if pi["test_coverage"] < pi["level"] else "about right"
    return (
        f"The range is empirical: the 10% to 90% spread of the model's errors on the validation years "
        f"{val[0]}-{val[1]}. On the {test[0]}-{test[1]} test rows it covered {pi['test_coverage']:.0%} of the actual "
        f"yields (nominal {pi['level']:.0%}), so treat it as {verdict}."
    )
