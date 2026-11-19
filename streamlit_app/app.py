"""Streamlit app: crop yield and ranking under drought (CT-378).

Every number on screen is output of the trained scikit-learn pipeline, a value from the
training data, or a saved evaluation result. Nothing is adjusted by hand.
Run locally:  streamlit run streamlit_app/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent


def find_base():
    """Repo root (this file is in streamlit_app/) or the folder that holds this file (Hugging Face bundle)."""
    for cand in (HERE, HERE.parent):
        if (cand / "models" / "crop_yield_pipeline.joblib").exists():
            return cand
    return None


BASE = find_base()
if BASE is not None:
    for cand in (BASE / "src", BASE):
        if (cand / "climate_machine_learning").is_dir():
            sys.path.insert(0, str(cand))
            break

st.set_page_config(page_title="Crop yield under drought", layout="wide")

if BASE is None:
    st.error("Model files not found. Run `python scripts/train.py` first (they are written to models/).")
    st.stop()

from climate_machine_learning import features as F  # noqa: E402
from climate_machine_learning import inference as I  # noqa: E402


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(BASE / "models" / "crop_yield_pipeline.joblib")
    I.single_thread(pipeline)
    reference = json.loads((BASE / "models" / "reference.json").read_text())
    metrics = json.loads((BASE / "models" / "metrics.json").read_text())
    fao_path = BASE / "data" / "reference" / "fao_crop_water_needs.json"
    fao = json.loads(fao_path.read_text()) if fao_path.exists() else {"crops": {}, "unit": ""}
    return pipeline, reference, metrics, fao


pipeline, ref, met, fao = load_artifacts()
interval = met["prediction_interval"]
yield_unit = "dataset yield units (the source does not state the unit)"

TYPICAL, DRIEST, WETTEST = "Typical (median) year", "Driest real year", "Wettest real year"
FIELDS = [  # key, label, format, optional
    ("precipitation_sum", "Annual rain (mm)", "%.1f", False),
    ("temperature_2m_mean", "Mean annual temperature (°C)", "%.2f", False),
    ("temperature_2m_max", "Mean monthly maximum temperature (°C)", "%.2f", True),
    ("temperature_2m_min", "Mean monthly minimum temperature (°C)", "%.2f", True),
    ("fao_evapotranspiration", "Evapotranspiration, yearly mean (mm/day)", "%.2f", True),
    ("shortwave_radiation", "Solar radiation, yearly mean (MJ/m²)", "%.1f", True),
    ("precipitation_hours", "Hours with rain, yearly total", "%.0f", True),
    ("soil_wetness_mean", "Root-zone soil wetness, yearly mean (0-1, NASA POWER)", "%.3f", True),
    ("soil_wetness_max", "Root-zone soil wetness, wettest month (0-1, NASA POWER)", "%.3f", True),
]

st.title("Climate-resilient crop selection under drought")
st.write(
    "Enter a year's climate for a province. The trained model predicts the yield of 13 crops and ranks them by how much "
    "of a typical year's yield each is expected to keep. Every number is model output, not adjusted by hand."
)

# ---------------------------------------------------------------- input form
st.header("1. Input form")
left, right = st.columns(2)
with left:
    provinces = list(ref["provinces"])
    province = st.selectbox("Province", provinces, index=provinces.index("Punjab") if "Punjab" in provinces else 0, key="province")
    crops = list(ref["crops"])
    crop = st.selectbox("Crop to look at", crops, index=crops.index("rice") if "rice" in crops else 0,
                        format_func=str.capitalize, key="crop")
with right:
    preset = st.radio("Fill the form from", [TYPICAL, DRIEST, WETTEST], key="preset")
    hist = ref["historical_years"][province]
    st.caption(
        f"Real years in the training data for {province}: driest {hist['driest']['year']} "
        f"({hist['driest']['inputs']['precipitation_sum']:.1f} mm of rain), "
        f"wettest {hist['wettest']['year']} ({hist['wettest']['inputs']['precipitation_sum']:.1f} mm). "
        "Real years fill every input from that year. Empty optional inputs use the province median."
    )

info = ref["provinces"][province]
if preset == TYPICAL:
    defaults = {k: round(info["median"][k], 1) for k in ("precipitation_sum", "temperature_2m_mean")}
else:
    defaults = dict(hist["driest" if preset == DRIEST else "wettest"]["inputs"])


def make_input(key, label, fmt, container):
    lo, hi = info["range"][key]
    return container.number_input(
        label, min_value=lo, max_value=hi, value=defaults.get(key), format=fmt,
        key=f"{key}_{province}_{preset}",
        help=f"The model was trained on {lo:.4g} to {hi:.4g} for {province}. Values outside are not allowed.",
    )


supplied = {}
cols = st.columns(2)
for (key, label, fmt, optional), col in zip([f for f in FIELDS if not f[3]], cols):
    supplied[key] = make_input(key, label, fmt, col)
with st.expander("Optional inputs (empty = province median)"):
    for key, label, fmt, optional in [f for f in FIELDS if f[3]]:
        supplied[key] = make_input(key, label, fmt, st)

# ---------------------------------------------------------------- prediction
try:
    result = I.predict_crops(pipeline, ref, province, supplied, log_quantiles=interval["log_residual_quantiles"])
except I.OutOfRangeError as err:
    st.error(str(err))
    st.stop()

chosen = next(i for i in result.items if i.crop == crop)

st.header("2. Prediction")
st.subheader(f"{crop.capitalize()} in {province}")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted yield", f"{chosen.predicted_yield:.2f}", help=f"In {yield_unit}.")
m2.metric(f"Likely range (nominal {interval['level']:.0%})", f"{chosen.low:.2f} to {chosen.high:.2f}")
m3.metric("Share of a typical year", f"{chosen.pct_of_normal:.1f}%",
          help=f"Typical {province} year: {chosen.normal_yield:.2f}. This is the model's yield in your scenario divided by its yield in a typical year.")
st.caption(f"Yield unit: {yield_unit}.")
test_years = met["split"]["test_years"]
val_years = met["split"]["val_years"]
coverage = interval["test_coverage"]
st.caption(
    f"The range is empirical: the 10% to 90% spread of the model's errors on the validation years "
    f"{val_years[0]}-{val_years[1]}. On the {test_years[0]}-{test_years[1]} test rows it covered {coverage:.0%} of the actual "
    f"yields (nominal {interval['level']:.0%}), so treat it as {'optimistic' if coverage < interval['level'] else 'about right'}."
)

# ---------------------------------------------------------------- explanatory chart + ranking
st.header("3. How the crops compare in this scenario")
frame = pd.DataFrame(
    {
        "Crop": [i.crop for i in result.items],
        "pct": [i.pct_of_normal for i in result.items],
        "Predicted yield": [i.predicted_yield for i in result.items],
    }
)
frame["Selected"] = frame["Crop"] == crop
bars = (
    alt.Chart(frame)
    .mark_bar()
    .encode(
        y=alt.Y("Crop:N", sort=alt.EncodingSortField(field="pct", order="descending"), title=None),
        x=alt.X("pct:Q", title="Share of a typical year's yield (%), model output"),
        color=alt.condition("datum.Selected", alt.value("#2f7d4f"), alt.value("#9aa5b1")),
        tooltip=["Crop", alt.Tooltip("pct:Q", format=".1f", title="% of typical"), alt.Tooltip("Predicted yield:Q", format=".2f")],
    )
)
line = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(strokeDash=[4, 4], color="#444").encode(x="x:Q")
st.altair_chart(bars + line, width="stretch")
st.caption("Bars longer than the dashed line (100%) mean the model expects more than a typical year, shorter means less. "
           "Crops with few training rows are noisier (see the table).")

possible = len(ref["provinces"]) * (ref["train_years"][1] - ref["train_years"][0] + 1)


def fao_text(name):
    entry = fao["crops"].get(name)
    if entry is None:
        return "not listed in the FAO table"
    return f"{entry['mm_min']}-{entry['mm_max']} mm per growing period, drought sensitivity {entry['drought_sensitivity']}"


table = pd.DataFrame(
    {
        "Rank": range(1, len(result.items) + 1),
        "Crop": [i.crop for i in result.items],
        "% of typical year": [round(i.pct_of_normal, 1) for i in result.items],
        "Predicted yield": [round(i.predicted_yield, 2) for i in result.items],
        "Likely range": [f"{i.low:.2f} to {i.high:.2f}" for i in result.items],
        "Training rows": [f"{ref['train_rows_per_crop'][i.crop]} of {possible}" for i in result.items],
        "FAO water need": [fao_text(i.crop) for i in result.items],
    }
)
st.dataframe(table, hide_index=True, width="stretch")
st.caption("FAO water needs (Brouwer and Heibloem 1986, Table 14) are context only. They do not change the ranking.")

with st.expander("Inputs used (and where each came from)"):
    labels = {k: label for k, label, _, _ in FIELDS}
    source_text = {
        "user": "entered or filled from a real year",
        "province_median_training_years": "province median (training years)",
        "user_mean_plus_province_median_offset": "your mean temperature plus the province's median offset",
    }
    st.dataframe(
        pd.DataFrame(
            {
                "Input": [labels[k] for k in F.BASE_INPUTS],
                "Value": [round(result.inputs[k], 3) for k in F.BASE_INPUTS],
                "Source": [source_text.get(result.sources[k], result.sources[k]) for k in F.BASE_INPUTS],
            }
        ),
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------- about
with st.expander("About this model and its limits"):
    chosen_model = met["models"][met["chosen"]["name"]]
    base = met["baselines"]["crop_province_mean"]["test"]
    st.write(
        f"**{chosen_model['label']}**, chosen on the validation years. Test years {test_years[0]}-{test_years[1]}: "
        f"R² {chosen_model['test']['r2']:.3f}, mean absolute error {chosen_model['test']['mae']:.1f}. "
        f"A baseline that predicts each crop's average yield per province scores R² {base['r2']:.3f} on the same years, "
        f"and the skill of the model against it is {chosen_model['test']['skill_vs_crop_province_mean_log']:+.2f} (above 0 is better)."
    )
    st.write(
        "Most of what the model knows is which crop it is. Weather and soil wetness change predictions only a little. "
        "It trains on one row per crop, province and year, so there is no trend over time and no district or farm detail. "
        "Inputs outside the trained range are not allowed because the model cannot extrapolate."
    )
    importance = met["permutation_importance_val_log_mae"]
    st.dataframe(
        pd.DataFrame({"Input": list(importance), "Importance (validation)": [round(v["mean"], 4) for v in importance.values()]}),
        hide_index=True,
        width="stretch",
    )

st.caption(
    "Data: Pakistan crop yield vs climate (MIT licence; origin Zenodo 15486470) and NASA POWER soil wetness "
    "(licence not specified; free to use, cite https://power.larc.nasa.gov/). CT-378 Machine Learning project."
)
