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
            if str(cand) not in sys.path:  # Streamlit re-runs this script on every click
                sys.path.insert(0, str(cand))
            break

st.set_page_config(page_title="Crop yield under drought", layout="wide")

if BASE is None:
    st.error("Model files not found. Run `python scripts/ml/train.py` first (they are written to models/).")
    st.stop()

from climate_machine_learning import features as F  # noqa: E402
from climate_machine_learning import inference as I  # noqa: E402
from climate_machine_learning import presentation as P  # noqa: E402


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

FORMATS = {  # number_input display format per input
    "precipitation_sum": "%.1f", "temperature_2m_mean": "%.2f", "temperature_2m_max": "%.2f", "temperature_2m_min": "%.2f",
    "fao_evapotranspiration": "%.2f", "shortwave_radiation": "%.1f", "precipitation_hours": "%.0f",
    "soil_wetness_mean": "%.3f", "soil_wetness_max": "%.3f",
}
FIELDS = [(key, label, FORMATS[key], optional) for key, label, optional in P.FIELDS]

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
    preset = st.radio("Fill the form from", P.PRESETS, key="preset")
    hist = ref["historical_years"][province]
    st.caption(
        f"Real years in {province}: driest {hist['driest']['year']} "
        f"({hist['driest']['inputs']['precipitation_sum']:.1f} mm of rain), "
        f"wettest {hist['wettest']['year']} ({hist['wettest']['inputs']['precipitation_sum']:.1f} mm). "
        "Each fills every input; empty optional inputs use the province median."
    )

info = ref["provinces"][province]
defaults = P.preset_inputs(ref, province, preset)


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
m1.metric("Predicted yield", f"{chosen.predicted_yield:.2f}", help=f"In {P.YIELD_UNIT}.")
m2.metric(f"Likely range (nominal {interval['level']:.0%})", f"{chosen.low:.2f} to {chosen.high:.2f}")
m3.metric("Share of a typical year", f"{chosen.pct_of_normal:.1f}%",
          help=f"Typical {province} year: {chosen.normal_yield:.2f}. This is the model's yield in your scenario divided by its yield in a typical year.")
st.caption(f"Yield unit: {P.YIELD_UNIT}.")
st.caption(I.absolute_yield_caveat(met))
st.caption(P.interval_note(met))

# ---------------------------------------------------------------- explanatory chart + ranking
st.header("3. How the crops compare in this scenario")
chosen_label = met["models"][met["chosen"]["name"]]["label"]
st.subheader(I.ranking_headline(result.items))
st.markdown("\n".join(f"- **{f['label']}:** {f['value']}" for f in I.scenario_facts(province, result.inputs, result.sources, chosen_label)))
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

table = P.ranking_table(result, ref, fao)
st.dataframe(table, hide_index=True, width="stretch")
st.caption("FAO water needs (Brouwer and Heibloem 1986, Table 14) are context only. They do not change the ranking.")

st.markdown("**How crops are ranked**")
st.write(I.RANKING_BASIS)
st.markdown("**About these numbers**")
st.write(I.model_caveat(met))  # the pinned-year note stays under the predicted yield above, once

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
    st.write(f"**{chosen_model['label']}**, chosen on the validation years. The win and its limits are stated under the ranking above.")
    st.write(
        "Most of what the model knows is which crop it is. Weather and soil wetness change predictions only a little. "
        "It trains on one row per crop, province and year. Year is not an input: each model starts from the crop-province "
        f"average of the training years with a year trend, pinned to {met['inference_year']} at prediction, which does not "
        "change the ranking. There is no district or farm detail. "
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
