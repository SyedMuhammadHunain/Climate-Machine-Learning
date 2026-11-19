"""Gradio app (Hugging Face Space): crop yield and ranking under drought (CT-378).

Every number is output of the trained scikit-learn pipeline, a value from the training data, or a
saved evaluation result. Nothing is adjusted by hand.  Run locally:  python hf_space/app.py
"""
import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
BASE = HERE if (HERE / "models" / "crop_yield_pipeline.joblib").exists() else HERE.parent
for _cand in (BASE / "src", BASE):
    if (_cand / "climate_machine_learning").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from climate_machine_learning import features as F  # noqa: E402
from climate_machine_learning import inference as I  # noqa: E402

pipeline = joblib.load(BASE / "models" / "crop_yield_pipeline.joblib")
I.single_thread(pipeline)
ref = json.loads((BASE / "models" / "reference.json").read_text())
met = json.loads((BASE / "models" / "metrics.json").read_text())
_fao_path = BASE / "data" / "reference" / "fao_crop_water_needs.json"
fao = json.loads(_fao_path.read_text()) if _fao_path.exists() else {"crops": {}}
interval = met["prediction_interval"]
yield_unit = "dataset yield units (the source does not state the unit)"

TYPICAL, DRIEST, WETTEST = "Typical (median) year", "Driest real year", "Wettest real year"
FIELDS = [  # key, label, optional
    ("precipitation_sum", "Annual rain (mm)", False),
    ("temperature_2m_mean", "Mean annual temperature (°C)", False),
    ("temperature_2m_max", "Mean monthly maximum temperature (°C)", True),
    ("temperature_2m_min", "Mean monthly minimum temperature (°C)", True),
    ("fao_evapotranspiration", "Evapotranspiration, yearly mean (mm/day)", True),
    ("shortwave_radiation", "Solar radiation, yearly mean (MJ/m²)", True),
    ("precipitation_hours", "Hours with rain, yearly total", True),
    ("soil_wetness_mean", "Root-zone soil wetness, yearly mean (0-1, NASA POWER)", True),
    ("soil_wetness_max", "Root-zone soil wetness, wettest month (0-1, NASA POWER)", True),
]
KEYS = [k for k, _, _ in FIELDS]
possible = len(ref["provinces"]) * (ref["train_years"][1] - ref["train_years"][0] + 1)


def fill(province, preset):
    """Values for the 9 input boxes: a typical year (optional boxes empty) or a real year from the training data."""
    if preset == TYPICAL:
        med = ref["provinces"][province]["median"]
        return [round(med["precipitation_sum"], 1), round(med["temperature_2m_mean"], 1)] + [None] * 7
    kind = "driest" if preset == DRIEST else "wettest"
    inputs = ref["historical_years"][province][kind]["inputs"]
    return [inputs[k] for k in KEYS]


def ranges_text(province):
    info = ref["provinces"][province]
    hist = ref["historical_years"][province]
    rows = "\n".join(
        f"| {label} | {info['range'][key][0]:.4g} to {info['range'][key][1]:.4g} | {info['median'][key]:.4g} |"
        for key, label, _ in FIELDS
    )
    return (
        f"**Trained ranges for {province}** (values outside are refused, the model cannot extrapolate). "
        f"Real years: driest {hist['driest']['year']}, wettest {hist['wettest']['year']}.\n\n"
        f"| Input | Trained range | Median |\n|---|---|---|\n{rows}"
    )


def fao_text(name):
    entry = fao["crops"].get(name)
    if entry is None:
        return "not listed in the FAO table"
    return f"{entry['mm_min']}-{entry['mm_max']} mm per growing period, drought sensitivity {entry['drought_sensitivity']}"


def predict(province, crop, *values):
    supplied = {key: value for key, value in zip(KEYS, values) if value is not None}
    try:
        result = I.predict_crops(pipeline, ref, province, supplied, log_quantiles=interval["log_residual_quantiles"])
    except I.OutOfRangeError as err:
        raise gr.Error(str(err))
    chosen = next(i for i in result.items if i.crop == crop)
    test_years, val_years = met["split"]["test_years"], met["split"]["val_years"]
    coverage = interval["test_coverage"]
    summary = (
        f"### {crop.capitalize()} in {province}\n\n"
        f"- **Predicted yield:** {chosen.predicted_yield:.2f} ({yield_unit})\n"
        f"- **Likely range (nominal {interval['level']:.0%}):** {chosen.low:.2f} to {chosen.high:.2f}\n"
        f"- **Share of a typical year:** {chosen.pct_of_normal:.1f}% (typical {province} year: {chosen.normal_yield:.2f})\n\n"
        f"The range is empirical: the 10% to 90% spread of the model's errors on the validation years "
        f"{val_years[0]}-{val_years[1]}. On the {test_years[0]}-{test_years[1]} test rows it covered {coverage:.0%} of the "
        f"actual yields (nominal {interval['level']:.0%}), so treat it as "
        f"{'optimistic' if coverage < interval['level'] else 'about right'}."
    )
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
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ordered = list(reversed(result.items))
    ax.barh([i.crop for i in ordered], [i.pct_of_normal for i in ordered],
            color=["#2f7d4f" if i.crop == crop else "#9aa5b1" for i in ordered])
    ax.axvline(100, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Share of a typical year's yield (%), model output")
    ax.set_title("How the crops compare in this scenario")
    fig.tight_layout()
    return summary, table, fig


def build_demo():
    with gr.Blocks(title="Crop yield under drought") as demo:
        gr.Markdown(
            "# Climate-resilient crop selection under drought\n"
            "Enter a year's climate for a province. The trained model predicts the yield of 13 crops and ranks them by how much "
            "of a typical year's yield each is expected to keep. Every number is model output, not adjusted by hand."
        )
        provinces = list(ref["provinces"])
        crops = list(ref["crops"])
        with gr.Row():
            province = gr.Dropdown(provinces, value="Punjab" if "Punjab" in provinces else provinces[0], label="Province")
            crop = gr.Dropdown(crops, value="rice" if "rice" in crops else crops[0], label="Crop to look at")
            preset = gr.Radio([TYPICAL, DRIEST, WETTEST], value=TYPICAL, label="Fill the form from")
        ranges = gr.Markdown(ranges_text(provinces[0]))
        boxes = []
        with gr.Row():
            for key, label, optional in FIELDS[:2]:
                boxes.append(gr.Number(label=label, value=None))
        with gr.Accordion("Optional inputs (empty = province median)", open=False):
            for key, label, optional in FIELDS[2:]:
                boxes.append(gr.Number(label=label, value=None))
        button = gr.Button("Predict", variant="primary")
        summary = gr.Markdown()
        plot = gr.Plot(label="Comparison")
        table = gr.Dataframe(label="Ranking of all crops (model output)", interactive=False)

        province.change(ranges_text, province, ranges)
        province.change(fill, [province, preset], boxes)
        preset.change(fill, [province, preset], boxes)
        demo.load(fill, [province, preset], boxes)
        demo.load(ranges_text, province, ranges)
        button.click(predict, [province, crop, *boxes], [summary, table, plot], api_name="predict")
        gr.Markdown(
            "Data: Pakistan crop yield vs climate (MIT licence; origin Zenodo 15486470) and NASA POWER soil wetness "
            "(licence not specified; free to use, cite https://power.larc.nasa.gov/). CT-378 Machine Learning project. "
            "Most of what the model knows is which crop it is; weather and soil wetness change predictions only a little."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
