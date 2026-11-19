"""Gradio app (Hugging Face Space): crop yield and ranking under drought (CT-378).

Every number is output of the trained scikit-learn pipeline, a value from the training data, or a
saved evaluation result. Nothing is adjusted by hand.  Run locally:  python hf_space/app.py
"""
import json
import sys
from pathlib import Path

import gradio as gr
import joblib
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
BASE = HERE if (HERE / "models" / "crop_yield_pipeline.joblib").exists() else HERE.parent
for _cand in (BASE / "src", BASE):
    if (_cand / "climate_machine_learning").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from climate_machine_learning import inference as I  # noqa: E402
from climate_machine_learning import presentation as P  # noqa: E402

pipeline = joblib.load(BASE / "models" / "crop_yield_pipeline.joblib")
I.single_thread(pipeline)
ref = json.loads((BASE / "models" / "reference.json").read_text())
met = json.loads((BASE / "models" / "metrics.json").read_text())
_fao_path = BASE / "data" / "reference" / "fao_crop_water_needs.json"
fao = json.loads(_fao_path.read_text()) if _fao_path.exists() else {"crops": {}}
interval = met["prediction_interval"]
FIELDS = P.FIELDS
KEYS = [k for k, _, _ in FIELDS]


def fill(province, preset):
    """Values for the 9 input boxes: a typical year (optional boxes empty) or a real year from the training data."""
    inputs = P.preset_inputs(ref, province, preset)
    return [inputs.get(key) for key in KEYS]


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


def predict(province, crop, *values):
    supplied = {key: value for key, value in zip(KEYS, values) if value is not None}
    try:
        result = I.predict_crops(pipeline, ref, province, supplied, log_quantiles=interval["log_residual_quantiles"])
    except I.OutOfRangeError as err:
        raise gr.Error(str(err))
    chosen = next(i for i in result.items if i.crop == crop)
    summary = (
        f"### {crop.capitalize()} in {province}\n\n"
        f"- **Predicted yield:** {chosen.predicted_yield:.2f} ({P.YIELD_UNIT})\n"
        f"- **Likely range (nominal {interval['level']:.0%}):** {chosen.low:.2f} to {chosen.high:.2f}\n"
        f"- **Share of a typical year:** {chosen.pct_of_normal:.1f}% (typical {province} year: {chosen.normal_yield:.2f})\n\n"
        + P.interval_note(met)
        + "\n\n" + I.model_caveat(met)
        + "\n\n" + I.absolute_yield_caveat(met)
    )
    table = P.ranking_table(result, ref, fao)
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
            preset = gr.Radio(P.PRESETS, value=P.TYPICAL, label="Fill the form from")
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
