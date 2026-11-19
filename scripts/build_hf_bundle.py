"""Build the Hugging Face upload bundle: a model repo and a Gradio Space, ready to upload.

Run:  python scripts/build_hf_bundle.py [--out hf_bundle]
Writes (self-contained, nothing is downloaded or uploaded here):
  <out>/model/  model card (README.md), pipeline, reference.json, metrics.json, inference code, example
  <out>/space/  Space README.md (config), app.py, requirements.txt, model files, inference code
The model card is generated from models/metrics.json, so its numbers always match the trained model.
Uploading needs your own Hugging Face account: see docs/deployment_guide.md.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "climate_machine_learning"
MODEL_FILES = ("crop_yield_pipeline.joblib", "reference.json", "metrics.json")
CODE_FILES = ("__init__.py", "features.py", "inference.py")


def _table(met: dict) -> str:
    """Evaluation table, generated from metrics.json only."""
    names = {"global_mean": "Baseline: overall mean yield", "crop_province_mean": "Baseline: crop-province mean yield"}
    lines = [
        "| Model | Val R² | Val MAE | Test R² | Test MAE | Test RMSE | Test skill vs crop-province mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [(names.get(k, k), v["val"], v["test"], False) for k, v in met["baselines"].items()]
    rows += [(v["label"], v["val"], v["test"], k == met["chosen"]["name"]) for k, v in met["models"].items()]
    for name, val, test, chosen in rows:
        label = f"**{name} (chosen)**" if chosen else name
        lines.append(
            f"| {label} | {val['r2']:.3f} | {val['mae']:.1f} | {test['r2']:.3f} | {test['mae']:.1f} | "
            f"{test['rmse']:.1f} | {test['skill_vs_crop_province_mean_log']:+.2f} |"
        )
    return "\n".join(lines)


def model_card(met: dict, ref: dict) -> str:
    chosen = met["models"][met["chosen"]["name"]]
    d, pi, split = met["data"], met["prediction_interval"], met["split"]
    sources = "\n".join(
        f"- **{s['role']}**: {s['name']}. Licence: {s['licence']}."
        + (f" Accessed {s['accessed']} (API {s['api_version']}, {s['points']} points)." if "accessed" in s else "")
        + f" File SHA-256 `{s['sha256'][:16]}...`."
        for s in d["sources"]
    )
    dropped = ", ".join(f"{c} {n}" for c, n in d["dropped_zero_yield_rows"].items())
    return f"""---
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - agriculture
  - drought
  - pakistan
  - scikit-learn
---

# Crop yield model for drought scenarios in Pakistan (CT-378)

A scikit-learn pipeline that predicts the yield of 13 crops in Punjab, Sindh and Balochistan from annual climate and soil wetness, and ranks the crops by how much of a typical year's yield each keeps in a scenario. It is a university course project (CT-378 Machine Learning).

- **Model:** {chosen['label']} inside a scikit-learn `Pipeline` (median imputation, standard scaling, one-hot encoding, regressor fitted on the log of the yield).
- **Inputs (annual):** rain, mean/max/min temperature, evapotranspiration, solar radiation, rain hours, root-zone soil wetness (mean and wettest month), plus the crop and the province. Features: {", ".join(met["features"])}.
- **Output:** yield in *dataset yield units* (the data source does not state the unit).
- **Code and app:** the project repository (<https://github.com/SyedMuhammadHunain/Climate-Machine-Learning>) has a FastAPI app, a Streamlit app and this model's training code.

## Intended use

Studying how a simple model responds to drought-like scenarios, and demonstrating an end-to-end machine learning lifecycle. **Not** for real farm or policy decisions.

## Training data

Two public sources are merged by province and year:

{sources}

The source file has 12 monthly rows per crop, province and year with one yearly yield repeated on all 12, so the model trains on one row per crop, province and year: {d['yearly_rows_after_drop']} rows after dropping {dropped} rows with zero area (not grown or not reported). Split by year: train {split['train_years'][0]}-{split['train_years'][1]} ({d['n_train']} rows), validation {split['val_years'][0]}-{split['val_years'][1]} ({d['n_val']} rows), test {split['test_years'][0]}-{split['test_years'][1]} ({d['n_test']} rows).

## Training procedure

`GridSearchCV` with 5-fold `GroupKFold` grouped by year, so a year never appears in two folds. Candidates: Ridge, Random Forest, Gradient Boosting, plus two baselines. The model was chosen on the validation years; the test years were scored once and not used to choose. Random seed {met['seed']}. Trained {met['trained_at']} with scikit-learn {met['versions']['scikit_learn']}.

## Evaluation

Yields are in dataset yield units.

{_table(met)}

"Skill" compares a model with the crop-province mean baseline on a log scale; above 0 means better than that baseline. Read the baseline row and the skill column before treating small differences between crops as meaningful. Most of the model's signal is which crop it is.

### Prediction interval

The "likely range" is the empirical {pi['level']:.0%} spread (10% to 90% quantiles of log(actual / predicted)) of the model's errors on the validation years. On the {split['test_years'][0]}-{split['test_years'][1]} test rows it covered {pi['test_coverage']:.0%} of the actual yields, below the nominal {pi['level']:.0%}, so treat it as optimistic.

## How to use

Download the files of this repository, install `requirements.txt`, and run the example (it adds this folder to the Python path, so the bundled `climate_machine_learning` code is used):

```bash
pip install -r requirements.txt
python inference_example.py
```

```python
import json, joblib
from climate_machine_learning import inference as I

pipeline = joblib.load("crop_yield_pipeline.joblib")
reference = json.load(open("reference.json"))
metrics = json.load(open("metrics.json"))
result = I.predict_crops(
    pipeline, reference, "Sindh",
    {{"precipitation_sum": 150.0, "temperature_2m_mean": 27.5}},     # other inputs: province median
    log_quantiles=metrics["prediction_interval"]["log_residual_quantiles"],
)
for item in result.items:
    print(item.crop, round(item.pct_of_normal, 1), "% of a typical year")
```

Inputs outside the range seen in training are refused (`OutOfRangeError`) because a tree model cannot extrapolate. `reference.json` holds the trained ranges, province medians and real driest/wettest training years.

## Limitations

- Few rows: {d['yearly_rows_after_drop']} usable yearly rows from 22 years and 3 provinces.
- Crop identity dominates; weather and soil wetness change predictions only a little. The soil wetness source adds little.
- No yearly trend in the model, so growth in yields over time is not modelled; this affects the test years.
- Weather and soil wetness are province-level (soil wetness from 3 hand-picked farming-area points per province), the same for every crop. There is no district or farm detail.
- Annual mean temperature varies by only about 1.5 deg C within a province in training, so many inputs fall outside the trained range and are refused.
- Some crops are missing years (grapes, pomegranate, apple) and are noisier.
- The yield unit is not confirmed by the data source.

## Licence

The group has not chosen a licence for this model yet. Data licences: the crop dataset is MIT (stated by its repository); the NASA POWER catalog entry says "License not specified" (free to use, citation requested: <https://power.larc.nasa.gov/docs/referencing/>).

## References

- Pakistan crop yield vs climate, <https://github.com/Abdullah-Masood-05/pakistan-crop-yield-vs-climate> (origin: Climate & Agriculture in Pakistan, Zenodo, <https://zenodo.org/records/15486470>).
- NASA Langley Research Center POWER Project, <https://power.larc.nasa.gov/>.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.

Course project of a group of three students.
"""


EXAMPLE = '''"""Predict with the bundled model. Run:  python inference_example.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib

from climate_machine_learning import inference as I

pipeline = joblib.load(HERE / "crop_yield_pipeline.joblib")
reference = json.loads((HERE / "reference.json").read_text())
metrics = json.loads((HERE / "metrics.json").read_text())

result = I.predict_crops(
    pipeline, reference, "Sindh",
    {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5},   # every other input: province median
    crops=["rice"],
    log_quantiles=metrics["prediction_interval"]["log_residual_quantiles"],
)
item = result.items[0]
print(f"{item.crop} in Sindh: predicted yield {item.predicted_yield:.2f} dataset yield units "
      f"(likely range {item.low:.2f} to {item.high:.2f}), {item.pct_of_normal:.1f}% of a typical year")
'''


def space_readme(sdk_version: str) -> str:
    return f"""---
title: Crop yield under drought
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {sdk_version}
python_version: "3.13"
app_file: app.py
pinned: false
short_description: Model-predicted crop yields and ranking for drought scenarios in Pakistan
---

CT-378 Machine Learning project. Every number is output of a trained scikit-learn model, not adjusted by hand.
Model card and files: see the model repository linked from this Space. Licence: not chosen by the group yet.
"""


def _copy_code(dest: Path) -> None:
    (dest / "climate_machine_learning").mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(PACKAGE / name, dest / "climate_machine_learning" / name)


def build(out: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    met = json.loads((ROOT / "models" / "metrics.json").read_text())
    ref = json.loads((ROOT / "models" / "reference.json").read_text())
    pins = dict(line.split("==") for line in (ROOT / "hf_space" / "requirements.txt").read_text().splitlines()
                if "==" in line and not line.startswith("#"))

    model = out / "model"
    model.mkdir(parents=True)
    for name in MODEL_FILES:
        shutil.copy2(ROOT / "models" / name, model / name)
    _copy_code(model)
    (model / "README.md").write_text(model_card(met, ref), encoding="utf-8")
    (model / "inference_example.py").write_text(EXAMPLE, encoding="utf-8")
    (model / "requirements.txt").write_text(
        "".join(f"{k}=={pins[k]}\n" for k in ("scikit-learn", "pandas", "numpy", "joblib")), encoding="utf-8")

    space = out / "space"
    (space / "models").mkdir(parents=True)
    (space / "data" / "reference").mkdir(parents=True)
    for name in MODEL_FILES:
        shutil.copy2(ROOT / "models" / name, space / "models" / name)
    shutil.copy2(ROOT / "data" / "reference" / "fao_crop_water_needs.json", space / "data" / "reference")
    _copy_code(space)
    shutil.copy2(ROOT / "hf_space" / "app.py", space / "app.py")
    shutil.copy2(ROOT / "hf_space" / "requirements.txt", space / "requirements.txt")
    (space / "README.md").write_text(space_readme(pins["gradio"]), encoding="utf-8")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "hf_bundle"
    build(target)
    print(f"built {target}/model and {target}/space")
    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(target)}  {path.stat().st_size:,} bytes")
