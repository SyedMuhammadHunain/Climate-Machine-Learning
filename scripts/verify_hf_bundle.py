"""Checks the Hugging Face bundle (model repo + Gradio Space) built by scripts/build_hf_bundle.py.

Needs gradio, gradio_client, matplotlib, pyyaml, so run it in a venv with hf_space/requirements.txt installed:
    pip install -r hf_space/requirements.txt pyyaml
    python scripts/verify_hf_bundle.py
Builds the bundle into a temp folder, then checks the model card, the Space config, file integrity,
and runs the Space app for real (server + client) against the trained model.
"""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import yaml

import build_hf_bundle as B
from climate_machine_learning import features as F
from climate_machine_learning import inference as I

MET = json.loads((ROOT / "models" / "metrics.json").read_text())
REF = json.loads((ROOT / "models" / "reference.json").read_text())
TMP = tempfile.TemporaryDirectory()
OUT = Path(TMP.name) / "hf_bundle"
B.build(OUT)
MODEL_DIR, SPACE_DIR = OUT / "model", OUT / "space"
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, f"{path} has no YAML front matter"
    return yaml.safe_load(m.group(1))


@test
def model_repo_has_the_files_a_user_needs():
    for name in ("README.md", "crop_yield_pipeline.joblib", "reference.json", "metrics.json", "inference_example.py",
                 "climate_machine_learning/features.py", "climate_machine_learning/inference.py",
                 "climate_machine_learning/__init__.py", "requirements.txt"):
        assert (MODEL_DIR / name).exists(), name
    for name in ("crop_yield_pipeline.joblib", "reference.json", "metrics.json"):
        assert sha(MODEL_DIR / name) == sha(ROOT / "models" / name), f"{name} differs from models/"


@test
def model_card_metadata_and_honest_content():
    meta = front_matter(MODEL_DIR / "README.md")
    assert meta["library_name"] == "sklearn" and meta["pipeline_tag"] == "tabular-regression"
    assert "license" not in meta, "licence is for the group to choose; the card must not pick one"
    text = (MODEL_DIR / "README.md").read_text(encoding="utf-8")
    for needle in ("## Intended use", "## Limitations", "## Training data", "## Evaluation", "## How to use",
                   "licence", "NASA POWER", "License not specified", "MIT", "2026-11-19"):
        assert needle.lower() in text.lower(), needle
    # evaluation numbers are generated from metrics.json, not typed
    chosen = MET["models"][MET["chosen"]["name"]]
    assert f"{chosen['test']['r2']:.3f}" in text
    base = MET["baselines"]["crop_province_mean"]["test"]
    assert f"{base['r2']:.3f}" in text, "the crop-average baseline must be in the card"
    assert f"{MET['prediction_interval']['test_coverage']:.0%}" in text, "interval coverage must be stated"
    for word in ("bajra", "0.892", "simulation", "fallback"):
        assert word not in text.lower(), word


@test
def space_config_is_a_native_gradio_space():
    meta = front_matter(SPACE_DIR / "README.md")
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py"
    req = (SPACE_DIR / "requirements.txt").read_text()
    pins = dict(line.split("==") for line in req.splitlines() if "==" in line and not line.startswith("#"))
    assert str(meta["sdk_version"]) == pins.get("gradio") or "gradio" not in pins
    for pkg in ("scikit-learn", "pandas", "numpy", "joblib", "matplotlib"):
        assert pkg in pins, f"{pkg} must be pinned"
    root_pins = dict(line.split("==") for line in (ROOT / "requirements.txt").read_text().splitlines() if "==" in line)
    for pkg in ("scikit-learn", "pandas", "numpy", "joblib"):
        assert pins[pkg] == root_pins[pkg], f"{pkg} pin differs from the pins the model was trained with"


@test
def space_is_self_contained_and_small_enough():
    for name in ("app.py", "models/crop_yield_pipeline.joblib", "models/reference.json", "models/metrics.json",
                 "data/reference/fao_crop_water_needs.json", "climate_machine_learning/inference.py"):
        assert (SPACE_DIR / name).exists(), name
    assert sha(SPACE_DIR / "models" / "crop_yield_pipeline.joblib") == sha(ROOT / "models" / "crop_yield_pipeline.joblib")
    biggest = max(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    assert biggest < 50 * 1024 * 1024, biggest


def load_space_app():
    sys.path.insert(0, str(SPACE_DIR))
    spec = importlib.util.spec_from_file_location("space_app", SPACE_DIR / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def space_functions_equal_the_model():
    app = load_space_app()
    q = MET["prediction_interval"]["log_residual_quantiles"]
    sc = REF["historical_years"]["Sindh"]["driest"]
    values = app.fill("Sindh", "Driest real year")
    assert np.isclose(values[0], sc["inputs"]["precipitation_sum"])
    assert np.isclose(values[7], sc["inputs"]["soil_wetness_mean"])
    summary, table, fig = app.predict("Sindh", "rice", *values)
    res = I.predict_crops(app.pipeline, REF, "Sindh", sc["inputs"], log_quantiles=q)
    item = next(i for i in res.items if i.crop == "rice")
    assert f"{item.predicted_yield:.2f}" in summary and f"{item.pct_of_normal:.1f}%" in summary
    assert list(table["Crop"]) == [i.crop for i in res.items]
    assert list(table["% of typical year"]) == [round(i.pct_of_normal, 1) for i in res.items]
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    assert f"{MET['prediction_interval']['test_coverage']:.0%}" in summary


@test
def space_refuses_out_of_range_input_with_the_range_in_the_message():
    app = load_space_app()
    values = list(app.fill("Sindh", "Typical (median) year"))
    values[0] = 5.0
    try:
        app.predict("Sindh", "rice", *values)
    except Exception as err:  # gradio.Error
        assert "precipitation_sum" in str(err) and "cannot extrapolate" in str(err), err
        return
    raise AssertionError("out-of-range input was accepted")


@test
def space_runs_for_real_through_a_gradio_server_and_client():
    from gradio_client import Client

    app = load_space_app()
    demo = app.build_demo()
    _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", server_port=7891, show_error=True, quiet=True)
    try:
        client = Client(url, verbose=False)
        values = app.fill("Punjab", "Typical (median) year")
        summary, table, _plot = client.predict("Punjab", "wheat", *values, api_name="/predict")
        assert "Predicted yield" in summary, summary[:200]
        assert len(table["data"]) == 13
    finally:
        demo.close()


@test
def model_repo_example_runs_with_plain_python():
    out = subprocess.run([sys.executable, "inference_example.py"], cwd=MODEL_DIR, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    q = MET["prediction_interval"]["log_residual_quantiles"]
    import joblib

    res = I.predict_crops(joblib.load(ROOT / "models" / "crop_yield_pipeline.joblib"), REF, "Sindh",
                          {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5}, crops=["rice"], log_quantiles=q)
    assert f"{res.items[0].predicted_yield:.2f}" in out.stdout, out.stdout


def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(err).__name__}: {str(err)[:300]}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    TMP.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
