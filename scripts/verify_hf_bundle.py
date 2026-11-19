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
    # SPEC-promote-offset-model.md: the win is stated with the log-MAE loss and the after-the-fact metric choice
    assert I.model_caveat(MET) in text, "the card must carry the generated caveat"
    assert I.absolute_yield_caveat(MET) in text, "the card must carry the pinned-year caveat"
    trend = MET["baselines"]["crop_province_mean_plus_trend"]["test"]
    assert f"{trend['r2']:.3f}" in text, "the trend-only baseline row must be in the card"
    assert "no yearly trend" not in text.lower(), "stale limitation"
    assert "gap" in text and str(MET["inference_year"]) in text, "card must say how year is handled"
    for word in ("bajra", "0.892", "simulation", "fallback"):
        assert word not in text.lower(), word


@test
def public_pages_contain_no_roll_numbers_or_secrets():
    """The Hub pages are public: no student roll numbers, no access tokens."""
    for path in list(MODEL_DIR.rglob("*.md")) + list(SPACE_DIR.rglob("*.md")) + [SPACE_DIR / "app.py", MODEL_DIR / "inference_example.py"]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"CT-24\d{3}", text), f"roll number in {path.name}"
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", text), f"token-like string in {path.name}"


@test
def build_refuses_a_folder_it_did_not_create():
    """Abuse case: `--out` pointed at a folder that holds someone's files must never delete them."""
    with tempfile.TemporaryDirectory() as tmp:
        victim = Path(tmp) / "my_documents"
        (victim / "important_subdir").mkdir(parents=True)
        (victim / "notes.txt").write_text("precious")
        (victim / "important_subdir" / "data.csv").write_text("more")
        (victim / "model").mkdir()                       # even a folder named like ours must not be deleted
        (victim / "model" / "mine.txt").write_text("not yours")
        try:
            B.build(victim)
        except B.UnsafeOutputDir as err:
            assert "not empty" in str(err) and "Refusing" in str(err), err
        else:
            raise AssertionError("build() accepted a foreign, non-empty folder")
        assert (victim / "notes.txt").read_text() == "precious"
        assert (victim / "important_subdir" / "data.csv").exists() and (victim / "model" / "mine.txt").exists()


@test
def build_rebuilds_its_own_folder_and_touches_only_model_and_space():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bundle"
        B.build(out)
        (out / "model" / "stale_file.txt").write_text("left over from an older build")
        (out / "README_mine.txt").write_text("a foreign file next to our folders")
        B.build(out)                                     # rebuild into our own folder is allowed
        assert not (out / "model" / "stale_file.txt").exists(), "rebuild must replace model/ completely"
        assert (out / "README_mine.txt").exists(), "only model/ and space/ may be removed"
        assert (out / "model" / "README.md").exists() and (out / "space" / "app.py").exists()
        empty = Path(tmp) / "empty_dir"
        empty.mkdir()
        B.build(empty)                                   # an existing EMPTY folder is fine
        assert (empty / "model" / "README.md").exists()


@test
def build_refuses_symlinked_targets():
    with tempfile.TemporaryDirectory() as tmp:
        out, elsewhere = Path(tmp) / "bundle", Path(tmp) / "elsewhere"
        B.build(out)
        elsewhere.mkdir()
        (elsewhere / "keep.txt").write_text("outside the bundle")
        import shutil
        shutil.rmtree(out / "model")
        (out / "model").symlink_to(elsewhere, target_is_directory=True)
        try:
            B.build(out)
        except B.UnsafeOutputDir as err:
            assert "symlink" in str(err), err
        else:
            raise AssertionError("build() followed a symlink")
        assert (elsewhere / "keep.txt").read_text() == "outside the bundle"


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
                 "data/reference/fao_crop_water_needs.json", "climate_machine_learning/inference.py",
                 "climate_machine_learning/presentation.py"):
        assert (SPACE_DIR / name).exists(), name
    # the model repo is already published and does not need UI code: it must stay as it is
    assert not (MODEL_DIR / "climate_machine_learning" / "presentation.py").exists()
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
    ties = I.tied_with(res.items)
    assert list(table["Tied with"]) == [ties[i.crop] for i in res.items]
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    assert f"{MET['prediction_interval']['test_coverage']:.0%}" in summary
    assert I.model_caveat(MET) in summary, "the win must come with the log-MAE loss and the after-the-fact note"
    assert I.absolute_yield_caveat(MET) in summary, "the predicted yield must come with the pinned-year caveat"


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
    # the model repo's requirements.txt has no web server: run the example with uvicorn blocked, as a stranger's env would be
    runner = "import sys, runpy; sys.modules['uvicorn'] = None; runpy.run_path('inference_example.py', run_name='__main__')"
    out = subprocess.run([sys.executable, "-c", runner], cwd=MODEL_DIR, capture_output=True, text=True, timeout=120)
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
