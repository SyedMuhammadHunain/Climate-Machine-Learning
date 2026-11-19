"""Headless checks of the Streamlit app (streamlit_app/app.py) with Streamlit's AppTest runner.

Needs Streamlit, so run it in a venv that has streamlit_app/requirements.txt installed:
    pip install -r streamlit_app/requirements.txt
    python scripts/verify_streamlit.py [path/to/app.py]     # default: streamlit_app/app.py

Every number on screen must equal what the trained pipeline predicts.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("streamlit is not installed: pip install -r streamlit_app/requirements.txt")
    sys.exit(2)

import joblib
import numpy as np

from climate_machine_learning import features as F
from climate_machine_learning import inference as I

APP = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "streamlit_app" / "app.py"
BASE = APP.parent if (APP.parent / "models").exists() else ROOT
PIPE = joblib.load(BASE / "models" / "crop_yield_pipeline.joblib")
REF = json.loads((BASE / "models" / "reference.json").read_text())
MET = json.loads((BASE / "models" / "metrics.json").read_text())
Q = MET["prediction_interval"]["log_residual_quantiles"]
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def fresh() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180).run()


def number_input(at, starts):
    matches = [w for w in at.number_input if w.label.startswith(starts)]
    assert matches, f"no number input starting with {starts!r}: {[w.label for w in at.number_input]}"
    return matches[0]


def metric_values(at):
    return {m.label: m.value for m in at.metric}


@test
def default_run_has_no_errors_and_a_title():
    at = fresh()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title and "crop" in at.title[0].value.lower()


@test
def default_numbers_equal_the_model():
    at = fresh()
    province = at.selectbox(key="province").value
    crop = at.selectbox(key="crop").value.lower()
    med = REF["provinces"][province]["median"]
    supplied = {"precipitation_sum": round(med["precipitation_sum"], 1),
                "temperature_2m_mean": round(med["temperature_2m_mean"], 1)}
    res = I.predict_crops(PIPE, REF, province, supplied, log_quantiles=Q)
    item = next(i for i in res.items if i.crop == crop)
    vals = metric_values(at)
    assert vals["Predicted yield"] == f"{item.predicted_yield:.2f}", vals
    assert vals["Share of a typical year"] == f"{item.pct_of_normal:.1f}%", vals
    assert vals["Likely range (nominal 80%)"] == f"{item.low:.2f} to {item.high:.2f}", vals


@test
def real_driest_year_fills_every_input_and_ranking_matches():
    at = fresh()
    at.selectbox(key="province").select("Sindh").run()
    at.radio(key="preset").set_value("Driest real year").run()
    assert not at.exception, at.exception
    sc = REF["historical_years"]["Sindh"]["driest"]
    assert np.isclose(number_input(at, "Annual rain").value, sc["inputs"]["precipitation_sum"])
    assert np.isclose(number_input(at, "Root-zone soil wetness, yearly mean").value, sc["inputs"]["soil_wetness_mean"])
    res = I.predict_crops(PIPE, REF, "Sindh", sc["inputs"], log_quantiles=Q)
    df = at.dataframe[0].value
    assert len(df) == 13
    assert list(df["Crop"]) == [i.crop for i in res.items], list(df["Crop"])
    assert list(df["% of typical year"]) == [round(i.pct_of_normal, 1) for i in res.items]
    ties = I.tied_with(res.items)
    assert list(df["Tied with"]) == [ties[i.crop] for i in res.items]
    assert max(ties.values()) >= 2, "this demo ranking has ties; the table must mark them"
    crop = at.selectbox(key="crop").value.lower()
    item = next(i for i in res.items if i.crop == crop)
    assert metric_values(at)["Predicted yield"] == f"{item.predicted_yield:.2f}"


@test
def number_inputs_are_limited_to_the_trained_range():
    at = fresh()
    at.selectbox(key="province").select("Sindh").run()
    rain = number_input(at, "Annual rain")
    lo, hi = REF["provinces"]["Sindh"]["range"]["precipitation_sum"]
    assert np.isclose(rain.min, lo) and np.isclose(rain.max, hi), (rain.min, rain.max, lo, hi)
    soil = number_input(at, "Root-zone soil wetness, yearly mean")
    slo, shi = REF["provinces"]["Sindh"]["range"]["soil_wetness_mean"]
    assert np.isclose(soil.min, slo) and np.isclose(soil.max, shi)


@test
def interval_is_explained_with_the_measured_test_coverage():
    at = fresh()
    text = " ".join(c.value for c in at.caption)
    cov = MET["prediction_interval"]["test_coverage"]
    assert f"{cov:.0%}" in text, text
    assert "validation" in text.lower() and "2020-2022" in text


@test
def table_shows_fao_context_only_where_fao_lists_the_crop():
    at = fresh()
    df = at.dataframe[0].value
    rice = df[df["Crop"] == "rice"].iloc[0]
    apple = df[df["Crop"] == "apple"].iloc[0]
    assert "450-700" in rice["FAO water need"] and "high" in rice["FAO water need"]
    assert "not listed" in apple["FAO water need"]
    assert {"Rank", "Crop", "% of typical year", "Predicted yield", "Likely range", "Training rows"} <= set(df.columns)


@test
def reruns_do_not_grow_sys_path():
    src = str(BASE / "src") if (BASE / "src" / "climate_machine_learning").is_dir() else str(BASE)
    at = fresh()
    before = sys.path.count(src)
    for _ in range(3):
        at.run()
    assert sys.path.count(src) == before, f"sys.path grew on rerun: {before} -> {sys.path.count(src)}"


@test
def one_explanatory_chart_is_shown():
    at = fresh()
    charts = at.get("arrow_vega_lite_chart") + at.get("vega_lite_chart")
    assert len(charts) == 1, f"expected exactly one chart, found {len(charts)}"


@test
def point_yield_comes_with_the_pinned_year_caveat():
    """Review R1: the predicted yield on screen is never shown without the pinned-year spread."""
    at = fresh()
    text = " ".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert I.absolute_yield_caveat(MET) in text, text[-600:]


@test
def about_section_states_the_win_with_the_caveat():
    """SPEC-promote-offset-model.md: the win is never shown without the log-MAE loss and the after-the-fact choice."""
    at = fresh()
    text = " ".join(m.value for m in at.markdown)
    assert I.model_caveat(MET) in text, text[-900:]
    ch = MET["chosen"]
    assert ch["beats_by_metric"]["log_mae"] is False and "log-MAE" in text
    assert "after seeing the test results" in text
    assert "no trend over time" not in text, "the old 'no trend' sentence is no longer true"
    assert f"{MET['inference_year']}" in text, "the pinned year must be stated"


@test
def app_source_has_no_fake_data_and_no_hardcoded_model_numbers():
    text = APP.read_text(encoding="utf-8")
    for word in ("bajra", "jowar", "fallback", "simulat", "dummy", "mock", "elasticit", "0.892"):
        assert word not in text.lower(), word
    tree = ast.parse(text)
    in_call = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "number_input", "set_page_config", "metric", "column_config"):
            in_call.update(id(n) for n in ast.walk(node))
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, float) and id(n) not in in_call and n.value not in (0.0, 1.0, 100.0)]
    assert not bad, f"hardcoded numbers in the app: {bad}"


def main() -> int:
    if not APP.exists():
        print(f"FAILED: {APP} does not exist")
        return 1
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(err).__name__}: {str(err)[:300]}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
