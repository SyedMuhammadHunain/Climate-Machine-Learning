"""Checks the report figures (and, later, the report PDF) against the real project files.

Run:   .venv/bin/python scripts/verify_report.py
Plain asserts; prints PASS/FAIL per test and exits non-zero on any failure.
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "report" / "figures"
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())

EDA_FIGURES = [
    "eda_yield_distribution.png",
    "eda_missing_rows.png",
    "eda_outliers_by_crop.png",
    "eda_correlation.png",
    "eda_rows_per_crop_province.png",
    "eda_rain_vs_yield_gap.png",
]
RESULT_FIGURES = ["results_model_comparison.png", "results_predicted_vs_actual.png"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _figure_values() -> dict:
    return json.loads((FIG_DIR / "figure_values.json").read_text())


def figures_exist_with_at_least_five_eda_plots():
    for name in EDA_FIGURES + RESULT_FIGURES:
        p = FIG_DIR / name
        assert p.exists(), f"missing figure {p}"
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {name}"
    assert len(EDA_FIGURES) >= 5


def comparison_figure_values_equal_metrics_json():
    vals = _figure_values()["model_comparison"]
    expected = {}
    for k, v in METRICS["baselines"].items():
        expected[k] = v["test"]
    for k, v in METRICS["models"].items():
        expected[k] = v["test"]
    assert set(vals) == set(expected), f"rows differ: {sorted(vals)} vs {sorted(expected)}"
    for k, row in vals.items():
        for metric in ("r2", "mae"):
            assert math.isclose(row[metric], expected[k][metric], rel_tol=0, abs_tol=1e-12), (k, metric)


def predicted_vs_actual_reproduces_saved_test_score():
    v = _figure_values()["predicted_vs_actual"]
    chosen = METRICS["models"][METRICS["chosen"]["name"]]["test"]
    assert v["n"] == METRICS["data"]["n_test"], (v["n"], METRICS["data"]["n_test"])
    assert math.isclose(v["r2"], chosen["r2"], abs_tol=1e-9), (v["r2"], chosen["r2"])
    assert math.isclose(v["mae"], chosen["mae"], abs_tol=1e-9), (v["mae"], chosen["mae"])


def figure_build_is_deterministic():
    names = EDA_FIGURES + RESULT_FIGURES + ["figure_values.json"]
    before = {n: _sha(FIG_DIR / n) for n in names}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_report_figures.py")], check=True,
                   capture_output=True)
    after = {n: _sha(FIG_DIR / n) for n in names}
    changed = [n for n in names if before[n] != after[n]]
    assert not changed, f"figures changed on rebuild: {changed}"


SCREENSHOTS = ["deploy_streamlit.png", "deploy_vercel.png", "deploy_local.png"]


def _expected_sindh_driest_rice() -> dict:
    """What the saved model says now for rice, Sindh, driest real year (through the real API code)."""
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        meta = client.get("/api/v1/metadata").json()
        dry = next(s for s in meta["scenarios"] if s["province"] == "Sindh" and s["kind"] == "driest")
        ranked = client.post("/api/v1/recommend", json={"province": "Sindh", **dry["inputs"]}).json()["ranked_crops"]
    rice = next(r for r in ranked if r["crop"] == "rice")
    return {"yield": f"{rice['predicted_yield']:.2f}", "pct": f"{rice['pct_of_normal']:.1f}%", "year": dry["year"]}


def deployment_screenshots_show_current_model_values():
    checks = json.loads((FIG_DIR / "screenshot_checks.json").read_text())
    exp = _expected_sindh_driest_rice()
    assert set(checks) == {s.removeprefix("deploy_").removesuffix(".png") for s in SCREENSHOTS}, sorted(checks)
    for name in SCREENSHOTS:
        p = FIG_DIR / name
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"missing screenshot {name}"
    for key, c in checks.items():
        assert c["expected"] == exp, f"{key}: captured against {c['expected']}, model now gives {exp}; recapture"
        assert c["found_on_page"] == [exp["yield"], exp["pct"]], f"{key}: page text did not show {exp}"


REPORT_DIR = ROOT / "docs" / "report"
PDF = REPORT_DIR / "report.pdf"
HTML = REPORT_DIR / "report.html"
TABLE_LABELS = {  # short row labels used in the report's comparison table
    "global_mean": "Baseline: overall mean",
    "crop_province_mean": "Baseline: crop-province mean",
    "crop_province_mean_plus_trend": "Baseline: mean + year trend",
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def _pdf_text(layout: bool = False) -> str:
    args = ["pdftotext"] + (["-layout"] if layout else []) + [str(PDF), "-"]
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _flat(text: str) -> str:
    """Whitespace and hyphens removed, so a sentence wrapped at a space or at a hyphen (pdftotext drops a
    hyphen at a line end, "2020-" + "2022" becomes "20202022") compares equal to its one-line form."""
    return "".join(ch for ch in text if not ch.isspace() and ch not in "-\u00ad\u2010\u2011")


def report_pdf_exists():
    assert PDF.exists() and PDF.read_bytes()[:5] == b"%PDF-", f"missing {PDF}"
    assert HTML.exists(), f"missing {HTML}"


def report_results_table_equals_metrics_json():
    rows = {}
    for key, v in METRICS["baselines"].items():
        rows[key] = v
    for key, v in METRICS["models"].items():
        rows[key] = v
    assert set(rows) == set(TABLE_LABELS), sorted(rows)
    lines = _pdf_text(layout=True).splitlines()
    for key, label in TABLE_LABELS.items():
        hits = [ln.strip() for ln in lines if ln.strip().startswith(label)]
        assert len(hits) == 1, f"table row {label!r} found {len(hits)} times"
        v = rows[key]
        want = [f"{v['val']['r2']:.3f}", f"{v['val']['mae']:.1f}", f"{v['test']['r2']:.3f}",
                f"{v['test']['mae']:.1f}", f"{v['test']['rmse']:.1f}", f"{v['test']['log_mae']:.3f}"]
        got = hits[0][len(label):].replace("(chosen)", "").split()
        assert got == want, f"{label}: PDF shows {got}, metrics.json gives {want}"


def report_states_the_caveats():
    sys.path.insert(0, str(ROOT / "src"))
    from climate_machine_learning import inference as I

    text = _flat(_pdf_text())
    for sentence in (I.model_caveat(METRICS), I.absolute_yield_caveat(METRICS)):
        assert _flat(sentence) in text, f"caveat missing from the PDF: {sentence[:90]}..."
    pi = METRICS["prediction_interval"]
    assert _flat(f"covered {pi['test_coverage']:.0%} of the test rows") in text, "likely-range coverage missing"
    assert _flat("dataset yield units") in text


TESTS = [
    figures_exist_with_at_least_five_eda_plots,
    comparison_figure_values_equal_metrics_json,
    predicted_vs_actual_reproduces_saved_test_score,
    figure_build_is_deterministic,
    deployment_screenshots_show_current_model_values,
    report_pdf_exists,
    report_results_table_equals_metrics_json,
    report_states_the_caveats,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
