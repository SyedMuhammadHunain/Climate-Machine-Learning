"""Static guard: frontend must not contain fake/fallback data.

Run: .venv/bin/python scripts/verify_frontend.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "climate-resilient-frontend"

BANNED = re.compile(r"bajra|jowar|renderFallback|fallback simulation|regional fallback", re.IGNORECASE)
PAGES = ["index.html", "predict.html", "metrics.html"]


def main() -> int:
    failures = []

    if (FRONTEND / "careers.html").exists():
        failures.append("careers.html still exists (dead copy of predict.html)")

    for name in PAGES:
        text = (FRONTEND / name).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if BANNED.search(line):
                failures.append(f"{name}:{lineno}: banned text: {line.strip()[:90]}")

    predict = (FRONTEND / "predict.html").read_text(encoding="utf-8")
    if "3500" in predict:
        failures.append("predict.html: old 3500 ms timeout still present")
    if "30000" not in predict:
        failures.append("predict.html: 30000 ms timeout missing")
    if 'id="btn-retry"' not in predict:
        failures.append("predict.html: Retry button missing")

    # T4: invented inputs and old semantics must be gone from the single-crop path
    for token in ("PRESET_CONFIGS", "field-area", "field-climate-index", "val + 6.0", "val >= 40.0",
                  "Total Expected Production", "Seasonal Precipitation", "is_drought_resilient",
                  "water_intensive_warnings", "ranked_drought_resilient", "kg/ha", "alert('"):
        if token in predict:
            failures.append(f"predict.html: old/invented logic still present: {token!r}")
    for token in ("api/v1/metadata", "pct_of_normal", "predicted_yield", "yield_unit", "inputs", "ranked_crops", "fao_water_need"):
        if token not in predict:
            failures.append(f"predict.html: expected T4 element missing: {token!r}")

    # T6: metrics page must read real results, not hold numbers or hand-made formulas
    metrics_page = (FRONTEND / "metrics.html").read_text(encoding="utf-8")
    for token in ("0.892", "32-year", "PROVINCE_DEFAULTS", "CROP_PHYSIOLOGY", "Elasticity", "elasticity",
                  "9.5 ms", "kg/ha", "Embedded Benchmarks", "1990", "17 features", "Kw"):
        if token in metrics_page:
            failures.append(f"metrics.html: hardcoded/old content present: {token!r}")
    for token in ("api/v1/metrics", "api/v1/metadata", "btn-retry"):
        if token not in metrics_page:
            failures.append(f"metrics.html: expected element missing: {token!r}")
    if re.search(r'class="metric-val"[^>]*>[^<]*\d', metrics_page):
        failures.append("metrics.html: a metric value is typed into the HTML")

    # T7: no demo/simulation wording or old hand-made concepts in any page
    for name in PAGES:
        text = (FRONTEND / name).read_text(encoding="utf-8").lower()
        for word in ("simulat", "demo", "dummy", "mock", "illustrat", "kg/ha", "physiolog", "elasticit",
                     "low-water, climate-resilient crops", "(low water)", "(high water)"):
            if word in text:
                failures.append(f"{name}: banned wording {word!r}")
    if "driest real year" not in predict.lower() or "scenarios" not in predict:
        failures.append("predict.html: quick-fill must use real-year scenarios from /metadata")
    if "training_rows_possible" not in predict:
        failures.append("predict.html: training row completeness not shown")

    # crop dropdown must list exactly the trained crops, with plain names
    ref_crops = json.loads((ROOT / "models" / "reference.json").read_text())["crops"]
    select = re.search(r'<select id="field-crop".*?</select>', predict, re.S).group(0)
    options = [v for v in re.findall(r'<option value="([^"]+)"', select) if v != "ALL"]
    if sorted(options) != sorted(ref_crops):
        failures.append(f"predict.html crop dropdown {sorted(options)} differs from trained crops {sorted(ref_crops)}")
    for value, label in re.findall(r'<option value="([^"]+)"[^>]*>([^<]*)</option>', select):
        if value != "ALL" and label.strip().lower() != value:
            failures.append(f"predict.html crop option label {label!r} is not the plain crop name {value!r}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: frontend has no fallback data, retry button present, 30 s timeout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
