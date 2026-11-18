"""Static guard: frontend must not contain fake/fallback data.

Run: .venv/bin/python scripts/verify_frontend.py
"""
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
                  "Total Expected Production", "Seasonal Precipitation", "is_drought_resilient"):
        if token in predict:
            failures.append(f"predict.html: old/invented logic still present: {token!r}")
    for token in ("api/v1/metadata", "pct_of_normal", "predicted_yield", "yield_unit", "inputs"):
        if token not in predict:
            failures.append(f"predict.html: expected T4 element missing: {token!r}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: frontend has no fallback data, retry button present, 30 s timeout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
