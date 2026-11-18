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

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: frontend has no fallback data, retry button present, 30 s timeout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
