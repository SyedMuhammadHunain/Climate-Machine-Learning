"""Checks docs/demo_and_viva_guide.md against the live app.

Every number the demo script promises is recomputed from the API and must appear in the guide.
Run: .venv/bin/python scripts/verify_guide.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "demo_and_viva_guide.md"
sys.path.insert(0, str(ROOT))


def main() -> int:
    if not GUIDE.exists():
        print("FAILED:\n  - docs/demo_and_viva_guide.md does not exist")
        return 1
    text = GUIDE.read_text(encoding="utf-8")
    failures = []

    def need(snippet: str, why: str):
        if snippet not in text:
            failures.append(f"{why}: expected {snippet!r}")

    from fastapi.testclient import TestClient
    from app.main import app

    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    with TestClient(app) as client:
        meta = client.get("/api/v1/metadata").json()
        scen = {(s["province"], s["kind"]): s for s in meta["scenarios"]}
        dry = scen[("Sindh", "driest")]
        ref = meta["reference"]["Sindh"]["median"]

        # --- demo step: driest real Sindh year, all crops
        rec = client.post("/api/v1/recommend", json={"province": "Sindh", **dry["inputs"]}).json()["ranked_crops"]
        need(f"Driest real year ({dry['year']})", "preset label")
        need(f"{dry['inputs']['precipitation_sum']:.1f} mm", "driest-year rain")
        for item in rec[:3] + rec[-1:]:
            need(f"{item['crop']} {item['pct_of_normal']:.1f}%", f"driest ranking row rank {item['rank']}")
        shares = [round(i["pct_of_normal"], 1) for i in rec]
        if len(set(shares)) < len(shares):
            need("tied", "the demo ranking has crops with equal shares, so the guide must say so")

        # --- demo step: typical year (median preset: rain and mean temperature only)
        med = client.post("/api/v1/recommend", json={
            "province": "Sindh",
            "precipitation_sum": round(ref["precipitation_sum"], 1),
            "temperature_2m_mean": round(ref["temperature_2m_mean"], 1),
        }).json()["ranked_crops"]
        need(f"{med[0]['crop']} {med[0]['pct_of_normal']:.1f}%", "typical-year top crop")
        need(f"{med[-1]['crop']} {med[-1]['pct_of_normal']:.1f}%", "typical-year last crop")

        # --- demo step: single crop
        one = client.post("/api/v1/predict", json={"province": "Sindh", "crop": "rice", **dry["inputs"]}).json()
        need(f"{one['predicted_yield']:.2f}", "rice predicted yield")
        need(f"{one['normal_yield']:.2f}", "rice typical yield")
        need(f"{one['pct_of_normal']:.1f}%", "rice % of typical")

        # --- demo step: refusal
        bad = client.post("/api/v1/predict", json={
            "province": "Sindh", "crop": "rice", "precipitation_sum": 5, "temperature_2m_mean": 27.8}).json()
        need(bad["message"], "422 message")

    # --- deployment guide: the numbers it tells you to expect on the live apps
    dep = ROOT / "docs" / "deployment_guide.md"
    dep_text = dep.read_text(encoding="utf-8") if dep.exists() else ""
    if not dep_text:
        failures.append("docs/deployment_guide.md does not exist")
    for snippet, why in ((f"{one['predicted_yield']:.2f}", "rice yield in deployment guide"),
                         (f"{one['pct_of_normal']:.1f}%", "rice share in deployment guide"),
                         (f"{metrics['prediction_interval']['test_coverage']:.0%}", "interval coverage in deployment guide")):
        if snippet not in dep_text:
            failures.append(f"{why}: expected {snippet!r}")
    need(f"{metrics['prediction_interval']['test_coverage']:.0%}", "interval coverage (viva: how confident is a prediction)")
    need("(deployment_guide.md)", "link to the deployment guide")

    # --- results part
    chosen = metrics["models"][metrics["chosen"]["name"]]
    need(f"{chosen['test']['r2']:.3f}", "chosen model test R2")
    base = metrics["baselines"]["crop_province_mean"]["test"]
    need(f"{base['r2']:.3f}", "baseline test R2 (viva needs the straight answer)")
    # SPEC-promote-offset-model.md: the win comes with the loss, the trend-only row and the after-the-fact choice
    need(f"{chosen['test']['mae']:.1f}", "chosen model test MAE")
    need(f"{base['mae']:.1f}", "baseline test MAE")
    need(f"{metrics['baselines']['crop_province_mean_plus_trend']['test']['r2']:.3f}", "trend-only baseline test R2")
    need(f"{chosen['test']['log_mae']:.3f}", "chosen model test log-MAE")
    need("tied with", "the pages mark tied crops (the guide may only claim what the pages do)")
    need("log-MAE", "the log-MAE loss")
    need("after seeing the test results", "the metric choice came after the test results")
    for stale in ("0.794", "no yearly trend", "random forest, scores"):
        if stale in text.lower():
            failures.append(f"stale claim about the earlier model: {stale!r}")

    # --- structure
    for heading in ("## Before you present", "## Part 1", "## Part 2", "## Part 3", "## Viva preparation",
                    "### Data and EDA", "### Pipeline, leakage and tuning", "### Results and the baseline",
                    "### Backend and API", "### Who answers what"):
        need(heading, "section")
    minutes = [int(m) for m in re.findall(r"^## Part \d[^\n]*\((\d+) min\)", text, flags=re.M)]
    if minutes != [3, 4, 3]:
        failures.append(f"timing must be 3 / 4 / 3 minutes, found {minutes}")
    for name in ("Muhammad Afzal", "Syed Muhammad Hunain", "Ayesha"):
        need(name, "group member")

    low = text.lower()
    for platform in ("streamlit", "vercel", "hugging face"):
        if platform not in low:
            failures.append(f"deployment checklist missing {platform}")
    if not re.search(r"- \[x\][^\n]*streamlit[^\n]*streamlit\.app", low):
        failures.append("Streamlit checklist item must be ticked and show the live streamlit.app URL")
    if not re.search(r"- \[x\][^\n]*vercel[^\n]*vercel\.app", low):
        failures.append("Vercel checklist item must be ticked and show the live vercel.app URL")
    if not re.search(r"- \[ \][^\n]*hugging face", low):
        failures.append("Hugging Face checklist item must stay open (the Space is not deployed)")
    for word in ("bajra", "jowar"):
        if word in low:
            failures.append(f"banned wording: {word}")

    for target in re.findall(r"\]\((?!https?://|#)([^)\s]+)\)", text):
        if not (GUIDE.parent / target.split("#")[0]).exists():
            failures.append(f"broken relative link: {target}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: guide numbers match the live app, timing is 3/4/3, sections and checklist present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
