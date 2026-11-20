"""Checks docs/guides/demo_and_viva_guide.md against the live app.

Every number the demo script promises is recomputed from the API and must appear in the guide.
Run: .venv/bin/python scripts/checks/verify_guide.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guides" / "demo_and_viva_guide.md"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))


def main() -> int:
    if not GUIDE.exists():
        print("FAILED:\n  - docs/guides/demo_and_viva_guide.md does not exist")
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
        driest = client.post("/api/v1/recommend", json={"province": "Sindh", **dry["inputs"]}).json()
        rec = driest["ranked_crops"]
        need(f"Driest real year ({dry['year']})", "preset label")
        need(f"{dry['inputs']['precipitation_sum']:.1f} mm", "driest-year rain")
        # the page groups crops that show the same share: one rank per group (SPEC-responsive.md R5, R8)
        for group in driest["groups"]:
            need(f"{group['pct_of_normal']:.1f}%", f"driest ranking group at rank {group['rank']}")
            if len(group["crops"]) == 1:
                need(f"{group['crops'][0]} {group['pct_of_normal']:.1f}%", f"driest ranking row rank {group['rank']}")
            else:
                need(f"{len(group['crops'])} crops tied", f"the tie group at rank {group['rank']} must be described as the page words it")
        if any(i["barely_grown"] for i in rec):
            need("barely grown here", "the page tags barely grown crops (apple in Sindh)")
        need(driest["headline"].split(" keeps ")[0].lower(), "the headline names the best crop")

        # --- demo step: typical year (the preset fills all nine inputs with the exact province medians)
        med = client.post("/api/v1/recommend", json={"province": "Sindh", **ref}).json()
        assert len(med["groups"]) == 1, "the typical year is one tie group"
        need(f"{med['groups'][0]['pct_of_normal']:.1f}%", "typical-year share")
        need(f"{len(med['ranked_crops'])} crops tied", "typical-year tie group")
        need("no difference between crops", "the headline the page shows for the typical year")

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
    dep = ROOT / "docs" / "guides" / "deployment_guide.md"
    dep_text = dep.read_text(encoding="utf-8") if dep.exists() else ""
    if not dep_text:
        failures.append("docs/guides/deployment_guide.md does not exist")
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
    spread = metrics["pinned_year_check"]
    need(f"{spread['ratio_min']['value']:.2f}", "the pinned-year spread: lowest ratio")
    need(f"{spread['ratio_max']['value']:.2f}", "the pinned-year spread: highest ratio")
    need("point number", "the guide must say the point yield is rough")
    need("crops tied", "the pages group tied crops (the guide may only claim what the pages do)")
    need("Start", "the landing page has one Start button")
    need("crop page", "the guide shows the page of one crop")
    need("verify_responsive.py", "the guide says how phone and large-screen layouts are checked")
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

    # --- "Who answers what": every topic row names its owner from scripts/lib/group_roles.py
    from group_roles import TOPIC_OWNERS, PLACEHOLDER
    if PLACEHOLDER in text:
        failures.append(f"'Who answers what' still has {PLACEHOLDER!r}")
    for topic, owner in TOPIC_OWNERS.items():
        need(f"| {topic} | {owner} | {owner} |", "who-answers-what row (scripts/lib/group_roles.py)")

    low = text.lower()
    for platform in ("streamlit", "vercel", "hugging face"):
        if platform not in low:
            failures.append(f"deployment checklist missing {platform}")
    if not re.search(r"- \[x\][^\n]*streamlit[^\n]*streamlit\.app", low):
        failures.append("Streamlit checklist item must be ticked and show the live streamlit.app URL")
    if not re.search(r"- \[x\][^\n]*vercel[^\n]*vercel\.app", low):
        failures.append("Vercel checklist item must be ticked and show the live vercel.app URL")
    hf_item = re.search(r"- \[(x| )\][^\n]*hugging face[^\n]*", low)
    if not hf_item or hf_item.group(1) != "x" or "huggingface.co/smhunain/crop-yield-drought-pakistan" not in hf_item.group(0) \
            or "group decision" not in hf_item.group(0) or "earlier model" in hf_item.group(0):
        failures.append("Hugging Face checklist item must be ticked, link the live model repo and say no Space by group decision")
    if not re.search(r"- \[x\][^\n]*report pdf", low):
        failures.append("Report PDF checklist item must be ticked (docs/report/report.pdf exists and is checked)")
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
