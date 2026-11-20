"""Checks README.md against the real project files.

Run:   .venv/bin/python scripts/checks/verify_readme.py          # check
       .venv/bin/python scripts/checks/verify_readme.py --write  # refresh the metrics block from models/metrics.json
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))


def metrics_block(m: dict) -> str:
    """Markdown generated only from models/metrics.json."""
    d = m["data"]
    lines = [
        START,
        f"Train {m['split']['train_years'][0]}-{m['split']['train_years'][1]} ({d['n_train']} rows), "
        f"validation {m['split']['val_years'][0]}-{m['split']['val_years'][1]} ({d['n_val']} rows), "
        f"test {m['split']['test_years'][0]}-{m['split']['test_years'][1]} ({d['n_test']} rows). "
        "Yields are in dataset yield units.",
        "",
        "| Model | Val R² | Val MAE | Test R² | Test MAE | Test RMSE | Test skill vs crop-province mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"global_mean": "Baseline: overall mean yield", "crop_province_mean": "Baseline: crop-province mean yield",
             "crop_province_mean_plus_trend": "Baseline: crop-province mean + year trend (no climate)"}
    rows = [(names.get(k, k), v["val"], v["test"], False) for k, v in m["baselines"].items()]
    rows += [(v["label"], v["val"], v["test"], k == m["chosen"]["name"]) for k, v in m["models"].items()]
    for name, val, test, chosen in rows:
        label = f"**{name} (chosen)**" if chosen else name
        lines.append(
            f"| {label} | {val['r2']:.3f} | {val['mae']:.1f} | {test['r2']:.3f} | {test['mae']:.1f} | "
            f"{test['rmse']:.1f} | {test['skill_vs_crop_province_mean_log']:+.2f} |"
        )
    from climate_machine_learning import inference as I

    pi = m["prediction_interval"]
    verdict = "optimistic" if pi["test_coverage"] < pi["level"] else "about right"
    lines += [
        "",
        I.model_caveat(m),
        "",
        I.absolute_yield_caveat(m),
        "",
        f"Likely range shown in the Streamlit and Hugging Face apps: nominal {pi['level']:.0%} "
        f"(x{math.exp(pi['log_residual_quantiles'][0]):.2f} to x{math.exp(pi['log_residual_quantiles'][1]):.2f} "
        f"of the prediction), from the validation-year errors. It covered {pi['test_coverage']:.0%} of the "
        f"{m['split']['test_years'][0]}-{m['split']['test_years'][1]} test rows, so it is {verdict}.",
        "",
        f"Chosen on the validation years: **{m['models'][m['chosen']['name']]['label']}**. "
        f"Trained {m['trained_at']} with scikit-learn {m['versions']['scikit_learn']}, "
        f"data SHA-256 `{d['sha256'][:16]}...`.",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    text = README.read_text(encoding="utf-8") if README.exists() else ""

    if "--write" in sys.argv:
        block = metrics_block(metrics)
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        else:
            print("README has no metrics markers; add them first.")
            return 1
        README.write_text(text, encoding="utf-8")
        print("metrics block refreshed")
        return 0

    failures = []
    if START not in text or END not in text:
        failures.append("metrics markers missing")
    else:
        current = text[text.index(START): text.index(END) + len(END)]
        if current != metrics_block(metrics):
            failures.append("metrics block differs from models/metrics.json (run with --write)")

    limits = text[text.index("## Limitations"): text.index("## References")] if "## Limitations" in text and "## References" in text else ""
    for needle in ("log-MAE", "after seeing the test results", "year trend"):
        if needle not in limits:
            failures.append(f"Limitations must mention {needle!r} (SPEC-promote-offset-model.md)")
    for stale in ("no yearly trend", "year is not an input, so growth"):
        if stale in text.lower():
            failures.append(f"stale limitation about the year trend: {stale!r}")
    ranking = text[text.index("## How the ranking works"): text.index("## API")] if "## How the ranking works" in text and "## API" in text else ""
    if "tied_with" not in ranking or "tied with" not in ranking:
        failures.append("How the ranking works must say tied crops are marked (`tied_with` in the API, 'tied with N other crops' on the pages)")
    method = text[text.index("## Method"): text.index("## Results")] if "## Method" in text and "## Results" in text else ""
    if "gap" not in method or "crop-province average" not in method:
        failures.append("Method must say the models learn the gap from the crop-province average")
    if "R² and MAE" not in text[text.index("## Results"): text.index("## How the ranking works")]:
        failures.append("Results must say the win is judged on R² and MAE")

    from group_roles import MEMBERS
    for name, roll, role in MEMBERS:
        if f"| {name} | {roll} | {role} |" not in text:
            failures.append(f"README group table row missing or wrong (scripts/lib/group_roles.py): | {name} | {roll} | {role} |")

    for heading in ("## Quick start", "## Data", "## Results", "## API", "## Deployment status", "## Limitations"):
        if heading not in text:
            failures.append(f"section missing: {heading}")

    # setup commands from the guidelines
    for needle in ("pip install -r requirements.txt", "scripts/ml/train.py", "uvicorn app.main:app", "127.0.0.1:8000"):
        if needle not in text:
            failures.append(f"setup command missing: {needle}")

    # every documented endpoint exists in the app, and every app endpoint is documented
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        paths = set(client.get("/openapi.json").json()["paths"])
    api_paths = {p for p in paths if p.startswith("/api/v1/")}
    for p in sorted(api_paths):
        if f"`{p}`" not in text:
            failures.append(f"endpoint not documented: {p}")
    for found in set(re.findall(r"`((?:GET|POST) )?(/api/v1/[a-z/]+)`", text)):
        if found[1] not in paths:
            failures.append(f"README documents an endpoint that does not exist: {found[1]}")

    # deployments: only the local app may be claimed as live
    low = text.lower()
    for platform in ("streamlit", "hugging face"):
        if platform not in low:
            failures.append(f"deployment status for {platform} missing")
    rows = {name: next((l for l in text.splitlines() if l.startswith(f"| {name}")), "") for name in ("Streamlit", "Vercel", "Hugging Face")}
    for name, row in rows.items():
        if not row:
            failures.append(f"README deployment table has no row starting with '| {name}'")
    if "https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/" not in rows["Streamlit"] or "not deployed" in rows["Streamlit"].lower():
        failures.append("README Streamlit row must show the live URL and must not say 'not deployed'")
    if "https://climate-machine-learning.vercel.app/" not in rows["Vercel"] or "not deployed" in rows["Vercel"].lower():
        failures.append("README Vercel row must show the live URL and must not say 'not deployed'")
    # SPEC-project-completion.md: model repo live with the current model; no Space, by the group's decision
    hf = rows["Hugging Face"].lower()
    if ("smhunain/crop-yield-drought-pakistan" not in hf or "no space" not in hf or "group decision" not in hf
            or "not deployed yet" in hf or "earlier model" in hf):
        failures.append("README Hugging Face row must link the live model repo, say 'No Space' and 'group decision', "
                        "and must not say 'not deployed yet' or 'earlier model'")

    # no fake / demo wording, no old model numbers
    for word in ("bajra", "jowar", "0.892", "elasticity", "simulation", "demo mode", "fallback"):
        if word in low:
            failures.append(f"banned wording in README: {word}")

    # data facts must match the real files
    if metrics["data"]["sha256"][:16] not in text:
        failures.append("data SHA-256 prefix missing")
    if "MIT" not in text or "zenodo.org/records/15486470" not in text:
        failures.append("dataset licence/source missing")

    # second data source (NASA POWER) must be documented honestly
    power = json.loads((ROOT / "data" / "reference" / "nasa_power_source.json").read_text())
    for needle in ("NASA POWER", "soil_wetness_mean", "soil_wetness_max", "scripts/ml/fetch_nasa_power.py",
                   "power.larc.nasa.gov/docs/referencing", power["csv_sha256"][:16],
                   power["source"]["api_version"], power["source"]["accessed"]):
        if needle not in text:
            failures.append(f"second source not documented: {needle}")
    if "licence not specified" not in low and "license not specified" not in low:
        failures.append("README must say the NASA POWER licence is not specified")
    for stale in ("single dataset", "one dataset", "were **not merged**"):
        if stale in low:
            failures.append(f"stale claim about data sources: {stale!r}")

    # the other two front ends and the deployment guide must be documented
    for needle in ("streamlit_app/app.py", "streamlit_app/requirements.txt", "hf_space/app.py",
                   "scripts/build/build_hf_bundle.py", "scripts/build/build_vercel_bundle.py", "scripts/checks/verify_vercel_bundle.py", "scripts/checks/verify_streamlit.py", "scripts/checks/verify_hf_bundle.py",
                   "docs/guides/deployment_guide.md", "src/climate_machine_learning/inference.py"):
        if needle not in text:
            failures.append(f"not documented: {needle}")
    if "(docs/guides/deployment_guide.md)" not in text:
        failures.append("README must link to docs/guides/deployment_guide.md")

    # relative links must point to real files
    for target in re.findall(r"\]\((?!https?://|#)([^)\s]+)\)", text):
        if not (ROOT / target.split("#")[0]).exists():
            failures.append(f"broken relative link: {target}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: README matches metrics.json, endpoints, setup, and deployment status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
