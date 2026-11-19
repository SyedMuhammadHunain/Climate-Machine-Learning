"""Checks README.md against the real project files.

Run:   .venv/bin/python scripts/verify_readme.py          # check
       .venv/bin/python scripts/verify_readme.py --write  # refresh the metrics block from models/metrics.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


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
    names = {"global_mean": "Baseline: overall mean yield", "crop_province_mean": "Baseline: crop-province mean yield"}
    rows = [(names.get(k, k), v["val"], v["test"], False) for k, v in m["baselines"].items()]
    rows += [(v["label"], v["val"], v["test"], k == m["chosen"]["name"]) for k, v in m["models"].items()]
    for name, val, test, chosen in rows:
        label = f"**{name} (chosen)**" if chosen else name
        lines.append(
            f"| {label} | {val['r2']:.3f} | {val['mae']:.1f} | {test['r2']:.3f} | {test['mae']:.1f} | "
            f"{test['rmse']:.1f} | {test['skill_vs_crop_province_mean_log']:+.2f} |"
        )
    lines += [
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

    for needle in ("Muhammad Afzal", "CT-24263", "Syed Muhammad Hunain", "CT-24281", "Ayesha", "CT-24309"):
        if needle not in text:
            failures.append(f"group member info missing: {needle}")

    for heading in ("## Quick start", "## Data", "## Results", "## API", "## Deployment status", "## Limitations"):
        if heading not in text:
            failures.append(f"section missing: {heading}")

    # setup commands from the guidelines
    for needle in ("pip install -r requirements.txt", "scripts/train.py", "uvicorn app.main:app", "127.0.0.1:8000"):
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
    if not re.search(r"streamlit[^\n]*not (yet )?deployed", low) or not re.search(r"hugging face[^\n]*not (yet )?deployed", low):
        failures.append("Streamlit and Hugging Face must be marked 'not deployed yet'")

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
    for needle in ("NASA POWER", "soil_wetness_mean", "soil_wetness_max", "scripts/fetch_nasa_power.py",
                   "power.larc.nasa.gov/docs/referencing", power["csv_sha256"][:16],
                   power["source"]["api_version"], power["source"]["accessed"]):
        if needle not in text:
            failures.append(f"second source not documented: {needle}")
    if "licence not specified" not in low and "license not specified" not in low:
        failures.append("README must say the NASA POWER licence is not specified")
    for stale in ("single dataset", "one dataset", "were **not merged**"):
        if stale in low:
            failures.append(f"stale claim about data sources: {stale!r}")

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
