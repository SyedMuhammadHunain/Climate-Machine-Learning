"""Checks the Vercel deploy bundle built by scripts/build_vercel_bundle.py.

Run: .venv/bin/python scripts/verify_vercel_bundle.py
Builds the bundle into a temp folder, checks what is in and out of it, and runs the bundled app on its own.
"""
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_vercel_bundle as B

TMP = tempfile.TemporaryDirectory()
OUT = Path(TMP.name) / "vercel_bundle"
B.build(OUT)
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def root_pins() -> dict:
    return dict(l.strip().split("==", 1) for l in (ROOT / "requirements.txt").read_text().splitlines()
                if "==" in l and not l.startswith("#"))


def bundle_pins() -> dict:
    return dict(l.strip().split("==", 1) for l in (OUT / "requirements.txt").read_text().splitlines()
                if "==" in l and not l.startswith("#"))


@test
def bundle_has_everything_the_app_needs():
    for name in ("app/main.py", "app/api/v1/endpoints.py", "app/services/model_service.py", "app/core/config.py",
                 "src/climate_machine_learning/__init__.py", "src/climate_machine_learning/features.py",
                 "src/climate_machine_learning/inference.py",
                 "models/crop_yield_pipeline.joblib", "models/reference.json", "models/metrics.json",
                 "data/reference/fao_crop_water_needs.json",
                 "frontend/climate-resilient-frontend/index.html", "frontend/climate-resilient-frontend/predict.html",
                 "frontend/climate-resilient-frontend/metrics.html", "frontend/climate-resilient-frontend/assets/wordmark.svg",
                 "requirements.txt", ".python-version"):
        assert (OUT / name).exists(), name


@test
def bundle_leaves_out_what_a_deployed_api_must_not_carry():
    gone = ("scripts", "notebooks", "tasks", "docs", "tests", "streamlit_app", "hf_space", "data/raw", ".git", ".env",
            "src/climate_machine_learning/training.py", "src/climate_machine_learning/presentation.py",
            "frontend/climate-resilient-frontend/package.json")
    for name in gone:
        assert not (OUT / name).exists(), f"{name} must not be in the bundle"
    assert not list(OUT.rglob("__pycache__")), "no bytecode caches"
    assert not list(OUT.rglob("*.ipynb")) and not list(OUT.rglob("*.parquet"))


@test
def requirements_are_lean_exactly_pinned_and_match_the_tested_versions():
    pins, root = bundle_pins(), root_pins()
    for must in ("fastapi", "pydantic", "scikit-learn", "pandas", "numpy", "joblib", "scipy", "starlette"):
        assert must in pins, f"{must} missing"
    for banned in ("jupyter", "jupyterlab", "notebook", "ipykernel", "matplotlib", "seaborn", "pyarrow", "uvicorn",
                   "streamlit", "gradio", "httpx", "pytest"):
        assert banned not in pins, f"{banned} must not ship to a serverless function"
    for name, version in pins.items():
        assert root[name] == version, f"{name}: bundle {version} differs from the tested {root.get(name)}"
    lines = [l for l in (OUT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.!+\-]+", l.strip()) for l in lines), "every line pinned with =="
    assert len(lines) < 60, len(lines)


@test
def payload_stays_small():
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    assert size < 40 * 1024 * 1024, f"{size/1e6:.1f} MB (model 20 MB + pages + code expected)"


@test
def python_version_is_one_the_model_was_tested_on():
    assert (OUT / ".python-version").read_text().strip() in ("3.12", "3.13")


@test
def bundled_app_runs_on_its_own_and_matches_the_repo():
    """Run the app from the bundle folder only, then compare one prediction with the repo's own model service."""
    code = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
bundle = sys.argv[1]
sys.path.insert(0, bundle)
import app.main as main
from fastapi.testclient import TestClient
from pathlib import Path
import climate_machine_learning.features as F
assert str(Path(main.__file__).resolve()).startswith(bundle), main.__file__
assert str(Path(F.__file__).resolve()).startswith(bundle), F.__file__
with TestClient(main.app) as c:
    out = {"health": c.get("/health").status_code, "index": c.get("/").status_code,
           "predict_page": c.get("/predict.html").status_code, "metrics_page": c.get("/metrics.html").status_code,
           "package_json": c.get("/package.json").status_code, "dotenv": c.get("/.env").status_code,
           "metrics_api": c.get("/api/v1/metrics").status_code}
    r = c.post("/api/v1/predict", json={"province": "Sindh", "crop": "rice", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5})
    out["predict"] = [r.status_code, r.json().get("predicted_yield")]
    r = c.post("/api/v1/recommend", json={"province": "Sindh", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5})
    out["recommend"] = [r.status_code, len(r.json().get("ranked_crops", []))]
print("RESULT " + json.dumps(out))
'''
    done = subprocess.run([sys.executable, "-c", code, str(OUT.resolve())], cwd=str(OUT), capture_output=True, text=True, timeout=300)
    line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, (done.stdout + done.stderr)[-600:]
    got = json.loads(line[0][7:])
    assert got["health"] == got["index"] == got["predict_page"] == got["metrics_page"] == got["metrics_api"] == 200, got
    assert got["package_json"] == 404 and got["dotenv"] == 404, got
    assert got["recommend"] == [200, 13], got
    # same number as the repo's own service
    sys.path.insert(0, str(ROOT))
    from app.services.model_service import ModelService
    from app.schemas.predict import SinglePredictionRequest

    expected = ModelService().predict_single(SinglePredictionRequest(
        province="Sindh", crop="rice", precipitation_sum=150.0, temperature_2m_mean=27.5)).predicted_yield
    assert got["predict"][0] == 200 and abs(got["predict"][1] - expected) < 1e-9, (got["predict"], expected)


@test
def rebuild_is_safe_and_never_deletes_foreign_folders():
    with tempfile.TemporaryDirectory() as tmp:
        victim = Path(tmp) / "my_documents"
        (victim / "app").mkdir(parents=True)
        (victim / "app" / "mine.py").write_text("precious")
        (victim / "notes.txt").write_text("precious")
        try:
            B.build(victim)
        except B.UnsafeOutputDir as err:
            assert "not empty" in str(err) and "Refusing" in str(err), err
        else:
            raise AssertionError("build() accepted a foreign, non-empty folder")
        assert (victim / "app" / "mine.py").read_text() == "precious" and (victim / "notes.txt").exists()
        out = Path(tmp) / "bundle"
        B.build(out)
        (out / "app" / "stale.py").write_text("old")
        (out / "foreign.txt").write_text("not ours")
        B.build(out)
        assert not (out / "app" / "stale.py").exists(), "rebuild must replace the folders it owns"
        assert (out / "foreign.txt").exists(), "files it does not own must stay"


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
