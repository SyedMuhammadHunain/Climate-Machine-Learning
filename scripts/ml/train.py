"""Train the crop-yield model and write models/*.

Run: .venv/bin/python scripts/ml/train.py [--out DIR]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from climate_machine_learning import training  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "models",
                    help="folder for the three artifacts (default: models/); use another folder to try a candidate")
    out = ap.parse_args().out
    training.train(out)
    print(f"saved: {out}/crop_yield_pipeline.joblib, metrics.json, reference.json")
