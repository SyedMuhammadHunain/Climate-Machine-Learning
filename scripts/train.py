"""Train the crop-yield model and write models/*.

Run: .venv/bin/python scripts/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from climate_machine_learning import training  # noqa: E402

if __name__ == "__main__":
    training.train(ROOT / "models")
    print("saved: models/crop_yield_pipeline.joblib, models/metrics.json, models/reference.json")
