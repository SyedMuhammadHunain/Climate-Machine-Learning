import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# The shared feature code lives in src/ (used by training and by the API).
# Make it importable even when the project package is not pip-installed.
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

# Everything the API serves comes from these files (written by scripts/train.py):
MODEL_PATH = BASE_DIR / "models" / "crop_yield_pipeline.joblib"
REFERENCE_PATH = BASE_DIR / "models" / "reference.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
# Cited context shown next to the ranking (not a model input):
FAO_WATER_PATH = BASE_DIR / "data" / "reference" / "fao_crop_water_needs.json"
