"""Download monthly NASA POWER data for 9 farming-area points and store it in the repo.

Run (needs internet, about 1 minute):  python scripts/ml/fetch_nasa_power.py
Writes:
  data/raw/nasa_power/<Province>_<Point>.json   raw API responses, untouched
  data/raw/nasa_power_monthly.csv               tidy table: Province, Point, Year, Month, GWETROOT, PRECTOTCORR
  data/reference/nasa_power_source.json         points, API version, access date, citation, sha256 of every file
After this, training needs no network.
"""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "nasa_power"
CSV_PATH = ROOT / "data" / "raw" / "nasa_power_monthly.csv"
DOC_PATH = ROOT / "data" / "reference" / "nasa_power_source.json"

API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
PARAMETERS = ["GWETROOT", "PRECTOTCORR"]
START, END = 2001, 2022

# Judgment: 3 farming-area points per province, equal weight (see SPEC-second-source-nasa-power.md)
POINTS = [
    ("Punjab", "Lahore", 31.5, 74.3), ("Punjab", "Faisalabad", 31.4, 73.1), ("Punjab", "Multan", 30.2, 71.5),
    ("Sindh", "Sukkur", 27.7, 68.9), ("Sindh", "Hyderabad", 25.4, 68.4), ("Sindh", "Mirpurkhas", 25.5, 69.0),
    ("Balochistan", "Quetta", 30.2, 67.0), ("Balochistan", "Jaffarabad", 28.5, 68.0), ("Balochistan", "Turbat", 26.0, 63.0),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(lat: float, lon: float) -> bytes:
    query = urllib.parse.urlencode({
        "parameters": ",".join(PARAMETERS), "community": "AG", "longitude": lon, "latitude": lat,
        "start": START, "end": END, "format": "JSON",
    })
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=90) as resp:
                return resp.read()
        except Exception as exc:  # network hiccup: wait and retry
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"NASA POWER request failed for ({lat}, {lon}): {last}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows, point_docs, header = [], [], None
    for province, name, lat, lon in POINTS:
        path = RAW_DIR / f"{province}_{name}.json"
        path.write_bytes(fetch(lat, lon))
        data = json.loads(path.read_text())
        header = data["header"]
        series = data["properties"]["parameter"]
        for key in sorted(series["GWETROOT"]):
            year, month = int(key[:4]), int(key[4:])
            if month == 13:  # annual summary key, we aggregate ourselves
                continue
            row = {"Province": province, "Point": name, "Year": year, "Month": month}
            for param in PARAMETERS:
                value = series[param][key]
                if value == header.get("fill_value", -999.0):
                    raise ValueError(f"fill value in {path.name} {param} {key}")
                row[param] = value
            rows.append(row)
        point_docs.append({
            "province": province, "name": name, "lat": lat, "lon": lon,
            "file": str(path.relative_to(ROOT)), "sha256": sha256(path),
        })
        print(f"fetched {province:12s} {name}")

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    doc = {
        "source": {
            "name": "NASA POWER (Prediction Of Worldwide Energy Resources), monthly point API",
            "url": API,
            "api_version": header["api"]["version"],
            "data_source": header.get("sources", ["MERRA2"]),
            "community": "AG",
            "parameters": {
                "GWETROOT": "Root Zone Soil Wetness (unit 1, 0-1)",
                "PRECTOTCORR": "Precipitation Corrected (mm/day)",
            },
            "years": [START, END],
            "accessed": date.today().isoformat(),
            "licence": "License not specified in the POWER catalog entry. Free to use; POWER asks users to cite its Reference and Data Reference.",
            "reference_url": "https://power.larc.nasa.gov/docs/referencing/",
            "point_choice": "3 farming-area points per province, equal weight; a judgment, not a measurement of the whole province",
        },
        "points": point_docs,
        "csv_file": str(CSV_PATH.relative_to(ROOT)),
        "csv_sha256": sha256(CSV_PATH),
    }
    DOC_PATH.write_text(json.dumps(doc, indent=2))
    print(f"wrote {CSV_PATH.name} ({len(rows)} rows) and {DOC_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
