from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "pakistan_crop_yield_climate.csv"

PROVINCES: List[str] = ["Punjab", "Sindh", "Balochistan"]

ALL_CROPS: List[str] = [
    "apple", "banana", "cotton", "dates", "grapes", "mango",
    "onion", "pomegranate", "potato", "rice", "sugarcane", "tomato", "wheat"
]

DROUGHT_RESILIENT_CROPS: List[str] = ["dates", "grapes", "pomegranate", "wheat", "onion"]
WATER_INTENSIVE_CROPS: List[str] = ["sugarcane", "rice", "banana", "cotton"]
MODERATE_WATER_CROPS: List[str] = ["apple", "mango", "potato", "tomato"]

# Benchmark historical climatic averages per province from genuine 2001-2022 dataset
PROVINCE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "Punjab": {
        "temperature_2m_mean": 24.9,
        "temperature_2m_max": 31.1,
        "temperature_2m_min": 19.1,
        "precipitation_sum": 34.8,
        "fao_evapotranspiration": 4.6,
        "shortwave_radiation": 18.7,
        "precipitation_hours": 43.4,
        "climate_index": 0.42,
    },
    "Sindh": {
        "temperature_2m_mean": 27.6,
        "temperature_2m_max": 34.0,
        "temperature_2m_min": 21.6,
        "precipitation_sum": 17.6,
        "fao_evapotranspiration": 6.3,
        "shortwave_radiation": 21.2,
        "precipitation_hours": 14.7,
        "climate_index": 0.39,
    },
    "Balochistan": {
        "temperature_2m_mean": 18.6,
        "temperature_2m_max": 24.3,
        "temperature_2m_min": 12.1,
        "precipitation_sum": 11.6,
        "fao_evapotranspiration": 5.6,
        "shortwave_radiation": 22.1,
        "precipitation_hours": 19.5,
        "climate_index": 0.44,
    },
}

# Pre-defined drought scenarios for rapid demonstration & simulation
PRESET_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "punjab_severe_drought",
        "name": "Severe Summer Dry Spell (Punjab)",
        "province": "Punjab",
        "area": 50.0,
        "precipitation_sum": 1.8,
        "temperature_2m_mean": 35.0,
        "fao_evapotranspiration": 7.8,
        "description": "Simulates acute monsoon failure with elevated atmospheric heat and extreme evapotranspiration in central Punjab."
    },
    {
        "id": "balochistan_arid_dry",
        "name": "Extreme Arid Drought (Balochistan)",
        "province": "Balochistan",
        "area": 25.0,
        "precipitation_sum": 0.5,
        "temperature_2m_mean": 28.0,
        "fao_evapotranspiration": 6.5,
        "description": "Prolonged dry winter/spring season in Balochistan uplands with near-zero rain."
    },
    {
        "id": "sindh_heatwave_drought",
        "name": "Thermal Heatwave & High Evaporation (Sindh)",
        "province": "Sindh",
        "area": 40.0,
        "precipitation_sum": 2.0,
        "temperature_2m_mean": 36.5,
        "fao_evapotranspiration": 8.5,
        "description": "Severe lower Indus water deficit combined with high thermal stress."
    },
    {
        "id": "punjab_moderate_drought",
        "name": "Moderate Seasonal Deficit (Punjab)",
        "province": "Punjab",
        "area": 50.0,
        "precipitation_sum": 15.0,
        "temperature_2m_mean": 27.0,
        "fao_evapotranspiration": 5.2,
        "description": "Moderate 50% rainfall deficit below the historical Punjab normal."
    }
]
