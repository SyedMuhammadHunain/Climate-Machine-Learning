from typing import List, Dict, Any
from pydantic import BaseModel


class MetadataResponse(BaseModel):
    provinces: List[str]
    all_crops: List[str]
    drought_resilient_crops: List[str]
    water_intensive_crops: List[str]
    moderate_water_crops: List[str]
    preset_scenarios: List[Dict[str, Any]]
    province_defaults: Dict[str, Dict[str, float]]
    model_version: str
    features_count: int
