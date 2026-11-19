from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.predict import ClimateInputs, InputValue


class DroughtScenarioRequest(ClimateInputs):
    scenario_name: Optional[str] = Field("Custom scenario", max_length=80, description="Label for this scenario")


class FaoWaterNeed(BaseModel):
    """FAO Training Manual 3, Table 14. Context only; not a model input."""
    fao_name: str
    mm_min: int
    mm_max: int
    drought_sensitivity: str
    unit: str
    source_url: str


class CropRecommendationItem(BaseModel):
    rank: int
    crop: str
    predicted_yield: float
    normal_yield: float
    pct_of_normal: float
    tied_with: int  # other crops with the same share at one decimal (0 = unique); tied crops are ordered by name only
    training_rows_for_crop: int
    training_rows_possible: int
    fao_water_need: Optional[FaoWaterNeed] = None  # None = crop is not in the FAO table


class RecommendationResponse(BaseModel):
    scenario_name: str
    province: str
    yield_unit: str
    ranking_basis: str
    inputs: Dict[str, InputValue]
    top_crop: CropRecommendationItem
    ranked_crops: List[CropRecommendationItem]
    recommendation_summary: str
    model_name: str
