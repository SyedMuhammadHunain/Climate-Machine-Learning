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
    barely_grown: bool  # typical predicted yield under 1% of the largest in this result (inference.BARELY_GROWN_SHARE)


class CropGroup(BaseModel):
    """Crops that show the same share of a typical year at one decimal: one rank for the whole group."""
    rank: int  # 1 = largest share, 2 = next different share, ...
    pct_of_normal: float  # the share shown, at one decimal
    crops: List[str]  # in ranking order (by name inside the group)


class ScenarioFact(BaseModel):
    label: str
    value: str


class RecommendationResponse(BaseModel):
    scenario_name: str
    province: str
    yield_unit: str
    ranking_basis: str
    inputs: Dict[str, InputValue]
    top_crop: CropRecommendationItem
    ranked_crops: List[CropRecommendationItem]
    groups: List[CropGroup]
    headline: str  # one sentence: largest and smallest share; a tie is named as a group, never as one best crop
    scenario_facts: List[ScenarioFact]
    recommendation_summary: str
    model_caveat: str  # inference.model_caveat(metrics.json): the win, the loss and the after-the-fact choice, as its own field
    model_name: str
    absolute_yield_note: str  # why one predicted number is rough (generated from metrics.json)
