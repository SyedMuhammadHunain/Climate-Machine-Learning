from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from app.schemas.common import ProvinceEnum


class DroughtScenarioRequest(BaseModel):
    province: ProvinceEnum = Field(..., description="Target agricultural province (Punjab, Sindh, Balochistan)")
    area: float = Field(50.0, ge=0.1, le=20000.0, description="Cultivated land area in hectares")
    precipitation_sum: float = Field(..., ge=0.0, le=600.0, description="Seasonal rainfall in mm (e.g. < 5.0 for drought)")
    temperature_2m_mean: float = Field(..., ge=-5.0, le=55.0, description="Mean seasonal temperature in °C")
    fao_evapotranspiration: Optional[float] = Field(None, ge=0.5, le=20.0, description="Evapotranspiration in mm/day")
    temperature_2m_max: Optional[float] = Field(None, ge=-5.0, le=60.0, description="Maximum temperature in °C")
    temperature_2m_min: Optional[float] = Field(None, ge=-10.0, le=45.0, description="Minimum temperature in °C")
    shortwave_radiation: Optional[float] = Field(None, ge=5.0, le=40.0, description="Shortwave solar radiation in MJ/m²")
    precipitation_hours: Optional[float] = Field(None, ge=0.0, le=500.0, description="Precipitation duration in hours")
    climate_index: Optional[float] = Field(None, ge=0.0, le=1.0, description="Climate dryness index (0 - 1)")
    scenario_name: Optional[str] = Field("Custom Drought Scenario", description="Descriptive label for this scenario")

    @model_validator(mode="after")
    def validate_temperatures(self) -> "DroughtScenarioRequest":
        if self.temperature_2m_max is not None and self.temperature_2m_min is not None:
            if self.temperature_2m_max < self.temperature_2m_min:
                raise ValueError(
                    f"Maximum temperature ({self.temperature_2m_max}°C) cannot be lower than minimum temperature ({self.temperature_2m_min}°C)."
                )
        return self


class CropRecommendationItem(BaseModel):
    rank: int
    crop: str
    predicted_yield_kg_ha: float
    water_category: str
    is_drought_resilient: bool
    risk_assessment: str


class WaterIntensiveWarning(BaseModel):
    crop: str
    predicted_yield_kg_ha: float
    warning: str


class RecommendationResponse(BaseModel):
    scenario_name: str
    province: str
    input_conditions: Dict[str, Any]
    top_resilient_crop: CropRecommendationItem
    ranked_drought_resilient: List[CropRecommendationItem]
    all_crops_ranked: List[CropRecommendationItem]
    water_intensive_warnings: List[WaterIntensiveWarning]
    recommendation_summary: str
