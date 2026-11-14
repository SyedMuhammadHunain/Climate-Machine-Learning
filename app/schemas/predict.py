from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from app.schemas.common import ProvinceEnum, CropEnum, WaterCategoryEnum


class SinglePredictionRequest(BaseModel):
    province: ProvinceEnum = Field(..., description="Target agricultural province (Punjab, Sindh, Balochistan)")
    crop: CropEnum = Field(..., description="Candidate agricultural crop name")
    area: float = Field(50.0, ge=0.1, le=20000.0, description="Cultivated area in hectares (0.1 - 20,000)")
    precipitation_sum: float = Field(..., ge=0.0, le=600.0, description="Total seasonal rainfall in mm (0 - 600)")
    temperature_2m_mean: float = Field(..., ge=-5.0, le=55.0, description="Mean temperature in °C (-5 - 55)")
    temperature_2m_max: Optional[float] = Field(None, ge=-5.0, le=60.0, description="Maximum temperature in °C")
    temperature_2m_min: Optional[float] = Field(None, ge=-10.0, le=45.0, description="Minimum temperature in °C")
    fao_evapotranspiration: Optional[float] = Field(None, ge=0.5, le=20.0, description="FAO reference evapotranspiration in mm/day")
    shortwave_radiation: Optional[float] = Field(None, ge=5.0, le=40.0, description="Solar shortwave radiation in MJ/m²")
    precipitation_hours: Optional[float] = Field(None, ge=0.0, le=500.0, description="Hours of precipitation in season")
    climate_index: Optional[float] = Field(None, ge=0.0, le=1.0, description="Composite climate/dryness index (0 - 1)")

    @model_validator(mode="after")
    def validate_temperatures(self) -> "SinglePredictionRequest":
        if self.temperature_2m_max is not None and self.temperature_2m_min is not None:
            if self.temperature_2m_max < self.temperature_2m_min:
                raise ValueError(
                    f"Maximum temperature ({self.temperature_2m_max}°C) cannot be lower than minimum temperature ({self.temperature_2m_min}°C)."
                )
        return self


class SinglePredictionResponse(BaseModel):
    crop: str
    province: str
    predicted_yield_kg_ha: float
    water_category: str
    is_drought_resilient: bool
    risk_assessment: str
    features_used: Dict[str, Any]
