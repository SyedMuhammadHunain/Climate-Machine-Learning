from typing import Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas.common import ProvinceEnum, CropEnum


class ClimateInputs(BaseModel):
    """Annual inputs shared by /predict and /recommend. Unknown fields are rejected, never silently ignored."""

    model_config = ConfigDict(extra="forbid")

    province: ProvinceEnum = Field(..., description="Province: Punjab, Sindh or Balochistan")
    precipitation_sum: float = Field(..., ge=0.0, le=5000.0, description="Total rain over the year (mm)")
    temperature_2m_mean: float = Field(..., ge=-5.0, le=55.0, description="Mean temperature over the year (deg C)")
    temperature_2m_max: Optional[float] = Field(None, ge=-5.0, le=60.0, description="Mean of the monthly maximum temperature (deg C)")
    temperature_2m_min: Optional[float] = Field(None, ge=-10.0, le=45.0, description="Mean of the monthly minimum temperature (deg C)")
    fao_evapotranspiration: Optional[float] = Field(None, ge=0.5, le=20.0, description="Mean FAO evapotranspiration (mm/day)")
    shortwave_radiation: Optional[float] = Field(None, ge=5.0, le=40.0, description="Mean shortwave radiation (MJ/m2)")
    precipitation_hours: Optional[float] = Field(None, ge=0.0, le=8760.0, description="Total hours with rain over the year")
    soil_wetness_mean: Optional[float] = Field(None, ge=0.0, le=1.0, description="NASA POWER root-zone soil wetness (0-1), mean of the 12 monthly values")
    soil_wetness_max: Optional[float] = Field(None, ge=0.0, le=1.0, description="NASA POWER root-zone soil wetness (0-1), wettest month")

    @model_validator(mode="after")
    def validate_temperature_order(self) -> "ClimateInputs":
        mean, tmax, tmin = self.temperature_2m_mean, self.temperature_2m_max, self.temperature_2m_min
        if tmax is not None and tmin is not None and tmax < tmin:
            raise ValueError(
                f"Maximum temperature ({tmax} C) cannot be lower than minimum temperature ({tmin} C)."
            )
        if tmax is not None and tmax < mean:
            raise ValueError(f"Maximum temperature ({tmax} C) cannot be lower than the mean ({mean} C).")
        if tmin is not None and tmin > mean:
            raise ValueError(f"Minimum temperature ({tmin} C) cannot be higher than the mean ({mean} C).")
        wet_mean, wet_max = self.soil_wetness_mean, self.soil_wetness_max
        if wet_mean is not None and wet_max is not None and wet_max < wet_mean:
            raise ValueError(f"Wettest-month soil wetness ({wet_max}) cannot be lower than the mean ({wet_mean}).")
        return self


class SinglePredictionRequest(ClimateInputs):
    crop: CropEnum = Field(..., description="Crop to predict")


class InputValue(BaseModel):
    value: float
    source: str  # user | province_median_training_years | user_mean_plus_province_median_offset


class SinglePredictionResponse(BaseModel):
    crop: str
    province: str
    predicted_yield: float
    yield_unit: str
    normal_yield: float
    pct_of_normal: float
    training_rows_for_crop: int
    training_rows_possible: int
    inputs: Dict[str, InputValue]
    model_name: str
    absolute_yield_note: str  # why one predicted number is rough (generated from metrics.json)
