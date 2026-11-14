from app.schemas.common import ProvinceEnum, CropEnum, WaterCategoryEnum
from app.schemas.predict import SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import (
    DroughtScenarioRequest,
    CropRecommendationItem,
    WaterIntensiveWarning,
    RecommendationResponse,
)
from app.schemas.metadata import MetadataResponse

__all__ = [
    "ProvinceEnum",
    "CropEnum",
    "WaterCategoryEnum",
    "SinglePredictionRequest",
    "SinglePredictionResponse",
    "DroughtScenarioRequest",
    "CropRecommendationItem",
    "WaterIntensiveWarning",
    "RecommendationResponse",
    "MetadataResponse",
]
