from fastapi import APIRouter, Request, status, HTTPException
from app.core.config import (
    PROVINCES,
    ALL_CROPS,
    DROUGHT_RESILIENT_CROPS,
    WATER_INTENSIVE_CROPS,
    MODERATE_WATER_CROPS,
    PRESET_SCENARIOS,
    PROVINCE_DEFAULTS,
)
from app.schemas.predict import SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import DroughtScenarioRequest, RecommendationResponse
from app.schemas.metadata import MetadataResponse

router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description="Verify backend operational status and confirmed in-memory ML pipeline readiness.",
    tags=["System"],
)
async def health_check(request: Request):
    model_service = getattr(request.app.state, "model_service", None)
    is_ready = model_service is not None and model_service.pipeline is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "framework": "FastAPI",
        "version": "0.1.0",
    }


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Agricultural & Climate Metadata",
    description="Fetch available provinces, crop classification categories, and historical default benchmarks.",
    tags=["Metadata"],
)
async def get_metadata():
    return MetadataResponse(
        provinces=PROVINCES,
        all_crops=ALL_CROPS,
        drought_resilient_crops=DROUGHT_RESILIENT_CROPS,
        water_intensive_crops=WATER_INTENSIVE_CROPS,
        moderate_water_crops=MODERATE_WATER_CROPS,
        preset_scenarios=PRESET_SCENARIOS,
        province_defaults=PROVINCE_DEFAULTS,
        model_version="crop_yield_pipeline.joblib (v1.0 - Real Pakistan Data)",
        features_count=17,
    )


@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    summary="Single Crop Yield Prediction",
    description="Predict expected harvest yield (kg/ha) for a single specific crop under given environmental conditions.",
    tags=["Inference"],
)
async def predict_single_crop(request: Request, payload: SinglePredictionRequest):
    model_service = getattr(request.app.state, "model_service", None)
    if not model_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model service is not initialized."
        )
    return model_service.predict_single(payload)


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Low-Water Crop Recommendation Engine",
    description="Simulate multiple crops under drought stress and rank low-water resilient crops while flagging water-intensive risks.",
    tags=["Inference"],
)
async def recommend_crops(request: Request, payload: DroughtScenarioRequest):
    model_service = getattr(request.app.state, "model_service", None)
    if not model_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model service is not initialized."
        )
    return model_service.recommend_crops(payload)
