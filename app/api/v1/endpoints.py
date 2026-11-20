from fastapi import APIRouter, Depends, Request, status, HTTPException
from app.schemas.predict import SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import DroughtScenarioRequest, RecommendationResponse
from app.schemas.metadata import MetadataResponse, Scenario

router = APIRouter()


def get_service(request: Request):
    """The loaded model service. Every model endpoint depends on this, so a missing service is always a 503."""
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model service is not initialized.",
        )
    return service


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
    description="Provinces, crops, real driest/wettest training years, and the climate ranges the model was trained on.",
    tags=["Metadata"],
)
async def get_metadata(svc=Depends(get_service)):
    ref = svc.reference
    scenarios = [
        Scenario(
            id=f"{province.lower()}_{kind}_{info['year']}",
            name=f"{kind.capitalize()} real year in the training data: {province} {info['year']}",
            province=province,
            year=info["year"],
            kind=kind,
            inputs=info["inputs"],
        )
        for province, kinds in ref["historical_years"].items()
        for kind, info in kinds.items()
    ]
    return MetadataResponse(
        provinces=list(ref["provinces"]),
        all_crops=list(ref["crops"]),
        scenarios=scenarios,
        model_version=f"{svc.model_name} (trained {svc.metrics['trained_at']})",
        features_count=len(ref["feature_cols"]),
        reference=ref["provinces"],
        train_rows_per_crop=ref["train_rows_per_crop"],
        training_rows_possible=svc.training_rows_possible,
    )


@router.get(
    "/metrics",
    summary="Saved Model Evaluation Results",
    description="Returns models/metrics.json exactly as written by scripts/ml/train.py: baselines, all tuned models, "
                "validation and test scores, permutation importance, data hash and split.",
    tags=["Metadata"],
)
async def get_metrics(model_service=Depends(get_service)):
    return model_service.metrics


@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    summary="Single Crop Yield Prediction",
    description="Predict yield (dataset yield units) for one crop from annual climate inputs. Inputs outside the trained range are rejected.",
    tags=["Inference"],
)
async def predict_single_crop(payload: SinglePredictionRequest, model_service=Depends(get_service)):
    return model_service.predict_single(payload)


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Low-Water Crop Recommendation Engine",
    description="Rank all 13 crops by model-predicted yield in your scenario relative to a typical year in the province. FAO water-need ranges are shown as context only.",
    tags=["Inference"],
)
async def recommend_crops(payload: DroughtScenarioRequest, model_service=Depends(get_service)):
    return model_service.recommend_crops(payload)
