from typing import Any, Dict, List
from pydantic import BaseModel


class Scenario(BaseModel):
    """A real observed year from the training data."""
    id: str
    name: str
    province: str
    year: int
    kind: str  # driest | wettest (by annual rain)
    inputs: Dict[str, float]


class MetadataResponse(BaseModel):
    provinces: List[str]
    all_crops: List[str]
    scenarios: List[Scenario]
    model_version: str
    features_count: int
    reference: Dict[str, Any]
    train_rows_per_crop: Dict[str, int]
    training_rows_possible: int
