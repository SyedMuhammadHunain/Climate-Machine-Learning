import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from climate_machine_learning import features as F
from app.core.config import FAO_WATER_PATH, METRICS_PATH, MODEL_PATH, REFERENCE_PATH
from app.core.exceptions import AgronomicValidationError
from app.schemas.predict import InputValue, SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import (
    CropRecommendationItem,
    DroughtScenarioRequest,
    FaoWaterNeed,
    RecommendationResponse,
)


class ModelService:
    """Serves the trained scikit-learn pipeline. Every yield returned is raw model output.

    No multipliers, no hand-made scores. Ranking uses only two model outputs per crop:
    the yield for the user's scenario and the yield for a typical (median) year.
    """

    YIELD_UNIT = "dataset yield units (unit not confirmed)"
    RANKING_BASIS = (
        "Model-predicted yield for your scenario divided by the model-predicted yield for a "
        "typical (median) year in the same province, as a percentage. Ties are ordered by crop name."
    )

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.reference: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.fao: Dict[str, Any] = {}
        self.load_model()

    @staticmethod
    def _single_thread(pipeline) -> None:
        """One thread at inference: avoids joblib overhead for small predictions."""
        reg = pipeline.named_steps.get("regressor")
        reg = getattr(reg, "regressor_", reg)
        if reg is not None and hasattr(reg, "n_jobs"):
            reg.n_jobs = 1

    def load_model(self) -> None:
        for path in (self.model_path, REFERENCE_PATH, METRICS_PATH, FAO_WATER_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. Run scripts/train.py to create the model files."
                )
        self.pipeline = joblib.load(self.model_path)
        self._single_thread(self.pipeline)
        self.reference = json.loads(Path(REFERENCE_PATH).read_text())
        self.metrics = json.loads(Path(METRICS_PATH).read_text())
        self.fao = json.loads(Path(FAO_WATER_PATH).read_text())

    # ------------------------------------------------------------------ helpers
    @property
    def training_rows_possible(self) -> int:
        """Rows a crop would have if every province-year were reported."""
        first, last = self.reference["train_years"]
        return len(self.reference["provinces"]) * (last - first + 1)

    @property
    def model_name(self) -> str:
        return self.metrics["models"][self.metrics["chosen"]["name"]]["label"]

    def _check_training_range(self, province: str, inputs: Dict[str, float]) -> None:
        """Refuse inputs the model never saw: a tree model cannot extrapolate."""
        ranges = self.reference["provinces"][province]["range"]
        for name, value in inputs.items():
            lo, hi = ranges[name]
            if value < lo or value > hi:
                raise AgronomicValidationError(
                    f"{name} = {value:g} is outside the range the model was trained on for {province} "
                    f"({lo:.1f} to {hi:.1f}, annual values {self.reference['train_years'][0]}-"
                    f"{self.reference['train_years'][1]}). The model cannot extrapolate, so no prediction is made.",
                    parameter=name,
                )

    @staticmethod
    def _supplied(request) -> Dict[str, float]:
        return {k: getattr(request, k) for k in F.BASE_INPUTS if getattr(request, k) is not None}

    def _fao_item(self, crop: str):
        entry = self.fao["crops"].get(crop)
        if entry is None:
            return None
        return FaoWaterNeed(
            fao_name=entry["fao_name"],
            mm_min=entry["mm_min"],
            mm_max=entry["mm_max"],
            drought_sensitivity=entry["drought_sensitivity"],
            unit=self.fao["unit"],
            source_url=self.fao["source"]["url"],
        )

    # ------------------------------------------------------------------ /predict
    def predict_single(self, request: SinglePredictionRequest) -> SinglePredictionResponse:
        province, crop = request.province.value, request.crop.value
        supplied = self._supplied(request)
        self._check_training_range(province, supplied)

        row, sources = F.build_inference_row(province, crop, supplied, self.reference)
        predicted = float(self.pipeline.predict(row)[0])

        median = self.reference["provinces"][province]["median"]
        normal_row, _ = F.build_inference_row(province, crop, dict(median), self.reference)
        normal = float(self.pipeline.predict(normal_row)[0])

        return SinglePredictionResponse(
            crop=crop,
            province=province,
            predicted_yield=predicted,
            yield_unit=self.YIELD_UNIT,
            normal_yield=normal,
            pct_of_normal=predicted / normal * 100.0,
            training_rows_for_crop=int(self.reference["train_rows_per_crop"][crop]),
            training_rows_possible=self.training_rows_possible,
            inputs={k: InputValue(value=float(row.iloc[0][k]), source=sources[k]) for k in F.BASE_INPUTS},
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------ /recommend
    def recommend_crops(self, request: DroughtScenarioRequest) -> RecommendationResponse:
        province = request.province.value
        supplied = self._supplied(request)
        self._check_training_range(province, supplied)

        crops: List[str] = list(self.reference["crops"])
        median = dict(self.reference["provinces"][province]["median"])

        scenario_rows, normal_rows, sources = [], [], {}
        for crop in crops:
            row, sources = F.build_inference_row(province, crop, supplied, self.reference)
            scenario_rows.append(row)
            normal_row, _ = F.build_inference_row(province, crop, median, self.reference)
            normal_rows.append(normal_row)

        batch = pd.concat(scenario_rows + normal_rows, ignore_index=True)
        preds = self.pipeline.predict(batch)
        scenario_preds, normal_preds = preds[: len(crops)], preds[len(crops):]

        records = [
            {
                "crop": crop,
                "predicted_yield": float(sp),
                "normal_yield": float(npred),
                "pct_of_normal": float(sp) / float(npred) * 100.0,
            }
            for crop, sp, npred in zip(crops, scenario_preds, normal_preds)
        ]
        # Highest % of normal first; exact ties are ordered by crop name (deterministic)
        records.sort(key=lambda r: (-r["pct_of_normal"], r["crop"]))

        items = [
            CropRecommendationItem(
                rank=i + 1,
                training_rows_for_crop=int(self.reference["train_rows_per_crop"][r["crop"]]),
                training_rows_possible=self.training_rows_possible,
                fao_water_need=self._fao_item(r["crop"]),
                **r,
            )
            for i, r in enumerate(records)
        ]
        top, bottom = items[0], items[-1]

        rain = float(batch.iloc[0]["precipitation_sum"])
        temp = float(batch.iloc[0]["temperature_2m_mean"])
        summary = (
            f"For {province} with {rain:.1f} mm of rain a year and a mean temperature of {temp:.1f} °C, "
            f"the model keeps the largest share of a typical year's yield for {top.crop.upper()} "
            f"({top.pct_of_normal:.1f}% of typical) and the smallest for {bottom.crop.upper()} "
            f"({bottom.pct_of_normal:.1f}% of typical). "
            "These are model outputs, not adjusted by hand."
        )
        if not self.metrics["chosen"]["beats_crop_province_mean_on_test"]:
            summary += (
                " Caution: on the 2020-2022 test years this model did not beat simply using each "
                "crop's average yield, so small differences between crops are weak evidence."
            )

        return RecommendationResponse(
            scenario_name=request.scenario_name or "Custom scenario",
            province=province,
            yield_unit=self.YIELD_UNIT,
            ranking_basis=self.RANKING_BASIS,
            inputs={k: InputValue(value=float(batch.iloc[0][k]), source=sources[k]) for k in F.BASE_INPUTS},
            top_crop=top,
            ranked_crops=items,
            recommendation_summary=summary,
            model_name=self.model_name,
        )
