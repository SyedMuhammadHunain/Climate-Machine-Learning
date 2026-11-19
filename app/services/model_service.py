import json
from pathlib import Path
from typing import Any, Dict

import joblib

# app.core.config first: it puts src/ on sys.path, so climate_machine_learning imports without being pip-installed.
from app.core.config import FAO_WATER_PATH, METRICS_PATH, MODEL_PATH, REFERENCE_PATH
from climate_machine_learning import features as F
from climate_machine_learning import inference as I
from app.core.exceptions import AgronomicValidationError
from app.schemas.predict import InputValue, SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import (
    CropRecommendationItem,
    DroughtScenarioRequest,
    FaoWaterNeed,
    RecommendationResponse,
)


def tie_notes(top, bottom) -> str:
    """One sentence for each end of the ranking that is tied (one in total when both ends are the same tie)."""
    same_tie = round(top.pct_of_normal, I.TIE_DIGITS) == round(bottom.pct_of_normal, I.TIE_DIGITS)
    ends = [("largest", top)] if same_tie else [("largest", top), ("smallest", bottom)]
    return "".join(
        f" The {label} share is tied: {end.crop.upper()} shares it with {end.tied_with} other "
        f"crop{'s' if end.tied_with > 1 else ''}, and tied crops are ordered by name only."
        for label, end in ends if end.tied_with
    )


class ModelService:
    """Serves the trained scikit-learn pipeline. Every yield returned is raw model output.

    No multipliers, no hand-made scores. The prediction and ranking logic lives in
    climate_machine_learning.inference (shared with the Streamlit app).
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

    def load_model(self) -> None:
        for path in (self.model_path, REFERENCE_PATH, METRICS_PATH, FAO_WATER_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. Run scripts/train.py to create the model files."
                )
        self.pipeline = joblib.load(self.model_path)
        I.single_thread(self.pipeline)
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

    @staticmethod
    def _supplied(request) -> Dict[str, float]:
        return {k: getattr(request, k) for k in F.BASE_INPUTS if getattr(request, k, None) is not None}

    def _predict(self, province: str, supplied: Dict[str, float], crops=None) -> I.Prediction:
        try:
            return I.predict_crops(self.pipeline, self.reference, province, supplied, crops=crops)
        except I.OutOfRangeError as err:
            raise AgronomicValidationError(str(err), parameter=err.parameter)

    def _inputs(self, result: I.Prediction) -> Dict[str, InputValue]:
        return {k: InputValue(value=result.inputs[k], source=result.sources[k]) for k in F.BASE_INPUTS}

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
        result = self._predict(province, self._supplied(request), crops=[crop])
        item = result.items[0]
        return SinglePredictionResponse(
            crop=crop,
            province=province,
            predicted_yield=item.predicted_yield,
            yield_unit=self.YIELD_UNIT,
            normal_yield=item.normal_yield,
            pct_of_normal=item.pct_of_normal,
            training_rows_for_crop=int(self.reference["train_rows_per_crop"][crop]),
            training_rows_possible=self.training_rows_possible,
            inputs=self._inputs(result),
            model_name=self.model_name,
            absolute_yield_note=I.absolute_yield_caveat(self.metrics),
        )

    # ------------------------------------------------------------------ /recommend
    def recommend_crops(self, request: DroughtScenarioRequest) -> RecommendationResponse:
        province = request.province.value
        result = self._predict(province, self._supplied(request))
        ties = I.tied_with(result.items)
        items = [
            CropRecommendationItem(
                rank=i + 1,
                crop=r.crop,
                predicted_yield=r.predicted_yield,
                normal_yield=r.normal_yield,
                pct_of_normal=r.pct_of_normal,
                tied_with=ties[r.crop],
                training_rows_for_crop=int(self.reference["train_rows_per_crop"][r.crop]),
                training_rows_possible=self.training_rows_possible,
                fao_water_need=self._fao_item(r.crop),
            )
            for i, r in enumerate(result.items)
        ]
        top, bottom = items[0], items[-1]
        rain, temp = result.inputs["precipitation_sum"], result.inputs["temperature_2m_mean"]
        summary = (
            f"For {province} with {rain:.1f} mm of rain a year and a mean temperature of {temp:.1f} °C, "
            f"the model keeps the largest share of a typical year's yield for {top.crop.upper()} "
            f"({top.pct_of_normal:.1f}% of typical) and the smallest for {bottom.crop.upper()} "
            f"({bottom.pct_of_normal:.1f}% of typical). "
            "These are model outputs, not adjusted by hand."
        )
        summary += tie_notes(top, bottom) + " " + I.model_caveat(self.metrics)
        return RecommendationResponse(
            scenario_name=request.scenario_name or "Custom scenario",
            province=province,
            yield_unit=self.YIELD_UNIT,
            ranking_basis=self.RANKING_BASIS,
            inputs=self._inputs(result),
            top_crop=top,
            ranked_crops=items,
            recommendation_summary=summary,
            model_name=self.model_name,
            absolute_yield_note=I.absolute_yield_caveat(self.metrics),
        )
