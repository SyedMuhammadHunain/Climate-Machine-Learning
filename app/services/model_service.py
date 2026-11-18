import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd

from climate_machine_learning import features as F
from app.core.exceptions import AgronomicValidationError
from app.core.config import (
    MODEL_PATH,
    REFERENCE_PATH,
    METRICS_PATH,
    LEGACY_MODEL_PATH,
    PROVINCES,
    ALL_CROPS,
    DROUGHT_RESILIENT_CROPS,
    WATER_INTENSIVE_CROPS,
    MODERATE_WATER_CROPS,
    PROVINCE_DEFAULTS,
)
from app.schemas.predict import SinglePredictionRequest, SinglePredictionResponse, InputValue
from app.schemas.recommend import (
    DroughtScenarioRequest,
    CropRecommendationItem,
    WaterIntensiveWarning,
    RecommendationResponse,
)


class ModelService:
    """Encapsulates the pre-trained Scikit-Learn pipeline for low-water crop yield inference."""

    # LEGACY (old /recommend only, removed in T5)
    LEGACY_FEATURE_COLS = [
        "Area",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "Combined_temperature_normalized",
        "precipitation_sum",
        "precipitation_sum_normalized",
        "shortwave_radiation",
        "shortwave_radiation_normalized",
        "fao_evapotranspiration",
        "fao_evapotranspiration_normalized",
        "precipitation_hours",
        "precipitation_hours_normalized",
        "Climate_Index",
        "Crop",
        "Province",
        "Is_Drought_Resilient",
    ]

    CROP_PHYSIOLOGY: Dict[str, Dict[str, float]] = {
        "dates": {"water_vulnerability": 0.08, "crit_temp": 42.0},
        "grapes": {"water_vulnerability": 0.12, "crit_temp": 36.0},
        "pomegranate": {"water_vulnerability": 0.18, "crit_temp": 40.0},
        "wheat": {"water_vulnerability": 0.32, "crit_temp": 32.0},
        "onion": {"water_vulnerability": 0.55, "crit_temp": 30.0},
        "apple": {"water_vulnerability": 0.40, "crit_temp": 32.0},
        "mango": {"water_vulnerability": 0.35, "crit_temp": 40.0},
        "potato": {"water_vulnerability": 0.55, "crit_temp": 30.0},
        "tomato": {"water_vulnerability": 0.48, "crit_temp": 35.0},
        "cotton": {"water_vulnerability": 0.65, "crit_temp": 38.0},
        "banana": {"water_vulnerability": 0.75, "crit_temp": 38.0},
        "sugarcane": {"water_vulnerability": 0.85, "crit_temp": 38.0},
        "rice": {"water_vulnerability": 0.85, "crit_temp": 36.0},
    }

    YIELD_UNIT = "dataset yield units (unit not confirmed)"

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.reference: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.legacy_pipeline = None
        self.load_model()

    @staticmethod
    def _single_thread(pipeline) -> None:
        """One thread at inference: avoids joblib overhead for single-row predictions."""
        reg = pipeline.named_steps.get("regressor")
        reg = getattr(reg, "regressor_", reg)
        if reg is not None and hasattr(reg, "n_jobs"):
            reg.n_jobs = 1

    def load_model(self) -> None:
        for path in (self.model_path, REFERENCE_PATH, METRICS_PATH, LEGACY_MODEL_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. Run scripts/train.py to create the model files."
                )
        self.pipeline = joblib.load(self.model_path)
        self._single_thread(self.pipeline)
        self.reference = json.loads(Path(REFERENCE_PATH).read_text())
        self.metrics = json.loads(Path(METRICS_PATH).read_text())
        # LEGACY: old /recommend only (removed in T5)
        self.legacy_pipeline = joblib.load(LEGACY_MODEL_PATH)
        self._single_thread(self.legacy_pipeline)

    @staticmethod
    def classify_crop(crop_name: str) -> Tuple[str, bool, str]:
        normalized_crop = crop_name.lower().strip()
        if normalized_crop in DROUGHT_RESILIENT_CROPS:
            return (
                "Low-Water Resilient",
                True,
                "Strong drought tolerance. Viable under severe moisture deficit.",
            )
        elif normalized_crop in WATER_INTENSIVE_CROPS:
            return (
                "High-Water Intensive",
                False,
                "High water demand. Extreme failure risk under drought conditions.",
            )
        else:
            return (
                "Moderate Water",
                False,
                "Moderate water requirement. Supplemental irrigation required under dry spells.",
            )

    def _resolve_environment(
        self,
        province: str,
        area: float,
        precipitation_sum: float,
        temperature_2m_mean: float,
        temperature_2m_max: float | None = None,
        temperature_2m_min: float | None = None,
        fao_evapotranspiration: float | None = None,
        shortwave_radiation: float | None = None,
        precipitation_hours: float | None = None,
        climate_index: float | None = None,
    ) -> Dict[str, float]:
        defaults = PROVINCE_DEFAULTS.get(province, PROVINCE_DEFAULTS["Punjab"])

        t_max = temperature_2m_max if temperature_2m_max is not None else max(temperature_2m_mean + 6.0, defaults["temperature_2m_max"])
        t_min = temperature_2m_min if temperature_2m_min is not None else min(temperature_2m_mean - 6.0, defaults["temperature_2m_min"])
        et0 = fao_evapotranspiration if fao_evapotranspiration is not None else defaults["fao_evapotranspiration"]
        sw_rad = shortwave_radiation if shortwave_radiation is not None else defaults["shortwave_radiation"]
        p_hours = precipitation_hours if precipitation_hours is not None else min(precipitation_sum * 1.5, defaults["precipitation_hours"])
        c_index = climate_index if climate_index is not None else defaults["climate_index"]

        return {
            "Area": float(area),
            "Province": province,
            "temperature_2m_max": float(t_max),
            "temperature_2m_min": float(t_min),
            "temperature_2m_mean": float(temperature_2m_mean),
            "Combined_temperature_normalized": float((temperature_2m_mean - 10.0) / 30.0),
            "precipitation_sum": float(precipitation_sum),
            "precipitation_sum_normalized": float(precipitation_sum / 200.0),
            "shortwave_radiation": float(sw_rad),
            "shortwave_radiation_normalized": float(sw_rad / 30.0),
            "fao_evapotranspiration": float(et0),
            "fao_evapotranspiration_normalized": float(et0 / 10.0),
            "precipitation_hours": float(p_hours),
            "precipitation_hours_normalized": float(p_hours / 100.0),
            "Climate_Index": float(c_index),
        }

    @classmethod
    def calculate_agroclimatic_elasticity(
        cls,
        crop: str,
        province: str,
        precipitation_sum: float,
        temperature_2m_max: float,
        fao_evapotranspiration: float,
    ) -> float:
        """Computes physiological yield response factor based on water deficit and heat stress."""
        crop_clean = crop.lower().strip()
        params = cls.CROP_PHYSIOLOGY.get(
            crop_clean, {"water_vulnerability": 0.40, "crit_temp": 35.0}
        )
        kw = params["water_vulnerability"]
        tcrit = params["crit_temp"]

        defaults = PROVINCE_DEFAULTS.get(province, PROVINCE_DEFAULTS["Punjab"])
        p_ref = defaults["precipitation_sum"]
        et0_ref = defaults["fao_evapotranspiration"]

        # 1. Moisture balance factor
        if precipitation_sum < p_ref:
            moisture_stress = max(0.0, (p_ref - precipitation_sum) / p_ref)
            w_factor = 1.0 - kw * min(1.0, moisture_stress)
        else:
            w_factor = 1.0 + 0.15 * min(1.0, (precipitation_sum - p_ref) / p_ref)

        # Evaporative stress
        if fao_evapotranspiration > et0_ref:
            et0_stress = min(0.20, ((fao_evapotranspiration - et0_ref) / et0_ref) * 0.15 * kw)
            w_factor = max(0.15, w_factor - et0_stress)

        # 2. Thermal stress factor
        if temperature_2m_max > tcrit:
            excess = temperature_2m_max - tcrit
            h_factor = max(0.25, 1.0 - 0.035 * excess)
        else:
            h_factor = 1.0

        return float(np.clip(w_factor * h_factor, a_min=0.10, a_max=1.50))

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

    def predict_single(self, request: SinglePredictionRequest) -> SinglePredictionResponse:
        province, crop = request.province.value, request.crop.value
        supplied = {k: getattr(request, k) for k in F.BASE_INPUTS if getattr(request, k) is not None}
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
            inputs={k: InputValue(value=float(row.iloc[0][k]), source=sources[k]) for k in F.BASE_INPUTS},
            model_name=self.metrics["models"][self.metrics["chosen"]["name"]]["label"],
        )

    def recommend_crops(self, request: DroughtScenarioRequest) -> RecommendationResponse:
        province_val = request.province.value
        env_dict = self._resolve_environment(
            province=province_val,
            area=request.area,
            precipitation_sum=request.precipitation_sum,
            temperature_2m_mean=request.temperature_2m_mean,
            temperature_2m_max=request.temperature_2m_max,
            temperature_2m_min=request.temperature_2m_min,
            fao_evapotranspiration=request.fao_evapotranspiration,
            shortwave_radiation=request.shortwave_radiation,
            precipitation_hours=request.precipitation_hours,
            climate_index=request.climate_index,
        )

        records = []
        for crop in ALL_CROPS:
            category, is_resilient, risk_desc = self.classify_crop(crop)
            records.append({
                **env_dict,
                "Crop": crop,
                "Is_Drought_Resilient": is_resilient,
                "_category": category,
                "_risk": risk_desc,
            })

        sim_df = pd.DataFrame(records)
        preds = self.legacy_pipeline.predict(sim_df[self.LEGACY_FEATURE_COLS])
        p_sum = env_dict["precipitation_sum"]
        t_max = env_dict["temperature_2m_max"]
        et0 = env_dict["fao_evapotranspiration"]

        elasticities = np.array([
            self.calculate_agroclimatic_elasticity(
                crop=r["Crop"],
                province=province_val,
                precipitation_sum=p_sum,
                temperature_2m_max=t_max,
                fao_evapotranspiration=et0,
            )
            for r in records
        ])
        predicted_yields = np.round(np.clip(preds * elasticities, a_min=0.0, a_max=None), 2)
        for i, r in enumerate(records):
            r["Predicted_Yield"] = float(predicted_yields[i])

        # 1. Drought-Resilient ranking based on Drought Resilience & Water-Use Efficiency
        defaults = PROVINCE_DEFAULTS.get(province_val, PROVINCE_DEFAULTS["Punjab"])
        p_ref = defaults["precipitation_sum"]
        deficit = max(0.0, (p_ref - request.precipitation_sum) / p_ref)
        w_prod = float(np.clip(1.0 - deficit, 0.08, 0.92))

        resilient_records = [r for r in records if r["Is_Drought_Resilient"]]
        max_resilient_yield = max((r["Predicted_Yield"] for r in resilient_records), default=1.0) or 1.0

        for r in resilient_records:
            c = r["Crop"]
            kw = self.CROP_PHYSIOLOGY.get(c, {}).get("water_vulnerability", 0.40)
            el = self.calculate_agroclimatic_elasticity(
                crop=c,
                province=province_val,
                precipitation_sum=p_sum,
                temperature_2m_max=t_max,
                fao_evapotranspiration=et0,
            )
            drought_resilience = el * (1.0 - deficit * kw)
            prod_norm = r["Predicted_Yield"] / max_resilient_yield
            r["_rank_score"] = (1.0 - w_prod) * drought_resilience + w_prod * prod_norm

        resilient_records.sort(key=lambda r: (r["_rank_score"], r["Predicted_Yield"]), reverse=True)

        ranked_resilient: List[CropRecommendationItem] = [
            CropRecommendationItem(
                rank=idx + 1,
                crop=r["Crop"],
                predicted_yield_kg_ha=r["Predicted_Yield"],
                water_category=r["_category"],
                is_drought_resilient=True,
                risk_assessment=r["_risk"],
            )
            for idx, r in enumerate(resilient_records)
        ]

        top_crop = ranked_resilient[0] if ranked_resilient else None

        # 2. All crops ranked for comparison chart
        all_sorted = sorted(records, key=lambda r: r["Predicted_Yield"], reverse=True)
        all_items: List[CropRecommendationItem] = [
            CropRecommendationItem(
                rank=idx + 1,
                crop=r["Crop"],
                predicted_yield_kg_ha=r["Predicted_Yield"],
                water_category=r["_category"],
                is_drought_resilient=bool(r["Is_Drought_Resilient"]),
                risk_assessment=r["_risk"],
            )
            for idx, r in enumerate(all_sorted)
        ]

        # 3. Water-intensive warnings
        water_intensive_rows = [r for r in records if r["Crop"] in WATER_INTENSIVE_CROPS]
        warnings: List[WaterIntensiveWarning] = [
            WaterIntensiveWarning(
                crop=r["Crop"],
                predicted_yield_kg_ha=r["Predicted_Yield"],
                warning=(
                    f"{r['Crop'].upper()} has excessive irrigation demand. Cultivation under "
                    f"drought stress risks severe crop failure and groundwater depletion."
                ),
            )
            for r in water_intensive_rows
        ]

        advisory_rationale = (
            f"{top_crop.crop.upper()} is prioritized due to superior drought resilience, "
            f"minimal water vulnerability, and deep root architecture under moisture deficit."
            if deficit > 0.4
            else f"{top_crop.crop.upper()} is prioritized due to high productivity and optimal water-use efficiency."
        )

        summary = (
            f"For {province_val} (Precipitation={request.precipitation_sum:.1f}mm, "
            f"Mean Temp={request.temperature_2m_mean:.1f}°C), the optimal low-water crop is "
            f"{top_crop.crop.upper()} with an expected yield of {top_crop.predicted_yield_kg_ha:,.1f} kg/ha. "
            f"{advisory_rationale} High-water crops ({', '.join(WATER_INTENSIVE_CROPS)}) should be avoided."
        )

        return RecommendationResponse(
            scenario_name=request.scenario_name or "Custom Drought Scenario",
            province=province_val,
            input_conditions=env_dict,
            top_resilient_crop=top_crop,
            ranked_drought_resilient=ranked_resilient,
            all_crops_ranked=all_items,
            water_intensive_warnings=warnings,
            recommendation_summary=summary,
        )
