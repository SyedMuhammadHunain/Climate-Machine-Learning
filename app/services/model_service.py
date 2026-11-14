from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd

from app.core.config import (
    MODEL_PATH,
    PROVINCES,
    ALL_CROPS,
    DROUGHT_RESILIENT_CROPS,
    WATER_INTENSIVE_CROPS,
    MODERATE_WATER_CROPS,
    PROVINCE_DEFAULTS,
)
from app.schemas.predict import SinglePredictionRequest, SinglePredictionResponse
from app.schemas.recommend import (
    DroughtScenarioRequest,
    CropRecommendationItem,
    WaterIntensiveWarning,
    RecommendationResponse,
)


class ModelService:
    """Encapsulates the pre-trained Scikit-Learn pipeline for low-water crop yield inference."""

    FEATURE_COLS = [
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

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {self.model_path}. "
                "Ensure models/crop_yield_pipeline.joblib exists."
            )
        self.pipeline = joblib.load(self.model_path)

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

    def predict_single(self, request: SinglePredictionRequest) -> SinglePredictionResponse:
        province_val = request.province.value
        crop_val = request.crop.value

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

        category, is_resilient, risk_desc = self.classify_crop(crop_val)
        row = {
            **env_dict,
            "Crop": crop_val,
            "Is_Drought_Resilient": is_resilient,
        }

        input_df = pd.DataFrame([row])[self.FEATURE_COLS]
        raw_pred = float(self.pipeline.predict(input_df)[0])
        predicted_yield = float(np.round(np.clip(raw_pred, a_min=0.0, a_max=None), 2))

        return SinglePredictionResponse(
            crop=crop_val,
            province=province_val,
            predicted_yield_kg_ha=predicted_yield,
            water_category=category,
            is_drought_resilient=is_resilient,
            risk_assessment=risk_desc,
            features_used=row,
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
        preds = self.pipeline.predict(sim_df[self.FEATURE_COLS])
        sim_df["Predicted_Yield"] = np.round(np.clip(preds, a_min=0.0, a_max=None), 2)

        # 1. Drought-Resilient ranking
        resilient_df = sim_df[sim_df["Is_Drought_Resilient"]].sort_values(
            by="Predicted_Yield", ascending=False
        ).reset_index(drop=True)

        ranked_resilient: List[CropRecommendationItem] = []
        for idx, r in resilient_df.iterrows():
            ranked_resilient.append(CropRecommendationItem(
                rank=int(idx + 1),
                crop=r["Crop"],
                predicted_yield_kg_ha=float(r["Predicted_Yield"]),
                water_category=r["_category"],
                is_drought_resilient=True,
                risk_assessment=r["_risk"],
            ))

        top_crop = ranked_resilient[0] if ranked_resilient else None

        # 2. All crops ranked for comparison chart
        all_sorted = sim_df.sort_values(by="Predicted_Yield", ascending=False).reset_index(drop=True)
        all_items: List[CropRecommendationItem] = []
        for idx, r in all_sorted.iterrows():
            all_items.append(CropRecommendationItem(
                rank=int(idx + 1),
                crop=r["Crop"],
                predicted_yield_kg_ha=float(r["Predicted_Yield"]),
                water_category=r["_category"],
                is_drought_resilient=bool(r["Is_Drought_Resilient"]),
                risk_assessment=r["_risk"],
            ))

        # 3. Water-intensive warnings
        water_intensive_df = sim_df[sim_df["Crop"].isin(WATER_INTENSIVE_CROPS)]
        warnings: List[WaterIntensiveWarning] = []
        for _, r in water_intensive_df.iterrows():
            warnings.append(WaterIntensiveWarning(
                crop=r["Crop"],
                predicted_yield_kg_ha=float(r["Predicted_Yield"]),
                warning=(
                    f"{r['Crop'].upper()} has excessive irrigation demand. Cultivation under "
                    f"drought stress risks severe crop failure and groundwater depletion."
                ),
            ))

        summary = (
            f"For {province_val} under drought stress (Precipitation={request.precipitation_sum:.1f}mm, "
            f"Mean Temp={request.temperature_2m_mean:.1f}°C), the optimal low-water crop is "
            f"{top_crop.crop.upper()} with an expected yield of {top_crop.predicted_yield_kg_ha:,.1f} kg/ha. "
            f"High-water crops ({', '.join(WATER_INTENSIVE_CROPS)}) should be avoided."
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
