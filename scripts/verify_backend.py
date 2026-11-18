import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import joblib
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from climate_machine_learning import features as F

REFERENCE = json.loads((ROOT_DIR / "models" / "reference.json").read_text())
PIPELINE = joblib.load(ROOT_DIR / "models" / "crop_yield_pipeline.joblib")

def run_verification():
    print("=" * 70)
    print("STARTING BACKEND VERIFICATION SUITE")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Health check test
        print("\n[TEST 1] GET /health")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["status"] == "healthy", f"Status degraded: {data}"
        assert data["model_loaded"] is True, "Model not loaded"
        print("  -> PASSED: Health check returned 200 OK and model is loaded in memory.")

        # 2. Metadata endpoint test
        print("\n[TEST 2] GET /api/v1/metadata")
        res = client.get("/api/v1/metadata")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert len(data["provinces"]) == 3, f"Unexpected provinces count: {data['provinces']}"
        assert len(data["all_crops"]) == 13, f"Unexpected crops count: {data['all_crops']}"
        assert len(data["preset_scenarios"]) >= 3, "Missing preset scenarios"
        print(f"  -> PASSED: Retrieved 3 provinces ({data['provinces']}) and 13 crops.")

        # 3. Single Prediction (valid, inside the trained range)
        print("\n[TEST 3] POST /api/v1/predict (valid Wheat input)")
        payload = {
            "province": "Punjab",
            "crop": "wheat",
            "precipitation_sum": 350.0,
            "temperature_2m_mean": 25.0,
            "fao_evapotranspiration": 4.6,
        }
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["crop"] == "wheat"
        assert data["province"] == "Punjab"
        assert data["predicted_yield"] > 0
        assert "unit" in data["yield_unit"]
        assert data["training_rows_for_crop"] > 0
        print(f"  -> PASSED: wheat yield {data['predicted_yield']:.2f} ({data['yield_unit']}), {data['pct_of_normal']:.1f}% of normal")

        # 4. Root alias /predict (Rubric compliance)
        print("\n[TEST 4] POST /predict (Root alias verification)")
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Expected 200 on /predict, got {res.status_code}"
        print("  -> PASSED: Root /predict endpoint is functional.")

        # 5. Recommendation Engine (Valid Punjab Severe Drought Scenario)
        print("\n[TEST 5] POST /api/v1/recommend (Severe Punjab Drought Scenario)")
        drought_payload = {
            "province": "Punjab",
            "area": 50.0,
            "precipitation_sum": 1.8,
            "temperature_2m_mean": 35.0,
            "fao_evapotranspiration": 7.8,
            "scenario_name": "Test Dry Spell"
        }
        res = client.post("/api/v1/recommend", json=drought_payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        top_crop = data["top_resilient_crop"]
        assert top_crop is not None
        assert top_crop["is_drought_resilient"] is True
        assert top_crop["crop"] in ["dates", "pomegranate"], f"Expected deep-root crop under severe drought, got {top_crop['crop']}"
        assert top_crop["crop"] != "onion", "Onion must not be unconditionally #1 in severe drought"
        assert len(data["ranked_drought_resilient"]) == 5
        assert len(data["all_crops_ranked"]) == 13
        assert len(data["water_intensive_warnings"]) >= 3
        print(f"  -> PASSED: Top recommended crop: {top_crop['crop'].upper()} ({top_crop['predicted_yield_kg_ha']} kg/ha)")
        print(f"  -> Flagged {len(data['water_intensive_warnings'])} water-intensive risks (e.g. Sugarcane, Rice).")

        # 6. Recommendation Engine (Valid Balochistan Arid Scenario)
        print("\n[TEST 6] POST /api/v1/recommend (Balochistan Arid Scenario)")
        balochistan_payload = {
            "province": "Balochistan",
            "area": 25.0,
            "precipitation_sum": 0.5,
            "temperature_2m_mean": 28.0,
            "fao_evapotranspiration": 6.5,
            "scenario_name": "Balochistan Arid Test"
        }
        res = client.post("/api/v1/recommend", json=balochistan_payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        b_data = res.json()
        b_top = b_data["top_resilient_crop"]
        assert b_top["crop"] in ["dates", "pomegranate"], f"Expected arid-hardy crop, got {b_top['crop']}"
        assert b_top["crop"] != "onion", "Onion must not be #1 in 0.5mm arid condition"
        print(f"  -> PASSED: Top Balochistan arid crop: {b_top['crop'].upper()} ({b_top['predicted_yield_kg_ha']} kg/ha)")

        # 7. Validation Error Handling: Temperature Inversion (T_max < T_min)
        print("\n[TEST 7] POST /api/v1/predict with Invalid Temperature Physics (T_max < T_min)")
        invalid_temp = {
            "province": "Punjab",
            "crop": "wheat",
            "precipitation_sum": 350.0,
            "temperature_2m_mean": 25.0,
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 30.0  # Physically impossible: min > max
        }
        res = client.post("/api/v1/predict", json=invalid_temp)
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"
        data = res.json()
        assert data["error"] == "InputValidationError"
        print(f"  -> PASSED: Correctly caught and returned 422 Unprocessable Entity with error detail.")

        # 8. Validation Error Handling: Negative Precipitation
        print("\n[TEST 8] POST /api/v1/predict with Negative Precipitation")
        invalid_precip = {
            "province": "Punjab",
            "crop": "wheat",
            "precipitation_sum": -15.0,  # Physically impossible
            "temperature_2m_mean": 25.0
        }
        res = client.post("/api/v1/predict", json=invalid_precip)
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"
        print("  -> PASSED: Correctly rejected negative precipitation with 422.")

        # 9. Validation Error Handling: Invalid Province
        print("\n[TEST 9] POST /api/v1/predict with Invalid Province")
        invalid_prov = {
            "province": "Mars",
            "crop": "wheat",
            "precipitation_sum": 350.0,
            "temperature_2m_mean": 25.0
        }
        res = client.post("/api/v1/predict", json=invalid_prov)
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"
        print("  -> PASSED: Correctly rejected unknown province.")

        # 10. Recommendation Dynamic Shift (Baseline Normal vs Severe Drought)
        print("\n[TEST 10] POST /api/v1/recommend (Baseline Normal vs Severe Drought Dynamic Shift)")
        normal_payload = {
            "province": "Punjab",
            "area": 50.0,
            "precipitation_sum": 50.0,
            "temperature_2m_mean": 24.0,
            "fao_evapotranspiration": 4.0,
            "scenario_name": "Punjab Baseline Normal"
        }
        res = client.post("/api/v1/recommend", json=normal_payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        normal_data = res.json()
        normal_top = normal_data["top_resilient_crop"]
        print(f"  -> Normal conditions top crop: {normal_top['crop'].upper()} ({normal_top['predicted_yield_kg_ha']} kg/ha)")
        assert normal_top["crop"] == "onion", f"Expected Onion elevated in normal rainfall, got {normal_top['crop']}"
        assert normal_top["crop"] != top_crop["crop"], (
            f"Top crop must differ between normal ({normal_top['crop']}) and drought ({top_crop['crop']})"
        )
        print("  -> PASSED: Ranking dynamically switches between drought (Dates) and normal moisture (Onion).")

        # 11. Static Frontend Integration & Unified Local Serving
        print("\n[TEST 11] GET Static HTML Routes & Assets (Unified Local Serving)")
        r_root = client.get("/")
        assert r_root.status_code == 200, f"Expected 200 on /, got {r_root.status_code}"
        assert "<!DOCTYPE html>" in r_root.text or "<html" in r_root.text

        r_predict = client.get("/predict.html")
        assert r_predict.status_code == 200, f"Expected 200 on /predict.html, got {r_predict.status_code}"
        assert "Crop Yield Prediction & Drought Advisory" in r_predict.text

        r_metrics = client.get("/metrics.html")
        assert r_metrics.status_code == 200, f"Expected 200 on /metrics.html, got {r_metrics.status_code}"
        assert "Model Performance & Agroclimatic Benchmarks" in r_metrics.text

        r_asset = client.get("/assets/wordmark.svg")
        assert r_asset.status_code == 200, f"Expected 200 on /assets/wordmark.svg, got {r_asset.status_code}"
        print("  -> PASSED: Successfully served root index, predict.html, metrics.html, and static SVG asset.")

        # 12. Error Contract Verification
        print("\n[TEST 12] Error Detail Contract Verification")
        res_err = client.post("/api/v1/predict", json=invalid_temp)
        err_body = res_err.json()
        assert "message" in err_body
        assert "details" in err_body or "parameter" in err_body
        print("  -> PASSED: Error response conforms to frontend error-parser contract.")

        # 13. API output == direct pipeline output (no post-processing)
        print("\n[TEST 13] /predict equals pipeline.predict on the same row")
        for prov, crop, rain, temp in [("Punjab", "wheat", 350.0, 25.0), ("Sindh", "rice", 120.0, 27.5),
                                       ("Balochistan", "dates", 90.0, 18.5)]:
            body = {"province": prov, "crop": crop, "precipitation_sum": rain, "temperature_2m_mean": temp}
            got = client.post("/api/v1/predict", json=body).json()
            row, _ = F.build_inference_row(prov, crop, {"precipitation_sum": rain, "temperature_2m_mean": temp}, REFERENCE)
            direct = float(PIPELINE.predict(row)[0])
            assert np.isclose(got["predicted_yield"], direct, rtol=1e-9), (prov, crop, got["predicted_yield"], direct)
            normal_row, _ = F.build_inference_row(prov, crop, dict(REFERENCE["provinces"][prov]["median"]), REFERENCE)
            normal = float(PIPELINE.predict(normal_row)[0])
            assert np.isclose(got["normal_yield"], normal, rtol=1e-9)
            assert np.isclose(got["pct_of_normal"], direct / normal * 100.0, rtol=1e-9)
        print("  -> PASSED: yield, normal yield and pct_of_normal equal direct model output (3 cases).")

        # 14. Out-of-range input: 422 with field name and allowed range
        print("\n[TEST 14] /predict rejects input outside the trained range")
        far = {"province": "Punjab", "crop": "wheat", "precipitation_sum": 5.0, "temperature_2m_mean": 25.0}
        res = client.post("/api/v1/predict", json=far)
        assert res.status_code == 422, res.text
        body = res.json()
        assert body["parameter"] == "precipitation_sum", body
        lo, hi = REFERENCE["provinces"]["Punjab"]["range"]["precipitation_sum"]
        assert f"{lo:.1f}" in body["message"] and f"{hi:.1f}" in body["message"], body["message"]
        hot = {"province": "Punjab", "crop": "wheat", "precipitation_sum": 350.0, "temperature_2m_mean": 42.0}
        res = client.post("/api/v1/predict", json=hot)
        assert res.status_code == 422 and res.json()["parameter"] == "temperature_2m_mean"
        print("  -> PASSED: 422 names the field and the allowed range.")

        # 15. Input sources are reported
        print("\n[TEST 15] /predict labels where each input came from")
        body = {"province": "Sindh", "crop": "mango", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5}
        data = client.post("/api/v1/predict", json=body).json()
        inp = data["inputs"]
        assert inp["precipitation_sum"] == {"value": 150.0, "source": "user"}
        assert inp["temperature_2m_mean"]["source"] == "user"
        assert inp["shortwave_radiation"]["source"] == "province_median_training_years"
        assert inp["temperature_2m_max"]["source"] == "user_mean_plus_province_median_offset"
        print("  -> PASSED: user vs province-median sources reported.")

        # 16. Removed / unknown fields are rejected, not silently ignored
        print("\n[TEST 16] /predict rejects unknown fields (area, climate_index)")
        for extra in ({"area": 50.0}, {"climate_index": 0.5}, {"bajra": 1}):
            res = client.post("/api/v1/predict", json={**payload, **extra})
            assert res.status_code == 422, (extra, res.status_code)
        print("  -> PASSED: unknown fields give 422.")

        # 17. Partial temperature order check
        print("\n[TEST 17] /predict rejects max below mean")
        bad = {**payload, "temperature_2m_max": 20.0}
        assert client.post("/api/v1/predict", json=bad).status_code == 422
        print("  -> PASSED: max < mean rejected.")

        # 18. Metadata exposes real ranges for the UI
        print("\n[TEST 18] /api/v1/metadata includes reference ranges")
        meta = client.get("/api/v1/metadata").json()
        assert set(meta["reference"]) == {"Punjab", "Sindh", "Balochistan"}
        assert meta["reference"]["Punjab"]["range"]["precipitation_sum"] == REFERENCE["provinces"]["Punjab"]["range"]["precipitation_sum"]
        assert meta["features_count"] == len(F.FEATURE_COLS)
        print("  -> PASSED: metadata carries training ranges and medians.")

    print("\n" + "=" * 70)
    print("ALL BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
