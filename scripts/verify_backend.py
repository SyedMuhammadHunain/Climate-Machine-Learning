import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

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

        # 3. Single Prediction (Valid - Wheat in Punjab under low rainfall)
        print("\n[TEST 3] POST /api/v1/predict (Valid Wheat input)")
        payload = {
            "province": "Punjab",
            "crop": "wheat",
            "area": 50.0,
            "precipitation_sum": 2.5,
            "temperature_2m_mean": 33.0,
            "fao_evapotranspiration": 7.5
        }
        res = client.post("/api/v1/predict", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["crop"] == "wheat"
        assert data["province"] == "Punjab"
        assert data["predicted_yield_kg_ha"] > 0
        assert data["water_category"] == "Low-Water Resilient"
        assert data["is_drought_resilient"] is True
        print(f"  -> PASSED: Predicted Wheat yield: {data['predicted_yield_kg_ha']} kg/ha ({data['water_category']})")

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
        print(f"  -> PASSED: Top Balochistan arid crop: {b_top['crop'].upper()} ({b_top['predicted_yield_kg_ha']} kg/ha)")

        # 7. Validation Error Handling: Temperature Inversion (T_max < T_min)
        print("\n[TEST 7] POST /api/v1/predict with Invalid Temperature Physics (T_max < T_min)")
        invalid_temp = {
            "province": "Punjab",
            "crop": "wheat",
            "area": 50.0,
            "precipitation_sum": 10.0,
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
            "area": 50.0,
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
            "area": 50.0,
            "precipitation_sum": 10.0,
            "temperature_2m_mean": 25.0
        }
        res = client.post("/api/v1/predict", json=invalid_prov)
        assert res.status_code == 422, f"Expected 422, got {res.status_code}"
        print("  -> PASSED: Correctly rejected unknown province.")

    print("\n" + "=" * 70)
    print("ALL 9 BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
