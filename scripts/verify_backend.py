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
        assert sorted(data["all_crops"]) == sorted(REFERENCE["crops"])
        for gone in ("drought_resilient_crops", "water_intensive_crops", "moderate_water_crops",
                     "preset_scenarios", "province_defaults"):
            assert gone not in data, f"hand-made field still served: {gone}"
        assert len(data["scenarios"]) == 6, "driest + wettest real training year per province"
        for sc in data["scenarios"]:
            assert sc["year"] <= 2016 and sc["kind"] in ("driest", "wettest")
            r = client.post("/api/v1/recommend", json={"province": sc["province"], **sc["inputs"]})
            assert r.status_code == 200, (sc["id"], r.text)
        print(f"  -> PASSED: Retrieved 3 provinces ({data['provinces']}), 13 crops, 6 real-year scenarios that the API accepts.")

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

        # 5. Recommendation: ranks all 13 crops by model output
        print("\n[TEST 5] POST /api/v1/recommend (driest Punjab year)")
        lo_p, hi_p = REFERENCE["provinces"]["Punjab"]["range"]["precipitation_sum"]
        dry_rain = float(np.ceil(lo_p * 10) / 10)
        rec_payload = {"province": "Punjab", "precipitation_sum": dry_rain, "temperature_2m_mean": 25.0,
                       "scenario_name": "Test driest Punjab"}
        res = client.post("/api/v1/recommend", json=rec_payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        items = data["ranked_crops"]
        assert [i["rank"] for i in items] == list(range(1, 14))
        assert sorted(i["crop"] for i in items) == sorted(REFERENCE["crops"]) and len(items) == 13
        assert data["top_crop"] == items[0]
        pcts = [i["pct_of_normal"] for i in items]
        assert pcts == sorted(pcts, reverse=True), "not sorted by pct_of_normal"
        for banned in ("water_intensive_warnings", "ranked_drought_resilient", "top_resilient_crop", "all_crops_ranked"):
            assert banned not in data, banned
        print(f"  -> PASSED: 13 crops ranked, top={items[0]['crop']} ({items[0]['pct_of_normal']:.1f}% of typical)")

        # 6. Every number equals direct model output
        print("\n[TEST 6] /recommend numbers equal pipeline.predict")
        user_in = {"precipitation_sum": dry_rain, "temperature_2m_mean": 25.0}
        for it in items:
            row, _ = F.build_inference_row("Punjab", it["crop"], user_in, REFERENCE)
            direct = float(PIPELINE.predict(row)[0])
            nrow, _ = F.build_inference_row("Punjab", it["crop"], dict(REFERENCE["provinces"]["Punjab"]["median"]), REFERENCE)
            normal = float(PIPELINE.predict(nrow)[0])
            assert np.isclose(it["predicted_yield"], direct, rtol=1e-9), it["crop"]
            assert np.isclose(it["normal_yield"], normal, rtol=1e-9), it["crop"]
            assert np.isclose(it["pct_of_normal"], direct / normal * 100.0, rtol=1e-9), it["crop"]
        assert items[0]["crop"].upper() in data["recommendation_summary"].upper()
        assert f"{items[0]['pct_of_normal']:.1f}%" in data["recommendation_summary"]
        assert data["inputs"]["precipitation_sum"]["source"] == "user"
        print("  -> PASSED: yields, normal yields, ratios and summary match the model.")

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

        # 10. Weather changes the ranking; ties are broken deterministically
        print("\n[TEST 10] /recommend: dry vs typical year changes the ranking; ties broken by crop name")
        for prov in ("Sindh", "Balochistan"):
            info = REFERENCE["provinces"][prov]
            dry = float(np.ceil(info["range"]["precipitation_sum"][0] * 10) / 10)
            t = round(info["median"]["temperature_2m_mean"], 1)
            order_dry = [i["crop"] for i in client.post("/api/v1/recommend", json={"province": prov, "precipitation_sum": dry, "temperature_2m_mean": t}).json()["ranked_crops"]]
            order_med = [i["crop"] for i in client.post("/api/v1/recommend", json={"province": prov, "precipitation_sum": round(info["median"]["precipitation_sum"], 1), "temperature_2m_mean": t}).json()["ranked_crops"]]
            assert order_dry != order_med, f"{prov}: ranking identical for driest and typical rain"
        med = REFERENCE["provinces"]["Punjab"]["median"]
        tie = client.post("/api/v1/recommend", json={"province": "Punjab", **med}).json()["ranked_crops"]
        assert all(abs(i["pct_of_normal"] - 100.0) < 1e-9 for i in tie), "median year must equal 100% for every crop"
        assert [i["crop"] for i in tie] == sorted(i["crop"] for i in tie), "ties must be ordered by crop name"
        again = client.post("/api/v1/recommend", json={"province": "Punjab", **med}).json()["ranked_crops"]
        assert [i["crop"] for i in again] == [i["crop"] for i in tie]
        print("  -> PASSED: ranking depends on weather; exact ties ordered alphabetically and stable.")

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

        # 19. /recommend input rules match /predict
        print("\n[TEST 19] /recommend rejects out-of-range and unknown fields")
        base = {"province": "Punjab", "precipitation_sum": 350.0, "temperature_2m_mean": 25.0}
        assert client.post("/api/v1/recommend", json=base).status_code == 200
        r = client.post("/api/v1/recommend", json={**base, "precipitation_sum": 5.0})
        assert r.status_code == 422 and r.json()["parameter"] == "precipitation_sum"
        for extra in ({"area": 50.0}, {"climate_index": 0.4}, {"crop": "wheat"}):
            assert client.post("/api/v1/recommend", json={**base, **extra}).status_code == 422, extra
        print("  -> PASSED: same validation as /predict.")

        # 20. FAO water-need context comes from a cited file, only where FAO lists the crop
        print("\n[TEST 20] FAO water-need context")
        fao = json.loads((ROOT_DIR / "data" / "reference" / "fao_crop_water_needs.json").read_text())
        assert fao["source"]["url"].startswith("https://www.fao.org/") and fao["source"]["accessed"]
        listed = set(fao["crops"])
        assert listed == {"banana", "cotton", "onion", "potato", "rice", "sugarcane", "tomato", "wheat"}, listed
        got = {i["crop"]: i for i in client.post("/api/v1/recommend", json=base).json()["ranked_crops"]}
        for crop, item in got.items():
            if crop in listed:
                assert item["fao_water_need"]["mm_min"] == fao["crops"][crop]["mm_min"], crop
                assert item["fao_water_need"]["drought_sensitivity"] == fao["crops"][crop]["drought_sensitivity"]
            else:
                assert item["fao_water_need"] is None, crop
        print("  -> PASSED: FAO values shown for 8 listed crops; none invented for the other 5.")

        # 21. /metrics returns the saved metrics.json untouched
        print("\n[TEST 21] GET /api/v1/metrics equals models/metrics.json")
        saved = json.loads((ROOT_DIR / "models" / "metrics.json").read_text())
        for path in ("/api/v1/metrics", "/metrics"):
            res = client.get(path)
            assert res.status_code == 200, (path, res.status_code)
            assert res.json() == saved, f"{path} body differs from models/metrics.json"
        assert res.json()["chosen"]["name"] in res.json()["models"]
        print("  -> PASSED: endpoint body equals the file (no recomputed or hardcoded values).")

        # 22. No hand-made logic or hardcoded numbers left in app/
        print("\n[TEST 22] app/ has no banned words and no hardcoded model numbers")
        import ast
        banned = ("bajra", "jowar", "fallback", "simulat", "demo", "dummy", "mock", "physiolog", "elasticit",
                  "drought_resilient", "water_intensive", "moderate_water", "province_defaults",
                  "preset_scenarios", "kg/ha", "legacy")
        allowed_floats = {0.0, 1.0, 100.0}  # 100.0 = percent conversion
        problems = []
        for path in sorted((ROOT_DIR / "app").rglob("*.py")):
            text = path.read_text()
            for word in banned:
                if word in text.lower():
                    problems.append(f"{path.relative_to(ROOT_DIR)}: banned word {word!r}")
            tree = ast.parse(text)
            in_field = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("Field", "Query"):
                    in_field.update(id(n) for n in ast.walk(node))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float) and id(node) not in in_field:
                    if node.value not in allowed_floats:
                        problems.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno}: hardcoded number {node.value}")
        assert not problems, "\n".join(problems)
        from app.schemas.common import CropEnum, ProvinceEnum
        assert sorted(c.value for c in CropEnum) == sorted(REFERENCE["crops"]), "CropEnum differs from trained crops"
        assert sorted(p.value for p in ProvinceEnum) == sorted(REFERENCE["provinces"]), "ProvinceEnum differs from trained provinces"
        print("  -> PASSED: clean. Enums match the trained crops and provinces.")

        # 23. Data completeness is reported per crop
        print("\n[TEST 23] responses say how many training rows exist vs possible")
        possible = len(REFERENCE["provinces"]) * (REFERENCE["train_years"][1] - REFERENCE["train_years"][0] + 1)
        r = client.post("/api/v1/recommend", json={"province": "Punjab", "precipitation_sum": 350.0, "temperature_2m_mean": 25.0}).json()
        for it in r["ranked_crops"]:
            assert it["training_rows_possible"] == possible, it
            assert it["training_rows_for_crop"] == REFERENCE["train_rows_per_crop"][it["crop"]]
        p = client.post("/api/v1/predict", json={"province": "Punjab", "crop": "grapes", "precipitation_sum": 350.0, "temperature_2m_mean": 25.0}).json()
        assert p["training_rows_possible"] == possible and p["training_rows_for_crop"] < possible
        print(f"  -> PASSED: grapes has {p['training_rows_for_crop']} of {possible} possible training rows.")

        # 24. Soil wetness inputs (second data source)
        print("\n[TEST 24] soil wetness inputs: accepted, labelled, validated")
        base = {"province": "Sindh", "crop": "rice", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5}
        none = client.post("/api/v1/predict", json=base).json()
        assert none["inputs"]["soil_wetness_mean"]["source"] == "province_median_training_years"
        assert none["inputs"]["soil_wetness_max"]["source"] == "province_median_training_years"
        lo, hi = REFERENCE["provinces"]["Sindh"]["range"]["soil_wetness_mean"]
        given = client.post("/api/v1/predict", json={**base, "soil_wetness_mean": (lo + hi) / 2}).json()
        assert given["inputs"]["soil_wetness_mean"] == {"value": (lo + hi) / 2, "source": "user"}
        assert set(REFERENCE["provinces"]["Sindh"]["range"]) >= {"soil_wetness_mean", "soil_wetness_max"}
        out = client.post("/api/v1/predict", json={**base, "soil_wetness_mean": hi + 0.1})
        assert out.status_code == 422 and out.json()["parameter"] == "soil_wetness_mean", out.text
        for bad in ({"soil_wetness_mean": 1.5}, {"soil_wetness_mean": 0.45, "soil_wetness_max": 0.40}):
            assert client.post("/api/v1/predict", json={**base, **bad}).status_code == 422, bad
        r = client.post("/api/v1/recommend", json={k: v for k, v in base.items() if k != "crop"} | {"soil_wetness_mean": (lo + hi) / 2})
        assert r.status_code == 200
        print("  -> PASSED: soil wetness accepted, labelled, range-checked.")

    print("\n" + "=" * 70)
    print("ALL BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
