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
        shown = [round(i["pct_of_normal"], 1) for i in items]   # the share on screen; equal shares are then in name order (TEST 34)
        assert shown == sorted(shown, reverse=True), "not sorted by the displayed pct_of_normal"
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
        # the message must show the soil range with enough digits to be usable (0.348 to 0.386, not "0.3 to 0.4")
        assert f"{lo:.4g}" in out.json()["message"] and f"{hi:.4g}" in out.json()["message"], out.json()["message"]
        assert f"{lo:.4g}" != f"{hi:.4g}"
        for bad in ({"soil_wetness_mean": 1.5}, {"soil_wetness_mean": 0.45, "soil_wetness_max": 0.40}):
            assert client.post("/api/v1/predict", json={**base, **bad}).status_code == 422, bad
        r = client.post("/api/v1/recommend", json={k: v for k, v in base.items() if k != "crop"} | {"soil_wetness_mean": (lo + hi) / 2})
        assert r.status_code == 200
        print("  -> PASSED: soil wetness accepted, labelled, range-checked.")

        # 25. CORS: no credentialed wildcard (the API has no cookies or auth)
        print("\n[TEST 25] CORS does not allow credentials with a wildcard origin")
        pre = client.options("/api/v1/predict", headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"})
        assert pre.headers.get("access-control-allow-credentials", "").lower() != "true", pre.headers
        assert client.get("/health", headers={"Origin": "http://example.com"}).headers.get("access-control-allow-credentials", "").lower() != "true"
        print("  -> PASSED: credentials are not allowed cross-origin.")

        # 26. Pages are revalidated, so an edited page is never served stale (Chrome did exactly that)
        print("\n[TEST 26] HTML pages carry Cache-Control: no-cache")
        for path in ("/", "/predict.html", "/metrics.html"):
            cc = client.get(path).headers.get("cache-control", "")
            assert "no-cache" in cc, (path, cc)
        assert "no-cache" not in client.get("/api/v1/metrics").headers.get("cache-control", ""), "API responses need no such header"
        print("  -> PASSED: pages revalidate on every load.")

        # 27. Whenever the API says how the model compares with the baseline, the caveat comes with it
        print("\n[TEST 27] /recommend summary carries the generated caveat (win, log-MAE loss, metric choice)")
        from climate_machine_learning import inference as I
        metrics_saved = json.loads((ROOT_DIR / "models" / "metrics.json").read_text())
        summary = client.post("/api/v1/recommend", json={"province": "Punjab", "precipitation_sum": 350.0,
                                                         "temperature_2m_mean": 25.0}).json()["recommendation_summary"]
        assert I.model_caveat(metrics_saved) in summary, summary
        ch, mt = metrics_saved["chosen"], metrics_saved["models"][metrics_saved["chosen"]["name"]]["test"]
        assert ch["beats_crop_province_mean_on_test"] is True and ch["beats_by_metric"]["log_mae"] is False
        assert f"log-MAE {mt['log_mae']:.3f}" in summary and "after seeing the test results" in summary
        print("  -> PASSED: caveat is built from metrics.json and shown with the ranking.")

        # 28. Equal shares are marked, in the items and in the summary (the guide says pages label ties)
        print("\n[TEST 28] /recommend marks crops with equal shares (tied_with) and says so in the summary")
        dry = REFERENCE["historical_years"]["Sindh"]["driest"]["inputs"]
        res = client.post("/api/v1/recommend", json={"province": "Sindh", **dry}).json()
        shown = [round(i["pct_of_normal"], 1) for i in res["ranked_crops"]]
        for it in res["ranked_crops"]:
            assert it["tied_with"] == shown.count(round(it["pct_of_normal"], 1)) - 1, it
        assert max(i["tied_with"] for i in res["ranked_crops"]) >= 2
        assert res["top_crop"]["tied_with"] == res["ranked_crops"][0]["tied_with"]
        med = REFERENCE["provinces"]["Sindh"]["median"]
        typ = client.post("/api/v1/recommend", json={"province": "Sindh", "precipitation_sum": round(med["precipitation_sum"], 1),
                                                     "temperature_2m_mean": round(med["temperature_2m_mean"], 1)}).json()
        assert all(i["tied_with"] == 12 for i in typ["ranked_crops"]), "typical year: all 13 crops tied"
        assert "tied" in typ["recommendation_summary"].lower() and "ordered by name" in typ["recommendation_summary"]
        print("  -> PASSED: tied_with is set from the displayed shares and the summary says so.")

        # 29. One dependency guards every model endpoint: no service means 503, never a 500 (review O2)
        print("\n[TEST 29] model endpoints answer 503 when the model service is missing; /health says degraded")
        saved_service = app.state.model_service
        app.state.model_service = None
        try:
            body = {"province": "Sindh", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5}
            for method, path, payload in (("get", "/api/v1/metadata", None), ("get", "/api/v1/metrics", None),
                                          ("post", "/api/v1/predict", {**body, "crop": "rice"}),
                                          ("post", "/api/v1/recommend", body)):
                res = client.get(path) if method == "get" else client.post(path, json=payload)
                assert res.status_code == 503, (path, res.status_code, res.text[:100])
            health = client.get("/api/v1/health").json()
            assert health["status"] == "degraded" and health["model_loaded"] is False, health
        finally:
            app.state.model_service = saved_service
        print("  -> PASSED: 503 on all four model endpoints, /health degraded.")

        # 30. No dead fallback in app/main.py: the frontend folder is mounted directly (review O3)
        print("\n[TEST 30] app/main.py has no dead static fallback")
        import app.main as app_main
        assert not hasattr(app_main, "STATIC_DIR") and not hasattr(app_main, "SERVE_DIR"), "dead fallback still present"
        assert "async def root" not in (ROOT_DIR / "app" / "main.py").read_text(), "JSON root fallback still present"
        print("  -> PASSED: frontend folder is mounted unconditionally.")

        # 31. Every response that shows a point yield also carries the pinned-year caveat (review R1)
        print("\n[TEST 31] /predict and /recommend carry absolute_yield_note from metrics.json")
        from climate_machine_learning import inference as I31
        note = I31.absolute_yield_caveat(json.loads((ROOT_DIR / "models" / "metrics.json").read_text()))
        rec = client.post("/api/v1/recommend", json={"province": "Sindh", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5}).json()
        one = client.post("/api/v1/predict", json={"province": "Sindh", "crop": "rice", "precipitation_sum": 150.0, "temperature_2m_mean": 27.5}).json()
        assert rec["absolute_yield_note"] == note and one["absolute_yield_note"] == note
        assert "pinned" in note and "rough" in note
        print("  -> PASSED: both responses carry the generated note.")

        # 32. Static files are compressed and cacheable; pages and the API are not (web performance quick wins)
        print("\n[TEST 32] static files: gzip for text-like files, cache headers; HTML no-cache; API untouched")
        frontend_dir = ROOT_DIR / "frontend" / "climate-resilient-frontend"
        js = client.get("/vendor/gsap.min.js", headers={"Accept-Encoding": "gzip"})
        assert js.status_code == 200 and js.headers.get("content-encoding") == "gzip", js.headers
        assert js.content == (frontend_dir / "vendor" / "gsap.min.js").read_bytes(), "decoded body must equal the file"
        assert js.num_bytes_downloaded < (frontend_dir / "vendor" / "gsap.min.js").stat().st_size, "gzip must make it smaller on the wire"
        for path in ("/vendor/gsap.min.js", "/assets/DenimINK-Regular.ttf", "/assets/logo-3d.glb", "/assets/fonts/WhitneySSm-Book_Web.woff2"):
            cache = client.get(path).headers.get("cache-control", "")
            assert "max-age=86400" in cache and "stale-while-revalidate" in cache and "public" in cache, (path, cache)
        woff = client.get("/assets/fonts/WhitneySSm-Book_Web.woff2", headers={"Accept-Encoding": "gzip"})
        assert "content-encoding" not in woff.headers, "woff2 is already compressed; gzip would only cost CPU"
        for missing in ("/assets/does-not-exist.png", "/vendor/missing.js", "/assets/fonts/nope.woff2"):
            gone = client.get(missing)
            assert gone.status_code == 404 and "max-age" not in gone.headers.get("cache-control", ""), (missing, gone.headers)
        assert "no-cache" in client.get("/index.html").headers["cache-control"]
        api = client.get("/api/v1/metrics")
        assert "max-age" not in api.headers.get("cache-control", ""), "API responses must not get the static cache header"
        print("  -> PASSED: gzip on js and ttf, none on woff2; assets cached for a day; HTML and API unchanged.")

        # 33. The sentence about tied ends of the ranking, in every combination (simplification safety net)
        print("\n[TEST 33] tie_notes: both ends in different ties, one shared tie, no ties, only the bottom tied")
        from types import SimpleNamespace as Item
        from app.services.model_service import tie_notes

        def end(crop, pct, tied):
            return Item(crop=crop, pct_of_normal=pct, tied_with=tied)

        both = tie_notes(end("apple", 100.04, 2), end("rice", 89.3, 1))
        assert both.count("share is tied") == 2 and both.index("largest") < both.index("smallest"), both
        assert "APPLE shares it with 2 other crops" in both and "RICE shares it with 1 other crop," in both, both
        one = tie_notes(end("apple", 100.0, 12), end("wheat", 100.0, 12))
        assert one.count("share is tied") == 1 and "largest" in one and "smallest" not in one, one
        assert "APPLE shares it with 12 other crops" in one
        assert tie_notes(end("apple", 99.2, 0), end("rice", 89.3, 0)) == ""
        bottom_only = tie_notes(end("apple", 99.2, 0), end("rice", 89.3, 3))
        assert bottom_only.count("share is tied") == 1 and "smallest" in bottom_only and "RICE shares it with 3 other crops" in bottom_only
        assert all(s.endswith("tied crops are ordered by name only.") for s in (both, one, bottom_only))
        print("  -> PASSED: all four combinations produce the expected sentences.")

        # 34. Ties are in name order through the API too, and the repo ignores .claude/ (/ship review B1 and hygiene)
        print("\n[TEST 34] ranking order inside displayed ties is by name; .gitignore hygiene")
        for province, kinds in REFERENCE["historical_years"].items():
            for kind, info in kinds.items():
                ranked = client.post("/api/v1/recommend", json={"province": province, **info["inputs"]}).json()["ranked_crops"]
                shares = [round(i["pct_of_normal"], 1) for i in ranked]
                assert shares == sorted(shares, reverse=True), (province, kind)
                for share in set(shares):
                    crops = [i["crop"] for i in ranked if round(i["pct_of_normal"], 1) == share]
                    assert crops == sorted(crops), (province, kind, share, crops)
        gitignore = (ROOT_DIR / ".gitignore").read_text().splitlines()
        assert ".claude/" in gitignore, ".claude/ must be ignored as a whole (it can hold history and tokens)"
        print("  -> PASSED: name order inside every tie, for every province and both real years.")

    # 35. Every app module imports on its own without the package being pip-installed. An editable install
    # (climate_machine_learning.pth in .venv) once hid an import-order bug that broke a fresh clone (T8).
    print("\n[TEST 35] app modules import on their own when climate_machine_learning is not installed")
    import subprocess
    probe = (
        "import sys, importlib\n"
        "sys.path = [p for p in sys.path if not p.rstrip('/').endswith('/src')]\n"
        "for name in list(sys.modules):\n"
        "    if name.startswith('climate_machine_learning'): del sys.modules[name]\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.startswith('climate_machine_learning') and not any(p.rstrip('/').endswith('/src') for p in sys.path):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(ROOT_DIR)!r})\n"
        "importlib.import_module(sys.argv[1])\n"
    )
    for module in ("app.services.model_service", "app.api.v1.endpoints", "app.main"):
        done = subprocess.run([sys.executable, "-c", probe, module], cwd=str(ROOT_DIR / "scripts"),
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, f"{module} does not import on its own: {done.stderr.strip().splitlines()[-1:]}"
    print("  -> PASSED: model_service, endpoints and main import with src/ added only by app.core.config.")

    print("\n" + "=" * 70)
    print("ALL BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
