"""Checks for the feature module and (later) the trained model artifacts.

Run: .venv/bin/python scripts/verify_model.py
Plain asserts, no pytest needed.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from climate_machine_learning import features as F  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "pakistan_crop_yield_climate.parquet"
RAW = pd.read_parquet(RAW_PATH)
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def raw_structure_is_858_groups_of_12_months():
    grouped = RAW.groupby(["Crop", "Province", "Year"], sort=False)
    assert grouped.ngroups == 858, grouped.ngroups
    assert set(grouped.size().unique()) == {12}


@test
def month_order_check_passes_on_real_data():
    with_month = F.add_month(RAW)
    assert sorted(with_month["Month"].unique()) == list(range(1, 13))
    F.check_month_order(with_month)  # must not raise


@test
def month_order_check_rejects_shuffled_data():
    shuffled = RAW.copy()
    rng = np.random.default_rng(0)
    parts = []
    for _, grp in shuffled.groupby(["Crop", "Province", "Year"], sort=False):
        parts.append(grp.iloc[rng.permutation(len(grp))])
    shuffled = pd.concat(parts, ignore_index=True)
    try:
        F.check_month_order(F.add_month(shuffled))
    except ValueError:
        return
    raise AssertionError("shuffled months were accepted")


@test
def yearly_table_shape_and_columns():
    table, dropped = F.build_yearly_table(RAW)
    assert len(table) == 858 - 85, len(table)
    assert dropped == {"apple": 22, "grapes": 41, "pomegranate": 22}, dropped
    assert (table["Yield"] > 0).all()
    expected = set(F.BASE_INPUTS) | {"Crop", "Province", "Yield", "Year"}
    assert expected <= set(table.columns), expected - set(table.columns)
    assert len(table.groupby(["Crop", "Province", "Year"])) == len(table)


@test
def banned_columns_are_not_features():
    for col in F.FEATURE_COLS:
        assert col not in ("Area", "Production", "Is_Drought_Resilient", "Climate_Index"), col
        assert not col.endswith("_normalized"), col


@test
def aggregation_matches_manual_math():
    table, _ = F.build_yearly_table(RAW)
    row = table[(table.Crop == "wheat") & (table.Province == "Punjab") & (table.Year == 2010)].iloc[0]
    raw = RAW[(RAW.Crop == "wheat") & (RAW.Province == "Punjab") & (RAW.Year == 2010)]
    assert np.isclose(row["precipitation_sum"], raw["precipitation_sum"].sum())
    assert np.isclose(row["temperature_2m_mean"], raw["temperature_2m_mean"].mean())
    assert np.isclose(row["fao_evapotranspiration"], raw["fao_evapotranspiration"].mean())
    assert np.isclose(row["precipitation_hours"], raw["precipitation_hours"].sum())
    assert np.isclose(row["Yield"], raw["Yield"].iloc[0])


@test
def reference_uses_only_training_years():
    table, _ = F.build_yearly_table(RAW)
    train = table[table.Year <= 2016]
    ref_a = F.build_reference(train)
    poisoned = table.copy()
    poisoned.loc[poisoned.Year >= 2020, "precipitation_sum"] = 99999.0
    ref_b = F.build_reference(poisoned[poisoned.Year <= 2016])
    assert ref_a == ref_b
    assert set(ref_a["provinces"]) == {"Punjab", "Sindh", "Balochistan"}
    for prov in ref_a["provinces"].values():
        assert prov["median"]["precipitation_sum"] > 0
        assert prov["mean_annual_rain"] > 0


@test
def inference_row_equals_training_row():
    """Same function path: raw user inputs -> same features as the training table."""
    table, _ = F.build_yearly_table(RAW)
    ref = F.build_reference(table[table.Year <= 2016])
    table = F.add_engineered(table, ref)
    sample = table[(table.Crop == "mango") & (table.Province == "Sindh") & (table.Year == 2005)].iloc[0]
    inputs = {k: float(sample[k]) for k in F.BASE_INPUTS}
    row, sources = F.build_inference_row("Sindh", "mango", inputs, ref)
    assert list(row.columns) == F.FEATURE_COLS
    for col in F.NUMERIC_FEATURES:
        assert np.isclose(row.iloc[0][col], sample[col]), col
    assert row.iloc[0]["Crop"] == "mango" and row.iloc[0]["Province"] == "Sindh"
    assert set(sources.values()) == {"user"}


@test
def engineered_features_match_hand_computed_values():
    """The three drought features, worked out by hand. The training and inference paths share one
    formula, so comparing them to each other proves nothing: pin the numbers instead."""
    _, ref = _train_reference()
    mean_rain = ref["provinces"]["Sindh"]["mean_annual_rain"]
    # one request: rain 150 mm, evapotranspiration 6 mm/day -> 2190 mm a year
    row, _ = F.build_inference_row("Sindh", "rice", {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5,
                                                     "fao_evapotranspiration": 6.0}, ref)
    got = row.iloc[0]
    assert got["water_balance_mm"] == 150.0 - 2190.0 == -2040.0
    assert np.isclose(got["aridity_index"], 150.0 / 2190.0) and np.isclose(got["aridity_index"], 0.0684931506849315)
    assert np.isclose(got["rain_deficit_pct"], (mean_rain - 150.0) / mean_rain * 100.0)
    # the training table: each row uses ITS province's long-term mean rain
    frame = pd.DataFrame({"Province": ["Punjab", "Sindh"], "precipitation_sum": [300.0, 100.0],
                          "fao_evapotranspiration": [4.0, 8.0]})
    out = F.add_engineered(frame, ref)
    for i, prov in enumerate(("Punjab", "Sindh")):
        m, rain, et = ref["provinces"][prov]["mean_annual_rain"], frame.precipitation_sum[i], frame.fao_evapotranspiration[i]
        assert np.isclose(out.rain_deficit_pct[i], (m - rain) / m * 100.0), prov
        assert out.water_balance_mm[i] == rain - et * 365.0, prov
        assert np.isclose(out.aridity_index[i], rain / (et * 365.0)), prov
    assert list(out.columns[-3:]) == F.ENGINEERED, "new columns must follow F.ENGINEERED order"


def _train_reference():
    table, _ = F.build_yearly_table(RAW)
    return table, F.build_reference(table[table.Year <= 2016])


@test
def row_building_stays_close_to_the_cost_of_one_frame():
    """Speed guard for the hot path (a recommendation builds 26 rows). Self-calibrating, so it does not
    depend on how fast the machine is: the row builder is timed against one bare pandas frame.
    Measured: about 1.0x when the formulas are plain arithmetic, about 2.8x when each row goes through
    pandas column operations (that version made a recommendation about 25 ms slower)."""
    import time

    _, ref = _train_reference()
    inputs = {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5}
    values = F.build_inference_row("Sindh", "rice", inputs, ref)[0].iloc[0].to_dict()

    def per_call(fn, n=200):
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) / n

    build = lambda: F.build_inference_row("Sindh", "rice", inputs, ref)
    bare = lambda: pd.DataFrame([values])[F.FEATURE_COLS]
    for _ in range(40):
        build(), bare()
    ratio = min(per_call(build) / per_call(bare) for _ in range(7))   # min of 7: noise only ever adds time
    assert ratio < 1.8, f"building one row costs {ratio:.1f}x a bare frame (expected about 1.0x)"


@test
def missing_inputs_come_from_reference_and_are_labeled():
    table, _ = F.build_yearly_table(RAW)
    ref = F.build_reference(table[table.Year <= 2016])
    row, sources = F.build_inference_row(
        "Punjab", "wheat", {"precipitation_sum": 150.0, "temperature_2m_mean": 25.0}, ref
    )
    assert sources["precipitation_sum"] == "user"
    assert sources["temperature_2m_mean"] == "user"
    missing = [k for k in F.BASE_INPUTS if k not in ("precipitation_sum", "temperature_2m_mean")]
    for k in missing:
        if k in ("temperature_2m_max", "temperature_2m_min"):
            assert sources[k] == "user_mean_plus_province_median_offset", (k, sources[k])
        else:
            assert sources[k] == "province_median_training_years", (k, sources[k])
    assert row.iloc[0]["precipitation_sum"] == 150.0
    # max/min temperature follow the user's mean via data-derived offsets
    off = ref["provinces"]["Punjab"]["temp_offsets"]
    assert np.isclose(row.iloc[0]["temperature_2m_max"], 25.0 + off["max_minus_mean"])
    assert np.isclose(row.iloc[0]["temperature_2m_min"], 25.0 + off["min_minus_mean"])


@test
def unknown_crop_or_province_is_rejected():
    table, _ = F.build_yearly_table(RAW)
    ref = F.build_reference(table[table.Year <= 2016])
    for prov, crop in [("Atlantis", "wheat"), ("Punjab", "bajra")]:
        try:
            F.build_inference_row(prov, crop, {"precipitation_sum": 10.0, "temperature_2m_mean": 20.0}, ref)
        except ValueError:
            continue
        raise AssertionError(f"accepted {prov}/{crop}")


# ----------------------------------------------------------------------- trained artifacts
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

import joblib  # noqa: E402

MODELS = ROOT / "models"


def _load_artifacts():
    pipe = joblib.load(MODELS / "crop_yield_pipeline.joblib")
    metrics = json.loads((MODELS / "metrics.json").read_text())
    ref = json.loads((MODELS / "reference.json").read_text())
    return pipe, metrics, ref


@test
def artifacts_exist_and_pipeline_predicts():
    pipe, _, ref = _load_artifacts()
    row, _ = F.build_inference_row(
        "Punjab", "wheat", {"precipitation_sum": 300.0, "temperature_2m_mean": 24.0}, ref
    )
    pred = float(pipe.predict(row)[0])
    assert np.isfinite(pred) and pred > 0, pred


@test
def metrics_are_complete_and_honest_about_splits():
    _, m, _ = _load_artifacts()
    assert m["split"]["train_years"] == [2001, 2016]
    assert m["split"]["val_years"] == [2017, 2019]
    assert m["split"]["test_years"] == [2020, 2022]
    assert {"global_mean", "crop_province_mean"} <= set(m["baselines"])
    assert {"ridge", "random_forest", "gradient_boosting"} <= set(m["models"])
    assert m["chosen"]["name"] in m["models"]
    for name, info in m["models"].items():
        for part in ("val", "test"):
            assert {"mae", "rmse", "r2", "log_mae", "skill_vs_crop_province_mean_log"} <= set(info[part]), name
        assert info["cv_folds"] == 5 and info["cv_grouped_by"] == "Year"
    table, _ = F.build_yearly_table(RAW)
    assert m["data"]["n_train"] == int((table.Year <= 2016).sum())
    assert m["data"]["n_test"] == int((table.Year >= 2020).sum())
    assert m["data"]["sha256"] == hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()


@test
def metrics_record_both_sources_with_hashes():
    _, m, _ = _load_artifacts()
    sources = m["data"]["sources"]
    assert len(sources) == 2, sources
    by_name = {s["role"]: s for s in sources}
    assert set(by_name) == {"crop yield and climate", "soil wetness"}
    assert by_name["crop yield and climate"]["sha256"] == hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    assert by_name["soil wetness"]["sha256"] == hashlib.sha256(POWER_CSV.read_bytes()).hexdigest()
    assert "not specified" in by_name["soil wetness"]["licence"].lower()
    assert m["data"]["raw_monthly_rows"] == len(RAW)
    assert "soil_wetness_mean" in m["features"] and "soil_wetness_max" in m["features"]


@test
def prediction_interval_is_calibrated_on_validation_and_checked_on_test():
    pipe, m, ref = _load_artifacts()
    from climate_machine_learning import training

    pi = m["prediction_interval"]
    assert pi["level"] == 0.8 and pi["fitted_on"] == "validation years"
    lo_q, hi_q = pi["log_residual_quantiles"]
    assert lo_q < 0 < hi_q
    _, _, _, splits = training.load_tables()
    va, te = splits["val"], splits["test"]
    # MODEL_COLS adds Year: training scored the real years, so the check must too (apps pin the year instead)
    resid = np.log(va["Yield"].to_numpy()) - np.log(pipe.predict(va[training.MODEL_COLS]))
    assert np.allclose(np.quantile(resid, [0.10, 0.90]), [lo_q, hi_q], atol=1e-9)
    assert pi["val_n"] == len(va)
    # coverage on the test years is reported, not tuned
    pred = pipe.predict(te[training.MODEL_COLS])
    lo, hi = F.prediction_interval(pred, pi["log_residual_quantiles"])
    assert (lo < pred).all() and (pred < hi).all()
    cover = float(((te["Yield"].to_numpy() >= lo) & (te["Yield"].to_numpy() <= hi)).mean())
    assert np.isclose(cover, pi["test_coverage"]), (cover, pi["test_coverage"])


@test
def inference_ranking_equals_direct_model_output():
    from climate_machine_learning import inference as I

    pipe, m, ref = _load_artifacts()
    q = m["prediction_interval"]["log_residual_quantiles"]
    supplied = {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5}
    res = I.predict_crops(pipe, ref, "Sindh", supplied, log_quantiles=q)
    assert len(res.items) == 13 and sorted(i.crop for i in res.items) == sorted(ref["crops"])
    shown = [round(i.pct_of_normal, I.TIE_DIGITS) for i in res.items]   # equal shares are then in name order
    assert shown == sorted(shown, reverse=True)
    for it in res.items:
        row, _ = F.build_inference_row("Sindh", it.crop, supplied, ref)
        nrow, _ = F.build_inference_row("Sindh", it.crop, dict(ref["provinces"]["Sindh"]["median"]), ref)
        direct, normal = float(pipe.predict(row)[0]), float(pipe.predict(nrow)[0])
        assert np.isclose(it.predicted_yield, direct) and np.isclose(it.normal_yield, normal)
        assert np.isclose(it.pct_of_normal, direct / normal * 100.0)
        assert np.isclose(it.low, direct * np.exp(q[0])) and np.isclose(it.high, direct * np.exp(q[1]))
    assert res.sources["precipitation_sum"] == "user" and res.sources["soil_wetness_mean"] == "province_median_training_years"
    assert np.isclose(res.inputs["precipitation_sum"], 150.0)


@test
def inference_ties_are_ordered_by_crop_name():
    from climate_machine_learning import inference as I

    pipe, _, ref = _load_artifacts()
    # crops go in REVERSE alphabetical order, so only the crop-name tie-break can put them right
    # (reference["crops"] is already alphabetical, which hid a missing tie-break in an earlier version)
    backwards = list(reversed(ref["crops"]))
    res = I.predict_crops(pipe, ref, "Punjab", dict(ref["provinces"]["Punjab"]["median"]), crops=backwards)
    assert all(abs(i.pct_of_normal - 100.0) < 1e-9 for i in res.items)
    assert [i.crop for i in res.items] == sorted(backwards), [i.crop for i in res.items]
    assert res.items[0].low is None and res.items[0].high is None  # no interval without quantiles


@test
def inference_single_crop_matches_its_ranking_entry():
    from climate_machine_learning import inference as I

    pipe, m, ref = _load_artifacts()
    q = m["prediction_interval"]["log_residual_quantiles"]
    supplied = {"precipitation_sum": 200.0, "temperature_2m_mean": 27.8}
    full = {i.crop: i for i in I.predict_crops(pipe, ref, "Sindh", supplied, log_quantiles=q).items}
    one = I.predict_crops(pipe, ref, "Sindh", supplied, crops=["rice"], log_quantiles=q).items
    assert len(one) == 1 and one[0].crop == "rice"
    assert np.isclose(one[0].predicted_yield, full["rice"].predicted_yield)
    assert np.isclose(one[0].pct_of_normal, full["rice"].pct_of_normal)


@test
def inference_refuses_inputs_outside_the_trained_range():
    from climate_machine_learning import inference as I

    pipe, _, ref = _load_artifacts()
    try:
        I.predict_crops(pipe, ref, "Sindh", {"precipitation_sum": 5.0, "temperature_2m_mean": 27.8})
    except I.OutOfRangeError as err:
        assert err.parameter == "precipitation_sum" and "14.7" in str(err) and "cannot extrapolate" in str(err)
        return
    raise AssertionError("out-of-range input was accepted")


@test
def package_imports_without_the_web_server_installed():
    """The shared package is used by the API, both UIs, the model repo and a lean serverless build.
    Only the API needs uvicorn, so importing the package must not (it once did, at the top of __init__)."""
    import subprocess

    code = ("import sys; sys.modules['uvicorn'] = None; sys.path.insert(0, 'src');"
            "import climate_machine_learning.features, climate_machine_learning.inference,"
            "climate_machine_learning.presentation; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-300:]


@test
def presentation_ranking_table_is_built_from_the_model_output():
    from climate_machine_learning import inference as I
    from climate_machine_learning import presentation as P

    pipe, m, ref = _load_artifacts()
    fao = json.loads((ROOT / "data" / "reference" / "fao_crop_water_needs.json").read_text())
    res = I.predict_crops(pipe, ref, "Sindh", {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5},
                          log_quantiles=m["prediction_interval"]["log_residual_quantiles"])
    table = P.ranking_table(res, ref, fao)
    assert list(table.columns) == ["Rank", "Crop", "% of typical year", "Tied with", "Predicted yield", "Likely range",
                                   "Training rows", "FAO water need"]
    ties = I.tied_with(res.items)
    assert list(table["Tied with"]) == [ties[i.crop] for i in res.items]
    assert list(table["Rank"]) == list(range(1, 14))
    assert list(table["Crop"]) == [i.crop for i in res.items]
    assert list(table["% of typical year"]) == [round(i.pct_of_normal, 1) for i in res.items]
    rice, apple = table[table.Crop == "rice"].iloc[0], table[table.Crop == "apple"].iloc[0]
    assert rice["FAO water need"] == "450-700 mm per growing period, drought sensitivity high"
    assert apple["FAO water need"] == "not listed in the FAO table"
    assert rice["Training rows"] == f"{ref['train_rows_per_crop']['rice']} of 48"
    assert P.fao_text(fao, "rice") == rice["FAO water need"]


@test
def presentation_presets_fill_typical_or_real_years():
    from climate_machine_learning import presentation as P

    _, _, ref = _load_artifacts()
    med = ref["provinces"]["Punjab"]["median"]
    assert P.preset_inputs(ref, "Punjab", P.TYPICAL) == {
        "precipitation_sum": round(med["precipitation_sum"], 1), "temperature_2m_mean": round(med["temperature_2m_mean"], 1)}
    assert P.preset_inputs(ref, "Sindh", P.DRIEST) == ref["historical_years"]["Sindh"]["driest"]["inputs"]
    assert P.preset_inputs(ref, "Sindh", P.WETTEST) == ref["historical_years"]["Sindh"]["wettest"]["inputs"]
    assert P.preset_inputs(ref, "Sindh", P.DRIEST) is not ref["historical_years"]["Sindh"]["driest"]["inputs"], "must be a copy"
    assert [f[0] for f in P.FIELDS] == F.BASE_INPUTS, "form fields must follow the model inputs, in order"


@test
def presentation_interval_note_states_the_measured_coverage():
    from climate_machine_learning import presentation as P

    _, m, _ = _load_artifacts()
    note = P.interval_note(m)
    pi = m["prediction_interval"]
    assert f"{pi['test_coverage']:.0%}" in note and f"{pi['level']:.0%}" in note
    assert "validation years 2017-2019" in note and "2020-2022 test rows" in note
    assert note.endswith("optimistic." if pi["test_coverage"] < pi["level"] else "about right.")
    better = json.loads(json.dumps(m)); better["prediction_interval"]["test_coverage"] = 0.95
    assert P.interval_note(better).endswith("about right.")


@test
def reference_json_matches_training_data():
    _, _, ref = _load_artifacts()
    table, _ = F.build_yearly_table(RAW)
    expected = F.build_reference(table[table.Year <= 2016])
    assert ref["provinces"].keys() == expected["provinces"].keys()
    for prov in expected["provinces"]:
        assert np.isclose(ref["provinces"][prov]["mean_annual_rain"], expected["provinces"][prov]["mean_annual_rain"])
    assert ref["crops"] == expected["crops"]
    assert sum(ref["train_rows_per_crop"].values()) == int((table.Year <= 2016).sum())


@test
def reference_has_real_driest_and_wettest_training_years():
    _, _, ref = _load_artifacts()
    table, _ = F.build_yearly_table(RAW)
    train = table[table.Year <= 2016].drop_duplicates(["Province", "Year"])
    for prov, grp in train.groupby("Province"):
        for kind, idx in (("driest", grp["precipitation_sum"].idxmin()), ("wettest", grp["precipitation_sum"].idxmax())):
            real = grp.loc[idx]
            got = ref["historical_years"][prov][kind]
            assert got["year"] == int(real["Year"]), (prov, kind)
            assert got["year"] <= 2016
            for k in F.BASE_INPUTS:
                assert np.isclose(got["inputs"][k], real[k]), (prov, kind, k)
                lo, hi = ref["provinces"][prov]["range"][k]
                assert lo <= got["inputs"][k] <= hi, (prov, kind, k)


# ----------------------------------------------------------------------- second source (NASA POWER)
POWER_CSV = ROOT / "data" / "raw" / "nasa_power_monthly.csv"
POWER_DOC = ROOT / "data" / "reference" / "nasa_power_source.json"


@test
def soil_inputs_are_model_inputs():
    for name in ("soil_wetness_mean", "soil_wetness_max"):
        assert name in F.BASE_INPUTS and name in F.NUMERIC_FEATURES, name
    assert "soil_wetness_min" not in F.BASE_INPUTS  # flat in every year, no signal


@test
def soil_wetness_is_merged_by_province_and_year():
    table, _ = F.build_yearly_table(RAW)
    assert table[["soil_wetness_mean", "soil_wetness_max"]].notna().all().all()
    soil = pd.read_csv(POWER_CSV)
    monthly = soil[(soil.Province == "Punjab") & (soil.Year == 2010)].groupby("Month")["GWETROOT"].mean()
    row = table[(table.Crop == "wheat") & (table.Province == "Punjab") & (table.Year == 2010)].iloc[0]
    assert np.isclose(row["soil_wetness_mean"], monthly.mean())
    assert np.isclose(row["soil_wetness_max"], monthly.max())
    same_year = table[(table.Province == "Sindh") & (table.Year == 2005)]
    assert same_year["soil_wetness_mean"].nunique() == 1, "soil wetness is province-level, same for every crop"


@test
def soil_merge_fails_loudly_when_a_year_is_missing():
    soil = pd.read_csv(POWER_CSV)
    soil = soil[~((soil.Province == "Sindh") & (soil.Year == 2010))]
    try:
        F.build_yearly_table(RAW, soil)
    except ValueError as err:
        assert "Sindh" in str(err) and "2010" in str(err), err
        return
    raise AssertionError("missing soil year was accepted")


@test
def nasa_power_monthly_data_is_complete_and_sane():
    df = pd.read_csv(POWER_CSV)
    assert set(df["Province"]) == {"Punjab", "Sindh", "Balochistan"}
    assert df.groupby(["Province", "Point"]).ngroups == 9
    for (prov, point), grp in df.groupby(["Province", "Point"]):
        assert len(grp) == 22 * 12, (prov, point, len(grp))
        assert sorted(grp["Year"].unique()) == list(range(2001, 2023)), (prov, point)
    assert not (df[["GWETROOT", "PRECTOTCORR"]] == -999).any().any(), "fill values left in data"
    assert df["GWETROOT"].between(0, 1).all() and (df["PRECTOTCORR"] >= 0).all()
    punjab = df[df.Province == "Punjab"].groupby("Month")["PRECTOTCORR"].mean()
    assert punjab[7] + punjab[8] > 5 * (punjab[1] + punjab[11]), "Punjab monsoon should be much wetter than winter"


@test
def nasa_power_source_doc_matches_files():
    doc = json.loads(POWER_DOC.read_text())
    assert doc["source"]["api_version"] and doc["source"]["accessed"] == "2026-11-19"
    assert "not specified" in doc["source"]["licence"].lower()
    assert doc["source"]["reference_url"].startswith("https://power.larc.nasa.gov/")
    assert len(doc["points"]) == 9
    for pt in doc["points"]:
        raw = ROOT / pt["file"]
        assert raw.exists(), pt["file"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == pt["sha256"], pt["file"]
    assert hashlib.sha256(POWER_CSV.read_bytes()).hexdigest() == doc["csv_sha256"]
    assert {pt["province"] for pt in doc["points"]} == {"Punjab", "Sindh", "Balochistan"}


@test
def weather_changes_predictions_sensitivity():
    """Rain from province min to max (other inputs at median) must move at least one crop."""
    pipe, _, ref = _load_artifacts()
    moved = {}
    for prov, info in ref["provinces"].items():
        lo, hi = info["range"]["precipitation_sum"]
        for crop in ref["crops"]:
            med = {"temperature_2m_mean": info["median"]["temperature_2m_mean"]}
            a, _ = F.build_inference_row(prov, crop, {**med, "precipitation_sum": lo}, ref)
            b, _ = F.build_inference_row(prov, crop, {**med, "precipitation_sum": hi}, ref)
            pa, pb = float(pipe.predict(a)[0]), float(pipe.predict(b)[0])
            if abs(pb - pa) / pa > 0.01:
                moved.setdefault(prov, []).append(crop)
    print("    crops moved >1% by rain (min->max):", {k: len(v) for k, v in moved.items()})
    assert moved, "model ignores rain for every crop in every province"


# --------------------------------------------------------------------------- OffsetYieldModel
# SPEC-beat-baseline.md: each model predicts the gap between log yield and a per-row offset
# (crop-province mean log yield, optional year trend) learned from the training rows only.
def _offset_frame(rows):
    """Tiny hand-made frame: rows are (crop, province, year, yield)."""
    return pd.DataFrame(rows, columns=["Crop", "Province", "Year", "Yield"])


def _offset_model(trend, inference_year=None):
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    inner = make_pipeline(ColumnTransformer([("n", StandardScaler(), ["a"])]), DummyRegressor())
    return F.OffsetYieldModel(inner, trend=trend, inference_year=inference_year)


def _hand_frame():
    e = np.e
    rows = [("wheat", "Punjab", 2000 + i, e ** (i + 1)) for i in range(3)]          # log yield 1, 2, 3
    rows += [("wheat", "Sindh", 2000 + i, e ** (2 * i + 4)) for i in range(3)]       # log yield 4, 6, 8
    rows += [("rice", "Sindh", 2000 + i, e ** 5) for i in range(3)]                  # log yield 5, 5, 5
    df = _offset_frame(rows)
    df["a"] = 0.0
    return df


@test
def offset_is_hand_computed_for_each_trend_variant():
    df = _hand_frame()
    probe = _offset_frame([("wheat", "Punjab", 2003, 1.0), ("wheat", "Sindh", 2003, 1.0)])
    probe["a"] = 0.0
    # no trend: crop-province mean log yield (Punjab 2, Sindh 6)
    m = _offset_model("none").fit(df, df["Yield"])
    assert np.allclose(m.log_offset(probe), [2.0, 6.0]), m.log_offset(probe)
    # per_crop: wheat slope = pooled within-group slope = (1*1 + 1*2)/(1 + 1 + 1 + 1) over both
    # provinces (dx = -1,0,1 in each, dy = -1,0,1 for Punjab and -2,0,2 for Sindh) -> 6/4 = 1.5
    m = _offset_model("per_crop").fit(df, df["Yield"])
    assert np.allclose(m.log_offset(probe), [2.0 + 1.5 * 2, 6.0 + 1.5 * 2]), m.log_offset(probe)
    # per_crop_province: Punjab slope 1, Sindh slope 2
    m = _offset_model("per_crop_province").fit(df, df["Yield"])
    assert np.allclose(m.log_offset(probe), [2.0 + 1.0 * 2, 6.0 + 2.0 * 2]), m.log_offset(probe)


@test
def offset_falls_back_from_crop_province_to_crop_to_global():
    df = _hand_frame()
    m = _offset_model("none").fit(df, df["Yield"])
    # unseen province for a seen crop -> that crop's mean log yield: wheat (1+2+3+4+6+8)/6 = 4
    # unseen crop -> global mean log yield: (24 + 15) / 9 = 13/3
    probe = _offset_frame([("wheat", "Balochistan", 2001, 1.0), ("maize", "Punjab", 2001, 1.0)])
    probe["a"] = 0.0
    assert np.allclose(m.log_offset(probe), [4.0, 13.0 / 3.0]), m.log_offset(probe)
    # a group with too few rows for its own slope uses the crop slope
    one = _offset_frame([("wheat", "Punjab", 2000, np.e), ("wheat", "Punjab", 2001, np.e ** 2),
                         ("wheat", "Sindh", 2000, np.e ** 4)])
    one["a"] = 0.0
    m = _offset_model("per_crop_province").fit(one, one["Yield"])
    assert np.all(np.isfinite(m.log_offset(one)))


@test
def offset_model_year_comes_from_the_year_column_or_the_pinned_year():
    df = _hand_frame()
    m = _offset_model("per_crop_province", inference_year=2005).fit(df, df["Yield"])
    row = df[(df.Crop == "wheat") & (df.Province == "Punjab")].iloc[[0]]
    with_year = row.assign(Year=2003)
    no_year = row.drop(columns="Year")
    assert np.allclose(m.log_offset(with_year), 2.0 + 1.0 * 2)           # Year column wins
    assert np.allclose(m.log_offset(no_year), 2.0 + 1.0 * 4)             # pinned year 2005
    unpinned = _offset_model("per_crop_province").fit(df, df["Yield"])
    try:
        unpinned.predict(no_year)
    except ValueError as exc:
        assert "inference_year" in str(exc)
    else:
        raise AssertionError("predict without Year and without inference_year must fail loudly")
    try:
        _offset_model("sideways").fit(df, df["Yield"])
    except ValueError as exc:
        assert "trend" in str(exc)
    else:
        raise AssertionError("unknown trend must be rejected")
    try:
        _offset_model("none").fit(df, df["Yield"] * 0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("non-positive yields must be rejected")


@test
def offset_model_predicts_exp_of_offset_plus_inner_gap_on_real_data():
    """Fitted on the real training years; the ratio of two scenarios must not depend on the pinned year."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    table, _ = F.build_yearly_table(RAW)
    tr = table[table.Year <= 2016].reset_index(drop=True)
    cols = ["precipitation_sum", "temperature_2m_mean"]
    inner = make_pipeline(ColumnTransformer([("n", StandardScaler(), cols)]), Ridge(alpha=1.0))
    m = F.OffsetYieldModel(inner, trend="per_crop", inference_year=2022)
    m.fit(tr[["Crop", "Province", "Year"] + cols], tr["Yield"])

    X = tr[["Crop", "Province", "Year"] + cols].iloc[:25]
    direct = np.exp(m.log_offset(X) + m.inner_.predict(X[cols]))
    assert np.allclose(m.predict(X), direct)

    pair = pd.DataFrame({"Crop": ["rice", "rice"], "Province": ["Sindh", "Sindh"],
                         "precipitation_sum": [50.0, 400.0], "temperature_2m_mean": [27.0, 27.0]})
    ratios = []
    for year in (2005, 2016, 2022, 2030):
        m.set_params(inference_year=year)
        p = m.predict(pair)
        ratios.append(p[0] / p[1])
    assert np.allclose(ratios, ratios[0]), ratios

    m.set_params(inference_year=2022)
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(m, Path(tmp) / "m.joblib")
        again = joblib.load(Path(tmp) / "m.joblib")
    assert np.array_equal(again.predict(pair), m.predict(pair))
    from sklearn.base import clone
    assert clone(m).get_params()["trend"] == "per_crop"


# --------------------------------------------------------------------------- ties are marked (Prove-It: the guide claimed pages label them)
@test
def tied_with_counts_other_crops_with_the_same_share_at_one_decimal():
    from climate_machine_learning import inference as I

    def item(crop, pct):
        return I.CropResult(crop, 1.0, 1.0, pct)

    items = [item("a", 99.24), item("b", 97.74), item("c", 97.66), item("d", 97.7), item("e", 89.3)]
    assert I.tied_with(items) == {"a": 0, "b": 2, "c": 2, "d": 2, "e": 0}
    assert I.tied_with([item("x", 100.04), item("y", 99.96)]) == {"x": 1, "y": 1}, "equal once rounded to 100.0"
    assert I.tied_with([]) == {}


@test
def tied_with_on_the_real_model_marks_the_sindh_driest_year_ties():
    from climate_machine_learning import inference as I

    pipe, m, ref = _load_artifacts()
    inputs = ref["historical_years"]["Sindh"]["driest"]["inputs"]
    res = I.predict_crops(pipe, ref, "Sindh", inputs)
    ties = I.tied_with(res.items)
    shown = [round(i.pct_of_normal, 1) for i in res.items]
    for i in res.items:
        assert ties[i.crop] == shown.count(round(i.pct_of_normal, 1)) - 1, i.crop
    assert max(ties.values()) >= 2, "the demo ranking has crops with equal shares; they must be marked"
    typical = I.predict_crops(pipe, ref, "Sindh", {"precipitation_sum": round(ref["provinces"]["Sindh"]["median"]["precipitation_sum"], 1),
                                                   "temperature_2m_mean": round(ref["provinces"]["Sindh"]["median"]["temperature_2m_mean"], 1)})
    assert all(v == len(typical.items) - 1 for v in I.tied_with(typical.items).values()), "typical year: all tied"


@test
def order_inside_a_displayed_tie_is_by_crop_name_on_real_years():
    """/ship review B1: the pages say tied crops are ordered by name only, so the sort must use the displayed share."""
    from climate_machine_learning import inference as I

    pipe, m, ref = _load_artifacts()
    checked = 0
    for province in ref["provinces"]:
        for kind in ("driest", "wettest"):
            res = I.predict_crops(pipe, ref, province, ref["historical_years"][province][kind]["inputs"])
            shown = [round(i.pct_of_normal, I.TIE_DIGITS) for i in res.items]
            assert shown == sorted(shown, reverse=True), (province, kind, "shares must be in descending displayed order")
            for share in set(shown):
                crops = [i.crop for i in res.items if round(i.pct_of_normal, I.TIE_DIGITS) == share]
                assert crops == sorted(crops), (province, kind, share, crops)
                checked += len(crops) > 1
    assert checked >= 3, "the real years must contain tie groups, or this test proves nothing"


@test
def sort_uses_the_displayed_share_for_near_ties():
    from climate_machine_learning import inference as I

    # the ranking sort key and the tie marker must agree: 97.66 and 97.74 both show 97.7, so name decides
    items = [I.CropResult("tomato", 1.0, 1.0, 97.74), I.CropResult("banana", 1.0, 1.0, 97.66), I.CropResult("rice", 1.0, 1.0, 89.3)]
    assert I.rank_key(items[0])[0] == I.rank_key(items[1])[0], "97.66 and 97.74 both show 97.7: same sort share"
    ordered = sorted(items, key=I.rank_key)
    assert [i.crop for i in ordered] == ["banana", "tomato", "rice"], [i.crop for i in ordered]
    assert I.tied_with(ordered) == {"banana": 1, "tomato": 1, "rice": 0}


# --------------------------------------------------------------------------- promote (SPEC-promote-offset-model.md)
@test
def beats_by_metric_needs_strictly_better_r2_and_mae_for_the_flag():
    from climate_machine_learning import training

    base = {"r2": 0.832, "mae": 34.7, "rmse": 70.1, "log_mae": 0.312}
    win = {"r2": 0.885, "mae": 29.7, "rmse": 55.0, "log_mae": 0.341}
    out = training.beats_by_metric(win, base)
    assert out == {"r2": True, "mae": True, "rmse": True, "log_mae": False}, out
    assert training.judged_win(out) is True
    only_r2 = training.beats_by_metric({**win, "mae": 40.0}, base)
    assert only_r2["r2"] and not only_r2["mae"] and training.judged_win(only_r2) is False
    tie = training.beats_by_metric(dict(base), base)
    assert not any(tie.values()) and training.judged_win(tie) is False, "a tie is not a win"


def _small_forest_pipeline(offset: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rf = RandomForestRegressor(n_estimators=40, random_state=0, n_jobs=-1)
    # step names as in training.make_pipeline: single_thread finds the forest under "regressor"
    inner = Pipeline([("preprocessor", ColumnTransformer([("n", StandardScaler(), ["a"])])), ("regressor", rf)])
    df = pd.DataFrame({"Crop": ["w"] * 12, "Province": ["P"] * 12, "Year": list(range(2000, 2012)),
                       "a": np.linspace(0, 1, 12), "Yield": np.linspace(10, 30, 12)})
    if offset:
        return F.OffsetYieldModel(inner, trend="per_crop", inference_year=2011).fit(df, df["Yield"]), df, rf
    from sklearn.compose import TransformedTargetRegressor
    reg = TransformedTargetRegressor(regressor=rf, func=np.log, inverse_func=np.exp)
    return Pipeline([("regressor", reg)]).fit(df[["a"]], df["Yield"]), df, rf


@test
def single_thread_reaches_the_regressor_inside_an_offset_model_and_an_old_pipeline():
    from climate_machine_learning import inference as I

    for offset in (True, False):
        pipe, _, _ = _small_forest_pipeline(offset)
        I.single_thread(pipe)
        if offset:
            rf = pipe.inner_.named_steps["regressor"]
        else:
            rf = pipe.named_steps["regressor"].regressor_
        assert rf.n_jobs == 1, f"offset={offset}: n_jobs={rf.n_jobs}"


@test
def single_thread_predictions_are_bit_identical_between_calls():
    from climate_machine_learning import inference as I

    pipe, df, _ = _small_forest_pipeline(True)
    I.single_thread(pipe)
    X = df[["Crop", "Province", "a"]].iloc[:6]          # no Year column: pinned year path, as the apps call it
    first = pipe.predict(X)
    for _ in range(5):
        assert np.array_equal(first, pipe.predict(X))


@test
def saved_model_is_the_promoted_offset_model():
    """models/ holds the model chosen under SPEC-promote-offset-model.md (structure and honesty, not typed-in numbers)."""
    pipe, m, ref = _load_artifacts()
    assert isinstance(pipe, F.OffsetYieldModel), type(pipe)
    ch = m["chosen"]
    assert pipe.trend == ch["trend"] and pipe.inference_year == m["inference_year"] == ref["inference_year"]
    assert ch["judged_on"] == ["r2", "mae"] and ch["judged_on_chosen_after_test"] is True
    mt, bt = m["models"][ch["name"]]["test"], m["baselines"]["crop_province_mean"]["test"]
    assert ch["beats_crop_province_mean_on_test"] is True and mt["r2"] > bt["r2"] and mt["mae"] < bt["mae"]
    assert ch["beats_by_metric"]["log_mae"] is False and mt["log_mae"] > bt["log_mae"], "the log-MAE loss must stay visible"
    assert any("after seeing the test" in n for n in m["notes"])
    d = m["data"]
    assert (d["n_train"], d["n_val"], d["n_test"]) == (565, 106, 102)
    assert d["sha256"] == hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    assert [s["sha256"] for s in d["sources"]] == [
        hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        hashlib.sha256((ROOT / "data" / "raw" / "nasa_power_monthly.csv").read_bytes()).hexdigest()]


@test
def model_caveat_states_wins_losses_and_the_after_the_fact_choice():
    from climate_machine_learning import inference as I

    def metrics(by, after=True, with_trend=True):
        base = {"r2": 0.832, "mae": 34.7, "rmse": 70.1, "log_mae": 0.312}
        model = {"r2": 0.885, "mae": 29.7, "rmse": 55.0, "log_mae": 0.341}
        m = {"split": {"test_years": [2020, 2022]},
             "chosen": {"name": "gb", "beats_by_metric": by, "judged_on": ["r2", "mae"],
                        "judged_on_chosen_after_test": after},
             "models": {"gb": {"test": model}}, "baselines": {"crop_province_mean": {"test": base}}}
        if with_trend:
            m["baselines"]["crop_province_mean_plus_trend"] = {"test": {**base, "r2": 0.875}}
        return m

    text = I.model_caveat(metrics({"r2": True, "mae": True, "rmse": True, "log_mae": False}))
    assert "2020-2022" in text
    assert "R² 0.885 against 0.832" in text and "MAE 29.7 against 34.7" in text
    assert "log-MAE 0.341 against 0.312" in text and text.index("was not better on") < text.index("log-MAE 0.341")
    assert "after seeing the test results" in text and "R² and MAE" in text
    assert "R² 0.875" in text and "no climate" in text
    assert "weak evidence" in text
    clean = I.model_caveat(metrics({"r2": True, "mae": True, "rmse": True, "log_mae": True}, after=False, with_trend=False))
    assert "was not better" not in clean and "weak evidence" not in clean
    assert "after seeing" not in clean and "no climate" not in clean
    loss = I.model_caveat(metrics({"r2": False, "mae": False, "rmse": False, "log_mae": False}))
    assert "was better on" not in loss and "was not better on" in loss


# --------------------------------------------------------------------------- R1: absolute yields at the pinned year
@test
def pinned_year_check_equals_an_independent_recomputation():
    """metrics.json says how far the pinned-year typical yield is from recent actuals; recompute it another way."""
    from climate_machine_learning import training

    pipe, m, ref = _load_artifacts()
    chk = m["pinned_year_check"]
    table, _ = F.build_yearly_table(RAW)
    recent = table[table.Year.between(*training.TEST_YEARS)].groupby(["Crop", "Province"])["Yield"].mean()
    ratios = {}
    for (crop, prov), actual in recent.items():
        row, _ = F.build_inference_row(prov, crop, dict(ref["provinces"][prov]["median"]), ref)
        ratios[(crop, prov)] = float(pipe.predict(row)[0]) / float(actual)
    vals = np.array(list(ratios.values()))
    assert chk["year"] == pipe.inference_year == ref["inference_year"]
    assert chk["compared_with_years"] == list(training.TEST_YEARS)
    assert chk["n_pairs"] == len(vals) and len(vals) > 20
    for key, q in (("ratio_median", 50), ("ratio_p25", 25), ("ratio_p75", 75)):
        assert np.isclose(chk[key], np.percentile(vals, q)), key
    lo, hi = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
    assert (chk["ratio_min"]["crop"], chk["ratio_min"]["province"]) == lo and np.isclose(chk["ratio_min"]["value"], ratios[lo])
    assert (chk["ratio_max"]["crop"], chk["ratio_max"]["province"]) == hi and np.isclose(chk["ratio_max"]["value"], ratios[hi])
    assert "not used to choose" in chk["note"]


@test
def absolute_yield_caveat_states_the_pinned_year_spread_from_metrics():
    from climate_machine_learning import inference as I

    metrics = {"pinned_year_check": {
        "year": 2022, "compared_with_years": [2020, 2022], "n_pairs": 34,
        "ratio_median": 0.93, "ratio_p25": 0.8, "ratio_p75": 1.15,
        "ratio_min": {"value": 0.21, "crop": "mango", "province": "Balochistan"},
        "ratio_max": {"value": 2.19, "crop": "cotton", "province": "Sindh"},
        "note": "Descriptive only; not used to choose the model."}}
    text = I.absolute_yield_caveat(metrics)
    for needle in ("2022", "2020-2022", "34", "0.80", "1.15", "0.21", "mango", "Balochistan", "2.19", "cotton", "Sindh"):
        assert needle in text, (needle, text)
    assert "rough" in text and "ranking" in text


# --------------------------------------------------------------------------- offset training (T2)
_CANDIDATE = {}


def _models_dir_hash():
    d = ROOT / "models"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def _candidate():
    """Train once per run into a scratch folder (about 3 minutes); models/ must stay untouched."""
    if not _CANDIDATE:
        from climate_machine_learning import training

        before = _models_dir_hash()
        tmp = tempfile.TemporaryDirectory()
        metrics = training.train(Path(tmp.name), verbose=False)
        _CANDIDATE.update(tmp=tmp, dir=Path(tmp.name), metrics=metrics, models_before=before,
                          models_after=_models_dir_hash())
    return _CANDIDATE


@test
def training_leaves_models_dir_untouched_when_given_another_folder():
    c = _candidate()
    assert c["models_before"] == c["models_after"]
    assert {"crop_yield_pipeline.joblib", "metrics.json", "reference.json"} <= {p.name for p in c["dir"].iterdir()}


@test
def candidate_metrics_report_trend_baseline_and_selection_on_validation():
    m = _candidate()["metrics"]
    base = m["baselines"]["crop_province_mean_plus_trend"]
    assert base["trend"] in F.TRENDS and set(base["variants"]) == set(F.TRENDS)
    assert set(base["val"]) >= {"mae", "rmse", "r2", "log_mae"} and set(base["test"]) >= {"mae", "rmse", "r2", "log_mae"}
    # the trend baseline's own variant is picked on validation log-MAE, never on test
    assert base["trend"] == min(base["variants"], key=lambda t: base["variants"][t]["val"]["log_mae"])
    assert base["val"] == base["variants"][base["trend"]]["val"]
    assert base["test"] == base["variants"][base["trend"]]["test"]
    # the chosen model is the lowest validation log-MAE, and its trend is one of the allowed variants
    chosen = min(m["models"], key=lambda k: m["models"][k]["val"]["log_mae"])
    assert m["chosen"]["name"] == chosen
    assert m["chosen"]["trend"] in F.TRENDS and m["models"][chosen]["best_params"]["trend"] == m["chosen"]["trend"]
    ch, base_test = m["chosen"], m["baselines"]["crop_province_mean"]["test"]
    mt = m["models"][chosen]["test"]
    assert ch["judged_on"] == ["r2", "mae"] and ch["judged_on_chosen_after_test"] is True
    assert ch["beats_by_metric"] == {"r2": mt["r2"] > base_test["r2"], "mae": mt["mae"] < base_test["mae"],
                                      "rmse": mt["rmse"] < base_test["rmse"],
                                      "log_mae": mt["log_mae"] < base_test["log_mae"]}
    assert ch["beats_crop_province_mean_on_test"] == (ch["beats_by_metric"]["r2"] and ch["beats_by_metric"]["mae"])
    assert any("after seeing the test" in n for n in m["notes"]), "notes must say the metric choice came after the test results"
    assert m["inference_year"] == 2022
    assert set(m["pinned_year_check"]) >= {"year", "n_pairs", "ratio_median", "ratio_p25", "ratio_p75", "ratio_min", "ratio_max", "note"}
    d = m["data"]
    assert (d["n_train"], d["n_val"], d["n_test"]) == (565, 106, 102), "row counts changed: no added or dropped data"


@test
def candidate_pipeline_is_an_offset_model_with_the_pinned_year_everywhere():

    c = _candidate()
    pipe = joblib.load(c["dir"] / "crop_yield_pipeline.joblib")
    ref = json.loads((c["dir"] / "reference.json").read_text())
    assert isinstance(pipe, F.OffsetYieldModel)
    assert pipe.inference_year == ref["inference_year"] == c["metrics"]["inference_year"] == 2022
    assert pipe.trend == c["metrics"]["chosen"]["trend"]
    assert "Year" not in F.FEATURE_COLS and "Year" not in ref["feature_cols"]
    # predicts from the app's feature columns alone (no Year column), as the apps call it
    row, _ = F.build_inference_row("Sindh", "rice", {"precipitation_sum": 150.0, "temperature_2m_mean": 27.5}, ref)
    assert list(row.columns) == F.FEATURE_COLS
    assert np.isfinite(pipe.predict(row)).all()


@test
def retraining_is_deterministic():
    from climate_machine_learning import training

    _, m, _ = _load_artifacts()
    with tempfile.TemporaryDirectory() as tmp:
        again = training.train(Path(tmp), verbose=False)
    for name in m["models"]:
        # parallel forest sums trees in a different order per run: ~1e-14 float noise only
        for k, v in m["models"][name]["test"].items():
            assert np.isclose(v, again["models"][name]["test"][k], rtol=1e-9, atol=1e-9), (name, k)
        assert m["models"][name]["best_params"] == again["models"][name]["best_params"], name
    assert m["chosen"]["name"] == again["chosen"]["name"]


def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
