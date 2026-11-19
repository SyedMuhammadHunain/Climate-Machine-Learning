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
    resid = np.log(va["Yield"].to_numpy()) - np.log(pipe.predict(va[F.FEATURE_COLS]))
    assert np.allclose(np.quantile(resid, [0.10, 0.90]), [lo_q, hi_q], atol=1e-9)
    assert pi["val_n"] == len(va)
    # coverage on the test years is reported, not tuned
    pred = pipe.predict(te[F.FEATURE_COLS])
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
    pcts = [i.pct_of_normal for i in res.items]
    assert pcts == sorted(pcts, reverse=True)
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
