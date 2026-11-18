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
