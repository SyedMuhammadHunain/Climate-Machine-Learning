"""Builds the project report: docs/report/report.html, printed to docs/report/report.pdf by headless Chrome.

Run:   .venv/bin/python scripts/build/build_report_figures.py   # figures first (if the data or model changed)
       .venv/bin/python scripts/build/capture_screenshots.py    # deployment screenshots (if an app changed)
       .venv/bin/python scripts/build/build_report.py           # HTML + PDF
       .venv/bin/python scripts/build/build_report.py --pdf-only  # re-print the existing HTML (used by checks)
Check: .venv/bin/python scripts/checks/verify_report.py

Every number in the report is read from models/metrics.json, the data files or the training code at build
time; caveat sentences come from climate_machine_learning.inference, the same text the apps show.
Prose lives in this file, next to the code that fills in its numbers.
"""
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
for _folder in ("lib", "checks", "build"):
    sys.path.insert(0, str(ROOT / "scripts" / _folder))
from climate_machine_learning import features as F  # noqa: E402
from climate_machine_learning import inference as I  # noqa: E402
from climate_machine_learning import training as T  # noqa: E402
from capture_screenshots import STREAMLIT_URL, VERCEL_URL  # noqa: E402
from group_roles import MEMBERS, TOPIC_OWNERS  # noqa: E402
from verify_report import HTML, PDF, TABLE_LABELS  # noqa: E402

METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())
STREAMLIT = STREAMLIT_URL.removesuffix("~/+/")
VERCEL = VERCEL_URL.removesuffix("predict.html")
HF_MODEL = "https://huggingface.co/SMHunain/crop-yield-drought-pakistan"
REPO = "https://github.com/SyedMuhammadHunain/Climate-Machine-Learning"


def esc(text) -> str:
    return html.escape(str(text))


def link(url: str) -> str:
    return f'<a href="{esc(url)}">{esc(url.removeprefix("https://"))}</a>'


def fig(name: str, caption: str) -> str:
    return (f'<figure><img src="figures/{name}" alt="{esc(caption)}">'
            f"<figcaption>{esc(caption)}</figcaption></figure>")


def row(*figures: str, cls: str = "") -> str:
    return f'<div class="figrow {cls}">{"".join(figures)}</div>'


def facts() -> dict:
    """Numbers the prose needs, computed from the data and the training code, never typed."""
    m = METRICS
    table, dropped, _ref, splits = T.load_tables()
    train = splits["train"]
    df = train[F.NUMERIC_FEATURES].copy()
    df["log_yield"] = np.log(train["Yield"])
    corr = df.corr()["log_yield"].drop("log_yield").abs().sort_values()
    gap = np.log(train["Yield"] / train.groupby(["Crop", "Province"])["Yield"].transform("mean"))
    n_total = m["data"]["n_train"] + m["data"]["n_val"] + m["data"]["n_test"]
    imp = m["permutation_importance_val_log_mae"]
    test = splits["test"]
    pipeline = joblib.load(ROOT / "models" / "crop_yield_pipeline.joblib")
    I.single_thread(pipeline)
    abs_log_err = np.abs(np.log(test["Yield"].to_numpy()) - np.log(pipeline.predict(test[T.MODEL_COLS])))
    by_province = pd.Series(abs_log_err).groupby(test["Province"].to_numpy()).mean().sort_values(ascending=False)
    return {
        "test_log_mae_by_province": by_province.to_dict(),
        "raw_rows": m["data"]["raw_monthly_rows"],
        "raw_cols": len(pd.read_parquet(T.RAW_PATH).columns),
        "power_rows": len(pd.read_csv(F.SOIL_CSV)),
        "rows": len(table),
        "dropped": dropped,
        "n_dropped": sum(dropped.values()),
        "crops": sorted(table["Crop"].unique()),
        "provinces": sorted(table["Province"].unique()),
        "years": (int(table["Year"].min()), int(table["Year"].max())),
        "n_features": len(F.FEATURE_COLS),
        "max_corr_name": corr.index[-1],
        "max_corr": float(corr.iloc[-1]),
        "rain_gap_corr": float(np.corrcoef(train["rain_deficit_pct"], gap)[0, 1]),
        "share": {k: m["data"][f"n_{k}"] / n_total for k in ("train", "val", "test")},
        "imp_crop": imp["Crop"]["mean"],
        "imp_province": imp["Province"]["mean"],
        "imp_top_weather": max((v["mean"], k) for k, v in imp.items() if k not in ("Crop", "Province")),
        "yield_range": (float(table["Yield"].min()), float(table["Yield"].max())),
    }


def criterion(m: dict) -> str:
    """The proposal's success test, checked from metrics.json: test R² above 0.7 and a clear margin over a
    mean-yield baseline."""
    c = m["models"][m["chosen"]["name"]]["test"]
    b = m["baselines"]["global_mean"]["test"]
    met = c["r2"] > 0.7 and c["r2"] > b["r2"] and c["mae"] < b["mae"]
    return (f"The proposal's success criterion is {'met' if met else 'not met'}: test R² {c['r2']:.3f} "
            f"(target above 0.7) against the overall mean-yield baseline's R² {b['r2']:.3f} and MAE {b['mae']:.1f} "
            f"(model MAE {c['mae']:.1f}).")


# --------------------------------------------------------------------------- sections
def title_block() -> str:
    names = ", ".join(f"{n} ({r})" for n, r, _ in MEMBERS)
    return f"""
<h1>Low-Water Crop Selection Under Drought in Pakistan</h1>
<p class="meta">CT-378 Machine Learning semester project. {esc(names)}. Code: {link(REPO)}</p>
"""


def abstract(f: dict) -> str:
    m = METRICS
    c = m["models"][m["chosen"]["name"]]
    cpm = m["baselines"]["crop_province_mean"]["test"]
    return f"""
<h2>Abstract</h2>
<p>We predict the yearly yield of {len(f['crops'])} crops in {', '.join(f['provinces'])} from annual climate and
soil wetness, and rank the crops by the share of a typical year's yield each keeps in a user's scenario. Two public
sources are merged: a crop yield and climate dataset ({f['raw_rows']:,} monthly rows, {f['years'][0]}-{f['years'][1]})
and NASA POWER soil wetness. After aggregation to one row per crop, province and year there are {f['rows']} rows.
A scikit-learn pipeline learns the log-scale gap from each crop-province average plus a year trend; models are
tuned with 5-fold cross-validation grouped by year and selected on validation years. The chosen {esc(c['label'])}
scores test R² {c['test']['r2']:.3f} and MAE {c['test']['mae']:.1f} (crop-province mean baseline:
{cpm['r2']:.3f} and {cpm['mae']:.1f}), but not a better log-MAE, and most of the gain comes from the year trend.
The model is served by a FastAPI app with HTML pages (local and on Vercel), a Streamlit app and a model repository
on the Hugging Face Hub.</p>
"""


def problem() -> str:
    return """
<h2>1. Problem and motivation</h2>
<p>Water-intensive crops such as rice and sugarcane are hard to sustain in drought years in Punjab, Sindh and
Balochistan. A farmer or agricultural extension officer needs to know which crops hold up when rain is scarce. The
user enters a year's climate for a province, either typed in or taken from a real driest, typical or wettest year
in the data, and receives every crop's predicted yield and its share of a typical year's yield, ranked. The task is
regression on yield; the ranking divides two predictions of the same model, so no number is set by hand.</p>
"""


def data_section(f: dict) -> str:
    m = METRICS
    src = {s["role"]: s for s in m["data"]["sources"]}
    power = src["soil wetness"]
    dropped = ", ".join(f"{k} {v}" for k, v in sorted(f["dropped"].items()))
    return f"""
<h2>2. Data collection</h2>
<table class="data">
<thead><tr><th></th><th>Crop yield and climate</th><th>NASA POWER soil wetness</th></tr></thead>
<tbody>
<tr><td>Source</td><td>{link("https://github.com/Abdullah-Masood-05/pakistan-crop-yield-vs-climate")}; upstream
{link("https://zenodo.org/records/15486470")}</td><td>{link("https://power.larc.nasa.gov/")}, monthly point API
{esc(power['api_version'])} (MERRA-2)</td></tr>
<tr><td>Collection method</td><td>Downloaded file; the repository does not state how it was collected or the weather
source</td><td>Fetched by <code>scripts/ml/fetch_nasa_power.py</code> for {power['points']} farming-area points, 3 per
province, equal weight</td></tr>
<tr><td>Licence</td><td>MIT (stated by the repository)</td><td>Licence not specified in the POWER catalog entry; free to use,
citation requested</td></tr>
<tr><td>Date</td><td>Added to the repository 2026-11-14</td><td>Accessed {esc(power['accessed'])}</td></tr>
<tr><td>Size</td><td>{f['raw_rows']:,} monthly rows, {f['raw_cols']} columns; {len(f['crops'])} crops, 3 provinces,
{f['years'][0]}-{f['years'][1]}</td><td>{f['power_rows']:,} monthly rows; root-zone soil wetness (0 to 1)</td></tr>
</tbody></table>
<p>The yield file has no month column and repeats each yearly yield on 12 monthly rows, with the same weather for
every crop in a province. Training on monthly rows would show each yield 12 times, so we aggregate to one row per
crop, province and year (rain and rain hours summed, the rest averaged) and merge soil wetness by province and year.
{f['n_dropped']} crop-province-years with Area = 0 (the crop was not grown or not reported: {dropped}) are dropped,
leaving {f['rows']} rows with {f['n_features']} input features. The raw data has no personal identifiers. In the
file, Yield = 25 x Production / Area, but the source does not state the unit, so we report
<i>dataset yield units</i>. The raw file meets the guideline minimum of 5,000 rows and 15 columns; the modelling
table is smaller because the yield is one number per year.</p>
"""


def eda_section(f: dict) -> str:
    lo, hi = f["yield_range"]
    return f"""
<h2>3. Exploratory data analysis</h2>
<p>Yields span {lo:.2f} to {hi:.2f} dataset yield units across crops, so the target is modelled on a log scale. There
are no empty cells; missing data shows up as Area = 0 years for three crops, which also leaves grapes with no Punjab
years and pomegranate with no Sindh years. Outliers are kept: they are real reported years, and the log scale limits
their pull. No single climate column correlates strongly with log yield on the training years (largest absolute
correlation {f['max_corr']:.2f}, {esc(f['max_corr_name'])}), and the rain deficit correlates only
{f['rain_gap_corr']:.2f} with a year's yield gap from its crop-province mean. Crop identity, not weather, carries most
of the signal.</p>
{row(fig("eda_yield_distribution.png", "Distribution of yearly yield (log10)."),
     fig("eda_missing_rows.png", "Missing data: crop-province-years kept and dropped."))}
{row(fig("eda_outliers_by_crop.png", "Yield by crop, log scale; points beyond 1.5 IQR."),
     fig("eda_correlation.png", "Correlation of inputs and log yield, training years."))}
{row(fig("eda_rows_per_crop_province.png", "Balance: years of data per crop and province."),
     fig("eda_rain_vs_yield_gap.png", "Rain deficit against the yield gap from the crop-province mean."))}
"""


def method_section(f: dict) -> str:
    m = METRICS
    sh = f["share"]
    grids = []
    for key, (_est, grid, label) in T.CANDIDATES.items():
        parts = [f"{k.split('__')[-1]} {', '.join(str(x) for x in v)}" for k, v in grid.items()]
        grids.append(f"{esc(label.split(' (')[0])}: {esc('; '.join(parts))}")
    tr, va, te = (m["split"][k] for k in ("train_years", "val_years", "test_years"))
    return f"""
<h2>4. Methodology</h2>
<p><b>Features.</b> Annual rain, mean, maximum and minimum temperature, evapotranspiration, solar radiation, rain
hours, soil wetness (mean and wettest month), three engineered drought features (rain deficit against the province's
training-year mean, water balance, aridity index), crop and province. <b>Pipeline.</b> Median imputation, standard
scaling and one-hot encoding inside a scikit-learn <code>Pipeline</code>, fitted on training rows only. The regressor
does not predict yield directly (crops differ about 100 times): it learns the log-scale gap between a year's yield and
the crop-province training average, plus a straight-line year trend (none, per crop, or per crop and province). Year
is never a model input; at prediction it is pinned to {m['inference_year']}. <b>Split by year.</b> Train
{tr[0]}-{tr[1]} ({m['data']['n_train']} rows, {sh['train']:.0%}), validation {va[0]}-{va[1]}
({m['data']['n_val']}, {sh['val']:.0%}), test {te[0]}-{te[1]} ({m['data']['n_test']}, {sh['test']:.0%}).
<b>Models.</b> Three baselines (overall mean, crop-province mean, crop-province mean plus year trend) and three
algorithms. <b>Tuning.</b> <code>GridSearchCV</code> with 5-fold <code>GroupKFold</code> grouped by year, so all
crops of one year stay in one fold (they share the same weather), scored on log-MAE. Grids:
{'. '.join(grids)}. <b>Selection</b> on validation log-MAE; the test years were scored once.
<b>Saved</b> with <code>joblib</code>; versions pinned in <code>requirements.txt</code>.</p>
"""


def results_section(f: dict) -> str:
    m = METRICS
    chosen = m["chosen"]["name"]
    rows = [(k, v) for k, v in m["baselines"].items()] + [(k, v) for k, v in m["models"].items()]
    body = []
    for key, v in rows:
        label = TABLE_LABELS[key] + (" (chosen)" if key == chosen else "")
        cells = [f"{v['val']['r2']:.3f}", f"{v['val']['mae']:.1f}", f"{v['test']['r2']:.3f}",
                 f"{v['test']['mae']:.1f}", f"{v['test']['rmse']:.1f}", f"{v['test']['log_mae']:.3f}"]
        cls = ' class="chosen"' if key == chosen else ""
        body.append(f"<tr{cls}><td>{esc(label)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    pi = m["prediction_interval"]
    verdict = "so it is optimistic" if pi["test_coverage"] < pi["level"] else "about as intended"
    best = m["models"][chosen]
    params = ", ".join(f"{k} {v}" for k, v in best["best_params"].items())
    val_gap = best["val"]["r2"] - best["test"]["r2"]
    top_w = f["imp_top_weather"]
    by_prov = ", ".join(f"{k} {v:.3f}" for k, v in f["test_log_mae_by_province"].items())
    return f"""
<h2>5. Results</h2>
<table class="results">
<thead><tr><th>Model</th><th>Val R²</th><th>Val MAE</th><th>Test R²</th><th>Test MAE</th><th>Test RMSE</th><th>Test log-MAE</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<p class="note">MAE and RMSE in dataset yield units. Chosen on validation log-MAE: {esc(best['label'])}
({esc(params)}).</p>
{row(fig("results_model_comparison.png", "Test R² and MAE of the baselines and models."), cls="wide")}
<p>{esc(I.model_caveat(m))} Validation R² is {val_gap:.3f} higher than test R² for the chosen model: some
over-fitting to the validation years, expected with three years per split. Test log-MAE by province:
{by_prov}. Permutation importance on the validation years: crop {f['imp_crop']:.3f}, province
{f['imp_province']:.3f}, the strongest weather input ({esc(top_w[1])}) only {top_w[0]:.4f}.</p>
<p>{esc(I.absolute_yield_caveat(m))} The likely range the apps show is the nominal {pi['level']:.0%} spread of
validation-year errors. It covered {pi['test_coverage']:.0%} of the test rows, {verdict}.</p>
{row(fig("results_predicted_vs_actual.png", "Predicted against actual yield, test years, log scale."),
     '<div class="side"><p class="note">Most crop-province pairs lie near the diagonal. The points far below it, '
     'mostly Balochistan, are years whose actual yield was much higher than predicted: a model built on '
     'crop-province averages and a straight-line trend does not follow such jumps.</p></div>')}
"""


def deployment_section() -> str:
    return f"""
<h2>6. Deployment</h2>
<p>All three front ends load the same saved pipeline through one shared package
(<code>src/climate_machine_learning/</code>), so they show the same numbers: below, all three show rice in Sindh in
its driest real year at the same predicted yield and share of a typical year. <b>Local web app:</b> FastAPI with
<code>/api/v1/predict</code>, <code>/api/v1/recommend</code> and <code>/api/v1/metrics</code>, HTML pages, input
validation (inputs outside the trained range are refused with a 422 that states the range) and error messages with no
fallback numbers; the same app is hosted at {link(VERCEL)}. <b>Streamlit Community Cloud:</b> input form,
prediction with a likely range, one chart: {link(STREAMLIT)}. <b>Hugging Face Hub:</b> model card and pipeline at
{link(HF_MODEL)}; no Space was built (group decision).</p>
{row(fig("deploy_streamlit.png", "Streamlit Community Cloud."),
     fig("deploy_vercel.png", "FastAPI pages on Vercel, all crops (rice ranked last).").replace("<figure>", '<figure class="bottom">'),
     fig("deploy_local.png", "Local FastAPI app, /predict for rice."), cls="three")}
"""


def proposal_section(f: dict) -> str:
    m = METRICS
    sh = f["share"]
    rows = [
        ("Target: yield in kg/ha", "Dataset yield units: the source does not state the unit, and values such as rice "
         "around 20 to 40 cannot be kg/ha."),
        ("Split 70/15/15, ordered by year", f"Year split {sh['train']:.0%}/{sh['val']:.0%}/{sh['test']:.0%}, so whole "
         "years stay together."),
        ("Several sources (national rain and temperature, city weather, PMD drought centre, Punjab irrigated area "
         "and district yields)", "Two sources used: the crop dataset and NASA POWER. The national series end in 2016 "
         "and are not provincial, the Punjab tables cover Punjab only, the city weather has no Balochistan, and the "
         "drought centre has no downloadable series."),
        ("Features: rain deficit, irrigated area share", "Rain deficit built; irrigated area share not built (no "
         "source has it for all three provinces)."),
        ("Baseline, linear regression, Random Forest, Gradient Boosting, 5-fold tuning", "Same, with Ridge as the "
         "linear model and folds grouped by year to avoid leakage across crops of one year."),
        ("Ranked list of low-water crops", "All 13 crops ranked by share of a typical year; FAO water need shown as "
         "context for the 8 crops it lists, so the user sees water use beside the ranking."),
        ("Optional best-crop classification", "Not built (future work)."),
    ]
    body = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows)
    return f"""
<h2>7. What we proposed and what we built</h2>
<table class="prop"><thead><tr><th>Proposal</th><th>Built</th></tr></thead><tbody>{body}</tbody></table>
<p>{esc(criterion(m))} It is checked against a stricter baseline too (section 5).</p>
"""


def limitations_section() -> str:
    return """
<h2>8. Limitations and future work</h2>
<p><b>Limitations.</b> Few rows (one per crop, province and year) from 22 years; weather and soil wetness are
province-level and identical for every crop in a province; within-province ranges are narrow, so inputs outside them
are refused rather than extrapolated; three crops have missing years; the yield unit is not confirmed; the win over
the crop-province mean holds on R² and MAE only, chosen after seeing the test results, and is mostly the year trend;
the likely range is optimistic; tied shares are ordered by name only; the soil wetness source adds little.
<b>Future work.</b> District or farm-level data and irrigation share, more years, a best-crop classifier, a model that
explains weather effects per crop, and a Hugging Face Space for the Gradio app already built in
<code>hf_space/</code>.</p>
"""


def references_section() -> str:
    refs = [
        "Abdullah-Masood-05. Pakistan crop yield vs climate (MIT). "
        "github.com/Abdullah-Masood-05/pakistan-crop-yield-vs-climate; upstream: Climate &amp; Agriculture in "
        "Pakistan, Zenodo, zenodo.org/records/15486470.",
        "NASA Langley Research Center POWER Project. Monthly and Annual API v2.10.1 (MERRA-2), accessed 2026-11-19. "
        "power.larc.nasa.gov; referencing: power.larc.nasa.gov/docs/referencing/.",
        "Brouwer, C. and Heibloem, M. (1986). Irrigation Water Management: Irrigation Water Needs. FAO Training "
        "manual no. 3, Table 14. fao.org/4/s2022e/s2022e07.htm.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
    ]
    return "<h2>9. References</h2><ol class=\"refs\">" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"


def contribution_section() -> str:
    body = []
    for name, roll, role in MEMBERS:
        parts = [t for t, owner in TOPIC_OWNERS.items() if owner == name]
        body.append(f"<tr><td>{esc(name)}</td><td>{esc(roll)}</td><td>{esc(role)}</td>"
                    f"<td>{esc('; '.join(p.split(' (')[0] for p in parts))}</td></tr>")
    return f"""
<h2>10. Individual contribution</h2>
<table class="contrib"><thead><tr><th>Name</th><th>Roll number</th><th>Role</th><th>Parts</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
"""


CSS = """
@page { size: A4; margin: 11mm 13mm 11mm 13mm; }
* { box-sizing: border-box; }
body { font-family: 'Liberation Serif', serif; font-size: 9.2pt; line-height: 1.28; color: #111; margin: 0; }
h1, h2, th, figcaption, .meta, .note { font-family: 'Liberation Sans', sans-serif; }
h1 { font-size: 14pt; margin: 0 0 1pt; }
h2 { font-size: 10pt; margin: 6pt 0 2pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; break-after: avoid; }
p { margin: 0 0 3pt; text-align: justify; hyphens: auto; }
a { color: #1c5cab; text-decoration: none; }
code { font-size: 8.2pt; }
.meta { font-size: 8pt; color: #444; margin-bottom: 3pt; }
.note { font-size: 7.6pt; color: #444; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 7.9pt; margin: 2pt 0 3pt; break-inside: avoid; }
th, td { padding: 1.3pt 4pt; border-bottom: 0.4pt solid #ccc; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 7.6pt; border-bottom: 0.8pt solid #555; }
table.data td, table.prop td, table.contrib td, table.data th, table.prop th, table.contrib th { text-align: left; }
table.data td:first-child { width: 17%; font-family: 'Liberation Sans', sans-serif; font-size: 7.6pt; }
table.prop td:first-child { width: 36%; }
tr.chosen td { font-weight: bold; }
.figrow { display: flex; gap: 2%; margin: 2pt 0 3pt; break-inside: avoid; align-items: flex-start; }
.figrow > * { flex: 0 0 49%; }
.figrow.wide > * { flex: 0 0 82%; margin-left: 9%; }
.figrow.three > * { flex: 0 0 32%; }
.side { padding-top: 14pt; }
figure { margin: 0; }
figure img { width: 100%; display: block; }
.figrow.three img { border: 0.4pt solid #bbb; height: 74mm; object-fit: cover; object-position: top; }
.figrow.three figure.bottom img { object-position: bottom; }
figcaption { font-size: 7.2pt; color: #444; margin-top: 1pt; }
ol.refs { margin: 0 0 3pt; padding-left: 14pt; font-size: 8.4pt; }
ol.refs li { margin-bottom: 1pt; }
"""


def build_html() -> str:
    f = facts()
    sections = [title_block(), abstract(f), problem(), data_section(f), eda_section(f), method_section(f),
                results_section(f), deployment_section(), proposal_section(f), limitations_section(),
                references_section(), contribution_section()]
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Low-Water Crop Selection Under Drought: Report</title>
<style>{CSS}</style></head>
<body>
{body}
</body></html>
"""


def print_pdf() -> None:
    exe = shutil.which("google-chrome") or shutil.which("chromium")
    assert exe, "google-chrome not found"
    with tempfile.TemporaryDirectory(prefix="chrome-report-") as profile:
        subprocess.run(
            [exe, "--headless=new", f"--user-data-dir={profile}", "--no-pdf-header-footer",
             "--disable-gpu", f"--print-to-pdf={PDF}", HTML.as_uri()],
            check=True, capture_output=True, timeout=120,
        )


def main() -> int:
    if "--pdf-only" not in sys.argv:
        HTML.parent.mkdir(parents=True, exist_ok=True)
        HTML.write_text(build_html(), encoding="utf-8")
    print_pdf()
    pages = subprocess.run(["pdfinfo", str(PDF)], check=True, capture_output=True, text=True).stdout
    n = next(line.split()[-1] for line in pages.splitlines() if line.startswith("Pages:"))
    print(f"wrote {HTML.relative_to(ROOT)} and {PDF.relative_to(ROOT)} ({n} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
