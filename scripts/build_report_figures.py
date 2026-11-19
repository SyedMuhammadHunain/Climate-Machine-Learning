"""Builds every figure the report uses, from the real data, the saved model and models/metrics.json.

Run:   .venv/bin/python scripts/build_report_figures.py
Writes docs/report/figures/*.png and figure_values.json (the numbers each results figure was drawn
from, so scripts/verify_report.py can compare them with metrics.json). No number is typed by hand.
Rebuilding gives byte-identical files (no timestamps in the PNGs, single-thread prediction).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from climate_machine_learning import features as F  # noqa: E402
from climate_machine_learning import inference as I  # noqa: E402
from climate_machine_learning import training as T  # noqa: E402

OUT = ROOT / "docs" / "report" / "figures"

# Palette: validated categorical slots (dataviz reference palette, light mode), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#a3a29c"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
PROVINCE_STYLE = {  # colour plus marker shape, so province is never colour alone
    "Balochistan": (BLUE, "o"),
    "Punjab": (ORANGE, "s"),
    "Sindh": (AQUA, "^"),
}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])

WIDTH = 3.4  # inches: two figures side by side on an A4 page
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 100,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, metadata={"Software": None})
    plt.close(fig)


def province_legend(ax, loc="best"):
    for prov, (c, m) in PROVINCE_STYLE.items():
        ax.scatter([], [], color=c, marker=m, s=14, label=prov)
    ax.legend(loc=loc, frameon=False, handletextpad=0.2)


# --------------------------------------------------------------------------- EDA
def yield_distribution(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.hist(np.log10(table["Yield"]), bins=30, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("log10 yield (dataset yield units)")
    ax.set_ylabel("crop-province-years")
    ax.set_title(f"Yield spans over two orders of magnitude (n = {len(table)})", loc="left")
    save(fig, "eda_yield_distribution.png")


def missing_rows(raw: pd.DataFrame, table: pd.DataFrame, dropped: dict) -> None:
    kept = table.groupby("Crop").size()
    crops = sorted(kept.index, key=lambda c: (-dropped.get(c, 0), c), reverse=True)
    k = np.array([kept[c] for c in crops])
    d = np.array([dropped.get(c, 0) for c in crops])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.barh(crops, k, color=BLUE, height=0.7, label="kept")
    ax.barh(crops, d, left=k, color=ORANGE, height=0.7, label="dropped (Area = 0)", edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("crop-province-years")
    n_nan = int(raw.isna().sum().sum())
    ax.set_title(f"Missing data: {n_nan} empty cells;\n{int(d.sum())} years with Area = 0 dropped", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "eda_missing_rows.png")


def outliers_by_crop(table: pd.DataFrame) -> None:
    order = table.groupby("Crop")["Yield"].median().sort_values().index.tolist()
    data = [table.loc[table["Crop"] == c, "Yield"].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.boxplot(data, orientation="horizontal", tick_labels=order, widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE, "linewidth": 0.8},
               medianprops={"color": BLUE, "linewidth": 1.2}, whiskerprops={"color": INK_2, "linewidth": 0.7},
               capprops={"color": INK_2, "linewidth": 0.7},
               flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": ORANGE, "markeredgewidth": 0})
    ax.set_xscale("log")
    ax.set_xlabel("yield (dataset yield units, log scale)")
    ax.set_title("Yield by crop (orange: beyond 1.5 IQR)", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "eda_outliers_by_crop.png")


def correlation(train: pd.DataFrame) -> None:
    cols = F.NUMERIC_FEATURES
    df = train[cols].copy()
    df["log_yield"] = np.log(train["Yield"])
    corr = df.corr().to_numpy()
    labels = cols + ["log_yield"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=5.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    ax.set_title(f"Pearson correlation, training years\n(n = {len(train)} crop-province-years)", loc="left")
    save(fig, "eda_correlation.png")


def rows_per_crop_province(table: pd.DataFrame) -> None:
    counts = table.pivot_table(index="Crop", columns="Province", values="Year", aggfunc="count").fillna(0)
    counts = counts[list(PROVINCE_STYLE)].sort_index()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    im = ax.imshow(counts.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=counts.to_numpy().max(), aspect="auto")
    ax.set_xticks(range(counts.shape[1]), counts.columns)
    ax.set_yticks(range(counts.shape[0]), counts.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    vmax = counts.to_numpy().max()
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = int(counts.iat[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                    color=SURFACE if v > 0.55 * vmax else INK)
    ax.set_title("Years of data per crop and province", loc="left")
    save(fig, "eda_rows_per_crop_province.png")


def rain_vs_yield_gap(train: pd.DataFrame) -> None:
    mean = train.groupby(["Crop", "Province"])["Yield"].transform("mean")
    gap = np.log(train["Yield"] / mean)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    for prov, (c, m) in PROVINCE_STYLE.items():
        sel = train["Province"] == prov
        ax.scatter(train.loc[sel, "rain_deficit_pct"], gap[sel], s=9, color=c, marker=m, alpha=0.75,
                   linewidths=0, label=prov)
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set_xlabel("rain deficit, % (positive = drier than the province mean)")
    ax.set_ylabel("log(yield / crop-province mean)")
    ax.set_title("Yield gap against rain deficit, training years", loc="left")
    ax.legend(frameon=False, loc="best", handletextpad=0.2)
    save(fig, "eda_rain_vs_yield_gap.png")


# --------------------------------------------------------------------------- results
def model_comparison(metrics: dict) -> dict:
    names = {"global_mean": "Overall mean", "crop_province_mean": "Crop-province mean",
             "crop_province_mean_plus_trend": "Crop-province mean + trend"}
    rows = [(k, names[k], v["test"]) for k, v in metrics["baselines"].items()]
    rows += [(k, v["label"], v["test"]) for k, v in metrics["models"].items()]
    chosen = metrics["chosen"]["name"]
    labels = [lab + (" (chosen)" if k == chosen else "") for k, lab, _ in rows]
    colors = [BLUE if k == chosen else GRAY for k, _, _ in rows]
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDTH, 2.3), sharey=True)
    for ax, metric, title in ((axes[0], "r2", "Test R² (higher is better)"),
                              (axes[1], "mae", "Test MAE (lower is better)")):
        vals = [t[metric] for _, _, t in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=colors, height=0.65)
        ax.axvline(0, color=INK_2, linewidth=0.6)
        for yi, v in zip(y, vals):
            fmt = f"{v:.3f}" if metric == "r2" else f"{v:.1f}"
            ax.text(v if v >= 0 else 0, yi, " " + fmt, va="center", fontsize=6.5, color=INK)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        if metric == "mae":
            ax.set_xlabel("dataset yield units")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(0, min(vals)) * 1.15, max(vals) * 1.25)
    save(fig, "results_model_comparison.png")
    return {k: {"label": lab, "r2": t["r2"], "mae": t["mae"]} for k, lab, t in rows}


def predicted_vs_actual(test: pd.DataFrame) -> dict:
    pipeline = joblib.load(ROOT / "models" / "crop_yield_pipeline.joblib")
    I.single_thread(pipeline)
    pred = pipeline.predict(test[T.MODEL_COLS])
    y = test["Yield"].to_numpy()
    r2, mae = float(r2_score(y, pred)), float(mean_absolute_error(y, pred))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for prov, (c, m) in PROVINCE_STYLE.items():
        sel = (test["Province"] == prov).to_numpy()
        ax.scatter(y[sel], pred[sel], s=10, color=c, marker=m, alpha=0.8, linewidths=0, label=prov)
    lo, hi = min(y.min(), pred.min()) * 0.8, max(y.max(), pred.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.8, linestyle="--", label="perfect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("actual yield (dataset yield units)")
    ax.set_ylabel("predicted yield")
    ax.set_title(f"Test years {T.TEST_YEARS[0]}-{T.TEST_YEARS[1]}: R² {r2:.3f}, MAE {mae:.1f} (n = {len(y)})",
                 loc="left")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    save(fig, "results_predicted_vs_actual.png")
    return {"n": int(len(y)), "r2": r2, "mae": mae}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    raw = pd.read_parquet(T.RAW_PATH)
    table, dropped, _reference, splits = T.load_tables()

    yield_distribution(table)
    missing_rows(raw, table, dropped)
    outliers_by_crop(table)
    correlation(splits["train"])
    rows_per_crop_province(table)
    rain_vs_yield_gap(splits["train"])
    values = {
        "model_comparison": model_comparison(metrics),
        "predicted_vs_actual": predicted_vs_actual(splits["test"]),
    }
    (OUT / "figure_values.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
