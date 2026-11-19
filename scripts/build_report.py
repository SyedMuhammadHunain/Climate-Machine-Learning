"""Builds the project report: docs/report/report.html, printed to docs/report/report.pdf by headless Chrome.

Run:   .venv/bin/python scripts/build_report_figures.py   # figures first (if the data or model changed)
       .venv/bin/python scripts/build_report.py           # HTML + PDF
       .venv/bin/python scripts/build_report.py --pdf-only  # re-print the existing HTML (used by checks)
Check: .venv/bin/python scripts/verify_report.py

Every number in the report is read from models/metrics.json, figure_values.json or the data at build
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from climate_machine_learning import inference as I  # noqa: E402
from verify_report import HTML, PDF, TABLE_LABELS  # noqa: E402

METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())


def esc(text) -> str:
    return html.escape(str(text))


def fig(name: str, caption: str, cls: str = "half") -> str:
    return (f'<figure class="{cls}"><img src="figures/{name}" alt="{esc(caption)}">'
            f"<figcaption>{esc(caption)}</figcaption></figure>")


# --------------------------------------------------------------------------- sections
def results_section() -> str:
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
    tv, te = m["split"]["val_years"], m["split"]["test_years"]
    pi = m["prediction_interval"]
    verdict = "so it is optimistic" if pi["test_coverage"] < pi["level"] else "about as intended"
    best = m["models"][chosen]
    params = ", ".join(f"{k} {v}" for k, v in best["best_params"].items())
    return f"""
<h2>6. Results</h2>
<table class="results">
<thead><tr><th>Model</th><th>Val R²</th><th>Val MAE</th><th>Test R²</th><th>Test MAE</th><th>Test RMSE</th><th>Test log-MAE</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<p class="note">Validation {tv[0]}-{tv[1]} ({m['data']['n_val']} rows), test {te[0]}-{te[1]} ({m['data']['n_test']} rows),
MAE and RMSE in dataset yield units. The model was selected on validation log-MAE; the test set was scored once.
Chosen: {esc(best['label'])} ({esc(params)}).</p>
{fig("results_model_comparison.png", "Test R² and MAE of the three baselines and three models (from metrics.json).", "full")}
<p>{esc(I.model_caveat(m))}</p>
<p>{esc(I.absolute_yield_caveat(m))}</p>
<p>The likely range the apps show is the nominal {pi['level']:.0%} spread of validation-year errors. It covered
{pi['test_coverage']:.0%} of the test rows, {verdict}.</p>
{fig("results_predicted_vs_actual.png", "Predicted against actual yield on the test years, log scale.")}
"""


SECTIONS = [results_section]

CSS = """
@page { size: A4; margin: 13mm 14mm 13mm 14mm; }
* { box-sizing: border-box; }
body { font-family: 'Liberation Serif', serif; font-size: 9.6pt; line-height: 1.32; color: #111; margin: 0; }
h1, h2, h3, th, figcaption, .meta, .note { font-family: 'Liberation Sans', sans-serif; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
h2 { font-size: 10.5pt; margin: 8pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
p { margin: 0 0 4pt; text-align: justify; hyphens: auto; }
.meta { font-size: 8.5pt; color: #444; margin-bottom: 6pt; }
.note { font-size: 7.8pt; color: #444; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3pt 0 4pt; }
th, td { padding: 1.6pt 4pt; border-bottom: 0.4pt solid #ccc; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 7.8pt; border-bottom: 0.8pt solid #555; }
tr.chosen td { font-weight: bold; }
figure { margin: 3pt 0 5pt; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure.half { display: inline-block; width: 49%; vertical-align: top; }
figure.full { width: 100%; }
figcaption { font-size: 7.4pt; color: #444; margin-top: 1pt; }
"""


def build_html() -> str:
    body = "\n".join(section() for section in SECTIONS)
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
