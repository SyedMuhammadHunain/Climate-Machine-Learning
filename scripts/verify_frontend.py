"""Static guard: frontend must not contain fake/fallback data.

Run: .venv/bin/python scripts/verify_frontend.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "climate-resilient-frontend"

BANNED = re.compile(r"bajra|jowar|renderFallback|fallback simulation|regional fallback", re.IGNORECASE)
PAGES = ["index.html", "predict.html", "metrics.html"]


def main() -> int:
    failures = []

    if (FRONTEND / "careers.html").exists():
        failures.append("careers.html still exists (dead copy of predict.html)")

    for name in PAGES:
        text = (FRONTEND / name).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if BANNED.search(line):
                failures.append(f"{name}:{lineno}: banned text: {line.strip()[:90]}")

    predict = (FRONTEND / "predict.html").read_text(encoding="utf-8")
    if "3500" in predict:
        failures.append("predict.html: old 3500 ms timeout still present")
    if "30000" not in predict:
        failures.append("predict.html: 30000 ms timeout missing")
    if 'id="btn-retry"' not in predict:
        failures.append("predict.html: Retry button missing")

    # T4: invented inputs and old semantics must be gone from the single-crop path
    for token in ("PRESET_CONFIGS", "field-area", "field-climate-index", "val + 6.0", "val >= 40.0",
                  "Total Expected Production", "Seasonal Precipitation", "is_drought_resilient",
                  "water_intensive_warnings", "ranked_drought_resilient", "kg/ha", "alert('"):
        if token in predict:
            failures.append(f"predict.html: old/invented logic still present: {token!r}")
    for token in ("api/v1/metadata", "pct_of_normal", "predicted_yield", "yield_unit", "inputs", "ranked_crops", "fao_water_need", "tied_with", "absolute_yield_note"):
        if token not in predict:
            failures.append(f"predict.html: expected T4 element missing: {token!r}")

    # T6: metrics page must read real results, not hold numbers or hand-made formulas
    metrics_page = (FRONTEND / "metrics.html").read_text(encoding="utf-8")
    for token in ("0.892", "32-year", "PROVINCE_DEFAULTS", "CROP_PHYSIOLOGY", "Elasticity", "elasticity",
                  "9.5 ms", "kg/ha", "Embedded Benchmarks", "1990", "17 features", "Kw"):
        if token in metrics_page:
            failures.append(f"metrics.html: hardcoded/old content present: {token!r}")
    for token in ("api/v1/metrics", "api/v1/metadata", "btn-retry"):
        if token not in metrics_page:
            failures.append(f"metrics.html: expected element missing: {token!r}")
    # SPEC-promote-offset-model.md: the win is shown per metric with the loss and the after-the-fact choice
    for token in ("beats_by_metric", "judged_on_chosen_after_test", "crop_province_mean_plus_trend"):
        if token not in metrics_page:
            failures.append(f"metrics.html: must read {token!r} from /metrics")
    if "Better than" in metrics_page and "beats_by_metric" not in metrics_page:
        failures.append("metrics.html: states 'Better than' without the per-metric comparison")
    if re.search(r'class="metric-val"[^>]*>[^<]*\d', metrics_page):
        failures.append("metrics.html: a metric value is typed into the HTML")

    # review O4: an older metrics.json must not blank the page (no eager access to newer fields)
    if re.search(r"crop_province_mean_plus_trend\.trend", metrics_page):
        failures.append("metrics.html: crop_province_mean_plus_trend.trend is read without optional chaining")
    if re.search(r"m\.chosen\.beats_by_metric\s*;", metrics_page):
        failures.append("metrics.html: m.chosen.beats_by_metric is read without a fallback")
    # review O1: preset buttons are built with DOM calls, not innerHTML with inline onclick
    if "onclick=\"applyPreset" in predict:
        failures.append("predict.html: preset buttons still use innerHTML with an inline onclick")

    # Rename (user request): the site is called "Low-Water Crops"; the old name must be gone from pages and assets
    site_name = "Low-Water Crops"
    old_brand = re.compile(r"infinite[\s_-]*field", re.IGNORECASE)
    for name in PAGES + ["package.json"]:
        if old_brand.search((FRONTEND / name).read_text(encoding="utf-8")):
            failures.append(f"{name}: still contains the old site name")
    for stale in sorted((FRONTEND / "assets").glob("*")):
        if old_brand.search(stale.name):
            failures.append(f"assets/{stale.name}: file name still carries the old site name")
    index_page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    for logo in ("dark-logo", "white-logo"):
        if not re.search(r'class="logo %s">\s*%s\s*<' % (logo, re.escape(site_name)), index_page):
            failures.append(f"index.html: header text for .{logo} must read {site_name!r}")
    for name in PAGES:
        title = re.search(r"<title>(.*?)</title>", (FRONTEND / name).read_text(encoding="utf-8"))
        if not title or site_name not in title.group(1):
            failures.append(f"{name}: <title> must contain {site_name!r}")
    if 'property="og:url" content="https://climate-machine-learning.vercel.app/"' not in index_page:
        failures.append("index.html: og:url must be the live site, not the old domain")
    for name in PAGES:
        for ref in re.findall(r'(?:href|content|src)="(assets/[^"]+)"', (FRONTEND / name).read_text(encoding="utf-8")):
            if not (FRONTEND / ref).exists():
                failures.append(f"{name}: references {ref}, which does not exist")
    if "<title>%s</title>" % site_name not in predict:
        failures.append("predict.html: the logo SVG must carry <title>%s</title>" % site_name)
    readme = (FRONTEND / "README.md").read_text(encoding="utf-8")
    if "Cloned from" not in readme or old_brand.match(readme):
        failures.append("frontend README must keep the honest 'Cloned from' line and not be headed by the old name")

    # the big zoom-in text is a flat mesh of the name: check the GLB is well formed and the size the page expects
    import struct
    glb = (FRONTEND / "assets" / "logo-3d.glb").read_bytes()
    if glb[:4] != b"glTF":
        failures.append("logo-3d.glb: not a binary glTF file")
    else:
        jlen = struct.unpack("<I", glb[12:16])[0]
        gltf = json.loads(glb[20:20 + jlen])
        acc = gltf["accessors"]
        pos, idx = acc[gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]], acc[gltf["meshes"][0]["primitives"][0]["indices"]]
        width, height = pos["max"][0] - pos["min"][0], pos["max"][2] - pos["min"][2]
        if len(gltf["meshes"]) != 1 or pos["count"] < 200 or idx["count"] % 3:
            failures.append("logo-3d.glb: expected one mesh with a triangle list")
        if pos["min"][1] != 0 or pos["max"][1] != 0:
            failures.append("logo-3d.glb: the text must lie flat in the XZ plane (y = 0)")
        if not (0.44 <= width <= 0.50 and 0.03 <= height <= 0.12 and abs(pos["max"][0] + pos["min"][0]) < 0.02):
            failures.append(f"logo-3d.glb: size {width:.3f} x {height:.3f} is not the size the page was built for (about 0.475 wide, centred)")

    # T7: no demo/simulation wording or old hand-made concepts in any page
    for name in PAGES:
        text = (FRONTEND / name).read_text(encoding="utf-8").lower()
        for word in ("simulat", "demo", "dummy", "mock", "illustrat", "kg/ha", "physiolog", "elasticit",
                     "low-water, climate-resilient crops", "(low water)", "(high water)"):
            if word in text:
                failures.append(f"{name}: banned wording {word!r}")
    if "driest real year" not in predict.lower() or "scenarios" not in predict:
        failures.append("predict.html: quick-fill must use real-year scenarios from /metadata")
    if "training_rows_possible" not in predict:
        failures.append("predict.html: training row completeness not shown")

    # crop dropdown must list exactly the trained crops, with plain names
    ref_crops = json.loads((ROOT / "models" / "reference.json").read_text())["crops"]
    select = re.search(r'<select id="field-crop".*?</select>', predict, re.S).group(0)
    options = [v for v in re.findall(r'<option value="([^"]+)"', select) if v != "ALL"]
    if sorted(options) != sorted(ref_crops):
        failures.append(f"predict.html crop dropdown {sorted(options)} differs from trained crops {sorted(ref_crops)}")
    for value, label in re.findall(r'<option value="([^"]+)"[^>]*>([^<]*)</option>', select):
        if value != "ALL" and label.strip().lower() != value:
            failures.append(f"predict.html crop option label {label!r} is not the plain crop name {value!r}")

    # second source (NASA POWER): the form must send the soil fields, the Metrics page must name both sources
    for token in ("field-soil-mean", "field-soil-max", "soil_wetness_mean", "soil_wetness_max"):
        if token not in predict:
            failures.append(f"predict.html: second-source field missing: {token!r}")
    for token in ("data.sources", "licence"):
        if token not in metrics_page:
            failures.append(f"metrics.html: data sources section missing: {token!r}")

    # HTML structure: unbalanced <div>s once pushed the buttons and results outside the centred column
    from html.parser import HTMLParser

    class Structure(HTMLParser):
        skip = {"meta", "link", "input", "br", "img", "hr", "path", "circle", "line", "source", "area", "base", "col"}

        def __init__(self):
            super().__init__()
            self.opened = self.closed = 0
            self.stack = []
            self.inside = {}

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag in self.skip:
                return
            self.stack.append((tag, a.get("class", "")))
            self.opened += tag == "div"
            if a.get("id"):
                self.inside[a["id"]] = any("job-board-container" in c.split() for _, c in self.stack)

        def handle_endtag(self, tag):
            if tag in self.skip:
                return
            self.closed += tag == "div"
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break

    for name in PAGES:
        parser = Structure()
        parser.feed((FRONTEND / name).read_text(encoding="utf-8"))
        if parser.opened != parser.closed:
            failures.append(f"{name}: unbalanced <div> tags ({parser.opened} opened, {parser.closed} closed)")
        if name == "predict.html":
            for element in ("field-province", "validation-alert", "btn-submit-prediction", "results-container"):
                if not parser.inside.get(element):
                    failures.append(f"predict.html: #{element} is outside the centred .job-board-container")

    # API text must be escaped before it goes into innerHTML
    for sink in ("single-crop-items", "team-items-1", "advisory-summary"):
        for block in re.findall(r'getElementById\("' + sink + r'"\)\.innerHTML = `(.*?)`;', predict, re.S):
            for raw in re.findall(r"\$\{([^}]*)\}", block):
                bare = raw.strip()
                safe = bare.startswith(("esc(", "pct(", "yl(", "fmt(", "inputsHtml", "item.rank", "fao", "data.training_rows", "item.training_rows"))
                if not safe and ("data." in bare or "item." in bare or "cropDisplay" in bare):
                    failures.append(f"predict.html: unescaped API text in {sink} innerHTML: ${{{bare}}}")

    # dev servers started by this repo must listen on localhost only (python's http.server binds 0.0.0.0 by default)
    for rel in ("scripts/run_frontend.sh", "frontend/climate-resilient-frontend/package.json"):
        for lineno, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if "http.server" in line and "--bind 127.0.0.1" not in line:
                failures.append(f"{rel}:{lineno}: http.server must use --bind 127.0.0.1 (it listens on all interfaces otherwise)")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASSED: frontend has no fallback data, retry button present, 30 s timeout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
