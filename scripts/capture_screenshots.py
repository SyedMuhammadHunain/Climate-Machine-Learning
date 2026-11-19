"""Captures the deployment screenshots for the report from the real, running apps.

Run:   .venv/bin/python scripts/capture_screenshots.py [--only streamlit,vercel,local]
Needs: google-chrome on PATH, internet for Streamlit and Vercel. Starts its own local uvicorn on port 8765.

Each app is driven like a visitor would (clean Chrome profile, not logged in): Sindh, driest real year.
Before saving a screenshot the script reads the page text and checks that rice's predicted yield and
share of a typical year, as the saved model gives them now, are on the page. The values it looked for and
what it found go to docs/report/figures/screenshot_checks.json; scripts/verify_report.py re-checks them.
Screenshots are never edited.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from verify_report import FIG_DIR, _expected_sindh_driest_rice  # noqa: E402
from chrome_cdp import Chrome as _Chrome, start_local_server as _start_server  # noqa: E402

STREAMLIT_URL = "https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/~/+/"
VERCEL_URL = "https://climate-machine-learning.vercel.app/predict.html"
LOCAL_PORT = 8765
LOCAL_URL = f"http://127.0.0.1:{LOCAL_PORT}/predict.html"
WIDTH, HEIGHT = 1280, 1220


def Chrome() -> _Chrome:
    """Report screenshots are taken at one desktop size."""
    return _Chrome(WIDTH, HEIGHT)


def found_on_page(chrome: _Chrome, exp: dict) -> list:
    text = chrome.eval("document.body.innerText")
    return [v for v in (exp["yield"], exp["pct"]) if v in text]


def has_both(exp: dict) -> str:
    return f"document.body.innerText.includes({json.dumps(exp['yield'])}) && document.body.innerText.includes({json.dumps(exp['pct'])})"


# --------------------------------------------------------------------------- one function per app
def capture_html_app(chrome: _Chrome, url: str, exp: dict, out: Path, crop: str) -> list:
    """The FastAPI pages (local and Vercel): Sindh, driest real year.

    crop="rice": the single-crop forecast (POST /api/v1/predict); the shot shows the form and the result.
    crop="ALL": all 13 crops ranked (POST /api/v1/recommend); the shot is scrolled so rice, ranked last,
    is at the bottom with the crops above it.
    """
    chrome.goto(url)
    chrome.wait_for("document.querySelectorAll('.preset-pill').length === 3", 90, "preset buttons")
    chrome.eval("(() => { const s = document.getElementById('field-province'); s.value = 'Sindh';"
                " s.dispatchEvent(new Event('change', {bubbles: true})); })()")
    chrome.wait_for("[...document.querySelectorAll('.preset-pill')].some(b => b.dataset.key === 'driest' && "
                    f"b.textContent.includes('{exp['year']}'))", 30, "Sindh preset buttons")
    chrome.click_at("document.querySelector('.preset-pill[data-key=driest]')")
    chrome.eval(f"document.getElementById('field-crop').value = {json.dumps(crop)}")
    chrome.click_at("document.getElementById('btn-submit-prediction')")
    chrome.wait_for(has_both(exp), 90, f"rice {exp['yield']} and {exp['pct']} on the page")
    if crop == "ALL":
        chrome.eval("(() => { const h = [...document.querySelectorAll('#results-container *')]"
                    ".find(e => e.children.length === 0 && e.textContent.trim() === 'rice');"
                    f" window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - {HEIGHT - 200}); }})()")
    else:
        chrome.eval("window.scrollTo(0, 0)")
    time.sleep(1.0)
    found = found_on_page(chrome, exp)
    chrome.screenshot(out, clip_to_js="document.getElementById('results-container')")
    return found


def capture_streamlit(chrome: _Chrome, exp: dict, out: Path) -> list:
    """Streamlit Community Cloud: pick Sindh in the province box, then the driest real year."""
    chrome.goto(STREAMLIT_URL)
    chrome.wait_for("document.body.innerText.includes('Fill the form from')", 240, "Streamlit app to wake and render")
    chrome.wait_for("!!document.querySelector('input[aria-label=Province]')", 60, "province box")
    chrome.click_at("document.querySelector('[data-testid=stSelectbox] input[aria-label=Province]')"
                    ".parentElement.querySelector('button')")  # the combobox's open button
    chrome.wait_for("[...document.querySelectorAll('[role=option]')].some(o => o.innerText.trim() === 'Sindh')",
                    30, "province options")
    chrome.click_at("[...document.querySelectorAll('[role=option]')].find(o => o.innerText.trim() === 'Sindh')")
    chrome.wait_for(f"document.body.innerText.includes('driest {exp['year']}')", 60, "Sindh caption")
    chrome.click_at("[...document.querySelectorAll('[data-testid=stRadio] label')]"
                    ".find(l => l.innerText.trim().startsWith('Driest'))")
    chrome.wait_for(has_both(exp), 90, f"rice {exp['yield']} and {exp['pct']} on the page")
    chrome.send("Emulation.setDeviceMetricsOverride",  # taller, so the comparison chart is in the shot
                {"width": WIDTH, "height": 1700, "deviceScaleFactor": 1, "mobile": False})
    time.sleep(3.0)  # let the chart finish drawing
    found = found_on_page(chrome, exp)
    chrome.screenshot(out)
    return found


def start_local_server():
    return _start_server(LOCAL_PORT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="streamlit,vercel,local")
    which = ap.parse_args().only.split(",")
    exp = _expected_sindh_driest_rice()
    print(f"expected for rice, Sindh, driest real year ({exp['year']}): {exp['yield']} and {exp['pct']}")
    checks_path = FIG_DIR / "screenshot_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    server = start_local_server() if "local" in which else None
    try:
        for key in which:
            chrome = Chrome()
            out = FIG_DIR / f"deploy_{key}.png"
            try:
                if key == "streamlit":
                    found = capture_streamlit(chrome, exp, out)
                    url = STREAMLIT_URL
                else:
                    url = VERCEL_URL if key == "vercel" else LOCAL_URL
                    found = capture_html_app(chrome, url, exp, out, crop="ALL" if key == "vercel" else "rice")
            finally:
                chrome.close()
            checks[key] = {"url": url, "expected": exp, "found_on_page": found,
                           "captured_at": time.strftime("%Y-%m-%d")}
            print(f"{key}: found {found} -> {out.relative_to(ROOT)}")
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
    checks_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
