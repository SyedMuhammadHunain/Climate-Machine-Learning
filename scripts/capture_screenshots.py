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
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client, already in requirements.txt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from verify_report import FIG_DIR, _expected_sindh_driest_rice  # noqa: E402

STREAMLIT_URL = "https://climate-machine-learning-nb5v2knz3htvseqqpuhvrk.streamlit.app/~/+/"
VERCEL_URL = "https://climate-machine-learning.vercel.app/predict.html"
LOCAL_PORT = 8765
LOCAL_URL = f"http://127.0.0.1:{LOCAL_PORT}/predict.html"
WIDTH, HEIGHT = 1280, 1220


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Chrome:
    """Minimal Chrome DevTools Protocol client: navigate, evaluate JS, click, screenshot."""

    def __init__(self):
        self.port = free_port()
        self.profile = tempfile.mkdtemp(prefix="chrome-profile-", dir=os.environ.get("CLAUDE_JOB_DIR_TMP"))
        exe = shutil.which("google-chrome") or shutil.which("chromium")
        assert exe, "google-chrome not found"
        self.proc = subprocess.Popen(
            [exe, "--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile}",
             f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome still starting
                time.sleep(0.1)
        else:
            raise RuntimeError("Chrome did not start")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
        self.next_id = 0
        self.send("Emulation.setDeviceMetricsOverride",
                  {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": 1, "mobile": False})

    def send(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        msg_id = self.next_id
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, js: str):
        r = self.send("Runtime.evaluate", {"expression": js, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(f"JS error: {d.get('exception', {}).get('description') or d.get('text')} in {js[:80]}")
        return r["result"].get("value")

    def goto(self, url: str) -> None:
        self.send("Page.navigate", {"url": url})

    def wait_for(self, js: str, timeout: float, what: str) -> None:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.eval(js):
                    return
            except RuntimeError:
                pass
            time.sleep(0.5)
        try:
            seen = self.eval("location.href + ' | ' + (document.body ? document.body.innerText.slice(0, 200) : '')")
        except RuntimeError as e:
            seen = str(e)
        raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}; page shows: {seen!r}")

    def click_at(self, rect_js: str) -> None:
        """Real mouse click at the centre of the element whose bounding rect rect_js returns."""
        r = self.eval(f"(() => {{ const r = ({rect_js}).getBoundingClientRect(); "
                      "return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })()")
        for kind in ("mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent",
                      {"type": kind, "x": r["x"], "y": r["y"], "button": "left", "clickCount": 1})

    def screenshot(self, path: Path, clip_to_js: str | None = None) -> None:
        """Viewport screenshot; with clip_to_js, only the horizontal band of that element (plus a margin),
        so the side whitespace of a centred page is not captured. Nothing is drawn or changed."""
        params = {"format": "png"}
        if clip_to_js:
            r = self.eval(f"(() => {{ const r = ({clip_to_js}).getBoundingClientRect(); "
                          "return {x: r.x + window.scrollX, y: window.scrollY, w: r.width, h: window.innerHeight}; })()")
            x = max(0, r["x"] - 32)
            # clip is in page coordinates, so y is the current scroll offset
            params["clip"] = {"x": x, "y": r["y"], "width": min(WIDTH - x, r["w"] + 64), "height": r["h"], "scale": 1}
        data = self.send("Page.captureScreenshot", params)["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            shutil.rmtree(self.profile, ignore_errors=True)


def found_on_page(chrome: Chrome, exp: dict) -> list:
    text = chrome.eval("document.body.innerText")
    return [v for v in (exp["yield"], exp["pct"]) if v in text]


def has_both(exp: dict) -> str:
    return f"document.body.innerText.includes({json.dumps(exp['yield'])}) && document.body.innerText.includes({json.dumps(exp['pct'])})"


# --------------------------------------------------------------------------- one function per app
def capture_html_app(chrome: Chrome, url: str, exp: dict, out: Path, crop: str) -> list:
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


def capture_streamlit(chrome: Chrome, exp: dict, out: Path) -> list:
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


def start_local_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(LOCAL_PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/health", timeout=1)
            return proc
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("local server did not start")


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
