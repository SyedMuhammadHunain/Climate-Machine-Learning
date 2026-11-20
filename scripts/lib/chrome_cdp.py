"""Minimal Chrome DevTools Protocol client shared by the browser scripts.

Used by scripts/build/capture_screenshots.py (report screenshots) and scripts/checks/verify_responsive.py
(layout checks at phone, tablet and desktop sizes). Needs google-chrome (or chromium) on PATH and
websocket-client (already in requirements.txt).
"""
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

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Chrome:
    """Headless Chrome driven over CDP: emulate a device, navigate, evaluate JS, click, screenshot."""

    def __init__(self, width: int = 1280, height: int = 1220, extra_args: tuple = ()):
        self.width, self.height = width, height
        self.port = free_port()
        self.profile = tempfile.mkdtemp(prefix="chrome-profile-", dir=os.environ.get("CLAUDE_JOB_DIR_TMP"))
        exe = shutil.which("google-chrome") or shutil.which("chromium")
        assert exe, "google-chrome not found"
        self.proc = subprocess.Popen(
            [exe, "--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile}",
             f"--window-size={width},{height}", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
             *extra_args, "about:blank"],
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
        self.emulate(width, height)

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

    def emulate(self, width: int, height: int, dpr: float = 1, mobile: bool = False, touch: bool = False,
                reduced_motion: bool = False) -> None:
        """Viewport size, device pixel ratio, mobile viewport handling, touch input and reduced motion.
        Takes effect for the next navigation (and immediately for layout)."""
        self.width, self.height = width, height
        self.send("Emulation.setDeviceMetricsOverride",
                  {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": mobile})
        self.send("Emulation.setTouchEmulationEnabled", {"enabled": touch, "maxTouchPoints": 5 if touch else 1})
        self.send("Emulation.setEmulatedMedia",
                  {"features": [{"name": "prefers-reduced-motion", "value": "reduce" if reduced_motion else ""}]})

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
            time.sleep(0.25)
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
            params["clip"] = {"x": x, "y": r["y"], "width": min(self.width - x, r["w"] + 64), "height": r["h"], "scale": 1}
        data = self.send("Page.captureScreenshot", params)["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            shutil.rmtree(self.profile, ignore_errors=True)


def start_local_server(port: int) -> subprocess.Popen:
    """uvicorn serving the API and the HTML pages, as `uvicorn app.main:app` does; returns once /health answers."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("local server did not start")
