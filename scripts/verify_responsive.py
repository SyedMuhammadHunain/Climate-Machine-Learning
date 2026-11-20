"""Renders every HTML page at phone, tablet and desktop sizes in headless Chrome and checks the layout.

Run:   .venv/bin/python scripts/verify_responsive.py [--only test1,test2] [--size 360x800] [--page predict]
                                                    [--shots DIR]
Needs: google-chrome on PATH. Starts its own uvicorn on a free port. Spec: SPEC-responsive.md.

Each page is loaded once per size; one JavaScript pass measures the layout and the tests below assert on
those measurements. Screenshots of every page and size go to --shots (default build/responsive/, git-ignored)
for a person to look at; they are never committed or used in the report.

Run one test without the CLI (same as the other verify_* scripts):
    import verify_responsive as V; [f() for f in V.TESTS if f.__name__ == "no_horizontal_page_scroll"]
"""
import argparse
import atexit
import json
import sys
import time
import urllib.error
import urllib.request

import websocket  # websocket-client, already in requirements.txt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from chrome_cdp import Chrome, free_port, start_local_server  # noqa: E402

# (name, width, height, device pixel ratio, phone): phone = mobile viewport + touch input
SIZES = [
    ("320x568", 320, 568, 2, True),       # smallest phones still in use
    ("360x800", 360, 800, 3, True),       # most common Android phone
    ("412x915", 412, 915, 2.625, True),   # large Android phone
    ("800x360", 800, 360, 3, True),       # Android phone in landscape
    ("768x1024", 768, 1024, 1, False),    # tablet portrait
    ("1024x768", 1024, 768, 1, False),    # tablet landscape, small laptop
    ("1440x900", 1440, 900, 1, False),    # laptop
    ("1920x1080", 1920, 1080, 1, False),  # desktop
    ("2560x1440", 2560, 1440, 1, False),  # large desktop
]

# page -> JS that is true once the page has drawn its content from the API
PAGES = {
    "index": "document.fonts.status === 'loaded' && !!document.querySelector('.paragraph-container')",
    "predict": "document.fonts.status === 'loaded' && document.querySelectorAll('.preset-pill').length === 3",
    "metrics": "document.fonts.status === 'loaded' && document.querySelectorAll('#cmp-body tr').length > 0",
}
# The predictor in the states a visitor sees: after ranking all crops, after one crop, with the optional inputs open.
# A name is "<page>-<state>"; the page is loaded, then SETUP[name] drives it (Sindh, driest real year, like the demo).
# The landing page is driven by scrolling: "index-text" scrolls to where the paragraph and the button are shown,
# "index-reduced" loads it with prefers-reduced-motion (everything shown at once, no scene, no scrolling).
PAGES["index-text"] = "document.fonts.status === 'loaded' && !!window.ScrollTrigger && ScrollTrigger.getAll().length > 0"
PAGES["index-reduced"] = "document.fonts.status === 'loaded' && !!document.querySelector('a.careers-button')"
# The crop detail page needs a scenario in its address: rice in Sindh, driest real year (see _Run.url)
PAGES["crop"] = "document.fonts.status === 'loaded' && document.body.dataset.ready === 'true'"
PAGES["predict-result"] = PAGES["predict-single"] = PAGES["predict-advanced"] = PAGES["predict"]
PREDICT_VARIANTS = ("predict", "predict-result", "predict-single", "predict-advanced")
# The landing page draws text over a WebGL scene, so colour contrast is not measurable from CSS there.
CONTRAST_PAGES = (*PREDICT_VARIANTS, "metrics", "crop")
TAB_PAGES = (*PREDICT_VARIANTS, "metrics", "crop")

MIN_TARGET_PX = 44     # WCAG 2.5.5 target size, at phone sizes
MIN_GUTTER_PX = 8      # tabs keep this much space from the screen edge, so none looks cut off
MIN_FORM_FONT_PX = 16  # under 16 px, mobile browsers zoom in when a field gets focus
MIN_CONTRAST = 4.5     # WCAG AA, normal text
MIN_CONTRAST_LARGE = 3.0  # WCAG AA, large text (24 px, or 18.66 px bold)

CHROME_ARGS = ("--use-angle=swiftshader", "--enable-unsafe-swiftshader")

# One pass over the rendered page. Returns plain data; all judging happens in Python.
# DEVICE_WIDTH is replaced with the emulated width: on a mobile viewport Chrome widens the layout viewport to
# fit content that is too wide (as phones do), so window.innerWidth grows with the overflow and cannot be used.
MEASURE_JS = r"""
(() => {
  const vw = DEVICE_WIDTH, vh = window.innerHeight;
  const isHtml = el => el instanceof HTMLElement;
  const shown = el => {
    if (!isHtml(el)) return false;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== "hidden" && cs.display !== "none";
  };
  const opacity = el => { let o = 1; for (let e = el; e && e !== document; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity); return o; };
  const ownText = el => [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join("").trim();
  const label = el => {
    let s = el.tagName.toLowerCase();
    if (el.id) s += "#" + el.id;
    else if (el.classList.length) s += "." + [...el.classList].slice(0, 2).join(".");
    if (el.tagName === "A" && el.getAttribute("href")) s += `[href=${el.getAttribute("href")}]`;
    return s;
  };
  const scrollParent = el => {
    for (let e = el.parentElement; e && e !== document.body; e = e.parentElement) {
      const ox = getComputedStyle(e).overflowX;
      if ((ox === "auto" || ox === "scroll") && e.scrollWidth > e.clientWidth) return e;
    }
    return null;
  };
  const textEls = [...document.body.querySelectorAll("*")].filter(el => shown(el) && ownText(el).length > 0
      && !el.closest("svg, script, style, noscript"));

  const outside = r => r.left < -1 || r.right > vw + 1;
  const boxes = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (el.closest("script, style, noscript") || (el.ownerSVGElement)) continue;  // inside an SVG: judged by its <svg>
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0 || getComputedStyle(el).display === "none" || !outside(r)) continue;
    if (scrollParent(el)) continue;
    const parent = el.parentElement;
    if (parent && parent !== document.body && outside(parent.getBoundingClientRect())) continue;  // parent reported
    boxes.push({ el: label(el), text: (el.innerText || el.getAttribute("aria-label") || "").trim().slice(0, 40),
                 left: Math.round(r.left), right: Math.round(r.right) });
  }

  const offscreen = [], clipped = [], lowContrast = [];
  const parse = c => { const m = c.match(/[\d.]+/g) || [0, 0, 0, 0]; return [+m[0], +m[1], +m[2], m.length > 3 ? +m[3] : 1]; };
  const lum = ([r, g, b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const blend = (top, under) => { const a = top[3]; return [0, 1, 2].map(i => top[i] * a + under[i] * (1 - a)).concat(1); };
  const background = el => {
    const layers = [];
    for (let e = el; e; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor);
      if (c[3] > 0) { layers.push(c); if (c[3] >= 1) break; }
    }
    let bg = [255, 255, 255, 1];
    for (let i = layers.length - 1; i >= 0; i--) bg = blend(layers[i], bg);
    return bg;
  };

  for (const el of textEls) {
    const r = el.getBoundingClientRect(), cs = getComputedStyle(el), text = ownText(el).slice(0, 40);
    const visible = opacity(el) > 0.05;
    if (!scrollParent(el) && (r.left < -1 || r.right > vw + 1)) offscreen.push({ el: label(el), text, left: Math.round(r.left), right: Math.round(r.right), visible });
    const ox = cs.overflowX;
    if ((ox === "hidden" || ox === "clip" || cs.textOverflow === "ellipsis") && el.scrollWidth > el.clientWidth + 1)
      clipped.push({ el: label(el), text, scrollWidth: el.scrollWidth, clientWidth: el.clientWidth });
    if (visible) {
      const fg = parse(cs.color), bg = background(el);
      const fgOnBg = blend(fg, bg);
      const [l1, l2] = [lum(fgOnBg), lum(bg)].sort((a, b) => b - a);
      const ratio = (l1 + 0.05) / (l2 + 0.05);
      const size = parseFloat(cs.fontSize), bold = parseInt(cs.fontWeight) >= 700;
      const large = size >= 24 || (bold && size >= 18.66);
      lowContrast.push({ el: label(el), text, ratio: Math.round(ratio * 100) / 100, large, color: cs.color, size });
    }
  }

  const isInlineLink = el => el.tagName === "A" && getComputedStyle(el).display === "inline"
      && [...el.parentElement.childNodes].some(n => n !== el && n.textContent.trim().length > 0);
  const targets = [...document.querySelectorAll("a[href], button, select, input:not([type=hidden]), textarea, [role=button], summary")]
      .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).display !== "none" && !isInlineLink(el); })
      .map(el => { const r = el.getBoundingClientRect(); return { el: label(el), text: (el.innerText || el.value || el.getAttribute("aria-label") || "").trim().slice(0, 30), w: Math.round(r.width), h: Math.round(r.height) }; });

  const formFonts = [...document.querySelectorAll("input:not([type=hidden]), select, textarea")]
      .filter(shown).map(el => ({ el: label(el), size: parseFloat(getComputedStyle(el).fontSize) }));

  const tabs = [...document.querySelectorAll(".site-nav .nav-pill")].map(el => {
    const r = el.getBoundingClientRect();
    return { text: el.innerText.trim(), left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height,
             clipped: el.scrollWidth > el.clientWidth + 1, shown: shown(el) };
  });

  const startButtons = [...document.querySelectorAll("a.careers-button")].map(el => {
    const r = el.getBoundingClientRect();
    return { text: el.innerText.trim(), href: el.getAttribute("href"), left: r.left, right: r.right, w: r.width, h: r.height };
  });

  const rect = el => { if (!el) return null; const r = el.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height }; };
  const actions = { submit: rect(document.getElementById("btn-submit-prediction")), reset: rect(document.querySelector(".btn-reset")), row: rect(document.querySelector(".form-actions-row")) };
  const gridLefts = [...document.querySelectorAll(".inputs-grid .input-group")].map(el => Math.round(el.getBoundingClientRect().left));
  const visible = id => { const el = document.getElementById(id); return !!el && shown(el); };
  const cropWrapper = document.querySelector(".crop-table-wrapper");
  const cropTable = cropWrapper ? { sw: cropWrapper.scrollWidth, cw: cropWrapper.clientWidth, wrap: [...document.querySelectorAll(".crop-table td:first-child")].filter(td => td.getClientRects().length).map(td => td.getClientRects().length > 0 && td.getBoundingClientRect().height > 2.2 * parseFloat(getComputedStyle(td).lineHeight || 20)).some(Boolean) } : null;
  const results = { container: visible("results-container"), rankedItems: document.querySelectorAll("#team-items-1 .rank-row").length,
                    singleItems: document.querySelectorAll("#single-crop-items .job-post-item").length,
                    rankedShown: visible("team-section-1"), singleShown: visible("single-crop-section"),
                    rankHeads: [...document.querySelectorAll("#team-items-1 .rank-group-head")].map(el => el.innerText.replace(/\s+/g, " ").trim()),
                    rankRows: [...document.querySelectorAll("#team-items-1 .rank-row")].map(el => { const r = el.getBoundingClientRect();
                        return { text: el.innerText.replace(/\s+/g, " ").trim(), h: r.height, left: r.left, right: r.right, tag: el.tagName, href: el.getAttribute("href") }; }),
                    advisory: (() => { const q = s => document.querySelector("#advisory-summary " + s); const tx = el => el ? el.innerText.replace(/\s+/g, " ").trim() : null;
                        return { headline: tx(q(".advisory-headline")), basis: tx(q(".advisory-basis")), caveat: tx(q(".advisory-caveat")), absolute: tx(q(".advisory-absolute")),
                                 subheads: [...document.querySelectorAll("#advisory-summary .advisory-subhead")].map(tx),
                                 facts: [...document.querySelectorAll("#advisory-summary .advisory-facts > div")].map(d => ({ label: tx(d.querySelector("dt")), value: tx(d.querySelector("dd")) })),
                                 factLefts: [...document.querySelectorAll("#advisory-summary .advisory-facts > div")].map(d => Math.round(d.getBoundingClientRect().left)) }; })(),
                    unitCount: ((document.getElementById("results-container") || { innerText: "" }).innerText.match(/dataset yield units/g) || []).length };
  const canvas = document.getElementById("mesh-canvas");
  const landing = document.querySelector(".paragraph-container") ? {
    header: rect(document.querySelector(".header")), footer: rect(document.querySelector(".footer")),
    paragraph: rect(document.querySelector(".paragraph-container")), paragraphOpacity: opacity(document.querySelector(".paragraph-container")),
    button: rect(document.querySelector("a.careers-button")), buttonOpacity: document.querySelector("a.careers-button") ? opacity(document.querySelector("a.careers-button")) : 0,
    canvas: canvas ? { w: canvas.width, h: canvas.height, rect: rect(canvas), fit: getComputedStyle(canvas).objectFit } : null,
    scrollSpace: rect(document.getElementById("scroll-space")), scrollHeight: document.documentElement.scrollHeight,
    footer1: (el => { if (!el) return null; const r = document.createRange(); r.selectNodeContents(el); return rect(r); })(document.querySelector(".white-footer-typer")),
    footer2: (el => { if (!el) return null; const r = document.createRange(); r.selectNodeContents(el); return rect(r); })(document.querySelectorAll(".white-footer-typer")[1]),
    footerLine: (el => el && getComputedStyle(el).display !== "none" ? rect(el) : null)(document.querySelector(".white-footer-horizontal-line")),
  } : null;
  return { vw, vh, actions, gridLefts, results, landing, cropTable, bodyText: document.body.innerText, innerWidth: window.innerWidth, scrollWidth: document.documentElement.scrollWidth, boxes, offscreen, clipped, lowContrast, targets, formFonts, tabs, startButtons };
})()
"""


# The five tables of metrics.html (R3, R6): 1 model comparison, 2 importance, 3 data sources, 4 rows per crop, 5 ranges.
TABLE_NAMES = ["model comparison", "importance", "data sources", "rows per crop", "climate ranges"]
CARD_TABLES = (0, 2, 4)        # under CARD_BELOW_PX these turn into one card per row
CARD_BELOW_PX = 600
FIT_WITHOUT_SCROLL = (0, 4)    # the two wide tables fit their column from this width up
FIT_FROM_PX = 1440
MIN_CELL_FONT_PX = 14
MIN_HEAD_FONT_PX = 13

TABLES_JS = r"""
(() => {
  const norm = s => s.replace(/\s+/g, " ").trim();
  return [...document.querySelectorAll(".data-table")].map((table, index) => {
    const wrapper = table.closest(".data-table-wrapper");
    const heads = [...table.querySelectorAll("thead th")];
    const headStyle = heads.length ? getComputedStyle(heads[0]) : null;
    const rows = [...table.querySelectorAll("tbody tr")].map(tr => {
      const cs = getComputedStyle(tr);
      const first = tr.cells[0] ? getComputedStyle(tr.cells[0]) : null;
      return {
        bg: first && first.backgroundColor !== "rgba(0, 0, 0, 0)" ? first.backgroundColor : cs.backgroundColor, weight: first ? parseInt(first.fontWeight) : 0,
        cells: [...tr.cells].map((td, i) => {
          const s = getComputedStyle(td);
          return { text: norm(td.innerText), label: td.getAttribute("data-label"), head: heads[i] ? norm(heads[i].textContent) : null,
                   num: td.classList.contains("num"), align: s.textAlign, font: parseFloat(s.fontSize), whiteSpace: s.whiteSpace,
                   position: s.position };
        })
      };
    });
    const section = table.closest(".section-block");
    return {
      index, rows,
      scrolls: wrapper.scrollWidth > wrapper.clientWidth + 1,
      wrapperWidth: wrapper.clientWidth, tableWidth: table.scrollWidth,
      headVisible: heads.length > 0 && headStyle.display !== "none" && heads[0].getBoundingClientRect().height > 0,
      heads: heads.map(h => ({ text: norm(h.textContent), font: parseFloat(getComputedStyle(h).fontSize), num: h.classList.contains("num"),
                               align: getComputedStyle(h).textAlign, position: getComputedStyle(h).position })),
      legend: section.querySelector(".table-legend") ? norm(section.querySelector(".table-legend").innerText) : null,
    };
  });
})()
"""


def _api_get(url: str) -> dict:
    return json.load(urllib.request.urlopen(url, timeout=30))


def _api_post(url: str, body: dict) -> tuple:
    """(status, json) for a POST; a 4xx answer is returned, not raised."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=30)
        return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


_SCENARIO_CACHE = {}


def _api_scenarios(base: str) -> dict:
    """Sindh's real driest and wettest years and the typical year (all nine medians), from /metadata."""
    if base not in _SCENARIO_CACHE:
        meta = _api_get(f"{base}/api/v1/metadata")
        pick = lambda kind: next(s for s in meta["scenarios"] if s["province"] == "Sindh" and s["kind"] == kind)
        _SCENARIO_CACHE[base] = {"driest": pick("driest"), "wettest": pick("wettest"), "crops": meta["all_crops"],
                                 "median": {"inputs": dict(meta["reference"]["Sindh"]["median"])}}
    return _SCENARIO_CACHE[base]


def crop_url(base: str, crop: str, province: str, inputs: dict) -> str:
    from urllib.parse import urlencode
    return f"{base}/crop.html?" + urlencode({"crop": crop, "province": province, **inputs})


def _click(chrome: Chrome, selector: str) -> None:
    chrome.eval(f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block: 'center'}})")
    chrome.click_at(f"document.querySelector({json.dumps(selector)})")


def _predict(chrome: Chrome, crop: str, done_js: str) -> None:
    """Sindh, driest real year, one crop or all (as the demo does), then wait for the result to be drawn."""
    chrome.eval("(() => { const s = document.getElementById('field-province'); s.value = 'Sindh';"
                " s.dispatchEvent(new Event('change', {bubbles: true})); })()")
    time.sleep(0.5)
    _click(chrome, ".preset-pill[data-key=driest]")
    chrome.eval(f"document.getElementById('field-crop').value = {json.dumps(crop)}")
    _click(chrome, "#btn-submit-prediction")
    chrome.wait_for(done_js, 90, f"the {crop} result")


def _open_advanced(chrome: Chrome) -> None:
    _click(chrome, "#toggle-advanced-btn")
    chrome.wait_for("document.getElementById('advanced-panel').classList.contains('open')", 10, "optional inputs")


def _scroll_to_text(chrome: Chrome) -> None:
    """Scroll through the intro until the paragraph and the button are both visible (opacity above 0.9)."""
    opacity = ("(el => { let v = 1; for (let e = el; e; e = e.parentElement) v *= parseFloat(getComputedStyle(e).opacity); return v; })")
    for i in range(45):
        chrome.eval(f"window.scrollTo(0, {0.40 + i * 0.013} * (document.documentElement.scrollHeight - innerHeight))")
        time.sleep(0.15)
        if chrome.eval(f"(() => {{ const o = {opacity}; return o(document.querySelector('.paragraph-container')) > 0.9"
                       f" && o(document.querySelector('a.careers-button')) > 0.9; }})()"):
            return
    raise AssertionError("the landing paragraph and button never became visible while scrolling the intro")


SETUP = {
    "index-text": _scroll_to_text,
    "predict-result": lambda c: _predict(c, "ALL", "document.querySelectorAll('#team-items-1 .rank-row').length > 0"),
    "predict-single": lambda c: _predict(c, "rice", "document.querySelectorAll('#single-crop-items .job-post-item').length > 0"),
    "predict-advanced": _open_advanced,
}


class _Run:
    """Lazily started server + browser, and the measurements per (page, size). Shared by all tests."""

    def __init__(self):
        self.port = None
        self.server = None
        self.chrome = None
        self.cache = {}
        self.sizes = SIZES
        self.pages = list(PAGES)
        self.shots = ROOT / "build" / "responsive"

    def base(self) -> str:
        if self.server is None:
            self.port = free_port()
            self.server = start_local_server(self.port)
            atexit.register(self.close)
        return f"http://127.0.0.1:{self.port}"

    def url(self, page: str) -> str:
        if page == "crop":
            return crop_url(self.base(), "rice", "Sindh", _api_scenarios(self.base())["driest"]["inputs"])
        return f"{self.base()}/{page.split('-')[0]}.html"

    def browser(self) -> Chrome:
        if self.chrome is None:
            self.chrome = Chrome(extra_args=CHROME_ARGS)
        return self.chrome

    def measure(self, page: str, size: tuple) -> dict:
        """Measurements for one page and size. The WebGL landing page sometimes takes headless Chrome down in a long run:
        start a fresh browser and measure that page again, once."""
        try:
            return self._measure(page, size)
        except (websocket.WebSocketException, OSError):
            try:
                self.chrome.close()
            except Exception:  # noqa: BLE001 - it is already gone
                pass
            self.chrome = None
            return self._measure(page, size)

    def _measure(self, page: str, size: tuple) -> dict:
        key = (page, size[0])
        if key not in self.cache:
            name, w, h, dpr, phone = size
            chrome = self.browser()
            chrome.emulate(w, h, dpr=dpr, mobile=phone, touch=phone, reduced_motion=page == "index-reduced")
            chrome.goto(self.url(page))
            chrome.wait_for(PAGES[page], 60, f"{page}.html to finish loading at {name}")
            if page in SETUP:
                SETUP[page](chrome)
            if page != "index-text":
                chrome.eval("window.scrollTo(0, 0)")
            time.sleep(0.5)  # let late layout (fonts swapped in, API text) settle
            self.cache[key] = chrome.eval(MEASURE_JS.replace("DEVICE_WIDTH", str(w)))
            if page == "metrics":
                self.cache[key]["tables"] = chrome.eval(TABLES_JS)
            self.shots.mkdir(parents=True, exist_ok=True)
            chrome.screenshot(self.shots / f"{page}_{name}.png")
        return self.cache[key]

    def each(self, pages=None):
        """(page, size, measurements) for every selected page and size."""
        for page in self.pages:
            if pages is not None and page not in pages:
                continue
            for size in self.sizes:
                yield page, size, self.measure(page, size)

    def close(self) -> None:
        if self.chrome:
            self.chrome.close()
            self.chrome = None
        if self.server:
            self.server.terminate()
            self.server.wait(timeout=10)
            self.server = None


RUN = _Run()


def _group(problems: list) -> list:
    """One line per (place, kind of problem) with a count and one example, so a failing table is not 400 lines.
    Each problem is a (place, kind, example) tuple."""
    seen = {}
    for place, kind, example in problems:
        entry = seen.setdefault((place, kind), [0, example])
        entry[0] += 1
    return [f"{place}: {kind} x{n}, e.g. {example}" for (place, kind), (n, example) in seen.items()]


def _fail(problems: list, title: str) -> None:
    if problems:
        shown = "\n    ".join(problems[:25])
        more = f"\n    ... and {len(problems) - 25} more" if len(problems) > 25 else ""
        raise AssertionError(f"{title}: {len(problems)} problem(s)\n    {shown}{more}")


# --------------------------------------------------------------------------- generic layout tests (R4)
def no_horizontal_page_scroll():
    """Page never scrolls sideways; only table wrappers may (R4.1)."""
    problems = [f"{page} {size[0]}: page is {m['scrollWidth']} px wide in a {m['vw']} px viewport"
                for page, size, m in RUN.each() if m["scrollWidth"] > m["vw"] + 1]
    _fail(problems, "sideways page scroll")


def boxes_stay_inside_viewport():
    """No element (text, logo SVG, image, form field, card) sticks out past the screen edge, unless it is
    inside a scrolling wrapper. Reported at the outermost element that sticks out (R4.1)."""
    problems = [f"{page} {size[0]}: {b['el']} {b['text']!r} spans {b['left']} to {b['right']} px"
                for page, size, m in RUN.each() for b in m["boxes"]]
    _fail(problems, "elements outside the viewport")


def text_stays_inside_viewport():
    """No text sticks out past the left or right edge, unless it is inside a scrolling wrapper (R4.1)."""
    problems = [f"{page} {size[0]}: {o['el']} {o['text']!r} spans {o['left']} to {o['right']} px"
                for page, size, m in RUN.each() for o in m["offscreen"]]
    _fail(problems, "text outside the viewport")


def no_clipped_text():
    """No text cut off by overflow: hidden or an ellipsis (R2, R6)."""
    problems = [f"{page} {size[0]}: {c['el']} {c['text']!r} needs {c['scrollWidth']} px, has {c['clientWidth']} px"
                for page, size, m in RUN.each() for c in m["clipped"]]
    _fail(problems, "clipped text")


def tap_targets_44px_on_phones():
    """Every link, button and field at least 44 x 44 px at phone sizes (R4.4)."""
    problems = [f"{page} {size[0]}: {t['el']} {t['text']!r} is {t['w']}x{t['h']} px"
                for page, size, m in RUN.each() if size[4]
                for t in m["targets"] if t["w"] < MIN_TARGET_PX or t["h"] < MIN_TARGET_PX]
    _fail(problems, f"tap targets under {MIN_TARGET_PX} px")


def form_text_16px_on_phones():
    """Form controls use at least 16 px text at phone sizes, so the browser does not zoom on focus (R4.5)."""
    problems = [f"{page} {size[0]}: {f['el']} has {f['size']:g} px text"
                for page, size, m in RUN.each() if size[4]
                for f in m["formFonts"] if f["size"] < MIN_FORM_FONT_PX]
    _fail(problems, f"form text under {MIN_FORM_FONT_PX} px")


def text_contrast_meets_wcag_aa():
    """All visible text at least 4.5:1 against its background, 3:1 for large text (R6)."""
    problems = []
    for page, size, m in RUN.each(CONTRAST_PAGES):
        for c in m["lowContrast"]:
            need = MIN_CONTRAST_LARGE if c["large"] else MIN_CONTRAST
            if c["ratio"] < need:
                problems.append(f"{page} {size[0]}: {c['el']} {c['text']!r} {c['color']} is {c['ratio']}:1, needs {need}:1")
    # the same text fails at every size; report each element once
    _fail(sorted(set(p.split(": ", 1)[1] for p in problems)), "low contrast text")


# --------------------------------------------------------------------------- tabs (R2)
def tabs_fully_visible():
    """The three tabs are inside the viewport with a small gutter, unclipped, not overlapping, and 44 px
    tall on phones (R2)."""
    problems = []
    for page, size, m in RUN.each(TAB_PAGES):
        where = f"{page} {size[0]}"
        tabs = m["tabs"]
        if len(tabs) != 3:
            problems.append(f"{where}: expected 3 tabs, found {len(tabs)}")
            continue
        for t in tabs:
            if not t["shown"]:
                problems.append(f"{where}: tab {t['text']!r} is not shown")
            if t["left"] < MIN_GUTTER_PX or t["right"] > m["vw"] - MIN_GUTTER_PX:
                problems.append(f"{where}: tab {t['text']!r} spans {t['left']:.0f} to {t['right']:.0f} px "
                                f"(needs {MIN_GUTTER_PX} px from each edge of {m['vw']} px)")
            if t["clipped"]:
                problems.append(f"{where}: tab {t['text']!r} text is clipped")
            if size[4] and t["h"] < MIN_TARGET_PX:
                problems.append(f"{where}: tab {t['text']!r} is {t['h']:.0f} px tall")
        for i, a in enumerate(tabs):
            for b in tabs[i + 1:]:
                if a["left"] < b["right"] - 1 and b["left"] < a["right"] - 1 and a["top"] < b["bottom"] - 1 and b["top"] < a["bottom"] - 1:
                    problems.append(f"{where}: tabs {a['text']!r} and {b['text']!r} overlap")
    _fail(problems, "tabs")


# --------------------------------------------------------------------------- tables (R3, R6)
def _tables():
    """(size, table) for every table of metrics.html at every selected size."""
    for page, size, m in RUN.each(("metrics",)):
        assert len(m["tables"]) == len(TABLE_NAMES), f"metrics {size[0]}: {len(m['tables'])} tables, expected {len(TABLE_NAMES)}"
        for tb in m["tables"]:
            yield size, tb


def tables_become_cards_on_phones():
    """Under 600 px the wide tables (1, 3, 5) show one card per row with a label on every value, taken from the
    column header; the narrow tables (2, 4) stay tables and fit the screen with no scroll (R3)."""
    problems = []
    for size, tb in _tables():
        if size[1] >= CARD_BELOW_PX:
            continue
        where = f"metrics {size[0]} table {tb['index'] + 1} ({TABLE_NAMES[tb['index']]})"
        if tb["index"] in CARD_TABLES:
            if tb["headVisible"]:
                problems.append((where, "header row still shown, expected cards", ""))
            for r in tb["rows"]:
                for c in r["cells"]:
                    if not c["label"] or c["label"] != c["head"]:
                        problems.append((where, "cell without the label of its column", f"{c['text'][:25]!r} has {c['label']!r}, header {c['head']!r}"))
        else:
            if not tb["headVisible"]:
                problems.append((where, "header row hidden, expected a normal table", ""))
            if tb["scrolls"]:
                problems.append((where, "table wider than its box", f"needs {tb['tableWidth']} px, has {tb['wrapperWidth']} px"))
    _fail(_group(problems), "phone tables")


def wide_tables_fit_or_scroll_inside_wrapper():
    """From 600 px: real tables. A table wider than its box scrolls inside it with the first column and the header
    row sticky; tables 1 and 5 need no scroll from 1440 px (R3, R6). The page itself never scrolls (other test)."""
    problems = []
    for size, tb in _tables():
        if size[1] < CARD_BELOW_PX:
            continue
        where = f"metrics {size[0]} table {tb['index'] + 1} ({TABLE_NAMES[tb['index']]})"
        if not tb["headVisible"]:
            problems.append(f"{where}: header row hidden")
        if tb["scrolls"]:
            first_col = all(r["cells"][0]["position"] == "sticky" for r in tb["rows"])
            sticky_head = all(h["position"] == "sticky" for h in tb["heads"])
            if not first_col:
                problems.append(f"{where}: scrolls ({tb['tableWidth']} > {tb['wrapperWidth']} px) but first column is not sticky")
            if not sticky_head:
                problems.append(f"{where}: scrolls but header cells are not sticky")
            if size[1] >= FIT_FROM_PX and tb["index"] in FIT_WITHOUT_SCROLL:
                problems.append(f"{where}: needs {tb['tableWidth']} px, has {tb['wrapperWidth']} px; must fit from {FIT_FROM_PX} px")
    _fail(problems, "wide tables")


def table_text_is_readable():
    """Body text 14 px, headers 13 px, numbers right-aligned under right-aligned headers, numbers never broken (R6)."""
    problems = []
    for size, tb in _tables():
        where = f"metrics {size[0]} table {tb['index'] + 1}"
        for h in tb["heads"]:
            if h["font"] < MIN_HEAD_FONT_PX:
                problems.append((where, f"header under {MIN_HEAD_FONT_PX} px", f"{h['text']!r} is {h['font']:g} px"))
            if h["num"] and h["align"] != "right" and tb["headVisible"]:
                problems.append((where, "numeric header not right-aligned", f"{h['text']!r} is {h['align']}"))
        for r in tb["rows"]:
            for c in r["cells"]:
                if c["font"] < MIN_CELL_FONT_PX:
                    problems.append((where, f"cell under {MIN_CELL_FONT_PX} px", f"{c['text'][:30]!r} is {c['font']:g} px"))
                if c["num"] and c["whiteSpace"] != "nowrap":
                    problems.append((where, "numeric cell may wrap", f"{c['text'][:30]!r} white-space {c['whiteSpace']}"))
                if c["num"] and size[1] >= CARD_BELOW_PX and c["align"] != "right":
                    problems.append((where, "numeric cell not right-aligned", f"{c['text'][:30]!r} is {c['align']}"))
    _fail(_group(problems), "table text")


def table_values_identical_at_every_size():
    """Every cell shows the same text at every size: only the presentation changes (R3)."""
    ref_size = next(s for s in SIZES if s[0] == "1440x900")
    ref = {tb["index"]: [[c["text"] for c in r["cells"]] for r in tb["rows"]] for tb in RUN.measure("metrics", ref_size)["tables"]}
    problems = []
    for size, tb in _tables():
        got = [[c["text"] for c in r["cells"]] for r in tb["rows"]]
        if got != ref[tb["index"]]:
            problems.append(f"metrics {size[0]} table {tb['index'] + 1}: cell texts differ from {ref_size[0]}")
    _fail(problems, "table values")


def chosen_model_row_is_highlighted():
    """The chosen model's row is tinted and bold, not only marked "(chosen)" in the text (R6)."""
    problems = []
    for size, tb in _tables():
        if tb["index"] != 0:
            continue
        chosen = [r for r in tb["rows"] if "(chosen)" in r["cells"][0]["text"]]
        where = f"metrics {size[0]}"
        if len(chosen) != 1:
            problems.append(f"{where}: {len(chosen)} rows marked chosen")
            continue
        r = chosen[0]
        if r["bg"] in ("rgba(0, 0, 0, 0)", "rgb(255, 255, 255)"):
            problems.append(f"{where}: chosen row has no tint ({r['bg']})")
        if r["weight"] < 600:
            problems.append(f"{where}: chosen row font weight is {r['weight']}")
    _fail(problems, "chosen row")


def jargon_headers_are_explained():
    """A one-line legend under the model comparison and importance tables explains each short header (R6)."""
    problems = []
    wanted = {0: ["R²", "MAE", "RMSE", "log-MAE", "skill", "Val", "Test"], 1: ["log-MAE", "weight"]}
    for size, tb in _tables():
        if size[0] != "1440x900" or tb["index"] not in wanted:
            continue
        legend = tb["legend"]
        if not legend:
            problems.append(f"table {tb['index'] + 1} ({TABLE_NAMES[tb['index']]}): no .table-legend")
            continue
        problems += [f"table {tb['index'] + 1} legend does not explain {w!r}" for w in wanted[tb["index"]] if w.lower() not in legend.lower()]
        if tb["index"] == 1 and "Relative weight" not in [h["text"] for h in tb["heads"]]:
            problems.append("table 2: bar column has no 'Relative weight' header")
    _fail(problems, "table legends")


# --------------------------------------------------------------------------- landing page (R1)
def predictor_form_layout():
    """Inputs one column under 600 px and two from 600 px; Predict and Reset stack full width under 400 px and sit
    side by side from 400 px (R4.6)."""
    problems = []
    for page, size, m in RUN.each(("predict",)):
        where = f"predict {size[0]}"
        columns = len(set(m["gridLefts"]))
        want = 1 if size[1] < 600 else 2
        if columns != want:
            problems.append(f"{where}: inputs in {columns} column(s), expected {want}")
        a = m["actions"]
        if size[1] < 400:
            for name in ("submit", "reset"):
                if a[name]["w"] < a["row"]["w"] - 2:
                    problems.append(f"{where}: {name} button is {a[name]['w']:.0f} px wide in a {a['row']['w']:.0f} px row, expected full width")
            if a["reset"]["top"] < a["submit"]["bottom"] - 1:
                problems.append(f"{where}: buttons are not stacked")
        elif abs(a["submit"]["top"] - a["reset"]["top"]) > 2:
            problems.append(f"{where}: buttons are not side by side")
    _fail(problems, "predictor form")


def prediction_results_are_drawn_at_every_size():
    """After a prediction the ranked list shows all 13 crops, and a single-crop prediction shows its forecast (R4.6)."""
    problems = []
    for page, size, m in RUN.each(("predict-result", "predict-single")):
        r = m["results"]
        where = f"{page} {size[0]}"
        if not r["container"]:
            problems.append(f"{where}: results container is not shown")
        if page == "predict-result" and (r["rankedItems"] != 13 or not r["rankedShown"]):
            problems.append(f"{where}: {r['rankedItems']} crops listed (shown: {r['rankedShown']}), expected 13")
        if page == "predict-single" and (r["singleItems"] != 1 or not r["singleShown"]):
            problems.append(f"{where}: {r['singleItems']} forecast items (shown: {r['singleShown']}), expected 1")
    _fail(problems, "prediction results")


def _expected_sindh_driest_ranking() -> dict:
    """The API's own ranking for the scenario the predict-result page runs (Sindh, driest real year)."""
    base = RUN.base()
    meta = json.load(urllib.request.urlopen(f"{base}/api/v1/metadata", timeout=30))
    inputs = next(s["inputs"] for s in meta["scenarios"] if s["province"] == "Sindh" and s["kind"] == "driest")
    request = urllib.request.Request(f"{base}/api/v1/recommend", data=json.dumps({"province": "Sindh", **inputs}).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=30))


def result_rows_show_only_crop_rank_share_and_yield():
    """Each crop row shows its name, rank, share of a typical year and yield, nothing else; tied crops sit under one
    heading with one rank; the yield unit is stated once (R8, R5)."""
    import re
    api = _expected_sindh_driest_ranking()
    by_crop = {c["crop"]: c for c in api["ranked_crops"]}
    unique = re.compile(r"^(\w+)( barely grown here)? Rank (\d+) \u00b7 (\d+\.\d)% of a typical year \u00b7 Yield (\d+\.\d\d)$")
    member = re.compile(r"^(\w+)( barely grown here)? Yield (\d+\.\d\d)$")
    head = re.compile(r"^Rank (\d+) \u00b7 (\d+\.\d)% of a typical year \u00b7 (\d+) crops tied$")
    tied_groups = [g for g in api["groups"] if len(g["crops"]) > 1]
    problems = []
    for page, size, m in RUN.each(("predict-result",)):
        where, r = f"{page} {size[0]}", m["results"]
        if len(r["rankRows"]) != 13:
            problems.append(f"{where}: {len(r['rankRows'])} crop rows, expected 13")
            continue
        heads = [head.match(h) for h in r["rankHeads"]]
        if not all(heads) or [(int(h[1]), float(h[2]), int(h[3])) for h in heads] != [(g["rank"], g["pct_of_normal"], len(g["crops"])) for g in tied_groups]:
            problems.append(f"{where}: tie group headings {r['rankHeads']} differ from the API's groups {[(g['rank'], g['pct_of_normal'], len(g['crops'])) for g in tied_groups]}")
        order = []
        for row in r["rankRows"]:
            mu, mm = unique.match(row["text"]), member.match(row["text"])
            if not (mu or mm):
                problems.append(f"{where}: row text {row['text']!r} is not crop, rank, share, yield")
                continue
            crop = (mu or mm)[1].lower()
            order.append(crop)
            c = by_crop[crop]
            yield_shown = (mu[5] if mu else mm[3])
            if yield_shown != f"{c['predicted_yield']:.2f}":
                problems.append(f"{where}: {crop} yield {yield_shown}, API {c['predicted_yield']:.2f}")
            if bool(mu or mm) and (" barely grown here" in row["text"]) != c["barely_grown"]:
                problems.append(f"{where}: {crop} barely grown tag does not match the API ({c['barely_grown']})")
            if mu and (int(mu[3]) != next(g["rank"] for g in api["groups"] if crop in g["crops"]) or float(mu[4]) != c["pct_of_normal"].__round__(1)):
                problems.append(f"{where}: {crop} rank/share {mu[3]} / {mu[4]} differ from the API")
            if any(word in row["text"] for word in ("FAO", "training", "Predicted yield", "tied with", "dataset yield units")):
                problems.append(f"{where}: row {row['text']!r} carries more than rank, share and yield")
            if row["h"] < MIN_TARGET_PX:
                problems.append(f"{where}: {crop} row is {row['h']:.0f} px tall")
            if row["left"] < -1 or row["right"] > m["vw"] + 1:
                problems.append(f"{where}: {crop} row spans {row['left']:.0f} to {row['right']:.0f} px")
        if order != [c["crop"] for c in api["ranked_crops"]]:
            problems.append(f"{where}: crop order {order} differs from the API's ranking")
        if r["unitCount"] != 1:
            problems.append(f"{where}: the yield unit is stated {r['unitCount']} times, expected once")
    _fail(problems, "result rows")


def advisory_card_is_structured():
    """The advisory is a headline, a facts grid, how crops are ranked, and the caveats, each its own block with the API's
    text unchanged; the facts grid is one column under 480 px and two from 480 px (R10)."""
    api = _expected_sindh_driest_ranking()
    problems = []
    for page, size, m in RUN.each(("predict-result",)):
        where, a = f"{page} {size[0]}", m["results"]["advisory"]
        if a["headline"] != api["headline"]:
            problems.append(f"{where}: headline {a['headline']!r}, API {api['headline']!r}")
        if a["subheads"] != ["How crops are ranked", "About these numbers"]:
            problems.append(f"{where}: sub-headings {a['subheads']}")
        if a["facts"] != [{"label": f["label"], "value": f["value"]} for f in api["scenario_facts"]]:
            problems.append(f"{where}: facts {a['facts']} differ from the API's {api['scenario_facts']}")
        for key, field in (("basis", "ranking_basis"), ("caveat", "model_caveat"), ("absolute", "absolute_yield_note")):
            if a[key] != " ".join(api[field].split()):
                problems.append(f"{where}: {key} text differs from the API's {field}")
        columns = len(set(a["factLefts"]))
        want = 1 if size[1] < 480 else 2
        if columns != want:
            problems.append(f"{where}: facts grid has {columns} column(s), expected {want}")
    _fail(problems, "advisory card")


def ranking_rows_link_to_the_crop_page():
    """Each ranking row is a link to crop.html with the crop, the province and the nine inputs the model used; clicking one
    opens that crop's page (R9)."""
    from urllib.parse import parse_qs, urlparse
    api = _expected_sindh_driest_ranking()
    problems = []
    for page, size, m in RUN.each(("predict-result",)):
        where = f"{page} {size[0]}"
        for row in m["results"]["rankRows"]:
            crop = row["text"].split(" ")[0].lower()
            if row["tag"] != "A" or not (row["href"] or "").startswith("crop.html?"):
                problems.append(f"{where}: {crop} row is <{row['tag'].lower()}> href={row['href']!r}, expected a link to crop.html")
                continue
            q = {k: v[0] for k, v in parse_qs(urlparse(row["href"]).query).items()}
            if q.get("crop") != crop or q.get("province") != "Sindh":
                problems.append(f"{where}: {crop} link has crop={q.get('crop')} province={q.get('province')}")
            for key, used in api["inputs"].items():
                if key not in q or abs(float(q[key]) - used["value"]) > 1e-9 * max(1.0, abs(used["value"])):
                    problems.append(f"{where}: {crop} link input {key}={q.get(key)}, the model used {used['value']}")
    chrome = RUN.browser()
    chrome.emulate(1440, 900, dpr=1, mobile=False, touch=False)
    chrome.goto(f"{RUN.base()}/predict.html")
    chrome.wait_for(PAGES["predict"], 60, "the predictor")
    SETUP["predict-result"](chrome)
    first = api["ranked_crops"][0]["crop"]
    _click(chrome, "a.rank-row")
    chrome.wait_for("document.body.dataset.ready === 'true' && location.pathname.endsWith('crop.html')", 30, "the crop page after a click")
    title = chrome.eval("document.getElementById('crop-title').innerText.trim().toLowerCase()")
    if title != first:
        problems.append(f"clicking the first row opened {title!r}, expected {first!r}")
    _fail(problems, "ranking links")


CROP_JS = r"""
(() => {
  const norm = s => s.replace(/\s+/g, " ").trim();
  const tx = el => el ? norm(el.innerText) : null;
  const result = {};
  document.querySelectorAll("#crop-result dt").forEach(dt => { result[tx(dt)] = tx(dt.nextElementSibling); });
  const inputs = [...document.querySelectorAll("#crop-inputs > div")].map(d => ({ label: tx(d.querySelector("dt")), value: tx(d.querySelector("dd")) }));
  const link = id => { const a = document.getElementById(id); return a ? { href: a.getAttribute("href"), text: tx(a) } : null; };
  return {
    title: tx(document.getElementById("crop-title")), province: tx(document.getElementById("crop-province")),
    result, years: [...document.querySelectorAll("#crop-years tbody tr")].map(tr => [...tr.cells].map(tx)),
    yearHeads: [...document.querySelectorAll("#crop-years thead th")].map(tx),
    water: tx(document.getElementById("crop-water")), waterLink: (document.querySelector("#crop-water a") || {}).href || null,
    data: tx(document.getElementById("crop-data")), inputs, caveat: tx(document.querySelector(".crop-caveat")), absolute: tx(document.querySelector(".crop-absolute")),
    prev: link("crop-prev"), next: link("crop-next"), back: link("crop-back"),
    unitCount: (document.body.innerText.match(/dataset yield units/g) || []).length,
    cards: document.querySelectorAll(".crop-card").length, error: tx(document.querySelector(".crop-error")),
    ready: document.body.dataset.ready, heading: document.querySelectorAll("h1").length,
  };
})()
"""


def crop_page_shows_the_apis_numbers():
    """For each of the 13 crops the page's rank, yields, share, water need, training rows, inputs, caveats and the
    driest / typical / wettest table equal the API's answers for the same inputs; prev and next walk the ranking (R9)."""
    import re
    base = RUN.base()
    sc = _api_scenarios(base)
    inputs = sc["driest"]["inputs"]
    api = _api_post(f"{base}/api/v1/recommend", {"province": "Sindh", **inputs})[1]
    years = {kind: _api_post(f"{base}/api/v1/recommend", {"province": "Sindh", **sc[kind]["inputs"]})[1] for kind in ("driest", "median", "wettest")}
    labels = {"driest": f"Driest real year ({sc['driest']['year']})", "median": "Typical (median) year", "wettest": f"Wettest real year ({sc['wettest']['year']})"}
    order = [c["crop"] for c in api["ranked_crops"]]
    chrome = RUN.browser()
    chrome.emulate(360, 800, dpr=2, mobile=True, touch=True)
    problems = []

    def group_text(response, crop):
        group = next(g for g in response["groups"] if crop in g["crops"])
        if len(group["crops"]) == 1:
            return group["rank"], f"Rank {group['rank']}"
        return group["rank"], f"Rank {group['rank']}, tied with {len(group['crops']) - 1} other crops at {group['pct_of_normal']:.1f}%"

    for i, crop in enumerate(order):
        chrome.goto(crop_url(base, crop, "Sindh", inputs))
        chrome.wait_for("document.body.dataset.ready === 'true'", 60, f"the {crop} page")
        page = chrome.eval(CROP_JS)
        item = next(c for c in api["ranked_crops"] if c["crop"] == crop)
        where = crop
        if page["cards"] == 0 or page["error"]:
            problems.append(f"{where}: no result cards (error: {page['error']!r})")
            continue
        if page["title"].lower() != crop or page["province"] != "Sindh" or page["heading"] != 1:
            problems.append(f"{where}: title {page['title']!r}, province {page['province']!r}, {page['heading']} h1")
        expected = {"Rank": group_text(api, crop)[1], "Predicted yield": f"{item['predicted_yield']:.2f}",
                    "Typical year yield": f"{item['normal_yield']:.2f}", "Share of a typical year": f"{item['pct_of_normal']:.1f}%"}
        for label, value in expected.items():
            if page["result"].get(label) != value:
                problems.append(f"{where}: {label} shows {page['result'].get(label)!r}, API {value!r}")
        for row, kind in zip(page["years"], ("driest", "median", "wettest")):
            it = next(c for c in years[kind]["ranked_crops"] if c["crop"] == crop)
            want = [labels[kind], str(group_text(years[kind], crop)[0]), f"{it['pct_of_normal']:.1f}%", f"{it['predicted_yield']:.2f}"]
            if row != want:
                problems.append(f"{where}: {kind} row {row}, API {want}")
        if len(page["years"]) != 3:
            problems.append(f"{where}: {len(page['years'])} real-year rows, expected 3")
        fao = item["fao_water_need"]
        if fao:
            want = f"{fao['mm_min']}-{fao['mm_max']} mm per growing period, drought sensitivity {fao['drought_sensitivity']}"
            if want not in page["water"] or page["waterLink"] != fao["source_url"]:
                problems.append(f"{where}: water need {page['water']!r} / {page['waterLink']!r}, API {want!r} / {fao['source_url']!r}")
        elif "not listed in the FAO table" not in page["water"]:
            problems.append(f"{where}: water need {page['water']!r}, expected 'not listed in the FAO table'")
        rows = f"{item['training_rows_for_crop']} of {item['training_rows_possible']}"
        if rows not in page["data"] or (item["training_rows_for_crop"] < item["training_rows_possible"]) != ("incomplete data" in page["data"]):
            problems.append(f"{where}: data block {page['data']!r}, API {rows}")
        if item["barely_grown"] != ("barely grown" in page["data"].lower()):
            problems.append(f"{where}: barely grown flag {item['barely_grown']} not shown correctly in {page['data']!r}")
        if [f["label"] for f in page["inputs"]] != [k for k in api["inputs"]] and len(page["inputs"]) != 9:
            problems.append(f"{where}: {len(page['inputs'])} inputs listed, expected 9")
        if page["caveat"] != " ".join(api["model_caveat"].split()) or page["absolute"] != " ".join(api["absolute_yield_note"].split()):
            problems.append(f"{where}: caveat text differs from the API's")
        if page["unitCount"] != 1:
            problems.append(f"{where}: yield unit stated {page['unitCount']} times, expected once")
        prev_crop, next_crop = order[(i - 1) % 13], order[(i + 1) % 13]
        for key, neighbour in (("prev", prev_crop), ("next", next_crop)):
            link = page[key]
            if not link or f"crop={neighbour}&" not in link["href"] or neighbour.lower() not in link["text"].lower():
                problems.append(f"{where}: {key} link {link} should go to {neighbour}")
        if not page["back"] or page["back"]["href"] != "predict.html":
            problems.append(f"{where}: back link {page['back']}")
    _fail(problems, "crop pages")


def crop_years_table_fits_its_box_at_every_size():
    """The driest / typical / wettest table never scrolls inside its box and its scenario names do not wrap into a tall
    column; on phones it is one card per row (R3, R6)."""
    problems = []
    for page, size, m in RUN.each(("crop",)):
        t = m["cropTable"]
        if t["sw"] > t["cw"] + 1:
            problems.append(f"{page} {size[0]}: table needs {t['sw']} px, has {t['cw']} px")
        if size[1] >= 600 and t["wrap"]:
            problems.append(f"{page} {size[0]}: a scenario name wraps onto three or more lines")
    _fail(problems, "crop years table")


def crop_page_refuses_bad_addresses_without_showing_numbers():
    """An unknown crop or province, a missing crop, or inputs outside the trained range show a plain message and a link back,
    never numbers; the out-of-range message is the API's own (R9)."""
    base = RUN.base()
    sc = _api_scenarios(base)
    inputs = sc["driest"]["inputs"]
    chrome = RUN.browser()
    chrome.emulate(360, 800, dpr=2, mobile=True, touch=True)
    out_of_range = {**inputs, "precipitation_sum": 5}
    api_message = _api_post(f"{base}/api/v1/recommend", {"province": "Sindh", **out_of_range})[1]["message"]
    cases = [
        ("unknown crop", crop_url(base, "quinoa", "Sindh", inputs), "quinoa"),
        ("unknown province", crop_url(base, "rice", "Atlantis", inputs), "Atlantis"),
        ("no crop", f"{base}/crop.html?province=Sindh", "crop"),
        ("out of range", crop_url(base, "rice", "Sindh", out_of_range), api_message),
        ("not a number", crop_url(base, "rice", "Sindh", {**inputs, "precipitation_sum": "lots"}), "precipitation_sum"),
    ]
    problems = []
    for name, url, expect in cases:
        chrome.goto(url)
        chrome.wait_for("document.body.dataset.ready === 'true'", 60, f"the {name} page")
        page = chrome.eval(CROP_JS)
        if not page["error"] or expect not in page["error"]:
            problems.append(f"{name}: error text {page['error']!r} should mention {expect!r}")
        if page["cards"] != 0 or page["result"]:
            problems.append(f"{name}: {page['cards']} result cards shown next to an error")
        if not page["back"] or page["back"]["href"] != "predict.html":
            problems.append(f"{name}: back link {page['back']}")
    _fail(problems, "bad crop addresses")


# Visible words per page at 1440 px, counted by the browser (innerText). Set from the counts after the text cuts of
# SPEC-responsive.md R7, with a few percent of room; a page that grows past its cap has gained text nobody asked for.
WORD_CAPS = {"predict": 95, "predict-result": 405, "predict-single": 221, "metrics": 775, "crop": 305}  # counts after the cuts: 90, 386, 210, 737, 290
DEVELOPER_PHRASES = ("Executing machine learning pipeline", "Backend Pipeline", "Climate Intelligence Engine", "Live ML Model Pipeline",
                     "FastAPI", "Inference completed", "Core Agricultural", "Nothing on this page is typed in by hand",
                     "Optional Climate and Soil Inputs")


def _text_at_1440(page):
    size = next(s for s in SIZES if s[0] == "1440x900")
    return RUN.measure(page, size)["bodyText"]


def pages_stay_under_their_word_caps():
    """No page carries more visible text than its cap (R7)."""
    problems = []
    for page, cap in WORD_CAPS.items():
        words = len(_text_at_1440(page).split())
        if words > cap:
            problems.append(f"{page}: {words} words, cap {cap}")
    _fail(problems, "word caps")


def pages_use_plain_labels_not_developer_wording():
    """Plain labels instead of pipeline and framework names; the footer is the course name only (R7)."""
    problems = []
    for page in WORD_CAPS:
        text = _text_at_1440(page)
        problems += [f"{page}: still says {phrase!r}" for phrase in DEVELOPER_PHRASES if phrase.lower() in text.lower()]
    for page, wanted in (("predict", ["Climate inputs", "Optional inputs (blank = province median)"]),
                         ("metrics", ["Server: online", "CT-378 Machine Learning Project"])):
        text = _text_at_1440(page)
        problems += [f"{page}: label {w!r} missing" for w in wanted if w not in text]
    if "Climate Intelligence" in _text_at_1440("metrics") or not _text_at_1440("metrics").rstrip().endswith("CT-378 Machine Learning Project"):
        problems.append("metrics: the footer must be exactly 'CT-378 Machine Learning Project'")
    _fail(problems, "wording")


def required_caveats_survive_the_text_cuts():
    """The words the project requires stay: the pinned-year note next to a single predicted yield, the unit label, and on
    the metrics page the loss on log-MAE and the after-the-fact choice of the pass test (R7, CLAUDE.md)."""
    problems = []
    single = _api_post(f"{RUN.base()}/api/v1/predict", {"province": "Sindh", "crop": "rice", **_api_scenarios(RUN.base())["driest"]["inputs"]})[1]
    text = " ".join(_text_at_1440("predict-single").split())
    if " ".join(single["absolute_yield_note"].split()) not in text:
        problems.append("predict-single: the pinned-year note is missing next to the predicted yield")
    if "dataset yield units" not in text:
        problems.append("predict-single: the 'dataset yield units' label is missing")
    metrics = " ".join(_text_at_1440("metrics").split())
    for needle in ("(not better)", "log-MAE", "chosen after seeing the test results", "dataset yield units"):
        if needle not in metrics:
            problems.append(f"metrics: {needle!r} is missing")
    _fail(problems, "required caveats")


def landing_text_phase_fits():
    """When the paragraph and the Start button are shown (scrolled intro, or reduced motion), header, paragraph, button
    and footer sit in that order without overlapping, all inside the screen (R4.8)."""
    problems = []
    for page, size, m in RUN.each(("index-text", "index-reduced")):
        where, L = f"{page} {size[0]}", m["landing"]
        if L["paragraphOpacity"] < 0.9 or L["buttonOpacity"] < 0.9:
            problems.append(f"{where}: paragraph opacity {L['paragraphOpacity']:.2f}, button opacity {L['buttonOpacity']:.2f}")
            continue
        pr, btn, head, foot = L["paragraph"], L["button"], L["header"], L["footer"]
        if pr["top"] < head["bottom"] - 1:
            problems.append(f"{where}: paragraph starts at {pr['top']:.0f} px, header ends at {head['bottom']:.0f} px")
        if btn["top"] < pr["bottom"] - 1:
            problems.append(f"{where}: button starts at {btn['top']:.0f} px, paragraph ends at {pr['bottom']:.0f} px")
        if btn["bottom"] > foot["top"] + 1:
            problems.append(f"{where}: button ends at {btn['bottom']:.0f} px, footer starts at {foot['top']:.0f} px")
        for name, r in (("paragraph", pr), ("button", btn)):
            if r["left"] < -1 or r["right"] > m["vw"] + 1:
                problems.append(f"{where}: {name} spans {r['left']:.0f} to {r['right']:.0f} px")
        f1, f2, line = L["footer1"], L["footer2"], L["footerLine"]
        if line and (f1["right"] > line["left"] + 0.5 or line["right"] > f2["left"] + 0.5):
            problems.append(f"{where}: footer line {line['left']:.0f} to {line['right']:.0f} px touches text ending at {f1['right']:.0f} / starting at {f2['left']:.0f} px")
        if not line and f1["right"] > f2["left"] + 0.5 and f1["top"] < f2["bottom"] and f2["top"] < f1["bottom"]:
            problems.append(f"{where}: footer texts overlap")
    _fail(problems, "landing text phase")


def landing_scene_is_lighter_on_phones():
    """On phones the scene is drawn at no more than 1.5x pixel ratio and the scroll space is shorter; desktop keeps both
    as they were (Decision 1)."""
    problems = []
    for page, size, m in RUN.each(("index-text",)):
        L = m["landing"]
        ratio = L["canvas"]["w"] / m["vw"]
        if size[4] and ratio > 1.5 + 0.01:
            problems.append(f"{page} {size[0]}: canvas drawn at {ratio:.2f}x pixel ratio, expected 1.5 at most")
        if size[4] and L["scrollSpace"]["h"] > 12000:
            problems.append(f"{page} {size[0]}: scroll space is {L['scrollSpace']['h']:.0f} px, expected 12000 at most on phones")
        if not size[4] and size[1] > 800 and L["scrollSpace"]["h"] != 30000:
            problems.append(f"{page} {size[0]}: scroll space is {L['scrollSpace']['h']:.0f} px, desktop keeps 30000")
    _fail(problems, "phone scene")


def landing_canvas_survives_address_bar():
    """Hiding or showing the Android address bar changes the height by about 100 px: the canvas keeps its drawing size
    and still fills the screen without distortion (object-fit: cover) (R4.8)."""
    problems = []
    chrome = RUN.browser()
    for name, w, h, dpr, phone in SIZES:
        if not phone or w > 500:
            continue
        chrome.emulate(w, h, dpr=dpr, mobile=True, touch=True)
        chrome.goto(f"{RUN.base()}/index.html")
        chrome.wait_for(PAGES["index-text"], 60, f"the landing scene at {name}")
        time.sleep(0.5)
        read = ("(() => { const c = document.getElementById('mesh-canvas'), r = c.getBoundingClientRect();"
                " return {w: c.width, h: c.height, rw: r.width, rh: r.height, ih: innerHeight, fit: getComputedStyle(c).objectFit}; })()")
        before = chrome.eval(read)
        for delta in (-110, 110):   # address bar shown, then hidden
            chrome.emulate(w, h + delta, dpr=dpr, mobile=True, touch=True)
            time.sleep(0.6)
            after = chrome.eval(read)
            if (after["w"], after["h"]) != (before["w"], before["h"]):
                problems.append(f"{name} {delta:+d} px: drawing size changed {before['w']}x{before['h']} to {after['w']}x{after['h']}")
            if abs(after["rh"] - after["ih"]) > 1 or abs(after["rw"] - w) > 1:
                problems.append(f"{name} {delta:+d} px: canvas is {after['rw']:.0f}x{after['rh']:.0f}, screen is {w}x{after['ih']}")
            if after["fit"] != "cover":
                problems.append(f"{name} {delta:+d} px: canvas object-fit is {after['fit']!r}, expected 'cover'")
        chrome.emulate(w, h, dpr=dpr, mobile=True, touch=True)
    _fail(problems, "address bar resize")


def landing_reduced_motion_shows_content_at_once():
    """With prefers-reduced-motion the page does not scroll, and the paragraph and button are shown without animation (R4.9)."""
    problems = []
    for page, size, m in RUN.each(("index-reduced",)):
        L = m["landing"]
        if L["scrollHeight"] > m["vh"] + 1:
            problems.append(f"{page} {size[0]}: page scrolls ({L['scrollHeight']} px in a {m['vh']} px screen)")
    _fail(problems, "reduced motion")


def landing_has_one_start_button():
    """index.html shows exactly one button, "Start", linking to predict.html relatively (R1)."""
    problems = []
    for page, size, m in RUN.each(("index",)):
        where = f"index {size[0]}"
        buttons = m["startButtons"]
        if len(buttons) != 1:
            problems.append(f"{where}: expected 1 button, found {len(buttons)} {[b['text'] for b in buttons]}")
            continue
        b = buttons[0]
        if b["text"].lower() != "start":
            problems.append(f"{where}: button text is {b['text']!r}, expected 'Start'")
        if b["href"] != "predict.html":
            problems.append(f"{where}: button href is {b['href']!r}, expected relative 'predict.html'")
        if b["left"] < -1 or b["right"] > m["vw"] + 1:
            problems.append(f"{where}: button spans {b['left']:.0f} to {b['right']:.0f} px")
        if size[4] and (b["w"] < MIN_TARGET_PX or b["h"] < MIN_TARGET_PX):
            problems.append(f"{where}: button is {b['w']:.0f}x{b['h']:.0f} px")
    status = urllib.request.urlopen(f"{RUN.base()}/predict.html", timeout=10).status
    if status != 200:
        problems.append(f"/predict.html answered {status}")
    _fail(problems, "landing Start button")


# --------------------------------------------------------------------------- Streamlit app (R4.11)
STREAMLIT_PYTHON = ROOT / ".venv-streamlit" / "bin" / "python"
STREAMLIT_SIZES = [s for s in SIZES if s[0] in ("320x568", "360x800", "412x915")]


def streamlit_page_has_no_sideways_scroll():
    """The Streamlit app at phone sizes: neither the document nor Streamlit's own scrolling containers scroll sideways
    (tables and charts scroll inside their own frames) (R4.11). Needs .venv-streamlit (run_checks.sh --all creates it)."""
    import subprocess
    assert STREAMLIT_PYTHON.exists(), f"{STREAMLIT_PYTHON} not found: run ./run_checks.sh --all once to create it"
    port = free_port()
    server = subprocess.Popen(
        [str(STREAMLIT_PYTHON), "-m", "streamlit", "run", str(ROOT / "streamlit_app" / "app.py"), "--server.headless", "true",
         "--server.port", str(port), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    problems = []
    chrome = RUN.browser()
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1)
                break
            except Exception:  # noqa: BLE001 - still starting
                time.sleep(0.5)
        else:
            raise AssertionError("Streamlit did not start")
        for name, w, h, dpr, phone in STREAMLIT_SIZES:
            chrome.emulate(w, h, dpr=dpr, mobile=True, touch=True)
            chrome.goto(f"http://127.0.0.1:{port}/")
            chrome.wait_for("document.body.innerText.includes('How crops are ranked')", 120, f"the Streamlit app at {name}")
            time.sleep(1.5)  # the chart and tables draw after the text
            m = chrome.eval("""(() => {
                const els = [["document", document.documentElement], ["stMain", document.querySelector('[data-testid="stMain"]')],
                             ["stApp", document.querySelector('[data-testid="stApp"]')]];
                return els.filter(([, el]) => el).map(([n, el]) => ({ n, sw: el.scrollWidth, cw: el.clientWidth }));
            })()""")
            problems += [f"{name}: {e['n']} is {e['sw']} px wide in {e['cw']} px" for e in m if e["sw"] > e["cw"] + 1]
            RUN.shots.mkdir(parents=True, exist_ok=True)
            chrome.screenshot(RUN.shots / f"streamlit_{name}.png")
    finally:
        server.terminate()
        server.wait(timeout=15)
    _fail(problems, "Streamlit sideways scroll")


TESTS = [
    no_horizontal_page_scroll,
    boxes_stay_inside_viewport,
    text_stays_inside_viewport,
    no_clipped_text,
    tap_targets_44px_on_phones,
    form_text_16px_on_phones,
    text_contrast_meets_wcag_aa,
    tabs_fully_visible,
    tables_become_cards_on_phones,
    wide_tables_fit_or_scroll_inside_wrapper,
    table_text_is_readable,
    table_values_identical_at_every_size,
    chosen_model_row_is_highlighted,
    jargon_headers_are_explained,
    predictor_form_layout,
    prediction_results_are_drawn_at_every_size,
    result_rows_show_only_crop_rank_share_and_yield,
    pages_stay_under_their_word_caps,
    pages_use_plain_labels_not_developer_wording,
    required_caveats_survive_the_text_cuts,
    ranking_rows_link_to_the_crop_page,
    crop_page_shows_the_apis_numbers,
    crop_years_table_fits_its_box_at_every_size,
    crop_page_refuses_bad_addresses_without_showing_numbers,
    advisory_card_is_structured,
    landing_text_phase_fits,
    landing_scene_is_lighter_on_phones,
    landing_canvas_survives_address_bar,
    landing_reduced_motion_shows_content_at_once,
    landing_has_one_start_button,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names")
    ap.add_argument("--size", help="comma-separated sizes, e.g. 360x800,1920x1080")
    ap.add_argument("--page", help="comma-separated pages: " + ",".join(PAGES))
    ap.add_argument("--shots", help="folder for screenshots (default build/responsive/)")
    ap.add_argument("--streamlit", action="store_true", help="check the Streamlit app instead of the HTML pages (needs .venv-streamlit)")
    args = ap.parse_args()
    if args.size:
        wanted = args.size.split(",")
        RUN.sizes = [s for s in SIZES if s[0] in wanted]
        assert RUN.sizes, f"unknown size {args.size}; choose from {[s[0] for s in SIZES]}"
    if args.page:
        RUN.pages = [p for p in args.page.split(",") if p in PAGES]
        assert RUN.pages, f"unknown page {args.page}; choose from {list(PAGES)}"
    if args.shots:
        RUN.shots = Path(args.shots)
    tests = [streamlit_page_has_no_sideways_scroll] if args.streamlit else TESTS
    if args.only:
        names = args.only.split(",")
        tests = [t for t in TESTS if t.__name__ in names]
        assert tests, f"unknown test {args.only}; choose from {[t.__name__ for t in TESTS]}"

    failed = 0
    try:
        for test in tests:
            try:
                test()
                print(f"PASS  {test.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {test.__name__}\n  {e}")
    finally:
        RUN.close()
    print(f"\n{len(tests) - failed}/{len(tests)} passed; screenshots in {RUN.shots}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
