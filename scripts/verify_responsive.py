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
import sys
import time
import urllib.request
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
# The landing page draws text over a WebGL scene, so colour contrast is not measurable from CSS there.
CONTRAST_PAGES = ("predict", "metrics")
TAB_PAGES = ("predict", "metrics")

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

  return { vw, vh, innerWidth: window.innerWidth, scrollWidth: document.documentElement.scrollWidth, boxes, offscreen, clipped, lowContrast, targets, formFonts, tabs, startButtons };
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

    def browser(self) -> Chrome:
        if self.chrome is None:
            self.chrome = Chrome(extra_args=CHROME_ARGS)
        return self.chrome

    def measure(self, page: str, size: tuple) -> dict:
        key = (page, size[0])
        if key not in self.cache:
            name, w, h, dpr, phone = size
            chrome = self.browser()
            chrome.emulate(w, h, dpr=dpr, mobile=phone, touch=phone)
            chrome.goto(f"{self.base()}/{page}.html")
            chrome.wait_for(PAGES[page], 60, f"{page}.html to finish loading at {name}")
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
    landing_has_one_start_button,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names")
    ap.add_argument("--size", help="comma-separated sizes, e.g. 360x800,1920x1080")
    ap.add_argument("--page", help="comma-separated pages: " + ",".join(PAGES))
    ap.add_argument("--shots", help="folder for screenshots (default build/responsive/)")
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
    tests = TESTS
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
