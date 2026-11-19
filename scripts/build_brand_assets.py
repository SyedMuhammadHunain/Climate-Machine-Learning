"""Build the site's name assets from one string, so renaming the site is a one-line change.

Run:  .venv/bin/python scripts/build_brand_assets.py
Writes (into frontend/climate-resilient-frontend/):
  assets/logo-3d.glb        the big zoom-in text on index.html: a flat triangle mesh of the name
  assets/wordmark.svg       the same outlines as an SVG
  predict.html              the inline logo SVG (between the brand-wordmark markers) is replaced
  assets/site_favicon.png, site_webclip.png, site_opengraph.png    icons and share image

Needs: fontTools and Pillow (already in the main venv), node (triangulation with the vendored three.js, see
brand_mesh.mjs) and google-chrome (renders the PNGs). The outlines come from Liberation Sans Bold, SIL Open
Font License 1.1 (/usr/share/fonts/liberation), a static font with no overlapping contours (overlaps break the
triangulation: a variable font's "W" came out as a solid block). Nothing here is downloaded.
"""
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from fontTools.pens.basePen import decomposeQuadraticSegment
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "climate-resilient-frontend"
ASSETS = FRONTEND / "assets"

SITE_NAME = "Low-Water Crops"
SITE_URL = "https://climate-machine-learning.vercel.app/"
FONT = Path("/usr/share/fonts/liberation/LiberationSans-Bold.ttf")   # static font, SIL OFL 1.1
ORANGE, INK = "#d96c2c", "#292a2a"                                 # the page's existing colours
MESH_WIDTH = 0.4751   # width of the mesh the page was built around (x range about +-0.24)
CURVE_STEPS = 10


def load_font(tmp: Path):
    font = TTFont(FONT)
    path = tmp / "brand.ttf"
    font.save(path)
    return font, path


def layout(font):
    """[(glyph name, x offset)] along one baseline, plus the total advance. No kerning."""
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    items, x = [], 0
    for ch in SITE_NAME:
        name = cmap[ord(ch)]
        items.append((name, x))
        x += hmtx[name][0]
    return items, x


def svg_wordmark(font, items):
    glyphs = font.getGlyphSet()
    scale = 0.2
    flip = (scale, 0, 0, -scale)   # font y goes up, SVG y goes down
    pen, bounds = SVGPathPen(glyphs, ntos=lambda v: f"{v:.1f}"), ControlBoundsPen(glyphs)
    for name, x in items:
        t = flip + (x * scale, 0)
        glyphs[name].draw(TransformPen(pen, t))
        glyphs[name].draw(TransformPen(bounds, t))
    xmin, ymin, xmax, ymax = bounds.bounds
    pad = 8
    view = f"{xmin - pad:.1f} {ymin - pad:.1f} {xmax - xmin + 2 * pad:.1f} {ymax - ymin + 2 * pad:.1f}"
    return view, pen.getCommands()


def flatten(font, items):
    """Per glyph: its contours as polygons (curves cut into short segments), in font units with the x offset applied."""
    glyphs, out = font.getGlyphSet(), []
    for name, x0 in items:
        rec = DecomposingRecordingPen(glyphs)
        glyphs[name].draw(rec)
        contours, current = [], []
        for op, pts in rec.value:
            if op == "moveTo":
                current = [pts[0]]
            elif op == "lineTo":
                current.append(pts[0])
            elif op == "qCurveTo":
                start = current[-1]
                for off, on in decomposeQuadraticSegment(pts):
                    for i in range(1, CURVE_STEPS + 1):
                        t = i / CURVE_STEPS
                        current.append(((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * off[0] + t * t * on[0],
                                        (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * off[1] + t * t * on[1]))
                    start = on
            elif op == "curveTo":
                start = current[-1]
                for i in range(1, CURVE_STEPS + 1):
                    t = i / CURVE_STEPS
                    a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
                    current.append((a * start[0] + b * pts[0][0] + c * pts[1][0] + d * pts[2][0],
                                    a * start[1] + b * pts[0][1] + c * pts[1][1] + d * pts[2][1]))
            elif op in ("closePath", "endPath"):
                if len(current) > 2:
                    contours.append([[x + x0, y] for x, y in current])
                current = []
        out.append(contours)
    return out


def triangulate(glyph_contours):
    done = subprocess.run(["node", str(ROOT / "scripts" / "brand_mesh.mjs")], input=json.dumps({"glyphs": glyph_contours}),
                          capture_output=True, text=True, check=True)
    mesh = json.loads(done.stdout)
    return mesh["positions"], mesh["indices"]


def write_glb(path: Path, positions, indices):
    """Scale to the width the page expects, centre it, lay it flat in XZ (text up = -z, as the old mesh), write a GLB."""
    xs, ys = positions[0::2], positions[1::2]
    s = MESH_WIDTH / (max(xs) - min(xs))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    verts = [((x - cx) * s, 0.0, -(y - cy) * s) for x, y in zip(xs, ys)]
    tris = [tuple(indices[i:i + 3]) for i in range(0, len(indices), 3)]

    def normal_y(a, b, c):
        ux, uz, vx, vz = b[0] - a[0], b[2] - a[2], c[0] - a[0], c[2] - a[2]
        return uz * vx - ux * vz   # y component of (b-a) x (c-a)

    # keep the winding the old mesh had, so the page's single-sided shader draws the front face
    old = path.read_bytes() if path.exists() else None
    want = 1
    if old:
        jlen = struct.unpack("<I", old[12:16])[0]
        gltf = json.loads(old[20:20 + jlen])
        blen = struct.unpack("<I", old[20 + jlen:24 + jlen])[0]
        binary = old[28 + jlen:28 + jlen + blen]
        v, i = gltf["bufferViews"][0], gltf["bufferViews"][1]
        pos = struct.unpack(f"<{v['byteLength'] // 4}f", binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]])
        ctype = gltf["accessors"][1]["componentType"]
        fmt, size = {5123: ("H", 2), 5125: ("I", 4)}[ctype]
        idx = struct.unpack(f"<{i['byteLength'] // size}{fmt}", binary[i.get("byteOffset", 0):i.get("byteOffset", 0) + i["byteLength"]])
        p = [(pos[k], pos[k + 1], pos[k + 2]) for k in range(0, len(pos), 3)]
        want = 1 if normal_y(p[idx[0]], p[idx[1]], p[idx[2]]) > 0 else -1
    if (1 if normal_y(*(verts[k] for k in tris[0])) > 0 else -1) != want:
        tris = [(a, c, b) for a, b, c in tris]
    flat = [k for t in tris for k in t]

    pos_bytes = struct.pack(f"<{len(verts) * 3}f", *(c for v in verts for c in v))
    idx_bytes = struct.pack(f"<{len(flat)}I", *flat)
    binary = pos_bytes + idx_bytes
    mins = [min(v[k] for v in verts) for k in range(3)]
    maxs = [max(v[k] for v in verts) for k in range(3)]
    meta = {
        "asset": {"version": "2.0", "generator": "scripts/build_brand_assets.py"},
        "scene": 0, "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "SiteName"}],
        "meshes": [{"name": "SiteName", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3", "min": mins, "max": maxs},
            {"bufferView": 1, "componentType": 5125, "count": len(flat), "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
                        {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(idx_bytes), "target": 34963}],
        "buffers": [{"byteLength": len(binary)}],
    }
    js = json.dumps(meta, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(binary))
                     + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    return len(verts), len(flat) // 3, [max(xs) - min(xs), max(ys) - min(ys)]


def chrome_png(html: str, size, out: Path, tmp: Path):
    page = tmp / "page.html"
    page.write_text(html, encoding="utf-8")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--force-device-scale-factor=1", f"--user-data-dir={tmp / 'profile'}", f"--window-size={size[0]},{size[1]}",
                    f"--screenshot={out}", page.as_uri()], check=True, capture_output=True)
    img = Image.open(out)
    if img.size != tuple(size):
        img.crop((0, 0) + tuple(size)).save(out)


def build_pngs(font_path: Path, tmp: Path):
    face = f"@font-face{{font-family:Brand;src:url('{font_path.as_uri()}')}}"
    base = f"{face}html,body{{margin:0;background:{ORANGE};color:{INK};font-family:Brand}}"
    first, _, rest = SITE_NAME.partition(" ")
    chrome_png(f"<style>{base}.t{{position:absolute;left:56px;top:48px;font-size:190px;line-height:1.02;letter-spacing:-3px}}"
               f".u{{position:absolute;left:56px;bottom:44px;font-size:34px}}</style>"
               f"<div class=t>{first}<br>{rest}</div><div class=u>{SITE_URL.removeprefix('https://').rstrip('/')}</div>",
               (1200, 630), ASSETS / "site_opengraph.png", tmp)
    chrome_png(f"<style>{base}.t{{position:absolute;left:20px;top:64px;font-size:42px;line-height:1.05;letter-spacing:-0.5px;white-space:nowrap}}</style>"
               f"<div class=t>{first}<br>{rest}</div>", (256, 256), ASSETS / "site_webclip.png", tmp)
    big = tmp / "icon.png"
    chrome_png(f"<style>{base}.t{{position:absolute;left:0;right:0;top:34px;text-align:center;font-size:190px}}</style>"
               f"<div class=t>LW</div>", (256, 256), big, tmp)
    Image.open(big).resize((32, 32), Image.LANCZOS).save(ASSETS / "site_favicon.png")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        font, font_path = load_font(tmp)
        items, _ = layout(font)

        view, d = svg_wordmark(font, items)
        (ASSETS / "wordmark.svg").write_text(
            f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img">'
            f"<title>{SITE_NAME}</title><path d=\"{d}\"/></svg>\n", encoding="utf-8")
        page = FRONTEND / "predict.html"
        html = page.read_text(encoding="utf-8")
        logo = (f'<svg id="site-wordmark" xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" '
                f'aria-label="{SITE_NAME}">\n                <title>{SITE_NAME}</title>\n                <path d="{d}"/>\n            </svg>')
        html, n = re.subn(r"<svg id=\"(?:Layer_1|site-wordmark)\".*?</svg>", lambda _: logo, html, count=1, flags=re.S)
        if n != 1:
            print("predict.html: logo SVG not found")
            return 1
        page.write_text(html, encoding="utf-8")

        verts, tris, size = write_glb(ASSETS / "logo-3d.glb", *triangulate(flatten(font, items)))
        print(f"logo-3d.glb: {verts} vertices, {tris} triangles, outline size {size[0]:.0f} x {size[1]:.0f} font units")
        build_pngs(font_path, tmp)
    print(f"built the assets for {SITE_NAME!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
