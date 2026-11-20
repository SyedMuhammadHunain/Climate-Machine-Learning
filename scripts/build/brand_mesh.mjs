// Triangulates glyph outlines with three.js (ShapeGeometry uses earcut, so counters in o, a, r are holes).
// Used only by scripts/build/build_brand_assets.py to build the 3D name mesh (frontend assets/logo-3d.glb).
//   stdin : JSON {"glyphs": [[[[x, y], ...contour...], ...contours of one glyph...], ...]}   (font units, y up)
//   stdout: JSON {"positions": [x, y, x, y, ...], "indices": [...]}
import fs from "node:fs";
import * as THREE from "../frontend/climate-resilient-frontend/vendor/three/three.module.min.js";

const { glyphs } = JSON.parse(fs.readFileSync(0, "utf8"));
const positions = [];
const indices = [];
for (const contours of glyphs) {
    if (!contours.length) continue;
    const path = new THREE.ShapePath();
    for (const contour of contours) {
        path.moveTo(contour[0][0], contour[0][1]);
        for (const [x, y] of contour.slice(1)) path.lineTo(x, y);
        path.lineTo(contour[0][0], contour[0][1]);
    }
    // TrueType: solid contours run clockwise, counters counter-clockwise (three's default)
    const geometry = new THREE.ShapeGeometry(path.toShapes(false));
    const base = positions.length / 2;
    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) positions.push(pos.getX(i), pos.getY(i));
    for (const i of geometry.index.array) indices.push(i + base);
}
process.stdout.write(JSON.stringify({ positions, indices }));
