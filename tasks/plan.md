# Implementation Plan: Offline-First Frontend & CDN Dependency Vendoring

## Overview
`index.html` currently crashes and renders blank when internet connectivity is lost because it relies on external CDN links (`cdn.jsdelivr.net` and `unpkg.com`) for `three.module.js`, `GLTFLoader.js`, `BufferGeometryUtils.js`, and `es-module-shims.js`. While `predict.html` is already completely self-contained, `index.html` needs these 3D/module dependencies stored locally in `./vendor/` so the entire frontend runs 100% offline.

## Architecture Decisions
- **Local Vendor Directory Structure**: Store Three.js and its JSM addons directly in `frontend/climate-resilient-frontend/vendor/`:
  - `vendor/es-module-shims.js` (polyfill for ES module import maps)
  - `vendor/three/three.module.js` (Three.js v0.164.0 core)
  - `vendor/three/addons/loaders/GLTFLoader.js` (GLTF/GLB 3D model loader)
  - `vendor/three/addons/utils/BufferGeometryUtils.js` (Required helper imported by GLTFLoader)
- **Local Import Map**: Update `<script type="importmap">` in `index.html` to map `"three"` to `"./vendor/three/three.module.js"` and `"three/addons/"` to `"./vendor/three/addons/"`.
- **Zero External Network Dependencies**: Eliminate all `https://cdn.jsdelivr.net` and `https://unpkg.com` requests from `index.html`.

## Task List

### Phase 1: Dependency Vendoring
- [ ] Task 1: Download and organize Three.js core, GLTFLoader, BufferGeometryUtils, and es-module-shims into `vendor/`
- [ ] Task 2: Update `index.html` import map and script tags to relative local paths

### Checkpoint: Offline Readiness
- [ ] No external CDN links remain in `index.html`
- [ ] Local static server serves all module scripts with 200 OK
- [ ] Three.js WebGL canvas and 3D logo render with network disconnected

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| GLTFLoader internal imports fail | High | Verify `BufferGeometryUtils.js` is positioned at exact relative path `../utils/BufferGeometryUtils.js` matching GLTFLoader's import specifier. |
| Browser import map caching issue | Low | Use standard relative path `./vendor/` compatible with native browser import maps. |
