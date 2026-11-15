# Tasks: Offline-First Frontend & CDN Dependency Vendoring

## Phase 1: Dependency Vendoring

### Task 1: Download Three.js, GLTFLoader, BufferGeometryUtils, and es-module-shims
**Description:** Download and vendor all external JavaScript libraries into `frontend/climate-resilient-frontend/vendor/` preserving required internal relative paths so ES module imports resolve offline.

**Acceptance criteria:**
- [x] `frontend/climate-resilient-frontend/vendor/es-module-shims.js` saved locally.
- [x] `frontend/climate-resilient-frontend/vendor/three/three.module.js` saved locally.
- [x] `frontend/climate-resilient-frontend/vendor/three/addons/loaders/GLTFLoader.js` saved locally.
- [x] `frontend/climate-resilient-frontend/vendor/three/addons/utils/BufferGeometryUtils.js` saved locally.

**Verification:**
- [x] Verify each file exists on disk and is non-empty.

**Dependencies:** None
**Files likely touched:**
- `frontend/climate-resilient-frontend/vendor/`
**Estimated scope:** S (4 vendor files)

---

### Task 2: Update `index.html` Import Map and Script Tags
**Description:** Update `index.html` to load local vendor files instead of remote CDNs (`unpkg.com` and `cdn.jsdelivr.net`).

**Acceptance criteria:**
- [x] `<script async src="./vendor/es-module-shims.js"></script>` points to local file.
- [x] Import map maps `"three"` to `"./vendor/three/three.module.js"`.
- [x] Import map maps `"three/addons/"` to `"./vendor/three/addons/"`.
- [x] Zero `cdn.jsdelivr.net` or `unpkg.com` references remain in `index.html`.

**Verification:**
- [x] `grep -E "jsdelivr|unpkg" frontend/climate-resilient-frontend/index.html` returns 0 results.
- [x] `curl -s http://localhost:3000/index.html` verifies local paths.
- [x] Curl each vendored script path on `http://localhost:3000/vendor/...` returns 200 OK.

**Dependencies:** Task 1
**Files likely touched:**
- `frontend/climate-resilient-frontend/index.html`
**Estimated scope:** XS (1 file)

---

## Checkpoint: Offline Readiness
- [x] `index.html` and `predict.html` have zero external CDN network dependencies.
- [x] All static assets serve 200 OK from local server.
- [x] WebGL canvas renders without internet access.
