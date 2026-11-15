# Tasks: Dedicated Model Metrics & Agroclimatic Benchmarks Page (`metrics.html`)

## Phase 1: Benchmarks Page & Navigation

### Task 1: Create `metrics.html` Page
**Description:** Implement `frontend/climate-resilient-frontend/metrics.html` containing model evaluation statistics ($R^2$, RMSE, MAE), provincial climate baseline tables (`Punjab`, `Sindh`, `Balochistan`), crop physiological classifications ($K_w$, $T_{\text{crit}}$), and agroclimatic elasticity formulas with local styling and API metadata sync.

**Acceptance criteria:**
- [x] `metrics.html` created in `frontend/climate-resilient-frontend/`.
- [x] Model metrics section displays $R^2 = 0.892$, $\text{MAE} = 4.82\text{ kg/ha}$, $\text{RMSE} = 7.65\text{ kg/ha}$, and 17 feature descriptions.
- [x] Provincial baselines table displays historical precipitation, ET0, solar radiation, and temperature defaults.
- [x] Crop physiology table shows low-water resilient crops ($K_w$, $T_{\text{crit}}$) and water-intensive risk flags.
- [x] Zero external CDN scripts (100% offline self-contained).

**Verification:**
- [x] `curl -s http://localhost:3000/metrics.html` returns 200 OK.
- [x] `grep -E "jsdelivr|unpkg" frontend/climate-resilient-frontend/metrics.html` returns 0 results.

**Dependencies:** None
**Files likely touched:**
- `frontend/climate-resilient-frontend/metrics.html`
**Estimated scope:** S (1 new HTML file)

---

### Task 2: Implement Cross-Page Header Navigation
**Description:** Add a unified sub-header navigation bar (`Overview`, `Predictor & Advisory`, `Model & Benchmarks`) linking across `index.html`, `predict.html`, `careers.html`, and `metrics.html`.

**Acceptance criteria:**
- [x] Navigation bar present in `predict.html` and `careers.html` linking to `metrics.html`.
- [x] Navigation bar in `metrics.html` highlights "Model & Benchmarks" as active.
- [x] `index.html` overlay includes link to `metrics.html`.

**Verification:**
- [x] Verify hyperlinks resolve between all pages.

**Dependencies:** Task 1
**Files likely touched:**
- `frontend/climate-resilient-frontend/predict.html`
- `frontend/climate-resilient-frontend/careers.html`
- `frontend/climate-resilient-frontend/index.html`
**Estimated scope:** S (3 files)

---

## Checkpoint: Verification
- [x] All pages load with 200 OK locally.
- [x] Navigation works bidirectionally.
- [x] Backend verification suite passes 10/10 tests.
- [x] Verified and ready for deployment.
