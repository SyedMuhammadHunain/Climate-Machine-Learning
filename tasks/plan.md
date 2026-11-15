# Implementation Plan: CT-378 Local Web App Finalization & Submission Readiness

## Overview
Audit, patch, and package the FastAPI + HTML Local Web App to ensure 100% compliance with the CT-378 Machine Learning Semester Project guidelines. This includes unifying localhost frontend-backend serving under FastAPI, fixing error message rendering on client-side and API validation, generating a pinned `requirements.txt`, producing a complete `README.md` with setup steps, and assembling a 10-minute demo and viva preparation guide.

## Architecture Decisions
- **Unified Single-Server Local Deployment**: Mount `frontend/climate-resilient-frontend` as static directory in FastAPI `app/main.py` with fallback route serving `index.html`. This allows examiners to run the complete system with one command (`uvicorn app.main:app`) on `http://127.0.0.1:8000`, while preserving standalone static serving (`python -m http.server 3000`).
- **Standardized Error Handling Contract**: Align client-side error handling in `predict.html` to parse both structured validation errors (`details` array from Pydantic `RequestValidationError`) and custom domain errors (`message` from `AgronomicValidationError`), displaying readable inline feedback instead of generic fallback text.
- **Strict Version Pinning**: Pin all exact dependencies in `requirements.txt` from the project's tested `.venv` environment to satisfy the scikit-learn/joblib reproduction requirement.
- **Academic & Demo Documentation**: Provide a complete `README.md` meeting all deliverable criteria (setup steps, dataset sources, model architecture, evaluation metrics, and API documentation) plus an actionable 10-minute demo and viva guide.

## Task List

### Phase 1: Core App Alignment & Error Handling
- [x] Task 1: Unify FastAPI static serving and fix frontend error message parsing (`predict.html`, `careers.html`, `app/main.py`)
- [x] Task 2: Generate pinned `requirements.txt` and verify clean environment installation

### Checkpoint: Local Serving & Validation
- [x] All 12 backend automated tests pass.
- [x] Single server on `http://127.0.0.1:8000` serves UI and API endpoints without CORS or path errors.
- [x] Invalid inputs (e.g. $T_{\text{max}} < T_{\text{min}}$ or negative precipitation) render clear, specific error text in the UI alert banner.

### Phase 2: Submission Deliverables & Viva Prep
- [ ] Task 3: Author comprehensive `README.md` with setup instructions, model metrics, and dataset citations
- [ ] Task 4: Create 10-minute live demonstration script and individual viva preparation guide (`docs/demo_and_viva_guide.md`)

### Checkpoint: Complete
- [ ] Repository has `data/`, `notebooks/`, `app/`, `models/`, `requirements.txt`, and complete `README.md`.
- [ ] Demo script cleanly maps to the 3 min problem, 4 min live demo, and 3 min results breakdown.
- [ ] Ready for final review and Week 15 presentation.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Static files mount alters relative asset paths in frontend | Medium | Mount at root or `/` using FastAPI `StaticFiles(directory=..., html=True)` while preserving `/api/v1` and `/predict` routes. |
| Examiner tests unexpected input combinations | High | Ensure both client-side JavaScript guardrails and backend Pydantic/Agronomic exception handlers catch invalid data and return user-friendly error banners. |
| Dependency conflicts during viva evaluation on clean machine | High | Generate pinned `requirements.txt` directly from working `.venv` with Python 3.13 / compatible wheels. |

## Open Questions
- None. Group confirmed choice of Local Web App deployment option per course guidelines.
