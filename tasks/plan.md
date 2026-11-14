# Implementation Plan: Clean Merge of `Hunain` and `afzal` Branches into `main`

## Overview
Safely merge the work of both Syed Muhammad Hunain (ML pipeline, real-world agroclimatology dataset, and Jupyter notebook) and Muhammad Afzal (FastAPI backend, Pydantic schemas, REST endpoints, and backend verification test suite) into the `main` branch with **zero merge conflicts**. Verify the unified repository end-to-end and publish the updated `main` branch to GitHub.

## Branch Lineage & Architecture Analysis
Git history reveals a clean linear ancestry:
- **`origin/main`** (`715ecdf`): Initial repo commit.
- **`origin/Hunain`** (`1c9af73`): 16 commits ahead of `main`. Contains the real 10,296-row dataset, 5 EDA plots, leakage-free preprocessing pipeline, 5-fold CV tuning ($R^2 = 0.8243$), saved joblib model (`models/crop_yield_pipeline.joblib`), and fully executed notebook.
- **`origin/afzal`** (`799181f`): Branched directly off the tip of `Hunain` (`1c9af73`). Adds the FastAPI application (`app/`), Pydantic schemas, configuration, model service loader, and `scripts/verify_backend.py`.

Because `afzal` is a direct descendant of `Hunain`, merging `Hunain` into `main` and then `afzal` into `main` is mathematically conflict-free.

## Merge Strategy
1. **Branch Checkout & Sync:** Checkout local `main` and ensure clean tracking of `origin/main`.
2. **Sequential Merge (No Conflicts):**
   - Step A: Merge `Hunain` into `main`.
   - Step B: Merge `afzal` into `main`.
3. **Dependency & Environment Alignment:** Run `uv sync` to ensure FastAPI and testclient dependencies from `pyproject.toml` are installed.
4. **End-to-End Verification:**
   - Run `scripts/verify_backend.py` to verify FastAPI endpoints and model inference.
   - Verify `models/crop_yield_pipeline.joblib` and `notebooks/01_low_water_crop_selection.ipynb`.
5. **Publish to Remote:** Push merged `main` to `origin/main`.

## Task List

### Phase 1: Git Merging & History Preservation
- [ ] Task 1: Checkout `main` and sequentially merge `Hunain` and `afzal` with zero conflicts

### Checkpoint: Clean Merge
- [ ] Git log shows `main` incorporates all commits from both `Hunain` and `afzal` without conflicts.

### Phase 2: System Verification & Testing
- [ ] Task 2: Synchronize dependencies and execute backend verification suite (`scripts/verify_backend.py`)

### Checkpoint: All Tests Green
- [ ] FastAPI backend tests pass 100%, model loads cleanly in memory, and notebook integrity is confirmed.

### Phase 3: Remote Synchronization
- [ ] Task 3: Push merged and verified `main` branch to `origin/main`

### Checkpoint: Complete Deliverable
- [ ] `origin/main` is up to date, conflict-free, and contains the complete project lifecycle (Data + ML Notebook + FastAPI API).

## Risks and Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Potential merge conflicts | High | None exist because `afzal` was branched directly from `Hunain`. Verified via `git merge-base`. |
| Missing FastAPI dependencies in venv | Medium | Run `uv sync` or install updated requirements from `pyproject.toml`. |
| Model path mismatch in backend | Medium | `scripts/verify_backend.py` asserts model loading from `models/crop_yield_pipeline.joblib`. |
