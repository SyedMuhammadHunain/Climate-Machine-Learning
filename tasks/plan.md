# Implementation Plan: Website Rebranding to Agcl & Project Slogan

## Overview
Rebrand the website from "Infinite Field" to **"Agcl"** and update generic template slogans ("There is no limit for those who think infinite") with project-specific messaging centered on climate-resilient agriculture, low-water crop selection, and drought adaptation.

## Proposed Copy & Branding
- **Brand Name**: **Agcl** (Agricultural Climate Intelligence)
- **Primary Slogan (2-Line Footer Animation on `index.html`)**:
  - Line 1: `Guarding harvests against drought`
  - Line 2: `with climate intelligence`
- **Hero Title**: `Agcl — Low-Water Crop Selection Under Drought`
- **Introductory Mission (`index.html`)**:
  `"At Agcl, we use machine learning and climate data to protect agriculture against drought. Our models predict crop yields and recommend low-water, climate-resilient crops to secure sustainable harvests across Pakistan."`
- **Predictor Wordmark (`predict.html` / `careers.html`)**:
  Clean modern wordmark badge: **Agcl** • *Climate AI*

## Architecture Decisions
- **SplitText Compatibility**: Maintain the exact DOM structure (`.dark-footer`, `.white-footer`, `.white-footer-typer`) so GSAP and SplitText animation runs smoothly on the new slogan without script breakage.
- **SVG / Wordmark Replacement**: Replace the static "Infinite Field" SVG path in `predict.html` with an accessible, crisp typographical wordmark for "Agcl" linking back to `index.html`.

## Task List

### Phase 1: Rebranding & Copy Updates
- [ ] Task 1: Update Landing Page (`index.html`) Brand & Slogan
- [ ] Task 2: Update Predictor & Advisory Interfaces (`predict.html`, `careers.html`)

### Checkpoint: Verification
- [ ] Both pages render "Agcl" correctly.
- [ ] Animated SplitText footer types the new slogan cleanly without errors.
- [ ] No layout regression on mobile or desktop viewports.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| SplitText char animation misalignment | Low | Keep line lengths balanced (~30 characters per line) to maintain centered typography. |
| Broken logo link navigation | Low | Retain `<a href="index.html">` wrapper around the new Agcl wordmark. |

## Open Questions
- None.
