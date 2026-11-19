# Climate Resilient Frontend (Low-Water Crops)

Cloned from [https://www.infinitefield.xyz/](https://www.infinitefield.xyz/) (page design and animation). The site name, header text, big zoom-in text, logo, icons and share image were replaced with "Low-Water Crops" by `scripts/build_brand_assets.py`.

## Tech Stack
- **Three.js** (v0.164.0) WebGL shaders & 3D GLB mesh rendering
- **GSAP** & **ScrollTrigger** for scroll-synced timeline animations
- **Lenis** smooth scrolling
- Custom fonts: DenimINK-Regular & LD_TechD-Regular

## Running Locally

Run any static server from this folder:

```bash
# Option 1: Python
python3 -m http.server 3000

# Option 2: Node
npx serve -l 3000
```

Then open `http://localhost:3000` in browser.
