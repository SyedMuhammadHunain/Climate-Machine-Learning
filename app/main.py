from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import MODEL_PATH
from app.core.exceptions import (
    AgronomicValidationError,
    agronomic_exception_handler,
    validation_exception_handler,
)
from app.services.model_service import ModelService
from app.api.v1.endpoints import router as v1_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "climate-resilient-frontend"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the serialized pipeline into shared application state
    print(f"[STARTUP] Loading machine learning pipeline from {MODEL_PATH}...")
    app.state.model_service = ModelService(MODEL_PATH)
    print("[STARTUP] Model pipeline successfully loaded into memory. Ready for requests.")
    yield
    # Shutdown
    print("[SHUTDOWN] Application shutting down.")


app = FastAPI(
    title="Climate-Resilient Low-Water Crop Selection API",
    description=(
        "FastAPI backend serving the CT-378 machine learning pipeline for predicting crop yield "
        "and recommending drought-adaptive crops in Pakistan."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable Cross-Origin Resource Sharing (CORS) for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies or auth in this API; a credentialed wildcard is never wanted
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def revalidate_pages(request, call_next):
    """HTML pages must be re-checked on every load: browsers otherwise reuse an edited page from cache."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# Register custom exception handlers for physical and schema validation errors
app.add_exception_handler(AgronomicValidationError, agronomic_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Mount API v1 router
app.include_router(v1_router, prefix="/api/v1")

# Convenience root aliases to satisfy exact rubric requirements (e.g. '/predict')
app.include_router(v1_router, prefix="")


# Serve static frontend files (prefers frontend/climate-resilient-frontend, falls back to app/static)
SERVE_DIR = FRONTEND_DIR if FRONTEND_DIR.exists() else (STATIC_DIR if STATIC_DIR.exists() else None)

if SERVE_DIR:
    app.mount("/", StaticFiles(directory=SERVE_DIR, html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "Climate-Resilient Low-Water Crop Selection API",
            "version": "0.1.0",
            "status": "online",
            "docs_url": "/docs",
            "endpoints": {
                "health": "/health",
                "metadata": "/api/v1/metadata",
                "predict": "/api/v1/predict",
                "recommend": "/api/v1/recommend",
            }
        }
