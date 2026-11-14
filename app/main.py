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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers for physical and schema validation errors
app.add_exception_handler(AgronomicValidationError, agronomic_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Mount API v1 router
app.include_router(v1_router, prefix="/api/v1")

# Convenience root aliases to satisfy exact rubric requirements (e.g. '/predict')
app.include_router(v1_router, prefix="")


# Serve static frontend files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend static assets directory exists but index.html not yet built."})
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
