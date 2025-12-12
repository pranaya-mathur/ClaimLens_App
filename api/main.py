"""
ClaimLens FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import fraud, health, analytics, ingest, cv_detection, ml_engine


# Create FastAPI app
app = FastAPI(
    title="ClaimLens API",
    description="AI-Powered Insurance Fraud Detection with Computer Vision & ML",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(fraud.router, prefix="/api/fraud", tags=["Fraud Detection"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Claim Ingestion"])
app.include_router(cv_detection.router, prefix="/api/cv", tags=["Computer Vision"])
app.include_router(ml_engine.router, prefix="/api/ml", tags=["ML Engine"])


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ClaimLens API...")
    logger.info("  - Fraud Detection: /api/fraud")
    logger.info("  - Claim Ingestion: /api/ingest")
    logger.info("  - Computer Vision: /api/cv")
    logger.info("  - Analytics: /api/analytics")
    logger.info("  - ML Engine: /api/ml")
    logger.success("✓ API ready")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ClaimLens API...")
    logger.success("✓ Shutdown complete")


@app.get("/")
def root():
    return {
        "message": "ClaimLens API - AI-Powered Insurance Fraud Detection",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "computer_vision": "/api/cv",
            "fraud_detection": "/api/fraud",
            "ml_engine": "/api/ml",
            "claim_ingestion": "/api/ingest",
            "analytics": "/api/analytics",
            "health": "/health"
        },
        "status": "active"
    }
