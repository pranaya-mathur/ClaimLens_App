"""
ClaimLens FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import fraud, health, analytics, ingest


# Create FastAPI app
app = FastAPI(
    title="ClaimLens API",
    description="AI-Powered Insurance Fraud Detection",
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


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ClaimLens API...")
    logger.success("✓ API ready")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ClaimLens API...")
    logger.success("✓ Shutdown complete")


@app.get("/")
def root():
    return {
        "message": "ClaimLens API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active"
    }
