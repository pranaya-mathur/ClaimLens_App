"""ClaimLens FastAPI Application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import fraud, health, analytics, ingest, cv_detection, ml_engine, document_verification, unified_fraud
from api.middleware.rate_limiter import RateLimitMiddleware


# Create FastAPI app
app = FastAPI(
    title="ClaimLens API",
    description="AI-Powered Insurance Fraud Detection with Computer Vision, ML, Graph Analytics & LLM",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(unified_fraud.router, prefix="/api/unified", tags=["Unified Fraud Analysis"])
app.include_router(fraud.router, prefix="/api/fraud", tags=["Fraud Detection"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Claim Ingestion"])
app.include_router(cv_detection.router, prefix="/api/cv", tags=["Computer Vision"])
app.include_router(ml_engine.router, prefix="/api/ml", tags=["ML Engine"])
app.include_router(document_verification.router, prefix="/api/documents", tags=["Document Verification"])


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ClaimLens API v2.0...")
    logger.info("  - 🎯 Unified Analysis: /api/unified ✅ NEW!")
    logger.info("  - Fraud Detection: /api/fraud")
    logger.info("  - Claim Ingestion: /api/ingest")
    logger.info("  - Computer Vision: /api/cv")
    logger.info("  - ML Engine: /api/ml")
    logger.info("  - Document Verification: /api/documents")
    logger.info("  - Analytics: /api/analytics")
    logger.info("  - Rate Limiting: ENABLED (100 req/min)")
    logger.success("✓ API ready")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ClaimLens API...")
    logger.success("✓ Shutdown complete")


@app.get("/")
def root():
    return {
        "message": "ClaimLens API - AI-Powered Insurance Fraud Detection",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "unified_analysis": {
                "base": "/api/unified",
                "endpoints": [
                    "/api/unified/analyze-complete",
                    "/api/unified/health"
                ],
                "description": "Complete fraud analysis with ML + CV + Graph + LLM",
                "status": "NEW"
            },
            "computer_vision": {
                "base": "/api/cv",
                "endpoints": [
                    "/api/cv/detect",
                    "/api/cv/detect-forgery",
                    "/api/cv/analyze-complete"
                ]
            },
            "document_verification": {
                "base": "/api/documents",
                "endpoints": [
                    "/api/documents/verify-pan",
                    "/api/documents/verify-aadhaar",
                    "/api/documents/verify-document",
                    "/api/documents/extract-text"
                ]
            },
            "ml_engine": {
                "base": "/api/ml",
                "endpoints": [
                    "/api/ml/score",
                    "/api/ml/batch",
                    "/api/ml/explain"
                ]
            },
            "fraud_detection": "/api/fraud",
            "claim_ingestion": "/api/ingest",
            "analytics": "/api/analytics",
            "health": "/health"
        },
        "features": {
            "unified_analysis": "All modules (ML + CV + Graph + LLM) in one endpoint",
            "smart_fallbacks": "Handles missing data gracefully",
            "multi_product": "Motor/Health/Life/Property",
            "fraud_rings": "Hospital/Claimant network detection",
            "rate_limiting": "100 requests per minute",
            "document_verification": "PAN/Aadhaar/Generic docs",
            "ocr_extraction": "Multi-language text extraction",
            "llm_explanations": "Groq-powered AI explanations",
            "neo4j_storage": "Automatic claim persistence for graph queries"
        },
        "status": "active"
    }
