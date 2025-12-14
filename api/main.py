"""ClaimLens FastAPI Application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import (
    fraud, health, analytics, ingest, cv_detection, ml_engine,
    document_verification, unified_fraud, llm_engine, cache, unified_analysis
)
from api.middleware.rate_limiter import RateLimitMiddleware


# Create FastAPI app
app = FastAPI(
    title="ClaimLens API",
    description="AI-Powered Insurance Fraud Detection with Computer Vision, ML, Graph Analytics & LLM",
    version="2.2.0"
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
app.include_router(cache.router, prefix="/api/cache", tags=["Cache"])
app.include_router(unified_analysis.router, prefix="/api/unified-analysis", tags=["Auto-Detection Analysis"])  # 🤖 NEW!
app.include_router(unified_fraud.router, prefix="/api/unified", tags=["Unified Fraud Analysis"])
app.include_router(fraud.router, prefix="/api/fraud", tags=["Fraud Detection"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Claim Ingestion"])
app.include_router(cv_detection.router, prefix="/api/cv", tags=["Computer Vision"])
app.include_router(ml_engine.router, prefix="/api/ml", tags=["ML Engine"])
app.include_router(document_verification.router, prefix="/api/documents", tags=["Document Verification"])
app.include_router(llm_engine.router, prefix="/api/llm", tags=["LLM Engine"])


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ClaimLens API v2.2...")
    logger.info("  - 🤖 Auto-Detection Analysis: /api/unified-analysis (NEW!)")
    logger.info("  - 🎯 Unified Analysis: /api/unified")
    logger.info("  - Fraud Detection: /api/fraud")
    logger.info("  - Claim Ingestion: /api/ingest")
    logger.info("  - Computer Vision: /api/cv")
    logger.info("  - ML Engine: /api/ml")
    logger.info("  - Document Verification: /api/documents")
    logger.info("  - LLM Engine: /api/llm")
    logger.info("  - ⚡ Cache Layer: /api/cache (Redis)")
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
        "version": "2.2.0",
        "docs": "/docs",
        "endpoints": {
            "auto_detection_analysis": {
                "base": "/api/unified-analysis",
                "endpoints": [
                    "/api/unified-analysis/analyze-complete",
                    "/api/unified-analysis/health"
                ],
                "description": "🤖 Smart auto-detection: Claim type detected from narrative + files, runs only relevant modules",
                "status": "NEW - v2.2"
            },
            "unified_analysis": {
                "base": "/api/unified",
                "endpoints": [
                    "/api/unified/analyze-complete",
                    "/api/unified/health"
                ],
                "description": "Complete fraud analysis with ML + CV + Graph + LLM"
            },
            "cache": {
                "base": "/api/cache",
                "endpoints": [
                    "/api/cache/health",
                    "/api/cache/stats",
                    "/api/cache/test",
                    "/api/cache/flush"
                ],
                "description": "Redis cache management and monitoring"
            },
            "llm_engine": {
                "base": "/api/llm",
                "endpoints": [
                    "/api/llm/explain",
                    "/api/llm/health",
                    "/api/llm/config"
                ],
                "description": "AI-powered natural language explanations using Groq LLM"
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
                    "/api/ml/score/detailed",
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
            "auto_detection": "🤖 Automatic claim type detection from Hinglish narrative + files",
            "smart_modules": "Only runs relevant modules (motor→vehicle damage, health→hospital bill)",
            "unified_analysis": "All modules (ML + CV + Graph + LLM) in one endpoint",
            "redis_caching": "High-performance caching layer for ML predictions and documents",
            "llm_explanations": "Natural language explanations powered by Groq Llama-3.3-70B",
            "smart_fallbacks": "Handles missing data gracefully",
            "multi_product": "Motor/Health/Life/Property",
            "fraud_rings": "Hospital/Claimant network detection",
            "rate_limiting": "100 requests per minute",
            "document_verification": "PAN/Aadhaar/Generic docs (ResNet50 + ELA)",
            "ocr_extraction": "Multi-language text extraction",
            "neo4j_storage": "Automatic claim persistence for graph queries"
        },
        "status": "active"
    }
