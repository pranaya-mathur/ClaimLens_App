"""
ClaimLens FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from config.settings import get_settings
from api.routes import fraud, health
from src.fraud_engine.fraud_detector import FraudDetector

settings = get_settings()

# Global fraud detector instance
fraud_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting ClaimLens API...")
    global fraud_detector
    fraud_detector = FraudDetector(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    logger.success("✓ Connected to Neo4j")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if fraud_detector:
        fraud_detector.close()
    logger.success("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ClaimLens API",
    description="AI-Powered Insurance Fraud Detection",
    version="1.0.0",
    lifespan=lifespan
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


@app.get("/")
def root():
    return {
        "message": "ClaimLens API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active"
    }


# Make fraud_detector accessible to routes
def get_fraud_detector():
    return fraud_detector
