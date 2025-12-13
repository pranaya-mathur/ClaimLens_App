"""
Health Check Routes
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "ClaimLens API",
        "version": "1.0.0"
    }


@router.get("/ready")
def readiness_check():
    # Add checks for Neo4j, Redis, etc.
    return {
        "ready": True,
        "services": {
            "api": "up",
            "neo4j": "connected",
            "redis": "connected"
        }
    }
