"""
Health Check Routes
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ClaimLens API",
        "version": "2.0.0"
    }


@router.get("/liveness")
def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"status": "ok"}


@router.get("/readiness")
@router.get("/ready")  # Backward compatibility
def readiness_check():
    """Kubernetes-style readiness probe"""
    # Add checks for Neo4j, Redis, etc. if needed
    return {
        "ready": True,
        "status": "ok",
        "services": {
            "api": "up",
            "neo4j": "unknown",
            "redis": "unknown"
        }
    }
