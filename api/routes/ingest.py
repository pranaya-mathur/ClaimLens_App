"""
Claim Ingestion API Routes
"""
from fastapi import APIRouter, HTTPException, status
from loguru import logger
import os
from dotenv import load_dotenv

from api.schemas.ingest import ClaimIngestRequest, ClaimIngestResponse
from src.fraud_engine.live_ingest import LiveClaimIngestor

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

# Neo4j connection from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# Global ingestor instance (connection pooling)
ingestor = None


def get_ingestor() -> LiveClaimIngestor:
    """Get or create LiveClaimIngestor instance"""
    global ingestor
    if ingestor is None:
        ingestor = LiveClaimIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return ingestor


@router.post(
    "/claim",
    response_model=ClaimIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest new claim into fraud graph",
    description="""
    Submit a new insurance claim for real-time ingestion into the Neo4j fraud graph.
    
    This endpoint:
    - Creates/updates claimant, policy, city, and document nodes
    - Creates new claim node with all relationships
    - Detects document reuse across claims
    - Updates fraud statistics for claimants
    - Enables immediate fraud scoring on the new claim
    
    **Integration Flow:**
    1. Submit claim via this endpoint
    2. Receive ingestion confirmation
    3. Call `/api/fraud/score` for fraud risk assessment
    4. Make decision based on risk score
    """
)
async def ingest_claim(request: ClaimIngestRequest):
    """
    Ingest a new claim into the fraud detection graph.
    
    Args:
        request: ClaimIngestRequest containing all claim data
    
    Returns:
        ClaimIngestResponse with ingestion statistics
    
    Raises:
        HTTPException: If claim already exists or ingestion fails
    """
    logger.info(f"Received ingestion request for claim {request.claim_id}")
    
    try:
        # Get ingestor instance
        claim_ingestor = get_ingestor()
        
        # Check if claim already exists
        if claim_ingestor.check_claim_exists(request.claim_id):
            logger.warning(f"Claim {request.claim_id} already exists in graph")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Claim {request.claim_id} already exists. Use update endpoint to modify."
            )
        
        # Convert Pydantic model to dict for ingestion
        claim_data = request.dict()
        
        # Ingest claim into graph
        stats = claim_ingestor.ingest_claim(claim_data)
        
        # Build response
        response = ClaimIngestResponse(
            status="success",
            claim_id=request.claim_id,
            graph_status="loaded",
            nodes_created=stats['nodes_created'],
            relationships_created=stats['relationships_created'],
            message=f"Claim successfully ingested. Ready for fraud scoring."
        )
        
        logger.success(f"✓ Claim {request.claim_id} ingested successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting claim {request.claim_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest claim: {str(e)}"
        )


@router.get(
    "/status/{claim_id}",
    summary="Check claim ingestion status",
    description="Check if a claim has been ingested into the fraud graph"
)
async def check_claim_status(claim_id: str):
    """
    Check if a claim exists in the fraud graph.
    
    Args:
        claim_id: Claim identifier
    
    Returns:
        Status dictionary with existence flag
    """
    try:
        claim_ingestor = get_ingestor()
        exists = claim_ingestor.check_claim_exists(claim_id)
        
        return {
            "claim_id": claim_id,
            "exists": exists,
            "status": "found" if exists else "not_found"
        }
        
    except Exception as e:
        logger.error(f"Error checking claim {claim_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check claim status: {str(e)}"
        )


@router.get(
    "/health",
    summary="Check ingestion service health",
    description="Verify Neo4j connection and ingestion service availability"
)
async def ingestion_health():
    """
    Health check for ingestion service.
    
    Returns:
        Health status dictionary
    """
    try:
        claim_ingestor = get_ingestor()
        
        # Test Neo4j connection with simple query
        with claim_ingestor.driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        
        return {
            "status": "healthy",
            "service": "claim_ingestion",
            "neo4j_connected": True,
            "message": "Ingestion service is operational"
        }
        
    except Exception as e:
        logger.error(f"Ingestion health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "claim_ingestion",
            "neo4j_connected": False,
            "error": str(e)
        }
