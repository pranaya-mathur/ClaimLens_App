"""
Fraud Detection API Routes
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from pydantic import BaseModel

from src.fraud_engine.fraud_detector import FraudDetector
from config.settings import get_settings


router = APIRouter()


class FraudScoreRequest(BaseModel):
    claim_id: int


class FraudScoreResponse(BaseModel):
    claim_id: int
    base_fraud_score: float
    final_risk_score: float
    risk_level: str
    graph_insights: Dict
    recommendation: str


def get_fraud_detector():
    """Dependency to get fraud detector instance"""
    settings = get_settings()
    return FraudDetector(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )


@router.post("/score", response_model=FraudScoreResponse)
def get_fraud_score(
    request: FraudScoreRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Get comprehensive fraud risk score for a claim
    
    Combines:
    - Base ML fraud score
    - Graph analytics (neighbor fraud, document sharing)
    - Claimant history
    """
    try:
        risk_data = detector.get_graph_risk_score(request.claim_id)
        
        if "error" in risk_data:
            raise HTTPException(status_code=404, detail=risk_data["error"])
        
        # Generate recommendation
        score = risk_data["final_risk_score"]
        if score >= 0.8:
            recommendation = "REJECT - High fraud risk"
        elif score >= 0.5:
            recommendation = "MANUAL_REVIEW - Medium risk"
        else:
            recommendation = "AUTO_APPROVE - Low risk"
        
        return FraudScoreResponse(
            claim_id=risk_data["claim_id"],
            base_fraud_score=risk_data["base_fraud_score"],
            final_risk_score=risk_data["final_risk_score"],
            risk_level=risk_data["risk_level"],
            graph_insights={
                "neighbor_fraud_count": risk_data["neighbor_fraud_count"],
                "doc_sharing_count": risk_data["doc_sharing_count"],
                "claimant_fraud_rate": risk_data["claimant_fraud_rate"]
            },
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/rings")
def find_fraud_rings(
    min_shared_docs: int = 2,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Find fraud rings (claimants sharing documents)"""
    try:
        rings = detector.find_fraud_rings(min_shared_docs=min_shared_docs)
        return {
            "total_rings_found": len(rings),
            "rings": rings[:20]  # Return top 20
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/serial-fraudsters")
def find_serial_fraudsters(
    min_fraud_claims: int = 3,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Find claimants with multiple high-fraud claims"""
    try:
        fraudsters = detector.find_serial_fraudsters(min_fraud_claims=min_fraud_claims)
        return {
            "total_found": len(fraudsters),
            "fraudsters": fraudsters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()


@router.get("/policy-abuse")
def detect_policy_abuse(
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Detect new policy + immediate high claim patterns"""
    try:
        cases = detector.detect_policy_abuse()
        return {
            "total_suspicious_cases": len(cases),
            "cases": cases
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        detector.close()