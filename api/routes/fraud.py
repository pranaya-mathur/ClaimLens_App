"""
Fraud Detection API Routes with LLM Integration

New Features:
- Semantic Aggregation using Groq LLM (Llama-3.3-70B)
- AI-Generated explanations with streaming support
- Intelligent verdict synthesis combining multiple fraud signals
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional
from pydantic import BaseModel
from loguru import logger

from src.fraud_engine.fraud_detector import FraudDetector
from src.llm_engine.semantic_aggregator import SemanticAggregator
from src.llm_engine.explanation_generator import ExplanationGenerator
from config.settings import get_settings

router = APIRouter()

# Initialize LLM components once at startup
_semantic_aggregator: Optional[SemanticAggregator] = None
_explanation_generator: Optional[ExplanationGenerator] = None


def get_llm_components():
    """
    Get or initialize LLM components (singleton pattern).
    
    Returns:
        Tuple of (SemanticAggregator, ExplanationGenerator)
    """
    global _semantic_aggregator, _explanation_generator
    
    settings = get_settings()
    
    if _semantic_aggregator is None:
        _semantic_aggregator = SemanticAggregator(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
    
    if _explanation_generator is None:
        _explanation_generator = ExplanationGenerator(
            api_key=settings.GROQ_API_KEY,
            model=settings.EXPLANATION_MODEL,
            temperature=settings.EXPLANATION_TEMPERATURE,
            max_tokens=settings.EXPLANATION_MAX_TOKENS
        )
    
    return _semantic_aggregator, _explanation_generator


class FraudScoreRequest(BaseModel):
    claim_id: str  # Changed from int to str for live ingestion compatibility


class FraudScoreResponse(BaseModel):
    claim_id: str
    base_fraud_score: float
    final_risk_score: float
    risk_level: str
    graph_insights: Dict
    recommendation: str
    # NEW: LLM fields
    llm_verdict: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_explanation: Optional[str] = None
    llm_used: Optional[bool] = False


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
    Get comprehensive fraud risk score for a claim with LLM semantic aggregation.
    
    Now combines:
    - Base ML fraud score
    - Graph analytics (neighbor fraud, document sharing)
    - Claimant history
    - LLM semantic aggregation (NEW!)
    
    Returns:
        FraudScoreResponse with verdict, risk score, and AI explanation
    """
    try:
        logger.info(f"Processing fraud score request for claim: {request.claim_id}")
        
        # Get base fraud detection results
        risk_data = detector.get_graph_risk_score(request.claim_id)
        
        if "error" in risk_data:
            logger.error(f"Fraud detection error: {risk_data['error']}")
            raise HTTPException(status_code=404, detail=risk_data["error"])
        
        settings = get_settings()
        response_data = {
            "claim_id": risk_data["claim_id"],
            "base_fraud_score": risk_data["base_fraud_score"],
            "final_risk_score": risk_data["final_risk_score"],
            "risk_level": risk_data["risk_level"],
            "graph_insights": {
                "neighbor_fraud_count": risk_data["neighbor_fraud_count"],
                "doc_sharing_count": risk_data["doc_sharing_count"],
                "claimant_fraud_rate": risk_data["claimant_fraud_rate"]
            },
            "llm_used": False
        }
        
        # Apply LLM semantic aggregation if enabled
        if settings.get_llm_enabled():
            logger.info("🤖 Applying LLM semantic aggregation...")
            
            try:
                semantic_agg, _ = get_llm_components()
                
                # Prepare component results for LLM
                component_results = {
                    "graph_analysis": {
                        "verdict": "FRAUD_RING_DETECTED" if risk_data["neighbor_fraud_count"] > 0 else "CLEAN",
                        "confidence": 0.88,
                        "score": risk_data["final_risk_score"],
                        "reason": f"{risk_data['neighbor_fraud_count']} fraud connections" if risk_data["neighbor_fraud_count"] > 0 else "No fraud network",
                        "red_flags": [f"{risk_data['neighbor_fraud_count']} fraud neighbors"] if risk_data["neighbor_fraud_count"] > 0 else []
                    }
                }
                
                # Get LLM aggregation
                llm_result = semantic_agg.aggregate(
                    component_results=component_results,
                    claim_data={
                        "claim_id": risk_data["claim_id"],
                        "final_risk_score": risk_data["final_risk_score"]
                    }
                )
                
                # Add LLM results to response
                response_data["llm_verdict"] = llm_result.get("verdict", "REVIEW")
                response_data["llm_confidence"] = llm_result.get("confidence", 0.75)
                response_data["final_risk_score"] = llm_result.get("final_risk_score", risk_data["final_risk_score"])
                response_data["llm_used"] = llm_result.get("llm_used", False)
                
                logger.success(f"✅ LLM verdict: {response_data['llm_verdict']}")
                
            except Exception as e:
                logger.warning(f"LLM aggregation failed, using fallback: {e}")
                response_data["llm_verdict"] = None
                response_data["llm_used"] = False
        
        # Generate AI explanation if enabled
        if settings.get_llm_explanation_enabled():
            logger.info("🧠 Generating AI explanation...")
            
            try:
                _, explanation_gen = get_llm_components()
                
                explanation = explanation_gen.generate(
                    verdict_data={
                        "verdict": response_data.get("llm_verdict", "REVIEW"),
                        "confidence": response_data.get("llm_confidence", 0.75),
                        "final_risk_score": response_data["final_risk_score"]
                    },
                    claim_data={"claim_id": request.claim_id},
                    component_results={},
                    audience="adjuster"
                )
                
                response_data["llm_explanation"] = explanation
                logger.success(f"✅ Explanation generated: {len(explanation)} chars")
                
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")
                response_data["llm_explanation"] = None
        
        # Set final recommendation based on verdict
        final_score = response_data["final_risk_score"]
        if response_data.get("llm_verdict"):
            response_data["recommendation"] = f"{response_data['llm_verdict']} - LLM Analysis (Confidence: {response_data.get('llm_confidence', 0):.0%})"
        else:
            if final_score >= 0.8:
                response_data["recommendation"] = "REJECT - High fraud risk"
            elif final_score >= 0.5:
                response_data["recommendation"] = "MANUAL_REVIEW - Medium risk"
            else:
                response_data["recommendation"] = "AUTO_APPROVE - Low risk"
        
        return FraudScoreResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fraud score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            detector.close()
        except:
            pass


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
        try:
            detector.close()
        except:
            pass


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
        try:
            detector.close()
        except:
            pass


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
        try:
            detector.close()
        except:
            pass
