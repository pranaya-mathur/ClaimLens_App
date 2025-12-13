"""
LLM Engine API Routes - AI-Powered Explanations using Groq

Provides natural language explanations for fraud detection verdicts.
Supports both technical (adjuster) and friendly (customer) modes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from loguru import logger

from src.llm_engine.explanation_generator import ExplanationGenerator
from config.settings import get_settings


router = APIRouter()

# Global instance (lazy loaded)
_explanation_generator: Optional[ExplanationGenerator] = None


class ExplanationRequest(BaseModel):
    """Request for LLM explanation generation"""
    claim_narrative: str = Field(..., description="Claim narrative/description")
    ml_fraud_prob: float = Field(..., ge=0, le=1, description="ML fraud probability (0-1)")
    document_risk: float = Field(default=0.0, ge=0, le=1, description="Document verification risk (0-1)")
    network_risk: float = Field(default=0.0, ge=0, le=1, description="Graph network risk (0-1)")
    claim_amount: float = Field(..., gt=0, description="Claim amount")
    premium: float = Field(default=0.0, ge=0, description="Policy premium")
    days_since_policy: int = Field(default=0, ge=0, description="Days since policy start")
    product_type: str = Field(default="motor", description="Product type")
    audience: str = Field(default="adjuster", description="Target audience: 'adjuster' or 'customer'")


class ExplanationResponse(BaseModel):
    """Response with AI-generated explanation"""
    claim_id: Optional[str] = None
    explanation: str
    verdict: str
    confidence: float
    risk_score: float
    llm_used: bool
    model: Optional[str] = None


def get_explanation_generator() -> ExplanationGenerator:
    """Lazy load explanation generator singleton"""
    global _explanation_generator
    
    if _explanation_generator is None:
        logger.info("Initializing ExplanationGenerator...")
        
        settings = get_settings()
        
        try:
            _explanation_generator = ExplanationGenerator(
                api_key=settings.GROQ_API_KEY,
                model=settings.EXPLANATION_MODEL,
                temperature=settings.EXPLANATION_TEMPERATURE,
                max_tokens=settings.EXPLANATION_MAX_TOKENS
            )
            
            if _explanation_generator.is_available():
                logger.success(f"✅ ExplanationGenerator ready with {settings.EXPLANATION_MODEL}")
            else:
                logger.warning("⚠️ LLM not available, will use template fallbacks")
                
        except Exception as e:
            logger.error(f"Failed to initialize ExplanationGenerator: {e}")
            # Return instance anyway - it will use template fallbacks
            _explanation_generator = ExplanationGenerator(
                api_key=None,
                model=settings.EXPLANATION_MODEL,
                temperature=settings.EXPLANATION_TEMPERATURE,
                max_tokens=settings.EXPLANATION_MAX_TOKENS
            )
    
    return _explanation_generator


@router.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """
    Generate AI-powered explanation for fraud detection verdict.
    
    **Audience Modes:**
    - `adjuster`: Technical explanation with ML details, risk scores, recommendations
    - `customer`: Friendly, empathetic explanation in simple language
    
    **How it works:**
    1. Analyzes ML fraud probability, document risk, and network risk
    2. Determines overall verdict (APPROVE/REVIEW/REJECT)
    3. Generates natural language explanation using Groq LLM
    4. Falls back to template if LLM unavailable
    
    **Returns:**
    - Clear, audience-appropriate explanation
    - Verdict and confidence level
    - Whether LLM was used or template fallback
    """
    try:
        logger.info(f"Generating {request.audience} explanation for claim")
        
        generator = get_explanation_generator()
        
        # Calculate overall risk score (weighted average)
        risk_score = (
            request.ml_fraud_prob * 0.5 + 
            request.document_risk * 0.3 + 
            request.network_risk * 0.2
        )
        
        # Determine verdict based on risk score
        if risk_score >= 0.7:
            verdict = "REJECT"
            confidence = 0.85
        elif risk_score >= 0.4:
            verdict = "REVIEW"
            confidence = 0.75
        else:
            verdict = "APPROVE"
            confidence = 0.80
        
        # Prepare data for explanation generator
        verdict_data = {
            "verdict": verdict,
            "confidence": confidence,
            "final_risk_score": risk_score,
            "primary_reason": _get_primary_reason(request),
            "critical_flags": _get_critical_flags(request)
        }
        
        claim_data = {
            "claim_id": "N/A",
            "product": request.product_type,
            "claim_amount": request.claim_amount,
            "premium": request.premium if request.premium > 0 else 10000,  # Default if missing
            "days_since_policy": request.days_since_policy,
            "narrative": request.claim_narrative
        }
        
        component_results = {
            "ml_fraud_score": {
                "verdict": "FRAUD" if request.ml_fraud_prob > 0.5 else "GENUINE",
                "confidence": max(request.ml_fraud_prob, 1 - request.ml_fraud_prob),
                "score": request.ml_fraud_prob,
                "reason": f"ML model predicted {request.ml_fraud_prob:.0%} fraud probability"
            },
            "document_verification": {
                "verdict": "SUSPICIOUS" if request.document_risk > 0.4 else "VERIFIED",
                "confidence": max(request.document_risk, 1 - request.document_risk),
                "score": request.document_risk,
                "reason": "Document authenticity check" if request.document_risk > 0 else "No documents uploaded"
            },
            "graph_analysis": {
                "verdict": "FRAUD_RING" if request.network_risk > 0.5 else "CLEAN",
                "confidence": max(request.network_risk, 1 - request.network_risk),
                "score": request.network_risk,
                "reason": "Network fraud patterns detected" if request.network_risk > 0.5 else "No fraud network"
            }
        }
        
        # Generate explanation
        explanation = generator.generate(
            verdict_data=verdict_data,
            claim_data=claim_data,
            component_results=component_results,
            audience=request.audience
        )
        
        llm_used = generator.is_available()
        model_name = generator.model if llm_used else "template_fallback"
        
        logger.success(
            f"Explanation generated: {len(explanation)} chars, "
            f"verdict={verdict}, llm_used={llm_used}"
        )
        
        return ExplanationResponse(
            explanation=explanation,
            verdict=verdict,
            confidence=confidence,
            risk_score=risk_score,
            llm_used=llm_used,
            model=model_name
        )
        
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


def _get_primary_reason(request: ExplanationRequest) -> str:
    """Determine primary reason for verdict"""
    if request.ml_fraud_prob > 0.7:
        return "High ML fraud probability"
    elif request.document_risk > 0.6:
        return "Document verification concerns"
    elif request.network_risk > 0.6:
        return "Fraud network detected"
    elif request.claim_amount / max(request.premium, 1) > 15:
        return "Unusually high claim-to-premium ratio"
    elif request.days_since_policy < 30:
        return "Very early claim filing"
    else:
        return "Multiple moderate risk factors"


def _get_critical_flags(request: ExplanationRequest) -> list:
    """Identify critical flags"""
    flags = []
    
    if request.ml_fraud_prob > 0.7:
        flags.append("ML_HIGH_RISK")
    
    if request.document_risk > 0.6:
        flags.append("DOCUMENT_SUSPICIOUS")
    
    if request.network_risk > 0.5:
        flags.append("FRAUD_NETWORK")
    
    if request.premium > 0 and (request.claim_amount / request.premium) > 20:
        flags.append("EXCESSIVE_CLAIM_RATIO")
    
    if request.days_since_policy < 30:
        flags.append("EARLY_CLAIM")
    
    return flags


@router.get("/health")
async def llm_health_check():
    """
    Health check for LLM Engine.
    
    Verifies:
    - ExplanationGenerator initialized
    - Groq API key configured
    - LLM model accessible
    """
    try:
        generator = get_explanation_generator()
        settings = get_settings()
        
        is_available = generator.is_available()
        has_api_key = bool(settings.GROQ_API_KEY)
        
        status = "healthy" if is_available else "degraded"
        
        return {
            "status": status,
            "llm_available": is_available,
            "api_key_configured": has_api_key,
            "model": settings.EXPLANATION_MODEL,
            "temperature": settings.EXPLANATION_TEMPERATURE,
            "max_tokens": settings.EXPLANATION_MAX_TOKENS,
            "fallback_mode": not is_available,
            "message": "LLM ready" if is_available else "Using template fallbacks (Groq API key missing or LangChain not installed)"
        }
        
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "LLM service unavailable"
        }


@router.get("/config")
async def get_llm_config():
    """
    Get current LLM configuration.
    
    Returns model settings and feature flags.
    """
    settings = get_settings()
    
    return {
        "model": settings.EXPLANATION_MODEL,
        "temperature": settings.EXPLANATION_TEMPERATURE,
        "max_tokens": settings.EXPLANATION_MAX_TOKENS,
        "semantic_aggregation_enabled": settings.ENABLE_SEMANTIC_AGGREGATION,
        "explanations_enabled": settings.ENABLE_LLM_EXPLANATIONS,
        "api_key_configured": bool(settings.GROQ_API_KEY),
        "llm_features_active": settings.get_llm_explanation_enabled()
    }
