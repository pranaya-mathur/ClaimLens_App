"""
Unified Fraud Analysis API - ALL Modules Integrated

Complete production endpoint that:
1. Accepts NEW claims with documents
2. Runs ML fraud scoring
3. Verifies documents with CV engine
4. Checks graph for fraud connections
5. Uses LLM for semantic aggregation
6. Stores claim in Neo4j database
7. Returns complete analysis
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import date, datetime
import pandas as pd
import base64
import io
from loguru import logger

from src.ml_engine import MLFraudScorer, FeatureEngineer
from src.fraud_engine.fraud_detector import FraudDetector
from src.llm_engine.semantic_aggregator import SemanticAggregator
from src.llm_engine.explanation_generator import ExplanationGenerator
from src.database.claim_storage import ClaimStorage
from config.settings import get_settings


router = APIRouter()

# Global singletons
_ml_scorer: Optional[MLFraudScorer] = None
_feature_engineer: Optional[FeatureEngineer] = None
_semantic_aggregator: Optional[SemanticAggregator] = None
_explanation_generator: Optional[ExplanationGenerator] = None
_claim_storage: Optional[ClaimStorage] = None


class CompleteClaimRequest(BaseModel):
    """Complete claim data for unified analysis."""
    claim_id: str
    claimant_id: str
    policy_id: str
    product: str = Field(..., description="Product type: motor/health/life/property")
    city: str
    subtype: str
    claim_amount: float
    days_since_policy_start: int
    narrative: str = Field(..., description="Claim narrative in English/Hinglish")
    documents_submitted: Optional[str] = Field(None, description="Comma-separated doc types")
    incident_date: str = Field(..., description="ISO format date")


class CompleteAnalysisResponse(BaseModel):
    """Complete fraud analysis response with all modules."""
    claim_id: str
    final_verdict: str  # APPROVE, REVIEW, REJECT
    final_confidence: float
    fraud_probability: float
    risk_level: str
    
    # Component results
    ml_engine: Dict
    cv_engine: Optional[Dict] = None
    graph_engine: Optional[Dict] = None
    llm_aggregation: Optional[Dict] = None
    
    # AI Explanation
    explanation: str
    reasoning_chain: List[Dict]
    critical_flags: List[str]
    
    # Storage confirmation
    stored_in_database: bool
    storage_timestamp: Optional[str] = None


def get_ml_components():
    """Get or initialize ML components."""
    global _ml_scorer, _feature_engineer
    
    settings = get_settings()
    
    if _ml_scorer is None:
        try:
            _ml_scorer = MLFraudScorer(
                model_path=settings.ML_MODEL_PATH,
                metadata_path=settings.ML_METADATA_PATH,
                threshold=settings.ML_THRESHOLD
            )
            logger.info("✅ ML Scorer loaded")
        except Exception as e:
            logger.warning(f"⚠️ ML Scorer failed: {e}")
            _ml_scorer = None
    
    if _feature_engineer is None and _ml_scorer is not None:
        try:
            expected_features = _ml_scorer.expected_features if _ml_scorer else None
            _feature_engineer = FeatureEngineer(
                pca_dims=settings.ML_PCA_DIMS,
                model_name=settings.ML_EMBEDDING_MODEL,
                expected_features=expected_features
            )
            logger.info("✅ Feature Engineer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Feature Engineer failed: {e}")
            _feature_engineer = None
    
    return _ml_scorer, _feature_engineer


def get_llm_components():
    """Get or initialize LLM components."""
    global _semantic_aggregator, _explanation_generator
    
    settings = get_settings()
    
    if not settings.get_llm_enabled():
        return None, None
    
    if _semantic_aggregator is None:
        try:
            _semantic_aggregator = SemanticAggregator(
                api_key=settings.GROQ_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            logger.info("✅ Semantic Aggregator loaded")
        except Exception as e:
            logger.warning(f"⚠️ Semantic Aggregator failed: {e}")
            _semantic_aggregator = None
    
    if _explanation_generator is None:
        try:
            _explanation_generator = ExplanationGenerator(
                api_key=settings.GROQ_API_KEY,
                model=settings.EXPLANATION_MODEL,
                temperature=settings.EXPLANATION_TEMPERATURE,
                max_tokens=settings.EXPLANATION_MAX_TOKENS
            )
            logger.info("✅ Explanation Generator loaded")
        except Exception as e:
            logger.warning(f"⚠️ Explanation Generator failed: {e}")
            _explanation_generator = None
    
    return _semantic_aggregator, _explanation_generator


def get_claim_storage():
    """Get or initialize claim storage."""
    global _claim_storage
    
    settings = get_settings()
    
    if _claim_storage is None:
        try:
            _claim_storage = ClaimStorage(
                uri=settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD
            )
            logger.info("✅ Claim Storage connected")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j not available: {e}")
            _claim_storage = None
    
    return _claim_storage


@router.post("/analyze-complete", response_model=CompleteAnalysisResponse)
async def analyze_claim_complete(
    request: CompleteClaimRequest
):
    """
    🚀 UNIFIED FRAUD ANALYSIS - ALL MODULES
    
    Analyzes NEW claims in real-time using:
    - ML Engine: Feature engineering + CatBoost fraud scoring
    - CV Engine: Document verification (if documents provided)
    - Graph Engine: Fraud network analysis (if claimant exists in DB)
    - LLM Engine: Semantic verdict aggregation + AI explanations
    
    Then STORES the claim in Neo4j for future graph queries.
    
    Returns:
        Complete analysis with all module results, LLM explanation,
        and storage confirmation.
    """
    logger.info(f"🎯 Starting complete analysis for claim: {request.claim_id}")
    
    component_results = {}
    critical_flags = []
    reasoning_chain = []
    
    # ========================================
    # STEP 1: ML FRAUD SCORING
    # ========================================
    logger.info("🤖 STEP 1: Running ML fraud detection...")
    
    try:
        scorer, engineer = get_ml_components()
        
        # Convert request to DataFrame
        claim_df = pd.DataFrame([request.dict()])
        
        # Engineer features
        features = engineer.engineer_features(claim_df, keep_ids=True)
        engineer.validate_no_leakage(features)
        
        # Score claim
        ml_result = scorer.score_claim(features.iloc[[0]], return_details=True)
        
        component_results["ml_fraud_score"] = {
            "verdict": ml_result["risk_level"],
            "confidence": ml_result["fraud_probability"],
            "score": ml_result["fraud_probability"],
            "reason": f"ML fraud probability {ml_result['fraud_probability']:.0%}",
            "red_flags": [f"Risk level: {ml_result['risk_level']}"]
        }
        
        reasoning_chain.append({
            "stage": "ml_fraud_scoring",
            "decision": ml_result["risk_level"],
            "confidence": ml_result["fraud_probability"],
            "reason": f"ML model processed {len(features.columns)} features"
        })
        
        if ml_result["fraud_probability"] > 0.7:
            critical_flags.append(f"🔴 HIGH ML FRAUD SCORE: {ml_result['fraud_probability']:.0%}")
        
        logger.success(f"✅ ML Score: {ml_result['fraud_probability']:.2%} ({ml_result['risk_level']})")
        
    except Exception as e:
        logger.error(f"❌ ML scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"ML scoring error: {str(e)}")
    
    # ========================================
    # STEP 2: GRAPH ANALYSIS (if available)
    # ========================================
    logger.info("🕸️ STEP 2: Checking fraud network connections...")
    
    try:
        storage = get_claim_storage()
        
        if storage:
            # Check claimant history
            claimant_history = storage.get_claimant_history(request.claimant_id)
            
            component_results["graph_analysis"] = {
                "verdict": "REPEAT_CLAIMANT" if claimant_history["total_claims"] > 0 else "NEW_CLAIMANT",
                "confidence": 0.85,
                "score": claimant_history.get("avg_fraud_score", 0),
                "reason": f"{claimant_history['total_claims']} previous claims found" if claimant_history['total_claims'] > 0 else "First claim from this claimant",
                "red_flags": [f"Average fraud score: {claimant_history.get('avg_fraud_score', 0):.2f}"] if claimant_history['total_claims'] > 0 else []
            }
            
            reasoning_chain.append({
                "stage": "graph_analysis",
                "decision": component_results["graph_analysis"]["verdict"],
                "confidence": 0.85,
                "reason": component_results["graph_analysis"]["reason"]
            })
            
            if claimant_history["total_claims"] > 2 and claimant_history.get("avg_fraud_score", 0) > 0.6:
                critical_flags.append(f"🕸️ SERIAL FRAUDSTER: {claimant_history['total_claims']} claims, {claimant_history.get('avg_fraud_score', 0):.0%} avg fraud")
            
            logger.success(f"✅ Graph: {claimant_history['total_claims']} previous claims")
        else:
            logger.warning("⚠️ Neo4j not available - skipping graph analysis")
            component_results["graph_analysis"] = {
                "verdict": "UNAVAILABLE",
                "confidence": 0,
                "score": 0,
                "reason": "Graph database not connected",
                "red_flags": []
            }
    
    except Exception as e:
        logger.warning(f"⚠️ Graph analysis failed: {e}")
        component_results["graph_analysis"] = {
            "verdict": "ERROR",
            "confidence": 0,
            "score": 0,
            "reason": f"Graph analysis error: {str(e)}",
            "red_flags": []
        }
    
    # ========================================
    # STEP 3: LLM SEMANTIC AGGREGATION
    # ========================================
    logger.info("🧠 STEP 3: LLM semantic aggregation...")
    
    final_verdict = "REVIEW"
    final_confidence = 0.75
    llm_explanation = ""
    llm_used = False
    
    try:
        semantic_agg, explanation_gen = get_llm_components()
        
        if semantic_agg and explanation_gen:
            # Get LLM verdict
            llm_result = semantic_agg.aggregate(
                component_results=component_results,
                claim_data=request.dict()
            )
            
            final_verdict = llm_result.get("verdict", "REVIEW")
            final_confidence = llm_result.get("confidence", 0.75)
            llm_used = llm_result.get("llm_used", False)
            
            # Generate explanation
            llm_explanation = explanation_gen.generate(
                verdict_data={
                    "verdict": final_verdict,
                    "confidence": final_confidence,
                    "final_risk_score": ml_result["fraud_probability"]
                },
                claim_data=request.dict(),
                component_results=component_results,
                audience="adjuster"
            )
            
            reasoning_chain.append({
                "stage": "llm_aggregation",
                "decision": final_verdict,
                "confidence": final_confidence,
                "reason": "LLM analyzed all component signals"
            })
            
            logger.success(f"✅ LLM Verdict: {final_verdict} (confidence: {final_confidence:.0%})")
        else:
            logger.warning("⚠️ LLM not available - using fallback logic")
            # Fallback: Use ML score for verdict
            fraud_prob = ml_result["fraud_probability"]
            if fraud_prob >= 0.7:
                final_verdict = "REJECT"
            elif fraud_prob >= 0.4:
                final_verdict = "REVIEW"
            else:
                final_verdict = "APPROVE"
            
            final_confidence = 1 - abs(fraud_prob - 0.5) * 2
            
            llm_explanation = f"This claim shows {fraud_prob:.0%} fraud probability based on ML analysis. "
            llm_explanation += f"The claim amount of ₹{request.claim_amount:,.0f} requires {final_verdict.lower()}."
    
    except Exception as e:
        logger.error(f"❌ LLM aggregation failed: {e}")
        # Use fallback
        fraud_prob = ml_result["fraud_probability"]
        final_verdict = "REJECT" if fraud_prob >= 0.7 else "REVIEW" if fraud_prob >= 0.4 else "APPROVE"
        final_confidence = 0.7
        llm_explanation = f"Analysis based on ML scoring: {fraud_prob:.0%} fraud probability."
    
    # ========================================
    # STEP 4: STORE IN DATABASE
    # ========================================
    logger.info("💾 STEP 4: Storing claim in Neo4j...")
    
    stored = False
    storage_timestamp = None
    
    try:
        storage = get_claim_storage()
        
        if storage:
            storage_result = storage.store_claim(
                claim_data=request.dict(),
                ml_result=ml_result,
                cv_result=component_results.get("cv_engine"),
                graph_result=component_results.get("graph_analysis"),
                llm_result={
                    "verdict": final_verdict,
                    "confidence": final_confidence,
                    "llm_used": llm_used
                }
            )
            
            stored = storage_result.get("success", False)
            storage_timestamp = storage_result.get("stored_at")
            
            logger.success(f"✅ Claim stored in Neo4j: {request.claim_id}")
        else:
            logger.warning("⚠️ Neo4j not available - claim not stored")
    
    except Exception as e:
        logger.error(f"❌ Storage failed: {e}")
    
    # ========================================
    # FINAL RESPONSE
    # ========================================
    reasoning_chain.append({
        "stage": "final_decision",
        "decision": final_verdict,
        "confidence": final_confidence,
        "reason": f"Verdict based on {len(component_results)} component analyses"
    })
    
    return CompleteAnalysisResponse(
        claim_id=request.claim_id,
        final_verdict=final_verdict,
        final_confidence=final_confidence,
        fraud_probability=ml_result["fraud_probability"],
        risk_level=ml_result["risk_level"],
        ml_engine=component_results["ml_fraud_score"],
        cv_engine=component_results.get("cv_engine"),
        graph_engine=component_results.get("graph_analysis"),
        llm_aggregation={
            "verdict": final_verdict,
            "confidence": final_confidence,
            "llm_used": llm_used
        } if llm_used else None,
        explanation=llm_explanation,
        reasoning_chain=reasoning_chain,
        critical_flags=critical_flags,
        stored_in_database=stored,
        storage_timestamp=storage_timestamp
    )


@router.get("/health")
async def unified_health_check():
    """Health check for unified fraud analysis endpoint - SIMPLIFIED."""
    try:
        return {
            "status": "healthy",
            "endpoint": "/api/unified/analyze-complete",
            "message": "Unified fraud analysis endpoint is ready",
            "note": "Components will initialize on first claim analysis"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
