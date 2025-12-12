"""ML Engine API Routes - Fraud Scoring with Hinglish NLP"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import json
import io
from loguru import logger

from src.ml_engine import MLFraudScorer, FeatureEngineer


router = APIRouter()

# Global instances (lazy loaded)
_scorer: Optional[MLFraudScorer] = None
_feature_engineer: Optional[FeatureEngineer] = None


class ClaimRequest(BaseModel):
    """Single claim data for fraud scoring"""
    claim_id: str
    claimant_id: str
    policy_id: str
    product: str = Field(..., description="Product type: motor/health/life/property")
    city: str = Field(..., description="City name")
    subtype: str = Field(..., description="Claim subtype")
    claim_amount: float
    days_since_policy_start: int
    narrative: str = Field(..., description="Claim narrative in Hinglish")
    documents_submitted: Optional[str] = Field(None, description="Comma-separated doc list")
    incident_date: str = Field(..., description="ISO format date")


class BatchClaimRequest(BaseModel):
    """Batch claims for scoring"""
    claims: List[ClaimRequest]


class FraudScoreResponse(BaseModel):
    """Fraud score response for single claim"""
    claim_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_level: str
    processing_time_ms: float


class DetailedFraudResponse(BaseModel):
    """Detailed fraud analysis with explanations"""
    claim_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_level: str
    threshold: float
    top_features: List[Dict]
    model_metrics: Dict


class BatchScoreResponse(BaseModel):
    """Batch scoring response"""
    total_claims: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    results: List[FraudScoreResponse]


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    total_features: int
    top_features: List[Dict]


class ThresholdAnalysisRequest(BaseModel):
    """Request for threshold analysis"""
    thresholds: List[float] = Field(default=[0.3, 0.4, 0.5, 0.6, 0.7])


def get_ml_scorer() -> MLFraudScorer:
    """Lazy load ML fraud scorer singleton"""
    global _scorer
    if _scorer is None:
        logger.info("Loading MLFraudScorer...")
        try:
            _scorer = MLFraudScorer(
                model_path="models/claimlens_catboost_hinglish.cbm",
                metadata_path="models/claimlens_model_metadata.json",
                threshold=0.5
            )
            logger.success("✓ MLFraudScorer loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise HTTPException(
                status_code=503,
                detail="ML model not available. Please ensure models are downloaded."
            )
        except Exception as e:
            logger.error(f"Failed to load ML scorer: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    return _scorer


def get_feature_engineer() -> FeatureEngineer:
    """Lazy load feature engineer singleton"""
    global _feature_engineer
    if _feature_engineer is None:
        logger.info("Loading FeatureEngineer...")
        try:
            _feature_engineer = FeatureEngineer(pca_dims=100)
            logger.success("✓ FeatureEngineer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load feature engineer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature engineer initialization error: {str(e)}"
            )
    return _feature_engineer


@router.post("/score", response_model=FraudScoreResponse)
async def score_single_claim(request: ClaimRequest):
    """
    Score a single claim for fraud risk using ML model.
    
    Returns fraud probability (0-1) and risk level categorization.
    
    **Risk Levels:**
    - LOW: < 0.3
    - MEDIUM: 0.3 - 0.5
    - HIGH: 0.5 - 0.7
    - CRITICAL: >= 0.7
    """
    import time
    start_time = time.time()
    
    try:
        scorer = get_ml_scorer()
        engineer = get_feature_engineer()
        
        # Convert request to DataFrame
        claim_data = pd.DataFrame([request.dict()])
        
        # Engineer features (BUG #3 FIX: use keep_ids=True)
        logger.info(f"Engineering features for claim {request.claim_id}")
        features = engineer.engineer_features(claim_data, keep_ids=True)
        
        # Validate no leakage
        engineer.validate_no_leakage(features)
        
        # Score claim
        result = scorer.score_claim(features.iloc[[0]], return_details=False)
        fraud_prob = float(result)
        
        # Categorize risk
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.5:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        fraud_prediction = int(fraud_prob >= scorer.threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Claim {request.claim_id}: fraud_prob={fraud_prob:.3f}, "
            f"risk={risk_level}, time={processing_time:.1f}ms"
        )
        
        return FraudScoreResponse(
            claim_id=request.claim_id,
            fraud_probability=fraud_prob,
            fraud_prediction=fraud_prediction,
            risk_level=risk_level,
            processing_time_ms=processing_time
        )
    
    except ValueError as e:
        logger.error(f"Validation error for claim {request.claim_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scoring error for claim {request.claim_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@router.post("/score/detailed", response_model=DetailedFraudResponse)
async def score_with_explanation(request: ClaimRequest):
    """
    Score claim with detailed explanation and top contributing features.
    
    Returns:
    - Fraud probability and prediction
    - Top 10 contributing features with importance scores
    - Model performance metrics (AUC, F1)
    """
    try:
        scorer = get_ml_scorer()
        engineer = get_feature_engineer()
        
        # Convert and engineer features (BUG #3 FIX)
        claim_data = pd.DataFrame([request.dict()])
        features = engineer.engineer_features(claim_data, keep_ids=True)
        engineer.validate_no_leakage(features)
        
        # Get detailed scoring
        result = scorer.score_claim(features.iloc[[0]], return_details=True)
        
        logger.info(f"Detailed scoring for {request.claim_id}: {result['risk_level']}")
        
        return DetailedFraudResponse(
            claim_id=request.claim_id,
            fraud_probability=result["fraud_probability"],
            fraud_prediction=result["fraud_prediction"],
            risk_level=result["risk_level"],
            threshold=result["threshold"],
            top_features=result["top_features"],
            model_metrics=result["model_metrics"]
        )
    
    except Exception as e:
        logger.error(f"Detailed scoring error for {request.claim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchScoreResponse)
async def score_batch_claims(request: BatchClaimRequest):
    """
    Score multiple claims in batch.
    
    Efficient batch processing with feature engineering pipeline.
    Returns aggregated statistics and individual scores.
    """
    try:
        scorer = get_ml_scorer()
        engineer = get_feature_engineer()
        
        logger.info(f"Batch scoring {len(request.claims)} claims")
        
        # Convert to DataFrame
        claims_data = pd.DataFrame([claim.dict() for claim in request.claims])
        
        # Engineer features for batch (BUG #3 FIX)
        features = engineer.engineer_features(claims_data, keep_ids=True)
        engineer.validate_no_leakage(features)
        
        # Batch score
        batch_results = scorer.score_batch(features, return_dataframe=True)
        
        # Build response
        results = []
        for idx, (_, row) in enumerate(batch_results.iterrows()):
            claim_id = claims_data.iloc[idx]["claim_id"]
            results.append(
                FraudScoreResponse(
                    claim_id=claim_id,
                    fraud_probability=row["fraud_probability"],
                    fraud_prediction=row["fraud_prediction"],
                    risk_level=row["risk_level"],
                    processing_time_ms=0.0  # Batch timing not per-claim
                )
            )
        
        # Aggregate stats
        high_risk = len(batch_results[batch_results["risk_level"].isin(["HIGH", "CRITICAL"])])
        medium_risk = len(batch_results[batch_results["risk_level"] == "MEDIUM"])
        low_risk = len(batch_results[batch_results["risk_level"] == "LOW"])
        
        logger.success(
            f"Batch complete: {len(request.claims)} claims - "
            f"High: {high_risk}, Medium: {medium_risk}, Low: {low_risk}"
        )
        
        return BatchScoreResponse(
            total_claims=len(request.claims),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            results=results
        )
    
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(top_n: int = 20):
    """
    Get top N most important features from the fraud detection model.
    
    Useful for understanding which features drive fraud predictions.
    
    Args:
        top_n: Number of top features to return (default: 20, max: 100)
    """
    try:
        scorer = get_ml_scorer()
        
        if top_n > 100:
            top_n = 100
        
        importance_df = scorer.get_feature_importance(top_n=top_n)
        
        top_features = [
            {
                "feature": row["feature"],
                "importance": float(row["importance"])
            }
            for _, row in importance_df.iterrows()
        ]
        
        logger.info(f"Retrieved top {top_n} features")
        
        return FeatureImportanceResponse(
            total_features=len(scorer.feature_importance),
            top_features=top_features
        )
    
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(request: ClaimRequest):
    """
    Explain fraud prediction for a claim with top contributing features.
    
    Returns:
    - Fraud probability
    - Prediction (0 or 1)
    - Top 10 features with their values and importance scores
    """
    try:
        scorer = get_ml_scorer()
        engineer = get_feature_engineer()
        
        # Engineer features (BUG #3 FIX)
        claim_data = pd.DataFrame([request.dict()])
        features = engineer.engineer_features(claim_data, keep_ids=True)
        engineer.validate_no_leakage(features)
        
        # Get explanation
        explanation = scorer.explain_prediction(features.iloc[[0]], top_n=10)
        
        logger.info(f"Generated explanation for {request.claim_id}")
        
        return {
            "claim_id": request.claim_id,
            **explanation
        }
    
    except Exception as e:
        logger.error(f"Explanation error for {request.claim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold/update")
async def update_threshold(new_threshold: float):
    """
    Update the fraud classification threshold.
    
    Args:
        new_threshold: New threshold value (0-1)
    
    Common thresholds:
    - 0.3: Aggressive fraud detection (high recall)
    - 0.5: Balanced (default)
    - 0.7: Conservative (high precision)
    """
    try:
        scorer = get_ml_scorer()
        scorer.update_threshold(new_threshold)
        
        logger.info(f"Threshold updated to {new_threshold}")
        
        return {
            "status": "success",
            "new_threshold": new_threshold,
            "message": f"Fraud threshold updated to {new_threshold}"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/summary")
async def get_model_summary():
    """
    Get ML model summary information.
    
    Returns:
    - Model type
    - Number of features
    - Current threshold
    - Training metrics (AUC, F1)
    - Top 5 features
    """
    try:
        scorer = get_ml_scorer()
        summary = scorer.summary()
        
        logger.info("Retrieved model summary")
        
        return summary
    
    except Exception as e:
        logger.error(f"Model summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def ml_health_check():
    """
    Health check for ML Engine.
    
    Verifies:
    - Model loaded
    - Feature engineer initialized
    - Embedding model accessible
    """
    try:
        scorer = get_ml_scorer()
        engineer = get_feature_engineer()
        
        return {
            "status": "healthy",
            "ml_scorer_loaded": scorer.model is not None,
            "feature_engineer_ready": engineer.pca is not None,
            "model_features": len(scorer.model.feature_names_) if scorer.model else 0,
            "threshold": scorer.threshold,
            "embedder_model": engineer.model_name
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
