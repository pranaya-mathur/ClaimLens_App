"""
Computer Vision Detection API Routes
Handles image uploads, damage detection, and forgery detection
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import os
import tempfile
from pathlib import Path
from loguru import logger

from src.cv_engine.damage_detector import DamageDetector
from src.cv_engine.forgery_detector import ForgeryDetector
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

# Global detector instances (lazy loading)
_damage_detector: Optional[DamageDetector] = None
_forgery_detector: Optional[ForgeryDetector] = None


def get_damage_detector() -> DamageDetector:
    """
    Get or initialize the damage detector instance
    Lazy loading to avoid loading models on import
    """
    global _damage_detector
    if _damage_detector is None:
        logger.info("Initializing DamageDetector...")
        _damage_detector = DamageDetector(
            parts_model_path=settings.PARTS_MODEL_PATH,
            damage_model_path=settings.DAMAGE_MODEL_PATH,
            severity_model_path=settings.SEVERITY_MODEL_PATH,
            device=settings.CV_DEVICE
        )
        logger.success("DamageDetector initialized")
    return _damage_detector


def get_forgery_detector() -> ForgeryDetector:
    """
    Get or initialize the forgery detector instance
    Lazy loading to avoid loading models on import
    """
    global _forgery_detector
    if _forgery_detector is None:
        logger.info("Initializing ForgeryDetector...")
        _forgery_detector = ForgeryDetector(
            model_path=settings.FORGERY_MODEL_PATH,
            config_path=settings.FORGERY_CONFIG_PATH,
            device=settings.CV_DEVICE
        )
        logger.success("ForgeryDetector initialized")
    return _forgery_detector


class DetectionResponse(BaseModel):
    """Response model for damage detection"""
    status: str
    parts_detected: list
    damages_detected: list
    summary: dict
    risk_assessment: dict


class ForgeryResponse(BaseModel):
    """Response model for forgery detection"""
    status: str
    image_path: str
    is_forged: bool
    forgery_probability: float
    threshold: float
    ela_score: float
    noise_variation: float
    confidence_level: str
    recommendation: str


class UnifiedAnalysisResponse(BaseModel):
    """Response model for combined damage + forgery analysis"""
    status: str
    forgery_analysis: dict
    damage_analysis: dict
    final_risk_assessment: dict


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: dict
    device: str
    message: str


@router.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect car damage from image",
    description="""
    Upload a car image to detect parts, damages, and severity levels.
    
    **Process:**
    1. Parts detection using YOLO11n-seg (23 car parts)
    2. Damage detection using YOLO11m (6 damage types)
    3. Severity classification using EfficientNet-B0 (3 levels)
    
    **Returns:**
    - List of detected car parts with bounding boxes
    - List of damages with type, location, and severity
    - Summary statistics
    - Risk assessment for claim processing
    """
)
async def detect_damage(
    file: UploadFile = File(..., description="Car damage image (JPG, PNG)"),
    parts_conf: float = 0.25,
    damage_conf: float = 0.25
):
    """
    Detect damage in uploaded car image
    
    Args:
        file: Uploaded image file
        parts_conf: Confidence threshold for parts detection (0-1)
        damage_conf: Confidence threshold for damage detection (0-1)
    
    Returns:
        Detection results with parts, damages, summary, and risk assessment
    """
    logger.info(f"Received damage detection request for file: {file.filename}")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    # Check file size
    file_size_mb = 0
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({file_size_mb:.1f}MB) exceeds limit ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Saved to temporary file: {tmp_path}")
        
        try:
            # Get detector instance
            detector = get_damage_detector()
            
            # Run detection
            results = detector.detect_damage(
                image_path=tmp_path,
                parts_conf=parts_conf,
                damage_conf=damage_conf,
                return_visualization=False
            )
            
            logger.success(f"Damage detection complete: {len(results['damages_detected'])} damages found")
            
            return DetectionResponse(
                status="success",
                parts_detected=results["parts_detected"],
                damages_detected=results["damages_detected"],
                summary=results["summary"],
                risk_assessment=results["risk_assessment"]
            )
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


@router.post(
    "/detect-forgery",
    response_model=ForgeryResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect image forgery and manipulation",
    description="""
    Upload an image to detect forgery, tampering, or manipulation.
    
    **Detection Methods:**
    1. Deep Learning - ResNet50 CNN classifier (83.6% accuracy)
    2. Error Level Analysis (ELA) - JPEG compression artifacts
    3. Noise Variation - Block-level inconsistency detection
    
    **Returns:**
    - Forgery probability (0-1)
    - Binary classification (authentic/forged)
    - ELA score and noise metrics
    - Confidence level and recommendation
    
    **Use Cases:**
    - Verify claim photos are authentic
    - Detect photoshopped/edited images
    - Identify spliced or copy-pasted regions
    """
)
async def detect_forgery(
    file: UploadFile = File(..., description="Image to analyze for forgery (JPG, PNG)")
):
    """
    Detect forgery in uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Forgery detection results with probability, ELA score, and recommendation
    """
    logger.info(f"Received forgery detection request for file: {file.filename}")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({file_size_mb:.1f}MB) exceeds limit ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Analyzing image for forgery: {tmp_path}")
        
        try:
            # Get forgery detector instance
            detector = get_forgery_detector()
            
            # Run forgery detection
            result = detector.analyze_image_as_dict(tmp_path)
            
            # Determine confidence level
            prob = result["forgery_prob"]
            if prob >= 0.8 or prob <= 0.2:
                confidence = "high"
            elif prob >= 0.6 or prob <= 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Generate recommendation
            if result["is_forged"]:
                if prob >= 0.8:
                    recommendation = "REJECT - High confidence forgery detected"
                else:
                    recommendation = "REVIEW - Possible forgery, manual inspection recommended"
            else:
                if prob <= 0.2:
                    recommendation = "APPROVE - Image appears authentic"
                else:
                    recommendation = "REVIEW - Low forgery signals, proceed with caution"
            
            logger.success(
                f"Forgery detection complete: is_forged={result['is_forged']}, "
                f"prob={prob:.3f}, confidence={confidence}"
            )
            
            return ForgeryResponse(
                status="success",
                image_path=file.filename,
                is_forged=result["is_forged"],
                forgery_probability=result["forgery_prob"],
                threshold=result["threshold"],
                ela_score=result["ela_score"],
                noise_variation=result["noise_variation"],
                confidence_level=confidence,
                recommendation=recommendation
            )
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing forgery detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing forgery detection: {str(e)}"
        )


@router.post(
    "/analyze-complete",
    response_model=UnifiedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete analysis: Damage + Forgery detection",
    description="""
    Comprehensive image analysis combining both damage detection and forgery detection.
    
    **Analysis Pipeline:**
    1. Forgery Detection - Verify image authenticity
    2. Damage Detection - Find vehicle damages if authentic
    3. Risk Fusion - Combine signals for final assessment
    
    **Returns:**
    - Complete forgery analysis
    - Complete damage analysis
    - Unified risk assessment with recommendation
    
    **Best for:**
    - Complete claim image validation
    - End-to-end fraud detection
    - Automated claim processing workflows
    """
)
async def analyze_complete(
    file: UploadFile = File(..., description="Claim image for complete analysis"),
    parts_conf: float = 0.25,
    damage_conf: float = 0.25
):
    """
    Complete analysis: Forgery + Damage detection
    
    Args:
        file: Uploaded image file
        parts_conf: Parts detection confidence threshold
        damage_conf: Damage detection confidence threshold
    
    Returns:
        Unified analysis with forgery, damage, and risk assessment
    """
    logger.info(f"Received complete analysis request for file: {file.filename}")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({file_size_mb:.1f}MB) exceeds limit ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Running complete analysis: {tmp_path}")
        
        try:
            # Get both detectors
            forgery_detector = get_forgery_detector()
            damage_detector = get_damage_detector()
            
            # Run forgery detection
            logger.info("Step 1/3: Forgery detection...")
            forgery_result = forgery_detector.analyze_image_as_dict(tmp_path)
            
            # Run damage detection
            logger.info("Step 2/3: Damage detection...")
            damage_result = damage_detector.detect_damage(
                image_path=tmp_path,
                parts_conf=parts_conf,
                damage_conf=damage_conf,
                return_visualization=False
            )
            
            # Fuse risk scores
            logger.info("Step 3/3: Risk fusion...")
            risk_factors = []
            risk_score = 0.0
            
            # Forgery component (60% weight)
            forgery_prob = forgery_result["forgery_prob"]
            if forgery_result["is_forged"]:
                risk_score += 0.6
                risk_factors.append(
                    f"Image forgery detected ({forgery_prob:.1%} confidence)"
                )
            else:
                risk_score += forgery_prob * 0.3
                if forgery_prob > 0.3:
                    risk_factors.append(
                        f"Moderate forgery signals ({forgery_prob:.1%})"
                    )
            
            # Damage component (40% weight)
            damage_risk = damage_result["risk_assessment"]["risk_score"]
            risk_score += damage_risk * 0.4
            
            if damage_result["risk_assessment"]["risk_level"] in ["HIGH", "MEDIUM"]:
                risk_factors.append(
                    f"{damage_result['risk_assessment']['risk_level']} damage risk"
                )
            
            # Special case: Forged image with damages
            if forgery_result["is_forged"] and damage_result["summary"]["total_damages"] > 0:
                risk_score = min(risk_score + 0.15, 1.0)
                risk_factors.append("Suspicious: Forged image showing damages")
            
            # Determine final risk level
            if risk_score >= 0.75:
                final_risk_level = "HIGH"
                recommendation = "REJECT - High fraud probability"
            elif risk_score >= 0.45:
                final_risk_level = "MEDIUM"
                recommendation = "REVIEW - Manual inspection required"
            else:
                final_risk_level = "LOW"
                recommendation = "APPROVE - Low fraud risk"
            
            final_risk_assessment = {
                "final_risk_level": final_risk_level,
                "final_risk_score": round(risk_score, 3),
                "risk_factors": risk_factors,
                "recommendation": recommendation
            }
            
            logger.success(
                f"Complete analysis done: risk={final_risk_level}, "
                f"score={risk_score:.3f}, recommendation={recommendation}"
            )
            
            return UnifiedAnalysisResponse(
                status="success",
                forgery_analysis=forgery_result,
                damage_analysis=damage_result,
                final_risk_assessment=final_risk_assessment
            )
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in complete analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in complete analysis: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check CV engine health",
    description="Verify that CV models are loaded and ready for inference"
)
async def health_check():
    """
    Health check for CV detection service
    
    Returns:
        Health status with model loading information
    """
    try:
        # Check both detectors
        models_status = {
            "damage_detector": False,
            "forgery_detector": False
        }
        
        try:
            damage_detector = get_damage_detector()
            models_status["damage_detector"] = True
        except Exception as e:
            logger.warning(f"Damage detector not available: {e}")
        
        try:
            forgery_detector = get_forgery_detector()
            models_status["forgery_detector"] = True
        except Exception as e:
            logger.warning(f"Forgery detector not available: {e}")
        
        all_loaded = all(models_status.values())
        
        return HealthResponse(
            status="healthy" if all_loaded else "degraded",
            models_loaded=models_status,
            device=settings.CV_DEVICE,
            message="CV detection service is operational" if all_loaded 
                    else "Some models failed to load"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            models_loaded={"damage_detector": False, "forgery_detector": False},
            device="unknown",
            message=f"Error: {str(e)}"
        )


@router.get(
    "/info",
    summary="Get CV model information",
    description="Get information about loaded models and supported classes"
)
async def model_info():
    """
    Get information about CV models
    
    Returns:
        Model configuration and class information
    """
    return {
        "models": {
            "parts_segmentation": {
                "architecture": "YOLO11n-seg",
                "classes": DamageDetector.PARTS_CLASSES,
                "num_classes": len(DamageDetector.PARTS_CLASSES)
            },
            "damage_detection": {
                "architecture": "YOLO11m",
                "classes": DamageDetector.DAMAGE_CLASSES,
                "num_classes": len(DamageDetector.DAMAGE_CLASSES)
            },
            "severity_classification": {
                "architecture": "EfficientNet-B0",
                "classes": DamageDetector.SEVERITY_CLASSES,
                "num_classes": len(DamageDetector.SEVERITY_CLASSES)
            },
            "forgery_detection": {
                "architecture": "ResNet50 + ELA + Noise Analysis",
                "classes": ["AUTHENTIC", "FORGED"],
                "num_classes": 2,
                "validation_accuracy": "83.6%",
                "threshold": 0.55
            }
        },
        "settings": {
            "parts_confidence_threshold": settings.PARTS_CONFIDENCE_THRESHOLD,
            "damage_confidence_threshold": settings.DAMAGE_CONFIDENCE_THRESHOLD,
            "max_image_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "device": settings.CV_DEVICE
        },
        "endpoints": {
            "damage_only": "/api/cv/detect",
            "forgery_only": "/api/cv/detect-forgery",
            "complete_analysis": "/api/cv/analyze-complete",
            "health_check": "/api/cv/health",
            "model_info": "/api/cv/info"
        }
    }
