"""
Computer Vision Detection API Routes
Handles image uploads and damage detection
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
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

# Global detector instance (lazy loading)
_detector: Optional[DamageDetector] = None


def get_detector() -> DamageDetector:
    """
    Get or initialize the damage detector instance
    Lazy loading to avoid loading models on import
    """
    global _detector
    if _detector is None:
        logger.info("Initializing DamageDetector...")
        _detector = DamageDetector(
            parts_model_path=settings.PARTS_MODEL_PATH,
            damage_model_path=settings.DAMAGE_MODEL_PATH,
            severity_model_path=settings.SEVERITY_MODEL_PATH,
            device=settings.CV_DEVICE
        )
        logger.success("DamageDetector initialized")
    return _detector


class DetectionResponse(BaseModel):
    """Response model for damage detection"""
    status: str
    parts_detected: list
    damages_detected: list
    summary: dict
    risk_assessment: dict


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
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
    logger.info(f"Received detection request for file: {file.filename}")
    
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
            detector = get_detector()
            
            # Run detection
            results = detector.detect_damage(
                image_path=tmp_path,
                parts_conf=parts_conf,
                damage_conf=damage_conf,
                return_visualization=False
            )
            
            logger.success(f"Detection complete: {len(results['damages_detected'])} damages found")
            
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
        # Try to get detector (will initialize if not already)
        detector = get_detector()
        
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            device=detector.device,
            message="CV detection service is operational"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
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
            }
        },
        "settings": {
            "parts_confidence_threshold": settings.PARTS_CONFIDENCE_THRESHOLD,
            "damage_confidence_threshold": settings.DAMAGE_CONFIDENCE_THRESHOLD,
            "max_image_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "device": settings.CV_DEVICE
        }
    }
