"""Document Verification API Routes
Handles PAN, Aadhaar, and generic document verification
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import os
import tempfile
from pathlib import Path
from loguru import logger
import re

from src.cv_engine.pan_detector import PANForgeryDetector
from src.cv_engine.aadhaar_detector import AadhaarForgeryDetector
from src.cv_engine.document_verifier import DocumentVerifier
from src.cv_engine.forgery_detector import ForgeryDetector  # 🔥 ADDED FOR GENERIC DOCS
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

# Global detector instances (lazy loading)
_pan_detector: Optional[PANForgeryDetector] = None
_aadhaar_detector: Optional[AadhaarForgeryDetector] = None
_doc_verifier: Optional[DocumentVerifier] = None
_forgery_detector: Optional[ForgeryDetector] = None  # 🔥 ADDED


def get_pan_detector() -> PANForgeryDetector:
    """Get or initialize PAN detector."""
    global _pan_detector
    if _pan_detector is None:
        logger.info("Initializing PANForgeryDetector...")
        _pan_detector = PANForgeryDetector()
        logger.success("PANForgeryDetector initialized")
    return _pan_detector


def get_aadhaar_detector() -> AadhaarForgeryDetector:
    """Get or initialize Aadhaar detector."""
    global _aadhaar_detector
    if _aadhaar_detector is None:
        logger.info("Initializing AadhaarForgeryDetector...")
        _aadhaar_detector = AadhaarForgeryDetector()
        logger.success("AadhaarForgeryDetector initialized")
    return _aadhaar_detector


def get_doc_verifier() -> DocumentVerifier:
    """Get or initialize document verifier."""
    global _doc_verifier
    if _doc_verifier is None:
        logger.info("Initializing DocumentVerifier...")
        _doc_verifier = DocumentVerifier()
        logger.success("DocumentVerifier initialized")
    return _doc_verifier


def get_forgery_detector() -> ForgeryDetector:
    """Get or initialize generic forgery detector."""
    global _forgery_detector
    if _forgery_detector is None:
        logger.info("Initializing ForgeryDetector (Generic)...")
        device = settings.CV_DEVICE if hasattr(settings, 'CV_DEVICE') else 'cpu'
        _forgery_detector = ForgeryDetector(device=device)
        logger.success("ForgeryDetector initialized (ResNet50 + ELA + Noise Analysis)")
    return _forgery_detector


class GenericDocVerificationResponse(BaseModel):
    """Response model for generic document verification."""
    status: str
    document_type: str
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_data: dict
    validation_checks: dict
    risk_score: float = Field(ge=0.0, le=1.0)
    red_flags: List[str]
    recommendation: str


@router.post(
    "/verify-document",
    response_model=GenericDocVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify any document (License, Passport, etc.)",
    description="""
    Generic document verification using ResNet50 + ELA forgery detection.
    
    **Supported Documents:**
    - Driving License
    - Passport  
    - Voter ID
    - Bank statements
    - Hospital bills
    - Death certificate
    - Any other official document
    
    **Verification Process:**
    1. ✅ ResNet50 CNN forgery probability
    2. ✅ ELA (Error Level Analysis) tampering detection
    3. ✅ Noise variation analysis
    4. ✅ Risk scoring and recommendation
    """
)
async def verify_document(
    file: UploadFile = File(..., description="Document image"),
    document_type: Literal["license", "passport", "voter_id", "bank_statement", "hospital_bill", "death_certificate", "other"] = Form(..., description="Type of document")
):
    """
    Verify generic document using ForgeryDetector.
    
    Args:
        file: Document image
        document_type: Type of document being verified
        
    Returns:
        Document verification result with forgery analysis
    """
    logger.info(f"📝 Generic document verification: {document_type} - {file.filename}")
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds limit ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # 🔥 USE FORGERY DETECTOR FOR GENERIC DOCUMENTS
            detector = get_forgery_detector()
            
            # Run generic forgery analysis
            logger.info(f"🔍 Analyzing {document_type} with ForgeryDetector (ResNet50 + ELA)...")
            result = detector.analyze_image(tmp_path)
            
            # Extract data (placeholder)
            extracted_data = {
                "document_type": document_type.upper().replace("_", " "),
                "file_name": file.filename
            }
            
            # 🔥 DETAILED VALIDATION CHECKS
            validation_checks = {
                "format_valid": True,  # File loaded successfully
                "quality_score": round(1.0 - result.forgery_prob, 3),  # Inverse of forgery prob
                "forgery_detected": result.is_forged,
                "cnn_forgery_probability": round(result.forgery_prob, 3),
                "ela_intensity_score": round(result.ela_score, 3),
                "noise_variation": round(result.noise_variation, 3),
                "threshold_used": round(result.threshold, 3)
            }
            
            # 🔥 CALCULATE RISK SCORE
            red_flags = []
            risk_score = 0.0
            
            # CNN forgery probability check
            if result.forgery_prob >= 0.7:
                red_flags.append(f"🚨 Very high forgery probability ({result.forgery_prob:.1%})")
                risk_score += 0.7
            elif result.forgery_prob >= 0.5:
                red_flags.append(f"⚠️ High forgery probability ({result.forgery_prob:.1%})")
                risk_score += 0.5
            elif result.forgery_prob >= 0.3:
                red_flags.append(f"🟡 Moderate forgery signals ({result.forgery_prob:.1%})")
                risk_score += 0.3
            
            # ELA anomaly detection
            if result.ela_score > 30.0:
                red_flags.append(f"🔍 High ELA intensity detected ({result.ela_score:.1f}) - possible tampering")
                risk_score += 0.2
            elif result.ela_score > 20.0:
                red_flags.append(f"🟡 Moderate ELA signals ({result.ela_score:.1f})")
                risk_score += 0.1
            
            # Noise analysis
            if result.noise_variation > 50.0:
                red_flags.append(f"🔊 High noise variation ({result.noise_variation:.1f}) - quality concerns")
                risk_score += 0.1
            
            # Cap risk score
            risk_score = min(risk_score, 1.0)
            
            # 🔥 OVERALL VALIDITY
            is_valid = not result.is_forged and result.forgery_prob < 0.4
            
            # Confidence from model
            confidence = 1.0 - result.forgery_prob  # Authenticity confidence
            
            # 🔥 RECOMMENDATION
            if risk_score >= 0.7:
                recommendation = "REJECT - High fraud risk detected by AI"
            elif risk_score >= 0.5:
                recommendation = "REVIEW - Manual verification strongly recommended"
            elif risk_score >= 0.3:
                recommendation = "REVIEW - Additional checks suggested"
            elif is_valid:
                recommendation = "APPROVE - Document appears authentic"
            else:
                recommendation = "REVIEW - Verification issues detected"
            
            logger.success(
                f"✅ {document_type} verification complete: valid={is_valid}, "
                f"confidence={confidence:.2f}, risk={risk_score:.2f}, "
                f"forgery_prob={result.forgery_prob:.3f}"
            )
            
            return GenericDocVerificationResponse(
                status="success",
                document_type=document_type.upper(),
                is_valid=is_valid,
                confidence=round(confidence, 3),
                extracted_data=extracted_data,
                validation_checks=validation_checks,
                risk_score=round(risk_score, 3),
                red_flags=red_flags,
                recommendation=recommendation
            )
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document verification failed for {document_type}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document verification error: {str(e)}"
        )


@router.get(
    "/info",
    summary="Get document verification capabilities"
)
async def service_info():
    """Get information about document verification capabilities."""
    return {
        "service": "Document Verification Suite",
        "version": "2.0.0",
        "capabilities": {
            "pan_verification": {
                "detector": "PANForgeryDetector",
                "accuracy": "99.19%",
                "auc": "0.9996",
                "technology": "ResNet50 + ELA"
            },
            "aadhaar_verification": {
                "detector": "AadhaarForgeryDetector",
                "accuracy": "99.62%",
                "auc": "0.9999",
                "technology": "ResNet50"
            },
            "generic_verification": {
                "detector": "ForgeryDetector",
                "technology": "ResNet50 + ELA + Noise Analysis",
                "supported_documents": [
                    "driving_license",
                    "passport",
                    "voter_id",
                    "bank_statement",
                    "hospital_bill",
                    "death_certificate",
                    "any_official_document"
                ],
                "features": [
                    "CNN forgery probability",
                    "ELA tampering detection",
                    "Noise variation analysis",
                    "Quality assessment"
                ]
            }
        },
        "endpoints": {
            "pan_verification": "/api/documents/verify-pan",
            "aadhaar_verification": "/api/documents/verify-aadhaar",
            "generic_verification": "/api/documents/verify-document",
            "service_info": "/api/documents/info"
        },
        "limits": {
            "max_file_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "supported_formats": ["JPG", "PNG", "PDF"]
        }
    }
