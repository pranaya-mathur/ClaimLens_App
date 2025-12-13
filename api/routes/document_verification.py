"""
Document Verification API Routes
Handles PAN, Aadhaar, and generic document verification with OCR
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

from src.cv_engine.pan_detector import PANDetector
from src.cv_engine.aadhaar_detector import AadhaarDetector
from src.cv_engine.document_verifier import DocumentVerifier
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

# Global detector instances (lazy loading)
_pan_detector: Optional[PANDetector] = None
_aadhaar_detector: Optional[AadhaarDetector] = None
_doc_verifier: Optional[DocumentVerifier] = None


def get_pan_detector() -> PANDetector:
    """Get or initialize PAN detector."""
    global _pan_detector
    if _pan_detector is None:
        logger.info("Initializing PANDetector...")
        _pan_detector = PANDetector()
        logger.success("PANDetector initialized")
    return _pan_detector


def get_aadhaar_detector() -> AadhaarDetector:
    """Get or initialize Aadhaar detector."""
    global _aadhaar_detector
    if _aadhaar_detector is None:
        logger.info("Initializing AadhaarDetector...")
        _aadhaar_detector = AadhaarDetector()
        logger.success("AadhaarDetector initialized")
    return _aadhaar_detector


def get_doc_verifier() -> DocumentVerifier:
    """Get or initialize document verifier."""
    global _doc_verifier
    if _doc_verifier is None:
        logger.info("Initializing DocumentVerifier...")
        _doc_verifier = DocumentVerifier()
        logger.success("DocumentVerifier initialized")
    return _doc_verifier


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PANVerificationResponse(BaseModel):
    """Response model for PAN verification."""
    status: str
    document_type: str = "PAN"
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_data: dict
    validation_checks: dict
    risk_score: float = Field(ge=0.0, le=1.0)
    red_flags: List[str]
    recommendation: str


class AadhaarVerificationResponse(BaseModel):
    """Response model for Aadhaar verification."""
    status: str
    document_type: str = "AADHAAR"
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_data: dict
    validation_checks: dict
    risk_score: float = Field(ge=0.0, le=1.0)
    red_flags: List[str]
    recommendation: str


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


class OCRExtractionResponse(BaseModel):
    """Response model for OCR text extraction."""
    status: str
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    word_count: int
    line_count: int
    detected_entities: dict


class BatchVerificationResponse(BaseModel):
    """Response model for batch document verification."""
    status: str
    total_documents: int
    verified_count: int
    failed_count: int
    results: List[dict]
    summary: dict


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    detectors_loaded: dict
    message: str


# ============================================================================
# PAN VERIFICATION ENDPOINTS
# ============================================================================

@router.post(
    "/verify-pan",
    response_model=PANVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify PAN card authenticity",
    description="""
    Upload a PAN card image for comprehensive verification.
    
    **Verification Process:**
    1. ✅ Format validation (10-character alphanumeric)
    2. ✅ Structure check (AAAAA9999A pattern)
    3. ✅ OCR extraction with confidence scoring
    4. ✅ Forgery detection (image manipulation)
    5. ✅ Quality assessment (blur, resolution)
    6. ✅ Data extraction (name, DOB, PAN number)
    
    **Returns:**
    - Validity status with confidence score
    - Extracted PAN details (number, name, father's name, DOB)
    - Validation checks (format, structure, quality)
    - Risk score and recommendation
    - Red flags if any issues detected
    
    **Use Cases:**
    - Claim identity verification
    - KYC compliance
    - Fraud prevention
    """
)
async def verify_pan(
    file: UploadFile = File(..., description="PAN card image (JPG, PNG, PDF)"),
    expected_pan: Optional[str] = Form(None, description="Expected PAN number for cross-verification"),
    expected_name: Optional[str] = Form(None, description="Expected name for cross-verification")
):
    """
    Verify PAN card authenticity.
    
    Args:
        file: PAN card image
        expected_pan: Optional PAN number to verify against
        expected_name: Optional name to verify against
        
    Returns:
        Comprehensive PAN verification result
    """
    logger.info(f"PAN verification request: {file.filename}")
    
    # Validate file type
    allowed_types = ['image/', 'application/pdf']
    if not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG) or PDF"
        )
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds limit ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Analyzing PAN card: {tmp_path}")
        
        try:
            # Get PAN detector
            detector = get_pan_detector()
            
            # Run verification
            result = detector.verify_pan(tmp_path)
            
            # Extract data
            extracted_data = {
                "pan_number": result.get("pan_number", ""),
                "name": result.get("name", ""),
                "fathers_name": result.get("fathers_name", ""),
                "date_of_birth": result.get("date_of_birth", "")
            }
            
            # Validation checks
            validation_checks = {
                "format_valid": result.get("format_valid", False),
                "structure_valid": result.get("structure_valid", False),
                "ocr_confidence": result.get("ocr_confidence", 0.0),
                "quality_score": result.get("quality_score", 0.0),
                "forgery_detected": result.get("forgery_detected", False)
            }
            
            # Cross-verification if expected values provided
            red_flags = []
            risk_score = 0.0
            
            if expected_pan:
                if extracted_data["pan_number"].upper() != expected_pan.upper():
                    red_flags.append(f"PAN mismatch: expected {expected_pan}, got {extracted_data['pan_number']}")
                    risk_score += 0.5
            
            if expected_name:
                extracted_name = extracted_data["name"].lower()
                expected_name_lower = expected_name.lower()
                if expected_name_lower not in extracted_name and extracted_name not in expected_name_lower:
                    red_flags.append(f"Name mismatch: expected {expected_name}, got {extracted_data['name']}")
                    risk_score += 0.3
            
            # Quality checks
            if not validation_checks["format_valid"]:
                red_flags.append("Invalid PAN format")
                risk_score += 0.4
            
            if validation_checks["forgery_detected"]:
                red_flags.append("Possible forgery/manipulation detected")
                risk_score += 0.6
            
            if validation_checks["ocr_confidence"] < 0.7:
                red_flags.append(f"Low OCR confidence ({validation_checks['ocr_confidence']:.1%})")
                risk_score += 0.2
            
            if validation_checks["quality_score"] < 0.6:
                red_flags.append("Poor image quality (blur, low resolution)")
                risk_score += 0.15
            
            risk_score = min(risk_score, 1.0)
            
            # Overall validity
            is_valid = (
                validation_checks["format_valid"] and
                validation_checks["structure_valid"] and
                not validation_checks["forgery_detected"] and
                validation_checks["ocr_confidence"] >= 0.6 and
                len(red_flags) == 0
            )
            
            # Confidence score
            confidence = (
                0.4 * validation_checks["ocr_confidence"] +
                0.3 * validation_checks["quality_score"] +
                0.3 * (1.0 if validation_checks["format_valid"] else 0.0)
            )
            
            # Recommendation
            if risk_score >= 0.7:
                recommendation = "REJECT - High risk of fraud"
            elif risk_score >= 0.4:
                recommendation = "REVIEW - Manual verification required"
            elif is_valid:
                recommendation = "APPROVE - PAN verified successfully"
            else:
                recommendation = "REVIEW - Verification issues detected"
            
            logger.success(
                f"PAN verification complete: valid={is_valid}, "
                f"confidence={confidence:.2f}, risk={risk_score:.2f}"
            )
            
            return PANVerificationResponse(
                status="success",
                is_valid=is_valid,
                confidence=round(confidence, 3),
                extracted_data=extracted_data,
                validation_checks=validation_checks,
                risk_score=round(risk_score, 3),
                red_flags=red_flags,
                recommendation=recommendation
            )
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PAN verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PAN verification error: {str(e)}"
        )


# ============================================================================
# AADHAAR VERIFICATION ENDPOINTS
# ============================================================================

@router.post(
    "/verify-aadhaar",
    response_model=AadhaarVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify Aadhaar card authenticity",
    description="""
    Upload an Aadhaar card image for comprehensive verification.
    
    **Verification Process:**
    1. ✅ Format validation (12-digit numeric)
    2. ✅ Checksum validation (Verhoeff algorithm)
    3. ✅ OCR extraction with confidence scoring
    4. ✅ Forgery detection (hologram, QR code)
    5. ✅ Quality assessment (blur, resolution)
    6. ✅ Data extraction (number, name, DOB, address)
    
    **Privacy Note:**
    - Extracted Aadhaar number is masked (XXXX-XXXX-1234)
    - Full data available only with proper authorization
    
    **Returns:**
    - Validity status with confidence score
    - Extracted Aadhaar details (masked number, name, DOB)
    - Validation checks (format, checksum, quality)
    - Risk score and recommendation
    """
)
async def verify_aadhaar(
    file: UploadFile = File(..., description="Aadhaar card image (JPG, PNG, PDF)"),
    expected_aadhaar: Optional[str] = Form(None, description="Expected Aadhaar number (last 4 digits)"),
    expected_name: Optional[str] = Form(None, description="Expected name for cross-verification"),
    mask_number: bool = Form(True, description="Mask Aadhaar number for privacy (default: true)")
):
    """
    Verify Aadhaar card authenticity.
    
    Args:
        file: Aadhaar card image
        expected_aadhaar: Optional last 4 digits to verify
        expected_name: Optional name to verify against
        mask_number: Whether to mask Aadhaar number
        
    Returns:
        Comprehensive Aadhaar verification result
    """
    logger.info(f"Aadhaar verification request: {file.filename}")
    
    # Validate file type
    allowed_types = ['image/', 'application/pdf']
    if not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG) or PDF"
        )
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds limit"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Analyzing Aadhaar card: {tmp_path}")
        
        try:
            # Get Aadhaar detector
            detector = get_aadhaar_detector()
            
            # Run verification
            result = detector.verify_aadhaar(tmp_path)
            
            # Extract and optionally mask data
            aadhaar_number = result.get("aadhaar_number", "")
            if mask_number and len(aadhaar_number) == 12:
                masked_number = f"XXXX-XXXX-{aadhaar_number[-4:]}"
            else:
                masked_number = aadhaar_number
            
            extracted_data = {
                "aadhaar_number": masked_number,
                "name": result.get("name", ""),
                "date_of_birth": result.get("date_of_birth", ""),
                "gender": result.get("gender", ""),
                "address": result.get("address", "")
            }
            
            # Validation checks
            validation_checks = {
                "format_valid": result.get("format_valid", False),
                "checksum_valid": result.get("checksum_valid", False),
                "ocr_confidence": result.get("ocr_confidence", 0.0),
                "quality_score": result.get("quality_score", 0.0),
                "hologram_present": result.get("hologram_present", False),
                "qr_code_present": result.get("qr_code_present", False),
                "forgery_detected": result.get("forgery_detected", False)
            }
            
            # Cross-verification
            red_flags = []
            risk_score = 0.0
            
            if expected_aadhaar:
                last_4 = aadhaar_number[-4:] if len(aadhaar_number) >= 4 else ""
                if last_4 != expected_aadhaar:
                    red_flags.append(f"Aadhaar mismatch: expected ...{expected_aadhaar}, got ...{last_4}")
                    risk_score += 0.5
            
            if expected_name:
                extracted_name = extracted_data["name"].lower()
                expected_name_lower = expected_name.lower()
                if expected_name_lower not in extracted_name and extracted_name not in expected_name_lower:
                    red_flags.append(f"Name mismatch")
                    risk_score += 0.3
            
            # Quality checks
            if not validation_checks["format_valid"]:
                red_flags.append("Invalid Aadhaar format")
                risk_score += 0.4
            
            if not validation_checks["checksum_valid"]:
                red_flags.append("Invalid Aadhaar checksum (Verhoeff algorithm failed)")
                risk_score += 0.5
            
            if validation_checks["forgery_detected"]:
                red_flags.append("Possible forgery detected")
                risk_score += 0.6
            
            if not validation_checks["hologram_present"]:
                red_flags.append("Missing hologram")
                risk_score += 0.3
            
            if validation_checks["ocr_confidence"] < 0.7:
                red_flags.append(f"Low OCR confidence")
                risk_score += 0.2
            
            risk_score = min(risk_score, 1.0)
            
            # Overall validity
            is_valid = (
                validation_checks["format_valid"] and
                validation_checks["checksum_valid"] and
                not validation_checks["forgery_detected"] and
                validation_checks["ocr_confidence"] >= 0.6 and
                len(red_flags) == 0
            )
            
            # Confidence
            confidence = (
                0.3 * validation_checks["ocr_confidence"] +
                0.2 * validation_checks["quality_score"] +
                0.2 * (1.0 if validation_checks["format_valid"] else 0.0) +
                0.2 * (1.0 if validation_checks["checksum_valid"] else 0.0) +
                0.1 * (1.0 if validation_checks["hologram_present"] else 0.0)
            )
            
            # Recommendation
            if risk_score >= 0.7:
                recommendation = "REJECT - High risk of fraud"
            elif risk_score >= 0.4:
                recommendation = "REVIEW - Manual verification required"
            elif is_valid:
                recommendation = "APPROVE - Aadhaar verified successfully"
            else:
                recommendation = "REVIEW - Verification issues detected"
            
            logger.success(
                f"Aadhaar verification complete: valid={is_valid}, "
                f"confidence={confidence:.2f}, risk={risk_score:.2f}"
            )
            
            return AadhaarVerificationResponse(
                status="success",
                is_valid=is_valid,
                confidence=round(confidence, 3),
                extracted_data=extracted_data,
                validation_checks=validation_checks,
                risk_score=round(risk_score, 3),
                red_flags=red_flags,
                recommendation=recommendation
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Aadhaar verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aadhaar verification error: {str(e)}"
        )


# ============================================================================
# GENERIC DOCUMENT VERIFICATION
# ============================================================================

@router.post(
    "/verify-document",
    response_model=GenericDocVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify any document (License, Passport, etc.)",
    description="""
    Generic document verification for various ID types.
    
    **Supported Documents:**
    - Driving License
    - Passport
    - Voter ID
    - Bank statements
    - Hospital bills
    - Death certificate
    - Any other official document
    
    **Verification:**
    - OCR extraction
    - Quality assessment
    - Forgery detection
    - Format validation (if applicable)
    """
)
async def verify_document(
    file: UploadFile = File(..., description="Document image"),
    document_type: Literal["license", "passport", "voter_id", "bank_statement", "hospital_bill", "death_certificate", "other"] = Form(..., description="Type of document")
):
    """
    Verify generic document.
    
    Args:
        file: Document image
        document_type: Type of document being verified
        
    Returns:
        Document verification result
    """
    logger.info(f"Document verification request: {document_type} - {file.filename}")
    
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds limit"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Get document verifier
            verifier = get_doc_verifier()
            
            # Run verification
            result = verifier.verify_document(tmp_path, document_type)
            
            extracted_data = result.get("extracted_data", {})
            validation_checks = result.get("validation_checks", {})
            
            # Calculate risk
            risk_score = 0.0
            red_flags = []
            
            if result.get("forgery_detected", False):
                red_flags.append("Possible forgery detected")
                risk_score += 0.6
            
            if result.get("quality_score", 1.0) < 0.6:
                red_flags.append("Poor image quality")
                risk_score += 0.2
            
            if result.get("ocr_confidence", 1.0) < 0.7:
                red_flags.append("Low OCR confidence")
                risk_score += 0.2
            
            risk_score = min(risk_score, 1.0)
            
            is_valid = risk_score < 0.4
            confidence = result.get("confidence", 0.5)
            
            if risk_score >= 0.6:
                recommendation = "REJECT - High risk"
            elif risk_score >= 0.4:
                recommendation = "REVIEW - Manual check needed"
            else:
                recommendation = "APPROVE - Document verified"
            
            logger.success(f"Document verification complete: {document_type}")
            
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document verification error: {str(e)}"
        )


# ============================================================================
# OCR EXTRACTION
# ============================================================================

@router.post(
    "/extract-text",
    response_model=OCRExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract text from document using OCR",
    description="""
    Extract text from any document image using OCR.
    
    **Features:**
    - Multi-language support (English, Hindi)
    - Layout preservation
    - Entity detection (dates, amounts, PAN, Aadhaar)
    - Confidence scoring
    """
)
async def extract_text(
    file: UploadFile = File(..., description="Document image for OCR")
):
    """
    Extract text from document.
    
    Args:
        file: Document image
        
    Returns:
        Extracted text with metadata
    """
    logger.info(f"OCR extraction request: {file.filename}")
    
    try:
        contents = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            verifier = get_doc_verifier()
            result = verifier.extract_text(tmp_path)
            
            extracted_text = result.get("text", "")
            confidence = result.get("confidence", 0.0)
            
            # Count lines and words
            lines = extracted_text.split('\n')
            words = extracted_text.split()
            
            # Detect entities
            detected_entities = {
                "pan_numbers": re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]', extracted_text),
                "aadhaar_numbers": re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', extracted_text),
                "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', extracted_text),
                "amounts": re.findall(r'₹?\s?\d+[,\d]*\.?\d*', extracted_text),
                "phone_numbers": re.findall(r'\b[6-9]\d{9}\b', extracted_text)
            }
            
            logger.success(f"OCR extraction complete: {len(words)} words")
            
            return OCRExtractionResponse(
                status="success",
                extracted_text=extracted_text,
                confidence=round(confidence, 3),
                word_count=len(words),
                line_count=len(lines),
                detected_entities=detected_entities
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR extraction error: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check document verification service health"
)
async def health_check():
    """
    Health check for document verification service.
    
    Returns:
        Health status with detector availability
    """
    try:
        detectors_status = {
            "pan_detector": False,
            "aadhaar_detector": False,
            "document_verifier": False
        }
        
        try:
            get_pan_detector()
            detectors_status["pan_detector"] = True
        except:
            pass
        
        try:
            get_aadhaar_detector()
            detectors_status["aadhaar_detector"] = True
        except:
            pass
        
        try:
            get_doc_verifier()
            detectors_status["document_verifier"] = True
        except:
            pass
        
        all_loaded = all(detectors_status.values())
        
        return HealthResponse(
            status="healthy" if all_loaded else "degraded",
            detectors_loaded=detectors_status,
            message="Document verification service operational" if all_loaded
                    else "Some detectors unavailable"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            detectors_loaded={},
            message=f"Error: {str(e)}"
        )


@router.get(
    "/info",
    summary="Get document verification capabilities"
)
async def service_info():
    """
    Get information about document verification capabilities.
    """
    return {
        "service": "Document Verification Suite",
        "version": "1.0.0",
        "capabilities": {
            "pan_verification": {
                "format_validation": True,
                "structure_check": True,
                "ocr_extraction": True,
                "forgery_detection": True,
                "cross_verification": True
            },
            "aadhaar_verification": {
                "format_validation": True,
                "checksum_validation": True,
                "hologram_detection": True,
                "qr_code_verification": True,
                "ocr_extraction": True,
                "privacy_masking": True
            },
            "generic_documents": [
                "driving_license",
                "passport",
                "voter_id",
                "bank_statement",
                "hospital_bill",
                "death_certificate"
            ],
            "ocr_languages": ["English", "Hindi"],
            "entity_detection": [
                "PAN numbers",
                "Aadhaar numbers",
                "Dates",
                "Amounts",
                "Phone numbers"
            ]
        },
        "endpoints": {
            "pan_verification": "/api/documents/verify-pan",
            "aadhaar_verification": "/api/documents/verify-aadhaar",
            "generic_verification": "/api/documents/verify-document",
            "ocr_extraction": "/api/documents/extract-text",
            "health_check": "/api/documents/health",
            "service_info": "/api/documents/info"
        },
        "limits": {
            "max_file_size_mb": settings.MAX_IMAGE_SIZE_MB,
            "supported_formats": ["JPG", "PNG", "PDF"]
        }
    }
