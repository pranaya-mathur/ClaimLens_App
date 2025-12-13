"""
Unified Document Verification Orchestrator

Centralized document verification supporting PAN and Aadhaar cards
with optional dual-check cross-validation.

Usage:
    from src.cv_engine import DocumentVerifier
    
    verifier = DocumentVerifier()
    result = verifier.verify("pan_card.jpg", doc_type="PAN")
    print(result["verdict"])  # "CLEAN" or "FORGED"
    
    # With dual-check for critical cases
    result = verifier.verify("aadhaar.jpg", doc_type="AADHAAR", dual_check=True)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Union, Optional
from loguru import logger

from .aadhaar_detector import AadhaarForgeryDetector, AadhaarVerificationResult
from .pan_detector import PANForgeryDetector, PANVerificationResult

PathLike = Union[str, Path]


@dataclass
class DocumentVerificationResult:
    """Unified document verification result with optional dual-check data"""
    
    document_type: str
    image_path: str
    verdict: str
    confidence: float
    primary_result: Dict
    dual_check_enabled: bool
    secondary_result: Optional[Dict] = None
    consensus_verdict: Optional[str] = None
    consensus_confidence: Optional[float] = None
    agreement: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return asdict(self)
    
    def is_forged(self) -> bool:
        """Check if document is forged (uses consensus if dual-check enabled)"""
        if self.dual_check_enabled and self.consensus_verdict:
            return self.consensus_verdict in ["FORGED", "SUSPICIOUS"]
        return self.verdict in ["FORGED", "SUSPICIOUS"]
    
    def is_suspicious(self) -> bool:
        """Check if dual-check disagreement indicates suspicious document"""
        return self.verdict == "SUSPICIOUS"


class DocumentVerifier:
    """
    Unified document verification system with dual-check capability.
    
    Supports:
    - PAN card verification (99.19% accuracy, AUC 0.9996)
    - Aadhaar card verification (99.62% accuracy, AUC 0.9999)
    - Dual-check mode for cross-validation
    - Batch processing
    
    Dual-Check Logic:
    - Both agree CLEAN → CLEAN (high confidence)
    - Both agree FORGED → FORGED (high confidence)
    - Disagree → SUSPICIOUS (requires manual review)
    
    Example:
        >>> verifier = DocumentVerifier()
        >>> 
        >>> # Standard verification
        >>> result = verifier.verify("pan.jpg", "PAN")
        >>> 
        >>> # High-stakes dual verification
        >>> result = verifier.verify("aadhaar.jpg", "AADHAAR", dual_check=True)
        >>> if result.is_suspicious():
        ...     print("Detectors disagree - manual review required")
    """
    
    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize document verifier with both detectors.
        
        Args:
            device: Device for inference ("cuda" or "cpu")
        """
        logger.info("Initializing DocumentVerifier")
        
        try:
            self.aadhaar_detector = AadhaarForgeryDetector(device=device)
            logger.success("Aadhaar detector loaded")
        except Exception as e:
            logger.warning(f"Aadhaar detector not available: {e}")
            self.aadhaar_detector = None
        
        try:
            self.pan_detector = PANForgeryDetector(device=device)
            logger.success("PAN detector loaded")
        except Exception as e:
            logger.warning(f"PAN detector not available: {e}")
            self.pan_detector = None
        
        if not self.aadhaar_detector and not self.pan_detector:
            raise RuntimeError(
                "No detectors available. Please ensure model files are present:"
                "\n- models/aadhaar_balanced_model.pth"
                "\n- models/resnet50_finetuned_after_strong_forgeries.pth"
            )
        
        logger.success("DocumentVerifier initialized successfully")
    
    def verify(
        self,
        image_path: PathLike,
        doc_type: str,
        dual_check: bool = False
    ) -> DocumentVerificationResult:
        """
        Verify a single document.
        
        Args:
            image_path: Path to document image
            doc_type: Document type ("PAN" or "AADHAAR")
            dual_check: Enable cross-validation with both detectors
        
        Returns:
            DocumentVerificationResult with verdict and confidence
        
        Raises:
            ValueError: If doc_type is invalid or detector unavailable
        """
        doc_type = doc_type.upper()
        image_path = str(image_path)
        
        if doc_type not in ["PAN", "AADHAAR"]:
            raise ValueError(
                f"Invalid document type: {doc_type}. Use 'PAN' or 'AADHAAR'"
            )
        
        logger.info(
            f"Verifying {doc_type} document: {Path(image_path).name} "
            f"(dual_check={dual_check})"
        )
        
        # Primary detection
        if doc_type == "AADHAAR":
            if not self.aadhaar_detector:
                raise ValueError("Aadhaar detector not available")
            primary_result = self.aadhaar_detector.analyze(image_path)
            primary_dict = primary_result.to_dict()
        else:  # PAN
            if not self.pan_detector:
                raise ValueError("PAN detector not available")
            primary_result = self.pan_detector.analyze(image_path)
            primary_dict = primary_result.to_dict()
        
        # Base result
        result = DocumentVerificationResult(
            document_type=doc_type,
            image_path=image_path,
            verdict=primary_result.verdict,
            confidence=primary_result.confidence,
            primary_result=primary_dict,
            dual_check_enabled=dual_check
        )
        
        # Dual-check (cross-validation)
        if dual_check:
            logger.info(f"Running dual-check for {doc_type} document")
            
            try:
                if doc_type == "AADHAAR":
                    if self.pan_detector:
                        secondary_result = self.pan_detector.analyze(image_path)
                    else:
                        logger.warning("PAN detector unavailable for dual-check")
                        secondary_result = None
                else:  # PAN
                    if self.aadhaar_detector:
                        secondary_result = self.aadhaar_detector.analyze(image_path)
                    else:
                        logger.warning("Aadhaar detector unavailable for dual-check")
                        secondary_result = None
                
                if secondary_result:
                    result.secondary_result = secondary_result.to_dict()
                    
                    # Consensus logic
                    primary_forged = primary_result.verdict in ["FORGED"]
                    secondary_forged = secondary_result.verdict in ["FORGED"]
                    
                    if primary_forged == secondary_forged:
                        # Agreement
                        result.agreement = True
                        result.consensus_verdict = "FORGED" if primary_forged else "CLEAN"
                        result.consensus_confidence = (
                            primary_result.confidence + secondary_result.confidence
                        ) / 2.0
                        
                        logger.info(
                            f"Dual-check AGREEMENT: {result.consensus_verdict} "
                            f"(consensus conf={result.consensus_confidence:.3f})"
                        )
                    else:
                        # Disagreement - flag as suspicious
                        result.agreement = False
                        result.consensus_verdict = "SUSPICIOUS"
                        result.consensus_confidence = abs(
                            primary_result.confidence - secondary_result.confidence
                        )
                        result.verdict = "SUSPICIOUS"  # Override primary verdict
                        
                        logger.warning(
                            f"Dual-check DISAGREEMENT: Primary={primary_result.verdict}, "
                            f"Secondary={secondary_result.verdict} → SUSPICIOUS"
                        )
            
            except Exception as e:
                logger.error(f"Dual-check failed: {e}")
                result.secondary_result = {"error": str(e)}
        
        return result
    
    def verify_batch(
        self,
        image_paths: List[PathLike],
        doc_type: str,
        dual_check: bool = False
    ) -> List[DocumentVerificationResult]:
        """
        Verify multiple documents of the same type.
        
        Args:
            image_paths: List of image paths
            doc_type: Document type ("PAN" or "AADHAAR")
            dual_check: Enable cross-validation
        
        Returns:
            List of verification results
        """
        logger.info(
            f"Batch verification: {len(image_paths)} {doc_type} documents "
            f"(dual_check={dual_check})"
        )
        
        results = []
        for path in image_paths:
            try:
                result = self.verify(path, doc_type, dual_check)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to verify {path}: {e}")
                # Create error result
                results.append(
                    DocumentVerificationResult(
                        document_type=doc_type,
                        image_path=str(path),
                        verdict="ERROR",
                        confidence=0.0,
                        primary_result={"error": str(e)},
                        dual_check_enabled=dual_check
                    )
                )
        
        logger.info(f"Batch verification complete: {len(results)} results")
        return results
    
    def verify_as_dict(
        self,
        image_path: PathLike,
        doc_type: str,
        dual_check: bool = False
    ) -> Dict:
        """
        Verify document and return result as dictionary.
        
        Args:
            image_path: Path to document image
            doc_type: Document type ("PAN" or "AADHAAR")
            dual_check: Enable cross-validation
        
        Returns:
            Dictionary with verification results
        """
        return self.verify(image_path, doc_type, dual_check).to_dict()
    
    def get_available_detectors(self) -> List[str]:
        """
        Get list of available detectors.
        
        Returns:
            List of available document types
        """
        available = []
        if self.aadhaar_detector:
            available.append("AADHAAR")
        if self.pan_detector:
            available.append("PAN")
        return available
    
    def get_detector_info(self, doc_type: str) -> Dict:
        """
        Get information about a specific detector.
        
        Args:
            doc_type: Document type ("PAN" or "AADHAAR")
        
        Returns:
            Dictionary with detector configuration and performance
        """
        doc_type = doc_type.upper()
        
        if doc_type == "AADHAAR" and self.aadhaar_detector:
            return self.aadhaar_detector.get_model_info()
        elif doc_type == "PAN" and self.pan_detector:
            return self.pan_detector.get_model_info()
        else:
            raise ValueError(f"Detector not available: {doc_type}")
