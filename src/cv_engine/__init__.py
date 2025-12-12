"""
Computer Vision Engine for ClaimLens
Handles car damage detection, severity classification, and document forgery detection

Document Verification:
- AadhaarForgeryDetector: 99.62% accuracy for Aadhaar cards
- PANForgeryDetector: 99.19% accuracy for PAN cards
- DocumentVerifier: Unified verification with dual-check support
"""

__version__ = "0.3.0"

# Existing modules (backward compatibility)
from .damage_detector import DamageDetector
from .forgery_detector import ForgeryDetector

# New document verification modules
from .aadhaar_detector import (
    AadhaarForgeryDetector,
    AadhaarVerificationResult
)
from .pan_detector import (
    PANForgeryDetector,
    PANVerificationResult
)
from .document_verifier import (
    DocumentVerifier,
    DocumentVerificationResult
)

# Model classes (for advanced usage)
from .forgery_models import (
    ForgeryDetectorCNN,
    AadhaarForgeryDetectorCNN,
    PANForgeryDetectorCNN
)

__all__ = [
    # Version
    "__version__",
    
    # Existing detectors
    "DamageDetector",
    "ForgeryDetector",
    
    # Document verification
    "AadhaarForgeryDetector",
    "PANForgeryDetector",
    "DocumentVerifier",
    
    # Result dataclasses
    "AadhaarVerificationResult",
    "PANVerificationResult",
    "DocumentVerificationResult",
    
    # Model architectures
    "ForgeryDetectorCNN",
    "AadhaarForgeryDetectorCNN",
    "PANForgeryDetectorCNN",
]
