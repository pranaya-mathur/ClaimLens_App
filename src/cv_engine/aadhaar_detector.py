"""
Aadhaar Card Forgery Detection Module

High-accuracy Aadhaar card verification using ResNet50 classifier.

Performance Metrics:
- Validation Accuracy: 99.62%
- Balanced Accuracy: 99.80%
- AUC: 0.9999
- Optimal Threshold: 0.5

Usage:
    from src.cv_engine import AadhaarForgeryDetector
    
    detector = AadhaarForgeryDetector()
    result = detector.analyze("path/to/aadhaar.jpg")
    print(result["verdict"])  # "AUTHENTIC" or "FORGED"
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Union, List
import torch
import numpy as np
from PIL import Image, ImageStat
from torchvision import transforms
from loguru import logger
import cv2

from .forgery_models import AadhaarForgeryDetectorCNN

PathLike = Union[str, Path]

# 🔧 FIX: Updated path to forgery_detection subdirectory
DEFAULT_MODEL_PATH = Path("models/forgery_detection/aadhaar_balanced_model.pth")


@dataclass
class AadhaarVerificationResult:
    """Aadhaar card verification result data structure"""
    
    document_type: str
    image_path: str
    verdict: str  # "AUTHENTIC" or "FORGED"
    confidence: float
    authentic_probability: float
    forged_probability: float
    threshold: float
    quality_score: float = 1.0  # 🆕 NEW: Image quality score (0-1)
    quality_warnings: List[str] = field(default_factory=list)  # 🆕 NEW: Quality issues
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return asdict(self)
    
    def is_forged(self) -> bool:
        """Check if document is forged"""
        return self.verdict == "FORGED"
    
    def is_authentic(self) -> bool:
        """Check if document is authentic"""
        return self.verdict == "AUTHENTIC"


class AadhaarForgeryDetector:
    """
    Aadhaar card forgery detector using balanced ResNet50 model.
    
    Model Architecture:
    - Backbone: ResNet50 (pretrained on ImageNet)
    - Input: 224x224 RGB images
    - Output: 2 classes [FORGED, AUTHENTIC]
    - Head: Simple Linear(2048 → 2)
    
    Training Details:
    - Dataset: 2,116 real + 2,116 synthetic forgeries
    - Validation: 530 real + 530 synthetic (balanced)
    - Class Weights: 1:1 (balanced after undersampling)
    - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
    - Epochs: 8 (best at epoch 4)
    
    Example:
        >>> detector = AadhaarForgeryDetector()
        >>> result = detector.analyze("aadhaar_card.jpg")
        >>> print(f"{result.verdict}: {result.confidence:.2%}")
        AUTHENTIC: 99.84%
    """
    
    # Embedded model configuration
    MODEL_CONFIG = {
        "architecture": "ResNet50",
        "input_size": (224, 224),
        "num_classes": 2,
        "threshold": 0.5,  # Balanced threshold for high-quality images
        "threshold_low_quality": 0.3,  # 🆕 NEW: Lenient threshold for poor quality
        "normalization": {
            "mean": [0.485, 0.456, 0.406],  # ImageNet stats
            "std": [0.229, 0.224, 0.225]
        },
        "performance": {
            "validation_accuracy": 0.9962,
            "balanced_accuracy": 0.9980,
            "auc": 0.9999,
            "false_positives": 7,
            "false_negatives": 0,
            "total_samples": 1060
        }
    }
    
    def __init__(
        self,
        model_path: PathLike = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        threshold: float = None,
        adaptive_threshold: bool = True  # 🆕 NEW: Enable adaptive thresholding
    ) -> None:
        """
        Initialize Aadhaar forgery detector.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to run inference on ("cuda" or "cpu")
            threshold: Custom threshold (default: 0.5 for balanced accuracy)
            adaptive_threshold: Adjust threshold based on image quality
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold or self.MODEL_CONFIG["threshold"]
        self.adaptive_threshold = adaptive_threshold
        self.input_size = self.MODEL_CONFIG["input_size"]
        
        logger.info(f"Initializing Aadhaar detector on {self.device}")
        
        # Load model
        self._load_model()
        
        # Setup image transform
        norm = self.MODEL_CONFIG["normalization"]
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"])
        ])
        
        logger.success(
            f"Aadhaar detector ready | "
            f"Threshold: {self.threshold} | "
            f"Adaptive: {self.adaptive_threshold} | "
            f"Accuracy: {self.MODEL_CONFIG['performance']['validation_accuracy']:.2%}"
        )
    
    def _load_model(self) -> None:
        """Load trained model weights with automatic key remapping"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please place 'aadhaar_balanced_model.pth' in the models/forgery_detection/ directory."
            )
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Initialize model
            self.model = AadhaarForgeryDetectorCNN(pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=False
            )
            
            # Extract state dict from checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info(
                        f"Loaded checkpoint with AUC: "
                        f"{checkpoint.get('best_auc', 'N/A')}"
                    )
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the dict is the state_dict itself
                    state_dict = checkpoint
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # 🔥 Add 'backbone.' prefix if missing (handles models trained before wrapper)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_state_dict[key] = value
                else:
                    new_state_dict[f'backbone.{key}'] = value
            
            # Use strict=False to ignore any remaining mismatches
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.success("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Aadhaar model: {e}")
            raise RuntimeError(
                f"Could not load Aadhaar model from {self.model_path}. "
                f"Error: {str(e)}"
            ) from e
    
    def _assess_image_quality(self, img: Image.Image) -> tuple[float, List[str]]:
        """
        🆕 NEW: Assess image quality and return score + warnings.
        
        Args:
            img: PIL Image
        
        Returns:
            (quality_score, warnings) tuple
            - quality_score: 0-1 (1 = perfect quality)
            - warnings: List of quality issues detected
        """
        warnings = []
        quality_score = 1.0
        
        # Convert to OpenCV format for analysis
        img_np = np.array(img.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. Check resolution
        width, height = img.size
        if width < 600 or height < 400:
            warnings.append("⚠️ Low resolution - image may lack detail")
            quality_score -= 0.2
        
        # 2. Check brightness
        stat = ImageStat.Stat(img.convert('L'))
        brightness = stat.mean[0]
        if brightness < 60:
            warnings.append("🌑 Too dark - poor lighting conditions")
            quality_score -= 0.25
        elif brightness > 220:
            warnings.append("☀️ Overexposed - too bright")
            quality_score -= 0.2
        
        # 3. Check blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            warnings.append("📸 Blurry image - focus issue detected")
            quality_score -= 0.3
        
        # 4. Check contrast
        contrast = stat.stddev[0]
        if contrast < 30:
            warnings.append("🔳 Low contrast - flat image")
            quality_score -= 0.15
        
        quality_score = max(quality_score, 0.0)
        
        if quality_score < 0.6:
            warnings.insert(0, "‼️ Poor image quality detected - consider retaking photo")
        
        return quality_score, warnings
    
    def _predict_probabilities(self, img: Image.Image) -> tuple[float, float]:
        """
        Run model inference and return class probabilities.
        
        Args:
            img: PIL Image (RGB)
        
        Returns:
            (forged_prob, authentic_prob) tuple
        """
        img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # Class mapping: 0=FORGED, 1=AUTHENTIC
        forged_prob = float(probs[0].item())
        authentic_prob = float(probs[1].item())
        
        return forged_prob, authentic_prob
    
    def analyze(self, image_path: PathLike) -> AadhaarVerificationResult:
        """
        Analyze Aadhaar card image for forgery with quality assessment.
        
        Args:
            image_path: Path to Aadhaar card image
        
        Returns:
            AadhaarVerificationResult with verdict, probabilities, and quality info
        
        Example:
            >>> result = detector.analyze("aadhaar.jpg")
            >>> if result.is_forged():
            ...     print(f"Warning: Forged document detected!")
            ...     print(f"Confidence: {result.confidence:.2%}")
            >>> if result.quality_warnings:
            ...     print("Quality issues:", result.quality_warnings)
        """
        image_path = str(image_path)
        
        try:
            pil_img = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Could not open image: {image_path}") from e
        
        # 🆕 Assess image quality
        quality_score, quality_warnings = self._assess_image_quality(pil_img)
        
        # 🆕 Adaptive threshold based on quality
        if self.adaptive_threshold and quality_score < 0.6:
            effective_threshold = self.MODEL_CONFIG["threshold_low_quality"]
            logger.info(f"Low quality image detected (score={quality_score:.2f}), using lenient threshold={effective_threshold}")
        else:
            effective_threshold = self.threshold
        
        # Get predictions
        forged_prob, authentic_prob = self._predict_probabilities(pil_img)
        
        # Determine verdict
        is_forged = authentic_prob < effective_threshold
        verdict = "FORGED" if is_forged else "AUTHENTIC"
        confidence = max(forged_prob, authentic_prob)
        
        logger.info(
            f"Aadhaar analysis: {Path(image_path).name} → "
            f"{verdict} (conf={confidence:.3f}, "
            f"auth={authentic_prob:.3f}, forged={forged_prob:.3f}, "
            f"quality={quality_score:.2f}, threshold={effective_threshold})"
        )
        
        return AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path=image_path,
            verdict=verdict,
            confidence=confidence,
            authentic_probability=authentic_prob,
            forged_probability=forged_prob,
            threshold=effective_threshold,
            quality_score=quality_score,
            quality_warnings=quality_warnings
        )
    
    def analyze_batch(
        self, 
        image_paths: list[PathLike]
    ) -> list[AadhaarVerificationResult]:
        """
        Analyze multiple Aadhaar card images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of verification results
        """
        return [self.analyze(path) for path in image_paths]
    
    def analyze_as_dict(self, image_path: PathLike) -> Dict:
        """
        Analyze image and return result as dictionary.
        
        Args:
            image_path: Path to Aadhaar card image
        
        Returns:
            Dictionary with verification results
        """
        return self.analyze(image_path).to_dict()
    
    def get_model_info(self) -> Dict:
        """
        Get model configuration and performance information.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_type": "AadhaarForgeryDetector",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "adaptive_threshold": self.adaptive_threshold,
            "config": self.MODEL_CONFIG
        }
