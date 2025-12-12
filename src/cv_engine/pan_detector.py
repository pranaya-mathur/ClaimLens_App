"""
PAN Card Forgery Detection Module

High-accuracy PAN card verification using 4-channel ResNet50 (RGB + ELA).

Performance Metrics:
- Accuracy: 99.19% @ threshold 0.5
- AUC: 0.9996
- F1 Score: 0.9942 @ threshold 0.49 (optimal)
- Precision: 95%+ @ threshold 0.48 (precision-oriented)

Usage:
    from src.cv_engine import PANForgeryDetector
    
    detector = PANForgeryDetector()
    result = detector.analyze("path/to/pan_card.jpg")
    print(result["verdict"])  # "CLEAN" or "FORGED"
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Union
import torch
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
from loguru import logger

from .forgery_models import PANForgeryDetectorCNN

PathLike = Union[str, Path]

DEFAULT_MODEL_PATH = Path("models/resnet50_finetuned_after_strong_forgeries.pth")


@dataclass
class PANVerificationResult:
    """PAN card verification result data structure"""
    
    document_type: str
    image_path: str
    verdict: str  # "CLEAN" or "FORGED"
    confidence: float
    forgery_probability: float
    clean_probability: float
    threshold: float
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return asdict(self)
    
    def is_forged(self) -> bool:
        """Check if document is forged"""
        return self.verdict == "FORGED"
    
    def is_clean(self) -> bool:
        """Check if document is clean/authentic"""
        return self.verdict == "CLEAN"


class PANForgeryDetector:
    """
    PAN card forgery detector using fine-tuned 4-channel ResNet50.
    
    Model Architecture:
    - Backbone: ResNet50 with modified conv1 for 4 channels
    - Input: 320x320 images (RGB + ELA)
    - Output: Single logit (BCEWithLogitsLoss)
    - Channels: [R, G, B, ELA]
    
    Training Details:
    - Initial: 640 samples (6 epochs)
    - Fine-tuning: +550 strong forgeries (2-3 epochs @ lr=5e-6)
    - Strong augmentations: copy-move, text overlays, print-scan, JPEG artifacts
    - Optimizer: AdamW (lr=1e-4 initial, 5e-6 fine-tune)
    
    Example:
        >>> detector = PANForgeryDetector(threshold=0.49)  # F1-optimal
        >>> result = detector.analyze("pan_card.jpg")
        >>> print(f"{result.verdict}: {result.confidence:.2%}")
        CLEAN: 98.50%
    """
    
    # Embedded model configuration
    MODEL_CONFIG = {
        "architecture": "ResNet50_4Channel",
        "input_size": (320, 320),
        "num_channels": 4,  # RGB + ELA
        "ela_quality": 90,
        "thresholds": {
            "balanced": 0.50,
            "f1_optimal": 0.49,
            "precision_oriented": 0.48
        },
        "performance": {
            "accuracy_at_0.5": 0.9919,
            "auc": 0.9996,
            "f1_score": 0.9942,
            "false_positives": 9,
            "false_negatives": 2,
            "total_samples": 1350
        }
    }
    
    def __init__(
        self,
        model_path: PathLike = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        threshold: float = 0.49,  # F1-optimal default
        ela_quality: int = 90
    ) -> None:
        """
        Initialize PAN forgery detector.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to run inference on ("cuda" or "cpu")
            threshold: Detection threshold (0.49=F1-optimal, 0.48=precision)
            ela_quality: JPEG quality for ELA generation (default: 90)
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.ela_quality = ela_quality
        self.input_size = self.MODEL_CONFIG["input_size"]
        
        logger.info(f"Initializing PAN detector on {self.device}")
        
        # Load model
        self._load_model()
        
        # Setup transform (no normalization for 4-channel input)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        
        logger.success(
            f"✓ PAN detector ready | "
            f"Threshold: {self.threshold} | "
            f"AUC: {self.MODEL_CONFIG['performance']['auc']:.4f}"
        )
    
    def _load_model(self) -> None:
        """Load trained model weights"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please place 'resnet50_finetuned_after_strong_forgeries.pth' "
                f"in the models/ directory."
            )
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Initialize model
            self.model = PANForgeryDetectorCNN(pretrained=False)
            
            # Load state dict directly (training saved state_dict only)
            state_dict = torch.load(
                self.model_path,
                map_location=self.device
            )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.success("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PAN model: {e}")
            raise RuntimeError(
                f"Could not load PAN model from {self.model_path}. "
                f"Error: {str(e)}"
            ) from e
    
    def _generate_ela(
        self, 
        img: Image.Image, 
        quality: int = None
    ) -> Image.Image:
        """
        Generate Error Level Analysis (ELA) image.
        
        ELA reveals areas of different compression levels in JPEG images,
        highlighting potential forgeries.
        
        Args:
            img: PIL Image (RGB)
            quality: JPEG compression quality (default: self.ela_quality)
        
        Returns:
            Grayscale ELA image resized to input_size
        """
        quality = quality or self.ela_quality
        
        # Compress image to JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        
        # Calculate difference
        ela = ImageChops.difference(img.convert("RGB"), compressed)
        
        # Amplify differences
        extrema = ela.getextrema()
        max_diff = max([e[1] for e in extrema])
        
        if max_diff == 0:
            scale = 1
        else:
            scale = 255.0 / max_diff
        
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        
        # Convert to grayscale and resize
        ela = ela.convert("L").resize(self.input_size)
        
        return ela
    
    def _predict_probability(self, img: Image.Image) -> float:
        """
        Run model inference and return forgery probability.
        
        Args:
            img: PIL Image (RGB)
        
        Returns:
            Forgery probability (0-1)
        """
        # Resize image
        img = img.convert("RGB").resize(self.input_size)
        
        # Generate ELA
        ela = self._generate_ela(img)
        
        # Transform RGB
        img_tensor = self.transform(img)  # 3 x H x W
        
        # Transform ELA (single channel)
        ela_tensor = transforms.ToTensor()(ela)  # 1 x H x W
        
        # Concatenate to 4 channels
        tensor_4ch = torch.cat([img_tensor, ela_tensor], dim=0)  # 4 x H x W
        tensor_4ch = tensor_4ch.unsqueeze(0).to(self.device)  # 1 x 4 x H x W
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor_4ch).cpu().numpy().ravel()[0]
            # Apply sigmoid
            prob = 1.0 / (1.0 + torch.exp(-torch.tensor(logits)).item())
        
        return float(prob)
    
    def analyze(self, image_path: PathLike) -> PANVerificationResult:
        """
        Analyze PAN card image for forgery.
        
        Args:
            image_path: Path to PAN card image
        
        Returns:
            PANVerificationResult with verdict and probabilities
        
        Example:
            >>> result = detector.analyze("pan.jpg")
            >>> if result.is_forged():
            ...     print(f"Warning: Forged PAN card detected!")
            ...     print(f"Forgery probability: {result.forgery_probability:.2%}")
        """
        image_path = str(image_path)
        
        try:
            pil_img = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Could not open image: {image_path}") from e
        
        # Get forgery probability
        forgery_prob = self._predict_probability(pil_img)
        clean_prob = 1.0 - forgery_prob
        
        # Determine verdict
        is_forged = forgery_prob >= self.threshold
        verdict = "FORGED" if is_forged else "CLEAN"
        confidence = forgery_prob if is_forged else clean_prob
        
        logger.info(
            f"PAN analysis: {Path(image_path).name} → "
            f"{verdict} (conf={confidence:.3f}, "
            f"forged={forgery_prob:.3f}, clean={clean_prob:.3f})"
        )
        
        return PANVerificationResult(
            document_type="PAN",
            image_path=image_path,
            verdict=verdict,
            confidence=confidence,
            forgery_probability=forgery_prob,
            clean_probability=clean_prob,
            threshold=self.threshold
        )
    
    def analyze_batch(
        self,
        image_paths: list[PathLike]
    ) -> list[PANVerificationResult]:
        """
        Analyze multiple PAN card images.
        
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
            image_path: Path to PAN card image
        
        Returns:
            Dictionary with verification results
        """
        return self.analyze(image_path).to_dict()
    
    def set_threshold(self, threshold: float, mode: str = None) -> None:
        """
        Update detection threshold.
        
        Args:
            threshold: New threshold value (0-1)
            mode: Preset mode ("balanced", "f1_optimal", "precision_oriented")
        
        Example:
            >>> detector.set_threshold(mode="precision_oriented")
            >>> # Now uses threshold=0.48 for 95%+ precision
        """
        if mode:
            if mode not in self.MODEL_CONFIG["thresholds"]:
                raise ValueError(
                    f"Unknown mode: {mode}. "
                    f"Use: {list(self.MODEL_CONFIG['thresholds'].keys())}"
                )
            threshold = self.MODEL_CONFIG["thresholds"][mode]
        
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")
    
    def get_model_info(self) -> Dict:
        """
        Get model configuration and performance information.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_type": "PANForgeryDetector",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "threshold": self.threshold,
            "config": self.MODEL_CONFIG
        }
