"""
Forgery Detector
High-level forgery detection combining CNN, ELA, and noise analysis
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from loguru import logger

from .forgery_models import ForgeryDetectorCNN
from .forgery_utils import ForgeryUtils

PathLike = Union[str, Path]

DEFAULT_MODEL_PATH = Path("models/forgery_detector_latest_run.pth")
DEFAULT_CONFIG_PATH = Path("models/forgery_detector_latest_run_config.json")


@dataclass
class ForgeryResult:
    """Forgery detection result data structure"""

    image_path: str
    is_forged: bool
    forgery_prob: float
    threshold: float
    ela_score: float
    noise_variation: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ForgeryDetector:
    """
    High-level forgery detector that wraps the trained CNN model and adds
    simple ELA + noise-based heuristics.

    Example:
        fd = ForgeryDetector()
        res = fd.analyze_image("some.jpg")
        print(res.is_forged, res.forgery_prob)
    """

    def __init__(
        self,
        model_path: PathLike = DEFAULT_MODEL_PATH,
        config_path: PathLike = DEFAULT_CONFIG_PATH,
        device: str = "cuda",
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing ForgeryDetector on {self.device}")

        self._load_config()
        self._load_model()
        self.utils = ForgeryUtils()

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.success("✓ ForgeryDetector loaded successfully")

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load_config(self) -> None:
        """Load configuration from JSON file"""
        # Defaults
        self.threshold: float = 0.55
        self.input_size = (224, 224)

        if not self.config_path.exists():
            logger.warning(
                f"Config file not found: {self.config_path}. Using defaults."
            )
            return

        with self.config_path.open("r") as f:
            cfg = json.load(f)

        self.threshold = float(cfg.get("threshold", self.threshold))
        size = cfg.get("input_size", list(self.input_size))
        self.input_size = (int(size[0]), int(size[1]))

        logger.info(f"Loaded config: threshold={self.threshold}, size={self.input_size}")

    def _load_model(self) -> None:
        """Load trained ResNet50 model weights with automatic key remapping"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Please place the .pth file in the models/ directory."
            )

        self.model = ForgeryDetectorCNN(pretrained=False)
        state = torch.load(self.model_path, map_location=self.device)
        
        # 🔥 Add 'backbone.' prefix if missing (handles models trained before wrapper)
        new_state = {}
        for key, value in state.items():
            if key.startswith('backbone.'):
                new_state[key] = value
            else:
                new_state[f'backbone.{key}'] = value
        
        # Use strict=False to ignore any remaining mismatches
        self.model.load_state_dict(new_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded model weights from {self.model_path}")

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #

    def _predict_prob(self, img: Image.Image) -> float:
        """Run CNN inference and return forgery probability"""
        img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # index 1 = FORGED class
        return float(probs[1].item())

    def analyze_image(self, image_path: PathLike) -> ForgeryResult:
        """
        Run full analysis on a single image.

        Args:
            image_path: Path to image file

        Returns:
            ForgeryResult with all detection metrics
        """
        image_path = str(image_path)
        pil_img = Image.open(image_path).convert("RGB")

        # CNN probability
        forgery_prob = self._predict_prob(pil_img)

        # ELA + noise metrics (for debugging / future fusion)
        ela_img = self.utils.calculate_ela(image_path)
        ela_score = self.utils.ela_intensity_score(ela_img)
        np_img = np.array(pil_img)
        noise_var = self.utils.compute_noise_variation(np_img)

        is_forged = forgery_prob >= self.threshold

        logger.info(
            f"Forgery analysis: prob={forgery_prob:.3f}, "
            f"is_forged={is_forged}, ela={ela_score:.3f}, noise={noise_var:.1f}"
        )

        return ForgeryResult(
            image_path=image_path,
            is_forged=is_forged,
            forgery_prob=forgery_prob,
            threshold=self.threshold,
            ela_score=ela_score,
            noise_variation=noise_var,
        )

    def analyze_batch(self, image_paths: List[PathLike]) -> List[ForgeryResult]:
        """Analyze multiple images"""
        return [self.analyze_image(p) for p in image_paths]

    # ------------------------------------------------------------------ #
    # Convenience for API layer
    # ------------------------------------------------------------------ #

    def analyze_image_as_dict(self, image_path: PathLike) -> Dict:
        """Return forgery analysis as dictionary"""
        return self.analyze_image(image_path).to_dict()

    def analyze_batch_as_dicts(self, image_paths: List[PathLike]) -> List[Dict]:
        """Return batch analysis as list of dictionaries"""
        return [r.to_dict() for r in self.analyze_batch(image_paths)]
