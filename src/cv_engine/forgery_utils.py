"""
Forgery Detection Utilities
Error Level Analysis (ELA) and noise variation analysis for image forensics
"""
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


class ForgeryUtils:
    """Utility helpers for forgery detection (ELA + noise analysis)."""

    @staticmethod
    def calculate_ela(image_path: str, quality: int = 90) -> np.ndarray:
        """
        Perform Error Level Analysis (ELA) on an image.

        Args:
            image_path: Path to input image.
            quality: JPEG quality used for recompression.

        Returns:
            ELA image as H×W×3 uint8 numpy array.
        """
        original = Image.open(image_path).convert("RGB")
        tmp_path = Path("/tmp") / "ela_temp.jpg"
        original.save(tmp_path, "JPEG", quality=quality)
        compressed = Image.open(tmp_path).convert("RGB")

        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max(ex[1] for ex in extrema)
        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return np.array(ela_image)

    @staticmethod
    def ela_intensity_score(ela_img: np.ndarray) -> float:
        """
        Simple scalar score from ELA image.
        Higher mean intensity tends to indicate more suspicious regions.
        Returns value in roughly 0–1 range (clipped).
        """
        mean_val = float(ela_img.mean())
        return float(min(mean_val / 100.0, 1.0))

    @staticmethod
    def compute_noise_variation(img: np.ndarray, block_size: int = 32) -> float:
        """
        Rough measure of noise pattern inconsistency across the image.
        Higher std of block variances can suggest patch-level manipulation.
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h, w = img.shape
        variances: List[float] = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y : y + block_size, x : x + block_size]
                variances.append(float(block.var()))

        return float(np.std(variances)) if variances else 0.0
