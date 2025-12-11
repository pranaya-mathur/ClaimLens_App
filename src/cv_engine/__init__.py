"""
Computer Vision Engine for ClaimLens
Handles car parts detection, damage detection, severity classification, and forgery detection
"""

__version__ = "0.2.0"

from .damage_detector import DamageDetector
from .forgery_detector import ForgeryDetector

__all__ = ["DamageDetector", "ForgeryDetector"]
