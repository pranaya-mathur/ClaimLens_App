"""ML Engine for fraud detection with Hinglish narrative analysis.

Provides CatBoost-based fraud scoring with feature engineering pipeline
including Bhasha-Embed for Hinglish text embeddings.
"""

from .ml_scorer import MLFraudScorer
from .feature_engineer import FeatureEngineer

__all__ = ["MLFraudScorer", "FeatureEngineer"]
