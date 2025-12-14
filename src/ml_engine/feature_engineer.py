"""Feature engineering pipeline for fraud detection.

Extracts features from claim narratives using Bhasha-Embed for Hinglish text,
combined with behavioral and document-based features.

FIXED VERSION - Resolves column name mismatch bug between dataset and model.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from typing import Dict, List, Optional
import warnings
from loguru import logger

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Feature engineering for insurance claim fraud detection.
    
    Generates 145+ features including:
    - Hinglish narrative embeddings (100 dims via PCA)
    - Time-aware aggregation features (claimant/policy history)
    - Behavioral patterns (claim frequency, amounts)
    - Document presence indicators
    - Categorical encodings (product, city, subtype)
    
    Attributes:
        embedder: SentenceTransformer model for Hinglish text
        pca: PCA model for dimensionality reduction
        pca_dims: Number of embedding dimensions (default: 100)
        is_fitted: Whether PCA has been fitted
        expected_features: List of feature names expected by the model (for alignment)
    """

    def __init__(
        self, 
        pca_dims: int = 100, 
        model_name: str = "AkshitaS/bhasha-embed-v0",
        expected_features: Optional[List[str]] = None
    ):
        """Initialize feature engineer.
        
        Args:
            pca_dims: Number of PCA dimensions for embeddings (default: 100)
            model_name: HuggingFace model for embeddings (default: Bhasha-Embed)
            expected_features: List of feature column names from trained model (for alignment)
        """
        self.pca_dims = pca_dims
        self.model_name = model_name
        self.embedder = None
        self.pca = PCA(n_components=pca_dims, random_state=42)
        self.is_fitted = False
        self.expected_features = expected_features  # Store for feature alignment
        
        # Feature column tracking
        self.numeric_features = [
            "days_since_policy_start",
            "claim_amount",
            "claimant_claim_count",
            "claimant_total_claimed",
            "claimant_avg_claim",
            "policy_claim_count",
            "policy_total_claimed",
            "policy_avg_claim",
            "days_since_last_claim",
            "rapid_claims",
            "is_first_claim",
            "claim_amount_log",
            "policy_age_months",
            "is_recent_policy",
            "num_docs",
            "has_fir",
            "has_photos",
            "has_death_cert",
            "has_discharge",
        ]
        
        # 🔧 FIXED: Updated to match dataset column names
        # Dataset has: product_type, city, claim_subtype
        # But we need to create dummies with prefixes: product_, city_, subtype_
        self.categorical_features = ["product_type", "city", "claim_subtype"]
        self.categorical_prefixes = ["product", "city", "subtype"]
        
        self.embedding_cols = [f"emb_{i}" for i in range(pca_dims)]
        
        logger.info(f"FeatureEngineer initialized with {pca_dims} PCA dimensions")
        if expected_features:
            logger.info(f"Feature alignment enabled: {len(expected_features)} expected features")

    def _load_embedder(self):
        """Lazy load embedding model."""
        if self.embedder is None:
            logger.info(f"Loading {self.model_name} embedding model...")
            try:
                self.embedder = SentenceTransformer(self.model_name)
                logger.success("Embedder ready!")
            except Exception as e:
                logger.error(f"Failed to load embedder: {e}")
                raise