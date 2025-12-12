"""Feature engineering pipeline for fraud detection.

Extracts features from claim narratives using Bhasha-Embed for Hinglish text,
combined with behavioral and document-based features.
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
    """

    def __init__(self, pca_dims: int = 100, model_name: str = "AkshitaS/bhasha-embed-v0"):
        """Initialize feature engineer.
        
        Args:
            pca_dims: Number of PCA dimensions for embeddings (default: 100)
            model_name: HuggingFace model for embeddings (default: Bhasha-Embed)
        """
        self.pca_dims = pca_dims
        self.model_name = model_name
        self.embedder = None
        self.pca = PCA(n_components=pca_dims, random_state=42)
        self.is_fitted = False
        
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
        
        self.categorical_features = ["product", "city", "subtype"]
        self.embedding_cols = [f"emb_{i}" for i in range(pca_dims)]

    def _load_embedder(self):
        """Lazy load embedding model."""
        if self.embedder is None:
            logger.info(f"Loading {self.model_name} embedding model...")
            self.embedder = SentenceTransformer(self.model_name)
            logger.success("Embedder ready!")

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-aware aggregation features without label leakage.
        
        Args:
            df: DataFrame with incident_date, claimant_id, policy_id, claim_amount
            
        Returns:
            DataFrame with added time-based features
            
        Raises:
            ValueError: If required columns are missing or invalid
        """
        # BUG #2 FIX: Validate incident_date column exists
        if "incident_date" not in df.columns:
            raise ValueError(
                "Missing required column: 'incident_date'. "
                "This column is required for time-aware feature engineering."
            )
        
        # BUG #2 FIX: Ensure incident_date is datetime
        try:
            df["incident_date"] = pd.to_datetime(df["incident_date"])
        except Exception as e:
            raise ValueError(
                f"Invalid 'incident_date' format. Expected ISO format (YYYY-MM-DD). "
                f"Error: {str(e)}"
            )
        
        # Check for null dates
        if df["incident_date"].isna().any():
            null_count = df["incident_date"].isna().sum()
            logger.warning(f"Found {null_count} null incident_date values. Filling with today's date.")
            df["incident_date"] = df["incident_date"].fillna(pd.Timestamp.now())
        
        df_sorted = df.sort_values("incident_date").reset_index(drop=True)
        
        # Claimant history features
        df_sorted["claimant_claim_count"] = df_sorted.groupby("claimant_id").cumcount() + 1
        df_sorted["claimant_total_claimed"] = df_sorted.groupby("claimant_id")["claim_amount"].cumsum()
        df_sorted["claimant_avg_claim"] = (
            df_sorted["claimant_total_claimed"] / df_sorted["claimant_claim_count"]
        )
        
        # Policy history features
        df_sorted["policy_claim_count"] = df_sorted.groupby("policy_id").cumcount() + 1
        df_sorted["policy_total_claimed"] = df_sorted.groupby("policy_id")["claim_amount"].cumsum()
        df_sorted["policy_avg_claim"] = (
            df_sorted["policy_total_claimed"] / df_sorted["policy_claim_count"]
        )
        
        # Time since last claim
        df_sorted["days_since_last_claim"] = (
            df_sorted.groupby("claimant_id")["incident_date"].diff().dt.days
        )
        df_sorted["days_since_last_claim"] = df_sorted["days_since_last_claim"].fillna(9999)
        
        # Rapid claims indicator (< 30 days)
        df_sorted["rapid_claims"] = (df_sorted["days_since_last_claim"] < 30).astype(int)
        df_sorted["is_first_claim"] = (df_sorted["claimant_claim_count"] == 1).astype(int)
        
        return df_sorted

    def create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric features from claim data.
        
        Args:
            df: DataFrame with days_since_policy_start, claim_amount
            
        Returns:
            DataFrame with added numeric features
        """
        df["claim_amount_log"] = np.log1p(df["claim_amount"])
        df["policy_age_months"] = df["days_since_policy_start"] / 30.0
        df["is_recent_policy"] = (df["days_since_policy_start"] < 180).astype(int)
        
        return df

    def create_document_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract document-based features.
        
        Args:
            df: DataFrame with documents_submitted column
            
        Returns:
            DataFrame with document presence indicators
        """
        # BUG #4 FIX: Correctly count documents (0 if empty/None)
        df["num_docs"] = df["documents_submitted"].fillna("").apply(
            lambda s: 1 + str(s).count(",") if str(s).strip() else 0  # FIXED: Returns 0 if empty
        )
        
        df["has_fir"] = df["documents_submitted"].fillna("").str.contains("FIR", na=False).astype(int)
        df["has_photos"] = df["documents_submitted"].fillna("").str.contains("photos", na=False).astype(int)
        df["has_death_cert"] = (
            df["documents_submitted"].fillna("").str.contains("death certificate", na=False).astype(int)
        )
        df["has_discharge"] = (
            df["documents_submitted"].fillna("").str.contains("discharge summary", na=False).astype(int)
        )
        
        return df

    def create_embeddings(self, narratives: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate Hinglish narrative embeddings with PCA.
        
        Args:
            narratives: List of claim narrative texts
            batch_size: Batch size for encoding (default: 64)
            
        Returns:
            PCA-reduced embeddings (n_samples, pca_dims)
        """
        self._load_embedder()
        
        logger.info(f"Generating embeddings for {len(narratives)} narratives...")
        embeddings = self.embedder.encode(
            narratives,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        
        if not self.is_fitted:
            logger.info(f"Fitting PCA to reduce to {self.pca_dims} dimensions...")
            reduced = self.pca.fit_transform(embeddings)
            self.is_fitted = True
            logger.success(f"PCA variance retained: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            reduced = self.pca.transform(embeddings)
        
        return reduced

    def engineer_features(
        self,
        df: pd.DataFrame,
        narrative_col: str = "narrative",
        keep_ids: bool = True,  # BUG #3 FIX: Renamed from drop_ids to keep_ids (clearer)
    ) -> pd.DataFrame:
        """Complete feature engineering pipeline.
        
        Args:
            df: Raw claim DataFrame
            narrative_col: Name of narrative text column
            keep_ids: Whether to KEEP ID columns in output (claim_id, claimant_id, policy_id)
                     Default: True (keeps IDs for tracking)
            
        Returns:
            Feature matrix with embeddings + engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # BUG #2 FIX: Validate and convert incident_date (moved to create_time_features)
        
        # Time-aware features
        logger.info("Creating time-aware aggregation features...")
        df = self.create_time_features(df)
        
        # Numeric features
        logger.info("Creating numeric features...")
        df = self.create_numeric_features(df)
        
        # Document features
        logger.info("Creating document features...")
        df = self.create_document_features(df)
        
        # Categorical encoding
        logger.info("One-hot encoding categorical features...")
        df_cat = pd.get_dummies(df[self.categorical_features], prefix=self.categorical_features)
        
        # Embeddings
        narratives = df[narrative_col].fillna("").tolist()
        embeddings = self.create_embeddings(narratives)
        df_emb = pd.DataFrame(embeddings, columns=self.embedding_cols)
        
        # Assemble feature matrix
        logger.info("Assembling feature matrix...")
        feature_df = pd.concat(
            [
                df[self.numeric_features],
                df_cat,
                df_emb,
            ],
            axis=1,
        )
        
        # BUG #3 FIX: Corrected logic - keep_ids=True means KEEP them
        if keep_ids:
            # Keep IDs in the feature matrix for tracking
            id_cols = ["claim_id", "claimant_id", "policy_id"]
            existing_ids = [c for c in id_cols if c in df.columns]
            if existing_ids:
                ids_df = df[existing_ids].copy()
                feature_df = pd.concat([ids_df, feature_df], axis=1)
                logger.info(f"Kept ID columns: {existing_ids}")
        else:
            logger.info("ID columns excluded from feature matrix")
        
        logger.success(f"Feature engineering complete! Shape: {feature_df.shape}")
        return feature_df

    def validate_no_leakage(self, feature_df: pd.DataFrame) -> bool:
        """Validate no label-leakage columns exist.
        
        Args:
            feature_df: Engineered feature DataFrame
            
        Returns:
            True if no leakage detected, raises ValueError otherwise
        """
        forbidden_substrings = ["fraud", "score", "redflag", "flag"]
        
        blocked_cols = []
        for col in feature_df.columns:
            col_lower = col.lower()
            if any(sub in col_lower for sub in forbidden_substrings):
                blocked_cols.append(col)
        
        if blocked_cols:
            raise ValueError(
                f"⚠️ LEAKAGE DETECTED! Found forbidden columns: {blocked_cols}\n"
                f"These columns may contain label information and must be removed."
            )
        
        logger.success("✅ Leakage validation passed: No forbidden columns found.")
        return True
