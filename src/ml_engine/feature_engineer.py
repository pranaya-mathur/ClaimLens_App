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
        
        logger.info(f"FeatureEngineer initialized with {pca_dims} PCA dimensions")

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

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-aware aggregation features without label leakage.
        
        Args:
            df: DataFrame with incident_date, claimant_id, policy_id, claim_amount
            
        Returns:
            DataFrame with added time-based features
            
        Raises:
            ValueError: If required columns are missing or invalid
        """
        logger.info("Creating time-aware features...")
        
        if "incident_date" not in df.columns:
            logger.error("Missing 'incident_date' column in DataFrame")
            raise ValueError(
                "Missing required column: 'incident_date'. "
                "This column is required for time-aware feature engineering."
            )
        
        try:
            df["incident_date"] = pd.to_datetime(df["incident_date"])
            logger.debug(f"Converted incident_date to datetime (min: {df['incident_date'].min()}, max: {df['incident_date'].max()})")
        except Exception as e:
            logger.error(f"Failed to parse incident_date: {e}")
            raise ValueError(
                f"Invalid 'incident_date' format. Expected ISO format (YYYY-MM-DD). "
                f"Error: {str(e)}"
            )
        
        null_count = df["incident_date"].isna().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null incident_date values. Filling with current date.")
            df["incident_date"] = df["incident_date"].fillna(pd.Timestamp.now())
        
        try:
            df_sorted = df.sort_values("incident_date").reset_index(drop=True)
            logger.debug(f"Sorted {len(df_sorted)} claims by incident_date")
            
            logger.debug("Computing claimant history features...")
            df_sorted["claimant_claim_count"] = df_sorted.groupby("claimant_id").cumcount() + 1
            df_sorted["claimant_total_claimed"] = df_sorted.groupby("claimant_id")["claim_amount"].cumsum()
            df_sorted["claimant_avg_claim"] = (
                df_sorted["claimant_total_claimed"] / df_sorted["claimant_claim_count"]
            )
            
            logger.debug("Computing policy history features...")
            df_sorted["policy_claim_count"] = df_sorted.groupby("policy_id").cumcount() + 1
            df_sorted["policy_total_claimed"] = df_sorted.groupby("policy_id")["claim_amount"].cumsum()
            df_sorted["policy_avg_claim"] = (
                df_sorted["policy_total_claimed"] / df_sorted["policy_claim_count"]
            )
            
            logger.debug("Computing time-based features...")
            df_sorted["days_since_last_claim"] = (
                df_sorted.groupby("claimant_id")["incident_date"].diff().dt.days
            )
            df_sorted["days_since_last_claim"] = df_sorted["days_since_last_claim"].fillna(9999)
            
            df_sorted["rapid_claims"] = (df_sorted["days_since_last_claim"] < 30).astype(int)
            df_sorted["is_first_claim"] = (df_sorted["claimant_claim_count"] == 1).astype(int)
            
            rapid_claim_count = df_sorted["rapid_claims"].sum()
            first_claim_count = df_sorted["is_first_claim"].sum()
            
            logger.success(
                f"Time features created: {rapid_claim_count} rapid claims, "
                f"{first_claim_count} first-time claims"
            )
            
            return df_sorted
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            raise

    def create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric features from claim data.
        
        Args:
            df: DataFrame with days_since_policy_start, claim_amount
            
        Returns:
            DataFrame with added numeric features
        """
        logger.debug("Creating numeric features...")
        
        try:
            df["claim_amount_log"] = np.log1p(df["claim_amount"])
            df["policy_age_months"] = df["days_since_policy_start"] / 30.0
            df["is_recent_policy"] = (df["days_since_policy_start"] < 180).astype(int)
            
            recent_policy_count = df["is_recent_policy"].sum()
            logger.debug(f"Numeric features: {recent_policy_count} recent policies (<180 days)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating numeric features: {e}")
            raise

    def create_document_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract document-based features.
        
        Args:
            df: DataFrame with documents_submitted column
            
        Returns:
            DataFrame with document presence indicators
        """
        logger.debug("Creating document features...")
        
        try:
            df["num_docs"] = df["documents_submitted"].fillna("").apply(
                lambda s: 1 + str(s).count(",") if str(s).strip() else 0
            )
            
            df["has_fir"] = df["documents_submitted"].fillna("").str.contains("FIR", na=False).astype(int)
            df["has_photos"] = df["documents_submitted"].fillna("").str.contains("photos", na=False).astype(int)
            df["has_death_cert"] = (
                df["documents_submitted"].fillna("").str.contains("death certificate", na=False).astype(int)
            )
            df["has_discharge"] = (
                df["documents_submitted"].fillna("").str.contains("discharge summary", na=False).astype(int)
            )
            
            avg_docs = df["num_docs"].mean()
            fir_count = df["has_fir"].sum()
            photo_count = df["has_photos"].sum()
            
            logger.debug(
                f"Document features: avg_docs={avg_docs:.1f}, "
                f"FIRs={fir_count}, photos={photo_count}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating document features: {e}")
            raise

    def create_embeddings(self, narratives: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate Hinglish narrative embeddings with PCA.
        
        Args:
            narratives: List of claim narrative texts
            batch_size: Batch size for encoding (default: 64)
            
        Returns:
            PCA-reduced embeddings (n_samples, pca_dims)
        """
        logger.info(f"Generating embeddings for {len(narratives)} narratives...")
        
        try:
            self._load_embedder()
            
            embeddings = self.embedder.encode(
                narratives,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            
            logger.debug(f"Raw embeddings shape: {embeddings.shape}")
            
            # 🔧 FIX BUG #9: Handle single claims (n_samples < pca_dims)
            n_samples = embeddings.shape[0]
            embedding_dim = embeddings.shape[1]
            
            if n_samples < self.pca_dims:
                logger.warning(
                    f"Sample size ({n_samples}) < PCA dims ({self.pca_dims}). "
                    f"Using adjusted embeddings without PCA reduction."
                )
                
                # Adjust embedding dimensions to match pca_dims
                if embedding_dim < self.pca_dims:
                    # Pad with zeros if embedding is smaller than pca_dims
                    padding = np.zeros((n_samples, self.pca_dims - embedding_dim))
                    reduced = np.hstack([embeddings, padding])
                    logger.debug(f"Padded embeddings from {embedding_dim} to {self.pca_dims} dimensions")
                else:
                    # Truncate if embedding is larger than pca_dims
                    reduced = embeddings[:, :self.pca_dims]
                    logger.debug(f"Truncated embeddings from {embedding_dim} to {self.pca_dims} dimensions")
                
                logger.info(f"Adjusted embeddings shape: {reduced.shape}")
                return reduced
            
            # Normal PCA flow for multiple samples
            if not self.is_fitted:
                logger.info(f"Fitting PCA to reduce to {self.pca_dims} dimensions...")
                
                # Ensure we don't try to fit with more components than samples
                effective_dims = min(self.pca_dims, n_samples)
                
                if effective_dims < self.pca_dims:
                    logger.warning(
                        f"Reducing PCA dims from {self.pca_dims} to {effective_dims} "
                        f"(limited by sample size)"
                    )
                    self.pca = PCA(n_components=effective_dims, random_state=42)
                
                reduced = self.pca.fit_transform(embeddings)
                self.is_fitted = True
                variance_retained = self.pca.explained_variance_ratio_.sum()
                logger.success(f"PCA fitted: {variance_retained:.2%} variance retained")
                
                # Pad to target dims if necessary
                if reduced.shape[1] < self.pca_dims:
                    padding = np.zeros((reduced.shape[0], self.pca_dims - reduced.shape[1]))
                    reduced = np.hstack([reduced, padding])
                    logger.debug(f"Padded PCA output to {self.pca_dims} dimensions")
                    
            else:
                logger.debug("Transforming embeddings with existing PCA...")
                reduced = self.pca.transform(embeddings)
                
                # Pad to target dims if necessary
                if reduced.shape[1] < self.pca_dims:
                    padding = np.zeros((reduced.shape[0], self.pca_dims - reduced.shape[1]))
                    reduced = np.hstack([reduced, padding])
            
            logger.debug(f"Final embeddings shape: {reduced.shape}")
            
            return reduced
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def engineer_features(
        self,
        df: pd.DataFrame,
        narrative_col: str = "narrative",
        keep_ids: bool = True,
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
        logger.info(f"Starting feature engineering pipeline for {len(df)} claims...")
        
        try:
            logger.info("Step 1/5: Creating time-aware aggregation features...")
            df = self.create_time_features(df)
            
            logger.info("Step 2/5: Creating numeric features...")
            df = self.create_numeric_features(df)
            
            logger.info("Step 3/5: Creating document features...")
            df = self.create_document_features(df)
            
            logger.info("Step 4/5: One-hot encoding categorical features...")
            df_cat = pd.get_dummies(df[self.categorical_features], prefix=self.categorical_features)
            logger.debug(f"Categorical features: {df_cat.shape[1]} columns created")
            
            logger.info("Step 5/5: Generating narrative embeddings...")
            narratives = df[narrative_col].fillna("").tolist()
            embeddings = self.create_embeddings(narratives)
            df_emb = pd.DataFrame(embeddings, columns=self.embedding_cols)
            
            logger.info("Assembling final feature matrix...")
            feature_df = pd.concat(
                [
                    df[self.numeric_features],
                    df_cat,
                    df_emb,
                ],
                axis=1,
            )
            
            # Keep IDs if requested
            if keep_ids:
                id_cols = ["claim_id", "claimant_id", "policy_id"]
                existing_ids = [c for c in id_cols if c in df.columns]
                if existing_ids:
                    ids_df = df[existing_ids].copy()
                    feature_df = pd.concat([ids_df, feature_df], axis=1)
                    logger.info(f"Kept ID columns: {existing_ids}")
            else:
                logger.info("ID columns excluded from feature matrix")
            
            logger.success(
                f"Feature engineering complete! "
                f"Shape: {feature_df.shape} ({feature_df.shape[0]} rows, {feature_df.shape[1]} features)"
            )
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            raise

    def validate_no_leakage(self, feature_df: pd.DataFrame) -> bool:
        """Validate no label-leakage columns exist.
        
        Args:
            feature_df: Engineered feature DataFrame
            
        Returns:
            True if no leakage detected, raises ValueError otherwise
        """
        logger.debug("Validating feature leakage...")
        
        forbidden_substrings = ["fraud", "score", "redflag", "flag"]
        
        blocked_cols = []
        for col in feature_df.columns:
            col_lower = col.lower()
            if any(sub in col_lower for sub in forbidden_substrings):
                blocked_cols.append(col)
        
        if blocked_cols:
            logger.error(f"LEAKAGE DETECTED! Forbidden columns: {blocked_cols}")
            raise ValueError(
                f"⚠️ LEAKAGE DETECTED! Found forbidden columns: {blocked_cols}\n"
                f"These columns may contain label information and must be removed."
            )
        
        logger.success("✅ Leakage validation passed: No forbidden columns found.")
        return True
