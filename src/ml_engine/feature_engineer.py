"""Feature engineering pipeline for fraud detection.

Extracts features from claim narratives using Bhasha-Embed for Hinglish text,
combined with behavioral and document-based features.

Note: Updated column alignment logic in Dec 2025 to fix model mismatch.
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
    - Time-aware aggregation features
    - Behavioral patterns
    - Document presence indicators
    - Categorical encodings
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
            model_name: HuggingFace model for embeddings
            expected_features: List of feature names from trained model for alignment
        """
        self.pca_dims = pca_dims
        self.model_name = model_name
        self.embedder = None
        self.pca = PCA(n_components=pca_dims, random_state=42)
        self.is_fitted = False
        self.expected_features = expected_features
        
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
        
        self.categorical_features = ["product_type", "city", "claim_subtype"]
        self.categorical_prefixes = ["product", "city", "subtype"]
        
        self.embedding_cols = [f"emb_{i}" for i in range(pca_dims)]
        
        logger.info(f"FeatureEngineer initialized ({pca_dims} PCA dims)")
        if expected_features:
            logger.debug(f"Feature alignment: {len(expected_features)} expected features")

    def _load_embedder(self):
        """Lazy load embedding model."""
        if self.embedder is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.embedder = SentenceTransformer(self.model_name)
                logger.info("Embedder loaded")
            except Exception as e:
                logger.error(f"Failed to load embedder: {e}")
                raise

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-aware aggregation features without label leakage.
        
        Args:
            df: DataFrame with incident_date, claimant_id, policy_id, claim_amount
            
        Returns:
            DataFrame with added time-based features
        """
        logger.info("Creating time-aware features...")
        
        # Support both incident_date and claim_date
        date_col = None
        if "incident_date" in df.columns:
            date_col = "incident_date"
        elif "claim_date" in df.columns:
            date_col = "claim_date"
            df["incident_date"] = df["claim_date"]
        else:
            logger.error("Missing date column in DataFrame")
            raise ValueError(
                "Missing required column: 'incident_date' or 'claim_date'"
            )
        
        try:
            df["incident_date"] = pd.to_datetime(df["incident_date"])
        except Exception as e:
            logger.error(f"Failed to parse {date_col}: {e}")
            raise ValueError(f"Invalid '{date_col}' format. Expected ISO format (YYYY-MM-DD).")
        
        null_count = df["incident_date"].isna().sum()
        if null_count > 0:
            df["incident_date"] = df["incident_date"].fillna(pd.Timestamp.now())
        
        try:
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
            
            # Time-based features
            df_sorted["days_since_last_claim"] = (
                df_sorted.groupby("claimant_id")["incident_date"].diff().dt.days
            )
            df_sorted["days_since_last_claim"] = df_sorted["days_since_last_claim"].fillna(9999)
            
            df_sorted["rapid_claims"] = (df_sorted["days_since_last_claim"] < 30).astype(int)
            df_sorted["is_first_claim"] = (df_sorted["claimant_claim_count"] == 1).astype(int)
            
            rapid_claim_count = df_sorted["rapid_claims"].sum()
            first_claim_count = df_sorted["is_first_claim"].sum()
            
            logger.info(
                f"Time features: {rapid_claim_count} rapid claims, "
                f"{first_claim_count} first-time"
            )
            
            return df_sorted
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            raise

    def create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric features from claim data."""
        try:
            df["claim_amount_log"] = np.log1p(df["claim_amount"])
            df["policy_age_months"] = df["days_since_policy_start"] / 30.0
            df["is_recent_policy"] = (df["days_since_policy_start"] < 180).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating numeric features: {e}")
            raise

    def create_document_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract document-based features."""
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating document features: {e}")
            raise

    def create_embeddings(self, narratives: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate Hinglish narrative embeddings with PCA.
        
        Args:
            narratives: List of claim narrative texts
            batch_size: Batch size for encoding
            
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
            
            n_samples = embeddings.shape[0]
            embedding_dim = embeddings.shape[1]
            
            # Handle small sample size
            if n_samples < self.pca_dims:
                logger.debug(f"Small sample ({n_samples}) - skipping PCA")
                
                if embedding_dim < self.pca_dims:
                    padding = np.zeros((n_samples, self.pca_dims - embedding_dim))
                    reduced = np.hstack([embeddings, padding])
                else:
                    reduced = embeddings[:, :self.pca_dims]
                
                return reduced
            
            # Fit or transform with PCA
            if not self.is_fitted:
                logger.info(f"Fitting PCA ({self.pca_dims} dims)...")
                
                effective_dims = min(self.pca_dims, n_samples)
                
                if effective_dims < self.pca_dims:
                    self.pca = PCA(n_components=effective_dims, random_state=42)
                
                reduced = self.pca.fit_transform(embeddings)
                self.is_fitted = True
                variance = self.pca.explained_variance_ratio_.sum()
                logger.info(f"PCA fitted ({variance:.2%} variance retained)")
                
                if reduced.shape[1] < self.pca_dims:
                    padding = np.zeros((reduced.shape[0], self.pca_dims - reduced.shape[1]))
                    reduced = np.hstack([reduced, padding])
                    
            else:
                reduced = self.pca.transform(embeddings)
                
                if reduced.shape[1] < self.pca_dims:
                    padding = np.zeros((reduced.shape[0], self.pca_dims - reduced.shape[1]))
                    reduced = np.hstack([reduced, padding])
            
            return reduced
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to match trained model expectations."""
        id_cols = [c for c in ['claim_id', 'claimant_id', 'policy_id'] if c in df.columns]
        categorical_prefixes = ('product_', 'city_', 'subtype_')
        embedding_prefix = 'emb_'
        
        rename_map = {}
        for col in df.columns:
            if col in id_cols:
                continue
            
            if col.startswith(categorical_prefixes):
                continue
            
            if col.startswith(embedding_prefix):
                continue
            
            normalized = col.replace('_', '').lower()
            if col != normalized:
                rename_map[col] = normalized
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Renamed {len(rename_map)} columns")
        
        return df

    def _align_features_with_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align engineered features with model's expected feature schema."""
        if self.expected_features is None:
            return df
        
        logger.info("Aligning features with model schema...")
        
        id_cols = [c for c in ['claim_id', 'claimant_id', 'policy_id'] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in id_cols]
        
        current_features = set(feature_cols)
        expected_features = set(self.expected_features)
        
        missing_features = expected_features - current_features
        extra_features = current_features - expected_features
        
        if missing_features:
            logger.debug(f"Adding {len(missing_features)} missing features")
            for feat in missing_features:
                df[feat] = 0
        
        if extra_features:
            logger.debug(f"Removing {len(extra_features)} extra features")
            df = df.drop(columns=list(extra_features))
        
        final_cols = id_cols + self.expected_features
        df = df[[c for c in final_cols if c in df.columns]]
        
        logger.info(f"Feature alignment complete ({len(self.expected_features)} features)")
        
        return df

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
            keep_ids: Whether to keep ID columns in output
            
        Returns:
            Feature matrix with embeddings + engineered features
        """
        logger.info(f"Feature engineering for {len(df)} claims...")
        
        try:
            df = self.create_time_features(df)
            df = self.create_numeric_features(df)
            df = self.create_document_features(df)
            
            logger.info("One-hot encoding categoricals...")
            df_cat = pd.get_dummies(
                df[self.categorical_features], 
                prefix=self.categorical_prefixes
            )
            
            logger.info("Generating narrative embeddings...")
            narratives = df[narrative_col].fillna("").tolist()
            embeddings = self.create_embeddings(narratives)
            df_emb = pd.DataFrame(embeddings, columns=self.embedding_cols)
            
            logger.info("Assembling feature matrix...")
            feature_df = pd.concat(
                [
                    df[self.numeric_features],
                    df_cat,
                    df_emb,
                ],
                axis=1,
            )
            
            if keep_ids:
                id_cols = ["claim_id", "claimant_id", "policy_id"]
                existing_ids = [c for c in id_cols if c in df.columns]
                if existing_ids:
                    ids_df = df[existing_ids].copy()
                    feature_df = pd.concat([ids_df, feature_df], axis=1)
            
            feature_df = self._normalize_column_names(feature_df)
            feature_df = self._align_features_with_model(feature_df)
            
            logger.info(
                f"Feature engineering complete: "
                f"{feature_df.shape[0]} rows, {feature_df.shape[1]} features"
            )
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def validate_no_leakage(self, feature_df: pd.DataFrame) -> bool:
        """Validate no label-leakage columns exist.
        
        Args:
            feature_df: Engineered feature DataFrame
            
        Returns:
            True if no leakage detected
        
        Raises:
            ValueError if leakage detected
        """
        forbidden_substrings = ["fraud", "score", "redflag", "flag"]
        
        blocked_cols = []
        for col in feature_df.columns:
            col_lower = col.lower()
            if any(sub in col_lower for sub in forbidden_substrings):
                blocked_cols.append(col)
        
        if blocked_cols:
            logger.error(f"Leakage detected: {blocked_cols}")
            raise ValueError(
                f"Leakage detected! Forbidden columns: {blocked_cols}"
            )
        
        return True
