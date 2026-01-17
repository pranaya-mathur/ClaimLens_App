"""ML-based fraud scoring using CatBoost classifier.

Provides fraud probability scoring for insurance claims using a trained
CatBoost model with leakage-free feature engineering.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from catboost import CatBoostClassifier
import warnings
from loguru import logger

warnings.filterwarnings("ignore")


class MLFraudScorer:
    """CatBoost-based fraud scorer for insurance claims.
    
    Trained on 50K Hinglish claims with 84.8% AUC and 42.8% F1 score.
    Uses 145 features including narrative embeddings, behavioral patterns,
    and document indicators.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
    ):
        """Initialize ML fraud scorer.
        
        Args:
            model_path: Path to trained CatBoost .cbm model file
            metadata_path: Path to model metadata JSON file
            threshold: Fraud probability threshold (default: 0.5)
        """
        self.model = None
        self.metadata = {}
        self.feature_importance = None
        self.threshold = threshold
        self.expected_features: Optional[List[str]] = None
        
        if model_path:
            self.load_model(model_path)
        if metadata_path:
            self.load_metadata(metadata_path)

    def load_model(self, model_path: Union[str, Path]):
        """Load trained CatBoost model."""
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading CatBoost model from {model_path}...")
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        
        self.expected_features = list(self.model.feature_names_)
        logger.info(f"Model loaded: {len(self.expected_features)} features")
        
        # Extract feature importance
        importances = self.model.get_feature_importance()
        self.feature_importance = pd.DataFrame({
            "feature": self.expected_features,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def load_metadata(self, metadata_path: Union[str, Path]):
        """Load model training metadata."""
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            logger.debug(f"Metadata file not found: {metadata_path}")
            return
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        logger.debug(f"Metadata loaded: AUC={self.metadata.get('auc_roc', 'N/A')}")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align input features with model's expected feature schema.
        
        Handles missing/extra features to ensure reliable predictions.
        Adds missing features with zeros and removes extras.
        """
        if self.expected_features is None:
            logger.error("Model not loaded - cannot align features")
            raise ValueError("Model must be loaded before feature alignment")
        
        current_features = set(features.columns)
        expected_features_set = set(self.expected_features)
        
        missing_features = expected_features_set - current_features
        extra_features = current_features - expected_features_set
        
        if missing_features:
            logger.debug(f"Adding {len(missing_features)} missing features with zeros")
            for feat in missing_features:
                features[feat] = 0
        
        if extra_features:
            logger.debug(f"Removing {len(extra_features)} extra features")
        
        # Reorder columns to match training
        features = features[self.expected_features]
        
        return features

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities with automatic feature alignment.
        
        Args:
            features: Feature DataFrame (will be automatically aligned)
            
        Returns:
            Array of fraud probabilities (0-1)
        """
        if self.model is None:
            logger.error("Model not loaded")
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Align features with model expectations
        features = self._align_features(features)
        
        # Predict probabilities (return fraud class probability)
        proba = self.model.predict_proba(features)[:, 1]
        
        return proba

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels based on threshold."""
        proba = self.predict_proba(features)
        return (proba >= self.threshold).astype(int)

    def score_claim(
        self,
        features: pd.DataFrame,
        return_details: bool = False,
    ) -> Union[float, Dict[str, any]]:
        """Score a single claim for fraud risk.
        
        Args:
            features: Feature DataFrame (single row)
            return_details: Whether to return detailed scoring info
            
        Returns:
            Fraud probability (0-1) or detailed dict if return_details=True
        """
        if len(features) != 1:
            raise ValueError("score_claim expects a single-row DataFrame")
        
        fraud_prob = self.predict_proba(features)[0]
        fraud_label = int(fraud_prob >= self.threshold)
        
        if not return_details:
            return float(fraud_prob)
        
        # Get top contributing features
        feature_values = features.iloc[0].to_dict()
        top_features = self.feature_importance.head(10).to_dict("records")
        
        # Risk level categorization
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.5:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            "fraud_probability": float(fraud_prob),
            "fraud_prediction": fraud_label,
            "risk_level": risk_level,
            "threshold": self.threshold,
            "top_features": top_features,
            "model_metrics": {
                "auc_roc": self.metadata.get("auc_roc"),
                "f1_score": self.metadata.get("f1_score"),
                "best_iteration": self.metadata.get("best_iteration"),
            },
        }

    def score_batch(
        self,
        features: pd.DataFrame,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Score multiple claims for fraud risk."""
        fraud_probs = self.predict_proba(features)
        fraud_labels = (fraud_probs >= self.threshold).astype(int)
        
        if not return_dataframe:
            return fraud_probs
        
        result_df = pd.DataFrame({
            "fraud_probability": fraud_probs,
            "fraud_prediction": fraud_labels,
        })
        
        result_df["risk_level"] = pd.cut(
            fraud_probs,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
        
        return result_df

    def analyze_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Analyze model performance across different thresholds."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        if thresholds is None:
            thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        
        results = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # False positive rate
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results.append({
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
            })
        
        return pd.DataFrame(results)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Load model first.")
        
        return self.feature_importance.head(top_n)

    def explain_prediction(
        self,
        features: pd.DataFrame,
        top_n: int = 10,
    ) -> Dict[str, any]:
        """Explain fraud prediction for a single claim."""
        if len(features) != 1:
            raise ValueError("explain_prediction expects a single-row DataFrame")
        
        fraud_prob = self.predict_proba(features)[0]
        
        top_features = self.feature_importance.head(top_n)
        feature_values = features.iloc[0]
        
        explanations = []
        for _, row in top_features.iterrows():
            feat_name = row["feature"]
            feat_importance = row["importance"]
            feat_value = feature_values.get(feat_name, "N/A")
            
            explanations.append({
                "feature": feat_name,
                "value": float(feat_value) if isinstance(feat_value, (int, float, np.number)) else str(feat_value),
                "importance": float(feat_importance),
            })
        
        return {
            "fraud_probability": float(fraud_prob),
            "fraud_prediction": int(fraud_prob >= self.threshold),
            "top_contributing_features": explanations,
        }

    def update_threshold(self, new_threshold: float):
        """Update fraud classification threshold."""
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = new_threshold
        logger.info(f"Threshold updated to {new_threshold}")

    def summary(self) -> Dict[str, any]:
        """Get model summary information."""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_type": "CatBoostClassifier",
            "n_features": len(self.expected_features) if self.expected_features else 0,
            "threshold": self.threshold,
            "training_metrics": self.metadata,
            "top_5_features": self.feature_importance.head(5).to_dict("records") if self.feature_importance is not None else None,
        }
