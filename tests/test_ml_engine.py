"""Comprehensive tests for ML Engine module.

Tests cover:
- FeatureEngineer: embedding generation, leakage validation, feature shapes
- MLFraudScorer: model loading, predictions, threshold behavior
- Integration: end-to-end feature engineering + scoring
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# ============================================================================
# FEATURE ENGINEER TESTS
# ============================================================================

class TestFeatureEngineer:
    """Tests for FeatureEngineer class"""

    @pytest.fixture
    def sample_claim_data(self):
        """Generate sample claim data for testing"""
        return pd.DataFrame([
            {
                "claim_id": "CLM-001",
                "claimant_id": "CLNT-001",
                "policy_id": "POL-001",
                "product": "motor",
                "city": "Mumbai",
                "subtype": "accident",
                "claim_amount": 50000,
                "days_since_policy_start": 365,
                "narrative": "Meri gaadi ka accident ho gaya highway pe",
                "documents_submitted": "FIR,photos,estimate",
                "incident_date": "2025-01-15"
            },
            {
                "claim_id": "CLM-002",
                "claimant_id": "CLNT-001",  # Same claimant
                "policy_id": "POL-001",
                "product": "motor",
                "city": "Delhi",
                "subtype": "theft",
                "claim_amount": 80000,
                "days_since_policy_start": 400,
                "narrative": "Car chori ho gayi parking se",
                "documents_submitted": "FIR,photos",
                "incident_date": "2025-02-20"
            }
        ])

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initializes correctly"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer(pca_dims=100)
        
        assert fe.pca_dims == 100
        assert fe.model_name == "AkshitaS/bhasha-embed-v0"
        assert fe.embedder is None  # Lazy loaded
        assert fe.pca is not None
        assert not fe.is_fitted

    def test_time_features_creation(self, sample_claim_data):
        """Test time-aware feature creation"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_time = fe.create_time_features(sample_claim_data)
        
        # Check new columns exist
        assert "claimant_claim_count" in df_with_time.columns
        assert "claimant_total_claimed" in df_with_time.columns
        assert "claimant_avg_claim" in df_with_time.columns
        assert "policy_claim_count" in df_with_time.columns
        assert "days_since_last_claim" in df_with_time.columns
        assert "rapid_claims" in df_with_time.columns
        assert "is_first_claim" in df_with_time.columns
        
        # Check values for second claim (same claimant)
        assert df_with_time.iloc[1]["claimant_claim_count"] == 2
        assert df_with_time.iloc[1]["claimant_total_claimed"] == 130000
        assert df_with_time.iloc[0]["is_first_claim"] == 1

    def test_numeric_features_creation(self, sample_claim_data):
        """Test numeric feature engineering"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_numeric = fe.create_numeric_features(sample_claim_data)
        
        assert "claim_amount_log" in df_with_numeric.columns
        assert "policy_age_months" in df_with_numeric.columns
        assert "is_recent_policy" in df_with_numeric.columns
        
        # Check log transformation
        assert df_with_numeric.iloc[0]["claim_amount_log"] == np.log1p(50000)
        
        # Check policy age calculation
        assert df_with_numeric.iloc[0]["policy_age_months"] == 365 / 30.0

    def test_document_features_creation(self, sample_claim_data):
        """Test document-based feature extraction"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_docs = fe.create_document_features(sample_claim_data)
        
        assert "num_docs" in df_with_docs.columns
        assert "has_fir" in df_with_docs.columns
        assert "has_photos" in df_with_docs.columns
        
        # Check document parsing
        assert df_with_docs.iloc[0]["num_docs"] == 3  # FIR, photos, estimate
        assert df_with_docs.iloc[0]["has_fir"] == 1
        assert df_with_docs.iloc[0]["has_photos"] == 1

    @patch('src.ml_engine.feature_engineer.SentenceTransformer')
    def test_embeddings_generation(self, mock_transformer, sample_claim_data):
        """Test narrative embedding generation with mocked model"""
        from src.ml_engine import FeatureEngineer
        
        # Mock embedding model
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(2, 384)  # 384-dim Bhasha-Embed output
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        fe = FeatureEngineer(pca_dims=100)
        narratives = sample_claim_data["narrative"].tolist()
        
        embeddings = fe.create_embeddings(narratives)
        
        # Check shape
        assert embeddings.shape == (2, 100)  # PCA reduced to 100 dims
        assert fe.is_fitted  # PCA should be fitted

    def test_leakage_validation_passes(self):
        """Test leakage validation passes for clean features"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer()
        clean_features = pd.DataFrame({
            "claim_amount": [50000],
            "claimant_claim_count": [2],
            "emb_0": [0.123],
            "product_motor": [1]
        })
        
        # Should not raise
        result = fe.validate_no_leakage(clean_features)
        assert result is True

    def test_leakage_validation_fails(self):
        """Test leakage validation detects forbidden columns"""
        from src.ml_engine import FeatureEngineer
        
        fe = FeatureEngineer()
        leaky_features = pd.DataFrame({
            "claim_amount": [50000],
            "fraud_score": [0.87],  # FORBIDDEN
            "red_flag_count": [3]  # FORBIDDEN
        })
        
        with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
            fe.validate_no_leakage(leaky_features)


# ============================================================================
# ML FRAUD SCORER TESTS
# ============================================================================

class TestMLFraudScorer:
    """Tests for MLFraudScorer class"""

    @pytest.fixture
    def mock_catboost_model(self):
        """Create mock CatBoost model"""
        mock_model = MagicMock()
        mock_model.feature_names_ = [
            "claim_amount", "claimant_claim_count", "emb_0", "emb_1"
        ]
        mock_model.get_feature_importance.return_value = np.array([0.5, 0.3, 0.15, 0.05])
        return mock_model

    @pytest.fixture
    def sample_features(self):
        """Sample feature DataFrame"""
        return pd.DataFrame({
            "claim_amount": [50000],
            "claimant_claim_count": [2],
            "emb_0": [0.123],
            "emb_1": [-0.456]
        })

    def test_scorer_initialization(self):
        """Test MLFraudScorer initialization"""
        from src.ml_engine import MLFraudScorer
        
        scorer = MLFraudScorer(threshold=0.5)
        
        assert scorer.model is None  # Not loaded yet
        assert scorer.threshold == 0.5
        assert scorer.metadata == {}

    @patch('src.ml_engine.ml_scorer.CatBoostClassifier')
    def test_predict_proba(self, mock_catboost_cls, mock_catboost_model, sample_features):
        """Test fraud probability prediction"""
        from src.ml_engine import MLFraudScorer
        
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.feature_names_ = ["claim_amount", "claimant_claim_count", "emb_0", "emb_1"]
        mock_instance.predict_proba.return_value = np.array([[0.75, 0.25]])  # [legit, fraud]
        mock_instance.get_feature_importance.return_value = np.array([0.5, 0.3, 0.15, 0.05])
        mock_catboost_cls.return_value = mock_instance
        
        scorer = MLFraudScorer()
        scorer.model = mock_instance
        scorer.feature_importance = pd.DataFrame({
            "feature": ["claim_amount", "claimant_claim_count"],
            "importance": [0.5, 0.3]
        })
        
        proba = scorer.predict_proba(sample_features)
        
        assert len(proba) == 1
        assert proba[0] == 0.25  # Fraud probability

    @patch('src.ml_engine.ml_scorer.CatBoostClassifier')
    def test_predict_binary(self, mock_catboost_cls, sample_features):
        """Test binary fraud prediction"""
        from src.ml_engine import MLFraudScorer
        
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.feature_names_ = ["claim_amount", "claimant_claim_count", "emb_0", "emb_1"]
        mock_instance.predict_proba.return_value = np.array([[0.25, 0.75]])  # High fraud prob
        mock_instance.get_feature_importance.return_value = np.array([0.5, 0.3, 0.15, 0.05])
        mock_catboost_cls.return_value = mock_instance
        
        scorer = MLFraudScorer(threshold=0.5)
        scorer.model = mock_instance
        scorer.feature_importance = pd.DataFrame({
            "feature": ["claim_amount"],
            "importance": [0.5]
        })
        
        prediction = scorer.predict(sample_features)
        
        assert len(prediction) == 1
        assert prediction[0] == 1  # Fraud (0.75 > 0.5)

    def test_threshold_update(self):
        """Test threshold update functionality"""
        from src.ml_engine import MLFraudScorer
        
        scorer = MLFraudScorer(threshold=0.5)
        scorer.update_threshold(0.7)
        
        assert scorer.threshold == 0.7

    def test_threshold_validation(self):
        """Test threshold validation (must be 0-1)"""
        from src.ml_engine import MLFraudScorer
        
        scorer = MLFraudScorer()
        
        with pytest.raises(ValueError):
            scorer.update_threshold(1.5)  # Invalid
        
        with pytest.raises(ValueError):
            scorer.update_threshold(-0.1)  # Invalid

    @patch('src.ml_engine.ml_scorer.CatBoostClassifier')
    def test_score_batch(self, mock_catboost_cls):
        """Test batch scoring functionality"""
        from src.ml_engine import MLFraudScorer
        
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.feature_names_ = ["claim_amount", "claimant_claim_count", "emb_0", "emb_1"]
        mock_instance.predict_proba.return_value = np.array([
            [0.8, 0.2],  # Low fraud
            [0.3, 0.7],  # High fraud
            [0.6, 0.4]   # Medium fraud
        ])
        mock_instance.get_feature_importance.return_value = np.array([0.5, 0.3, 0.15, 0.05])
        mock_catboost_cls.return_value = mock_instance
        
        scorer = MLFraudScorer(threshold=0.5)
        scorer.model = mock_instance
        scorer.feature_importance = pd.DataFrame({
            "feature": ["claim_amount"],
            "importance": [0.5]
        })
        
        batch_features = pd.DataFrame({
            "claim_amount": [50000, 80000, 60000],
            "claimant_claim_count": [1, 3, 2],
            "emb_0": [0.1, 0.2, 0.3],
            "emb_1": [-0.1, -0.2, -0.3]
        })
        
        results = scorer.score_batch(batch_features, return_dataframe=True)
        
        assert len(results) == 3
        assert "fraud_probability" in results.columns
        assert "fraud_prediction" in results.columns
        assert "risk_level" in results.columns
        
        # Check risk categorization
        assert results.iloc[0]["risk_level"] == "LOW"     # 0.2
        assert results.iloc[1]["risk_level"] == "HIGH"    # 0.7
        assert results.iloc[2]["risk_level"] == "MEDIUM"  # 0.4


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMLEngineIntegration:
    """Integration tests for end-to-end ML Engine workflow"""

    @pytest.fixture
    def integration_claim_data(self):
        """Full claim data for integration testing"""
        return pd.DataFrame([{
            "claim_id": "CLM-INT-001",
            "claimant_id": "CLNT-INT-001",
            "policy_id": "POL-INT-001",
            "product": "motor",
            "city": "Mumbai",
            "subtype": "accident",
            "claim_amount": 75000,
            "days_since_policy_start": 180,
            "narrative": "Highway pe accident ho gaya late night",
            "documents_submitted": "FIR,photos,estimate",
            "incident_date": "2025-03-01"
        }])

    @patch('src.ml_engine.feature_engineer.SentenceTransformer')
    @patch('src.ml_engine.ml_scorer.CatBoostClassifier')
    def test_end_to_end_scoring(self, mock_catboost_cls, mock_transformer, integration_claim_data):
        """Test complete workflow: data -> features -> score"""
        from src.ml_engine import FeatureEngineer, MLFraudScorer
        
        # Mock embedding model
        mock_embed_model = MagicMock()
        mock_embed_model.encode.return_value = np.random.rand(1, 384)
        mock_transformer.return_value = mock_embed_model
        
        # Mock CatBoost model
        mock_cb_model = MagicMock()
        # Generate expected feature names (148 total)
        feature_names = (
            ["days_since_policy_start", "claim_amount", "claimant_claim_count",
             "claimant_total_claimed", "claimant_avg_claim", "policy_claim_count",
             "policy_total_claimed", "policy_avg_claim", "days_since_last_claim",
             "rapid_claims", "is_first_claim", "claim_amount_log",
             "policy_age_months", "is_recent_policy", "num_docs",
             "has_fir", "has_photos", "has_death_cert", "has_discharge"] +
            ["product_motor", "city_Mumbai", "subtype_accident"] +  # Categorical
            [f"emb_{i}" for i in range(100)]  # Embeddings
        )
        mock_cb_model.feature_names_ = feature_names
        mock_cb_model.predict_proba.return_value = np.array([[0.65, 0.35]])
        mock_cb_model.get_feature_importance.return_value = np.random.rand(len(feature_names))
        mock_catboost_cls.return_value = mock_cb_model
        
        # Feature engineering
        fe = FeatureEngineer(pca_dims=100)
        features = fe.engineer_features(integration_claim_data, drop_ids=False)
        
        # Leakage validation
        fe.validate_no_leakage(features)
        
        # Scoring
        scorer = MLFraudScorer(threshold=0.5)
        scorer.model = mock_cb_model
        scorer.feature_importance = pd.DataFrame({
            "feature": feature_names[:10],
            "importance": np.random.rand(10)
        })
        
        fraud_prob = scorer.predict_proba(features)
        
        # Assertions
        assert len(fraud_prob) == 1
        assert 0 <= fraud_prob[0] <= 1
        assert features.shape[1] >= 100  # At least 100+ features


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestMLEngineErrors:
    """Test error handling in ML Engine"""

    def test_scorer_missing_features_error(self):
        """Test error when features don't match model expectations"""
        from src.ml_engine import MLFraudScorer
        
        mock_model = MagicMock()
        mock_model.feature_names_ = ["feature_a", "feature_b", "feature_c"]
        
        scorer = MLFraudScorer()
        scorer.model = mock_model
        
        wrong_features = pd.DataFrame({
            "feature_a": [1],
            "feature_x": [2]  # Missing feature_b and feature_c
        })
        
        with pytest.raises(ValueError, match="Missing features"):
            scorer.predict_proba(wrong_features)

    def test_scorer_not_loaded_error(self):
        """Test error when trying to predict without loading model"""
        from src.ml_engine import MLFraudScorer
        
        scorer = MLFraudScorer()
        features = pd.DataFrame({"dummy": [1]})
        
        with pytest.raises(ValueError, match="Model not loaded"):
            scorer.predict_proba(features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
