"""
Unit tests for PAN Forgery Detector

Tests cover:
- 4-channel model initialization
- ELA generation
- Inference pipeline
- Threshold modes
- Batch processing
- Error handling
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.cv_engine.pan_detector import (
    PANForgeryDetector,
    PANVerificationResult
)


class TestPANVerificationResult:
    """Test PANVerificationResult dataclass"""
    
    def test_result_creation(self):
        """Test creating verification result"""
        result = PANVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.92,
            forgery_probability=0.08,
            clean_probability=0.92,
            threshold=0.49
        )
        
        assert result.document_type == "PAN"
        assert result.verdict == "CLEAN"
        assert result.confidence == 0.92
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = PANVerificationResult(
            document_type="PAN",
            image_path="pan.jpg",
            verdict="FORGED",
            confidence=0.88,
            forgery_probability=0.88,
            clean_probability=0.12,
            threshold=0.49
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["verdict"] == "FORGED"
        assert result_dict["forgery_probability"] == 0.88
    
    def test_is_forged(self):
        """Test is_forged method"""
        forged_result = PANVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="FORGED",
            confidence=0.95,
            forgery_probability=0.95,
            clean_probability=0.05,
            threshold=0.49
        )
        assert forged_result.is_forged() is True
        
        clean_result = PANVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.98,
            forgery_probability=0.02,
            clean_probability=0.98,
            threshold=0.49
        )
        assert clean_result.is_forged() is False
    
    def test_is_clean(self):
        """Test is_clean method"""
        clean_result = PANVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.96,
            forgery_probability=0.04,
            clean_probability=0.96,
            threshold=0.49
        )
        assert clean_result.is_clean() is True


class TestPANForgeryDetector:
    """Test PANForgeryDetector class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model
    
    @pytest.fixture
    def mock_detector(self, mock_model, tmp_path):
        """Create detector with mocked model loading"""
        model_file = tmp_path / "resnet50_finetuned_after_strong_forgeries.pth"
        model_file.touch()
        
        with patch('src.cv_engine.pan_detector.PANForgeryDetectorCNN') as mock_cnn:
            mock_cnn.return_value = mock_model
            
            with patch('torch.load') as mock_load:
                mock_load.return_value = {}  # State dict
                
                detector = PANForgeryDetector(
                    model_path=model_file,
                    device="cpu",
                    threshold=0.49
                )
                detector.model = mock_model
                
                return detector
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create sample test image"""
        img = Image.new('RGB', (320, 320), color='green')
        img_path = tmp_path / "test_pan.jpg"
        img.save(img_path)
        return img_path
    
    def test_initialization_defaults(self, mock_detector):
        """Test detector initializes with correct defaults"""
        assert mock_detector.device.type == "cpu"
        assert mock_detector.threshold == 0.49  # F1-optimal
        assert mock_detector.input_size == (320, 320)
        assert mock_detector.ela_quality == 90
    
    def test_model_config(self, mock_detector):
        """Test model configuration"""
        config = mock_detector.MODEL_CONFIG
        assert config["architecture"] == "ResNet50_4Channel"
        assert config["num_channels"] == 4
        assert config["thresholds"]["f1_optimal"] == 0.49
        assert config["performance"]["auc"] == 0.9996
    
    def test_threshold_modes(self, mock_detector):
        """Test different threshold modes"""
        config = mock_detector.MODEL_CONFIG["thresholds"]
        
        assert config["balanced"] == 0.50
        assert config["f1_optimal"] == 0.49
        assert config["precision_oriented"] == 0.48
    
    def test_set_threshold_with_mode(self, mock_detector):
        """Test setting threshold by mode"""
        mock_detector.set_threshold(mode="precision_oriented")
        assert mock_detector.threshold == 0.48
        
        mock_detector.set_threshold(mode="balanced")
        assert mock_detector.threshold == 0.50
    
    def test_set_threshold_custom(self, mock_detector):
        """Test setting custom threshold"""
        mock_detector.set_threshold(0.6)
        assert mock_detector.threshold == 0.6
    
    def test_set_threshold_invalid_mode(self, mock_detector):
        """Test error on invalid threshold mode"""
        with pytest.raises(ValueError) as exc_info:
            mock_detector.set_threshold(mode="invalid_mode")
        
        assert "Unknown mode" in str(exc_info.value)
    
    def test_ela_generation(self, mock_detector, sample_image):
        """Test ELA generation"""
        img = Image.open(sample_image)
        ela = mock_detector._generate_ela(img)
        
        assert isinstance(ela, Image.Image)
        assert ela.mode == "L"  # Grayscale
        assert ela.size == (320, 320)
    
    def test_ela_with_custom_quality(self, mock_detector, sample_image):
        """Test ELA with custom quality"""
        img = Image.open(sample_image)
        ela = mock_detector._generate_ela(img, quality=50)
        
        assert isinstance(ela, Image.Image)
        assert ela.size == (320, 320)
    
    def test_analyze_clean(self, mock_detector, sample_image):
        """Test analyzing clean document"""
        # Mock model to return low forgery probability
        mock_logit = torch.tensor([[-2.0]])  # Sigmoid ≈ 0.12
        mock_detector.model.return_value = mock_logit
        
        result = mock_detector.analyze(sample_image)
        
        assert isinstance(result, PANVerificationResult)
        assert result.document_type == "PAN"
        assert result.verdict == "CLEAN"
        assert result.forgery_probability < 0.49
    
    def test_analyze_forged(self, mock_detector, sample_image):
        """Test analyzing forged document"""
        # Mock model to return high forgery probability
        mock_logit = torch.tensor([[3.0]])  # Sigmoid ≈ 0.95
        mock_detector.model.return_value = mock_logit
        
        result = mock_detector.analyze(sample_image)
        
        assert result.verdict == "FORGED"
        assert result.forgery_probability >= 0.49
        assert result.clean_probability < 0.51
    
    def test_analyze_invalid_image(self, mock_detector, tmp_path):
        """Test error handling for invalid image"""
        invalid_path = tmp_path / "not_image.bin"
        invalid_path.write_bytes(b"\x00\x01\x02")
        
        with pytest.raises(ValueError) as exc_info:
            mock_detector.analyze(invalid_path)
        
        assert "Could not open image" in str(exc_info.value)
    
    def test_batch_analysis(self, mock_detector, tmp_path):
        """Test batch processing"""
        images = []
        for i in range(3):
            img = Image.new('RGB', (320, 320), color='yellow')
            img_path = tmp_path / f"pan_{i}.jpg"
            img.save(img_path)
            images.append(img_path)
        
        # Mock clean predictions
        mock_logit = torch.tensor([[-1.5]])  # Low forgery prob
        mock_detector.model.return_value = mock_logit
        
        results = mock_detector.analyze_batch(images)
        
        assert len(results) == 3
        assert all(isinstance(r, PANVerificationResult) for r in results)
        assert all(r.verdict == "CLEAN" for r in results)
    
    def test_analyze_as_dict(self, mock_detector, sample_image):
        """Test dictionary output"""
        mock_logit = torch.tensor([[0.0]])  # Sigmoid = 0.5
        mock_detector.model.return_value = mock_logit
        
        result_dict = mock_detector.analyze_as_dict(sample_image)
        
        assert isinstance(result_dict, dict)
        assert "verdict" in result_dict
        assert "confidence" in result_dict
        assert "forgery_probability" in result_dict
        assert "clean_probability" in result_dict
    
    def test_threshold_boundary_f1_optimal(self, mock_detector, sample_image):
        """Test threshold boundary at 0.49 (F1-optimal)"""
        # Exactly at threshold
        mock_logit = torch.tensor([[-0.04]])  # Sigmoid ≈ 0.49
        mock_detector.model.return_value = mock_logit
        
        result = mock_detector.analyze(sample_image)
        
        # At 0.49, forgery_prob < 0.49 means CLEAN
        assert result.verdict == "CLEAN"
    
    def test_probability_sum(self, mock_detector, sample_image):
        """Test that probabilities sum to 1"""
        mock_logit = torch.tensor([[1.0]])
        mock_detector.model.return_value = mock_logit
        
        result = mock_detector.analyze(sample_image)
        
        prob_sum = result.forgery_probability + result.clean_probability
        assert abs(prob_sum - 1.0) < 1e-5
    
    def test_confidence_calculation(self, mock_detector, sample_image):
        """Test confidence equals appropriate probability"""
        # High forgery probability
        mock_logit = torch.tensor([[2.5]])  # Sigmoid ≈ 0.92
        mock_detector.model.return_value = mock_logit
        
        result = mock_detector.analyze(sample_image)
        
        assert result.verdict == "FORGED"
        # Confidence should equal forgery_prob when forged
        assert abs(result.confidence - result.forgery_probability) < 1e-5
    
    def test_get_model_info(self, mock_detector):
        """Test model info retrieval"""
        info = mock_detector.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_type"] == "PANForgeryDetector"
        assert info["threshold"] == 0.49
        assert "config" in info
    
    def test_missing_model_file(self, tmp_path):
        """Test error when model file is missing"""
        fake_path = tmp_path / "nonexistent.pth"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            PANForgeryDetector(model_path=fake_path, device="cpu")
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_custom_ela_quality_init(self, tmp_path):
        """Test initialization with custom ELA quality"""
        model_file = tmp_path / "model.pth"
        model_file.touch()
        
        with patch('src.cv_engine.pan_detector.PANForgeryDetectorCNN'):
            with patch('torch.load'):
                detector = PANForgeryDetector(
                    model_path=model_file,
                    device="cpu",
                    ela_quality=80
                )
                
                assert detector.ela_quality == 80
    
    def test_sigmoid_activation(self, mock_detector, sample_image):
        """Test sigmoid activation is applied correctly"""
        # Test known logit values
        test_cases = [
            (0.0, 0.5),    # Sigmoid(0) = 0.5
            (2.0, 0.88),   # Sigmoid(2) ≈ 0.88
            (-2.0, 0.12),  # Sigmoid(-2) ≈ 0.12
        ]
        
        for logit, expected_prob in test_cases:
            mock_logit = torch.tensor([[logit]])
            mock_detector.model.return_value = mock_logit
            
            result = mock_detector.analyze(sample_image)
            
            assert abs(result.forgery_probability - expected_prob) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
