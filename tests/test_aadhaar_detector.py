"""
Unit tests for Aadhaar Forgery Detector

Tests cover:
- Model initialization
- Inference pipeline
- Result format validation
- Batch processing
- Error handling
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.cv_engine.aadhaar_detector import (
    AadhaarForgeryDetector,
    AadhaarVerificationResult
)


class TestAadhaarVerificationResult:
    """Test AadhaarVerificationResult dataclass"""
    
    def test_result_creation(self):
        """Test creating verification result"""
        result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="AUTHENTIC",
            confidence=0.95,
            authentic_probability=0.95,
            forged_probability=0.05,
            threshold=0.5
        )
        
        assert result.document_type == "AADHAAR"
        assert result.verdict == "AUTHENTIC"
        assert result.confidence == 0.95
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="FORGED",
            confidence=0.85,
            authentic_probability=0.15,
            forged_probability=0.85,
            threshold=0.5
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["verdict"] == "FORGED"
        assert result_dict["confidence"] == 0.85
    
    def test_is_forged(self):
        """Test is_forged method"""
        forged_result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="FORGED",
            confidence=0.9,
            authentic_probability=0.1,
            forged_probability=0.9,
            threshold=0.5
        )
        assert forged_result.is_forged() is True
        
        authentic_result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="AUTHENTIC",
            confidence=0.95,
            authentic_probability=0.95,
            forged_probability=0.05,
            threshold=0.5
        )
        assert authentic_result.is_forged() is False
    
    def test_is_authentic(self):
        """Test is_authentic method"""
        authentic_result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="AUTHENTIC",
            confidence=0.98,
            authentic_probability=0.98,
            forged_probability=0.02,
            threshold=0.5
        )
        assert authentic_result.is_authentic() is True


class TestAadhaarForgeryDetector:
    """Test AadhaarForgeryDetector class"""
    
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
        # Create fake model file
        model_file = tmp_path / "aadhaar_balanced_model.pth"
        model_file.touch()
        
        with patch('src.cv_engine.aadhaar_detector.AadhaarForgeryDetectorCNN') as mock_cnn:
            mock_cnn.return_value = mock_model
            
            with patch('torch.load') as mock_load:
                mock_load.return_value = {
                    'model_state_dict': {},
                    'best_auc': 0.9999
                }
                
                detector = AadhaarForgeryDetector(
                    model_path=model_file,
                    device="cpu"
                )
                detector.model = mock_model
                
                return detector
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create sample test image"""
        img = Image.new('RGB', (224, 224), color='red')
        img_path = tmp_path / "test_aadhaar.jpg"
        img.save(img_path)
        return img_path
    
    def test_initialization_device(self, mock_detector):
        """Test detector initializes with correct device"""
        assert mock_detector.device.type == "cpu"
        assert mock_detector.threshold == 0.5
        assert mock_detector.input_size == (224, 224)
    
    def test_model_config(self, mock_detector):
        """Test model configuration"""
        config = mock_detector.MODEL_CONFIG
        assert config["architecture"] == "ResNet50"
        assert config["num_classes"] == 2
        assert config["threshold"] == 0.5
        assert config["performance"]["validation_accuracy"] == 0.9962
    
    def test_missing_model_file(self, tmp_path):
        """Test error when model file is missing"""
        fake_path = tmp_path / "nonexistent_model.pth"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            AadhaarForgeryDetector(model_path=fake_path, device="cpu")
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_analyze_authentic(self, mock_detector, sample_image):
        """Test analyzing authentic document"""
        # Mock model to return authentic prediction
        mock_logits = torch.tensor([[0.1, 0.9]])  # [FORGED, AUTHENTIC]
        mock_detector.model.return_value = mock_logits
        
        result = mock_detector.analyze(sample_image)
        
        assert isinstance(result, AadhaarVerificationResult)
        assert result.document_type == "AADHAAR"
        assert result.verdict == "AUTHENTIC"
        assert result.authentic_probability > 0.5
        assert result.image_path == str(sample_image)
    
    def test_analyze_forged(self, mock_detector, sample_image):
        """Test analyzing forged document"""
        # Mock model to return forged prediction
        mock_logits = torch.tensor([[0.95, 0.05]])  # [FORGED, AUTHENTIC]
        mock_detector.model.return_value = mock_logits
        
        result = mock_detector.analyze(sample_image)
        
        assert result.verdict == "FORGED"
        assert result.forged_probability > 0.5
        assert result.authentic_probability < 0.5
    
    def test_analyze_invalid_image(self, mock_detector, tmp_path):
        """Test error handling for invalid image"""
        invalid_path = tmp_path / "invalid.txt"
        invalid_path.write_text("not an image")
        
        with pytest.raises(ValueError) as exc_info:
            mock_detector.analyze(invalid_path)
        
        assert "Could not open image" in str(exc_info.value)
    
    def test_batch_analysis(self, mock_detector, tmp_path):
        """Test batch processing"""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color='blue')
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            images.append(img_path)
        
        # Mock predictions
        mock_logits = torch.tensor([[0.2, 0.8]])  # AUTHENTIC
        mock_detector.model.return_value = mock_logits
        
        results = mock_detector.analyze_batch(images)
        
        assert len(results) == 3
        assert all(isinstance(r, AadhaarVerificationResult) for r in results)
        assert all(r.verdict == "AUTHENTIC" for r in results)
    
    def test_analyze_as_dict(self, mock_detector, sample_image):
        """Test dictionary output"""
        mock_logits = torch.tensor([[0.1, 0.9]])
        mock_detector.model.return_value = mock_logits
        
        result_dict = mock_detector.analyze_as_dict(sample_image)
        
        assert isinstance(result_dict, dict)
        assert "verdict" in result_dict
        assert "confidence" in result_dict
        assert "authentic_probability" in result_dict
    
    def test_threshold_boundary(self, mock_detector, sample_image):
        """Test threshold boundary behavior"""
        # Exactly at threshold (0.5)
        mock_logits = torch.tensor([[0.5, 0.5]])
        mock_detector.model.return_value = mock_logits
        
        result = mock_detector.analyze(sample_image)
        
        # At threshold = 0.5, authentic_prob < 0.5 means FORGED
        assert result.verdict == "FORGED"
    
    def test_custom_threshold(self, tmp_path):
        """Test custom threshold initialization"""
        model_file = tmp_path / "model.pth"
        model_file.touch()
        
        with patch('src.cv_engine.aadhaar_detector.AadhaarForgeryDetectorCNN'):
            with patch('torch.load'):
                detector = AadhaarForgeryDetector(
                    model_path=model_file,
                    device="cpu",
                    threshold=0.7
                )
                
                assert detector.threshold == 0.7
    
    def test_get_model_info(self, mock_detector):
        """Test model info retrieval"""
        info = mock_detector.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_type"] == "AadhaarForgeryDetector"
        assert "config" in info
        assert "device" in info
    
    def test_probability_sum(self, mock_detector, sample_image):
        """Test that probabilities sum to 1"""
        mock_logits = torch.tensor([[0.3, 0.7]])
        mock_detector.model.return_value = mock_logits
        
        result = mock_detector.analyze(sample_image)
        
        prob_sum = result.authentic_probability + result.forged_probability
        assert abs(prob_sum - 1.0) < 1e-5  # Allow floating point error
    
    def test_confidence_is_max_probability(self, mock_detector, sample_image):
        """Test that confidence equals max probability"""
        mock_logits = torch.tensor([[0.25, 0.75]])
        mock_detector.model.return_value = mock_logits
        
        result = mock_detector.analyze(sample_image)
        
        expected_confidence = max(
            result.authentic_probability,
            result.forged_probability
        )
        assert abs(result.confidence - expected_confidence) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
