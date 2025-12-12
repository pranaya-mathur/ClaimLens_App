"""
Integration tests for Document Verifier

Tests cover:
- Unified verification API
- Dual-check mode
- Consensus logic
- Batch processing
- Error handling
- Detector routing
"""
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.cv_engine.document_verifier import (
    DocumentVerifier,
    DocumentVerificationResult
)
from src.cv_engine.aadhaar_detector import AadhaarVerificationResult
from src.cv_engine.pan_detector import PANVerificationResult


class TestDocumentVerificationResult:
    """Test DocumentVerificationResult dataclass"""
    
    def test_result_creation(self):
        """Test creating verification result"""
        result = DocumentVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.95,
            primary_result={"verdict": "CLEAN"},
            dual_check_enabled=False
        )
        
        assert result.document_type == "PAN"
        assert result.verdict == "CLEAN"
        assert result.dual_check_enabled is False
    
    def test_dual_check_fields(self):
        """Test dual-check specific fields"""
        result = DocumentVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.92,
            primary_result={},
            dual_check_enabled=True,
            secondary_result={},
            consensus_verdict="CLEAN",
            consensus_confidence=0.94,
            agreement=True
        )
        
        assert result.dual_check_enabled is True
        assert result.agreement is True
        assert result.consensus_verdict == "CLEAN"
    
    def test_is_forged_single_check(self):
        """Test is_forged without dual-check"""
        result = DocumentVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="FORGED",
            confidence=0.88,
            primary_result={},
            dual_check_enabled=False
        )
        
        assert result.is_forged() is True
    
    def test_is_forged_dual_check_consensus(self):
        """Test is_forged with dual-check consensus"""
        result = DocumentVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="CLEAN",  # Primary says CLEAN
            confidence=0.9,
            primary_result={},
            dual_check_enabled=True,
            consensus_verdict="FORGED",  # But consensus is FORGED
            agreement=True
        )
        
        # Should use consensus when dual-check enabled
        assert result.is_forged() is True
    
    def test_is_suspicious(self):
        """Test suspicious flag for detector disagreement"""
        result = DocumentVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="SUSPICIOUS",
            confidence=0.5,
            primary_result={},
            dual_check_enabled=True,
            agreement=False
        )
        
        assert result.is_suspicious() is True
        assert result.is_forged() is True  # SUSPICIOUS counts as forged


class TestDocumentVerifier:
    """Test DocumentVerifier class"""
    
    @pytest.fixture
    def mock_aadhaar_result(self):
        """Mock Aadhaar detector result"""
        return AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path="test.jpg",
            verdict="AUTHENTIC",
            confidence=0.96,
            authentic_probability=0.96,
            forged_probability=0.04,
            threshold=0.5
        )
    
    @pytest.fixture
    def mock_pan_result(self):
        """Mock PAN detector result"""
        return PANVerificationResult(
            document_type="PAN",
            image_path="test.jpg",
            verdict="CLEAN",
            confidence=0.94,
            forgery_probability=0.06,
            clean_probability=0.94,
            threshold=0.49
        )
    
    @pytest.fixture
    def mock_verifier(self, mock_aadhaar_result, mock_pan_result):
        """Create verifier with mocked detectors"""
        with patch('src.cv_engine.document_verifier.AadhaarForgeryDetector') as mock_a:
            with patch('src.cv_engine.document_verifier.PANForgeryDetector') as mock_p:
                # Mock detector instances
                aadhaar_detector = MagicMock()
                aadhaar_detector.analyze.return_value = mock_aadhaar_result
                
                pan_detector = MagicMock()
                pan_detector.analyze.return_value = mock_pan_result
                
                mock_a.return_value = aadhaar_detector
                mock_p.return_value = pan_detector
                
                verifier = DocumentVerifier(device="cpu")
                
                return verifier
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create sample test image"""
        img = Image.new('RGB', (224, 224), color='blue')
        img_path = tmp_path / "test_doc.jpg"
        img.save(img_path)
        return img_path
    
    def test_initialization_both_detectors(self, mock_verifier):
        """Test verifier initializes with both detectors"""
        assert mock_verifier.aadhaar_detector is not None
        assert mock_verifier.pan_detector is not None
    
    def test_get_available_detectors(self, mock_verifier):
        """Test listing available detectors"""
        available = mock_verifier.get_available_detectors()
        
        assert "AADHAAR" in available
        assert "PAN" in available
        assert len(available) == 2
    
    def test_verify_aadhaar_single_check(self, mock_verifier, sample_image):
        """Test single-check Aadhaar verification"""
        result = mock_verifier.verify(sample_image, "AADHAAR", dual_check=False)
        
        assert isinstance(result, DocumentVerificationResult)
        assert result.document_type == "AADHAAR"
        assert result.verdict == "AUTHENTIC"
        assert result.dual_check_enabled is False
        assert result.secondary_result is None
    
    def test_verify_pan_single_check(self, mock_verifier, sample_image):
        """Test single-check PAN verification"""
        result = mock_verifier.verify(sample_image, "PAN", dual_check=False)
        
        assert result.document_type == "PAN"
        assert result.verdict == "CLEAN"
        assert result.dual_check_enabled is False
    
    def test_verify_invalid_doc_type(self, mock_verifier, sample_image):
        """Test error on invalid document type"""
        with pytest.raises(ValueError) as exc_info:
            mock_verifier.verify(sample_image, "PASSPORT")
        
        assert "Invalid document type" in str(exc_info.value)
    
    def test_dual_check_agreement_clean(self, mock_verifier, sample_image):
        """Test dual-check when both detectors agree (CLEAN)"""
        # Both return CLEAN/AUTHENTIC
        result = mock_verifier.verify(sample_image, "AADHAAR", dual_check=True)
        
        assert result.dual_check_enabled is True
        assert result.agreement is True
        assert result.consensus_verdict == "CLEAN"
        assert result.secondary_result is not None
    
    def test_dual_check_agreement_forged(self, mock_verifier, sample_image):
        """Test dual-check when both detectors agree (FORGED)"""
        # Mock both detectors to return FORGED
        forged_aadhaar = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path=str(sample_image),
            verdict="FORGED",
            confidence=0.92,
            authentic_probability=0.08,
            forged_probability=0.92,
            threshold=0.5
        )
        
        forged_pan = PANVerificationResult(
            document_type="PAN",
            image_path=str(sample_image),
            verdict="FORGED",
            confidence=0.89,
            forgery_probability=0.89,
            clean_probability=0.11,
            threshold=0.49
        )
        
        mock_verifier.aadhaar_detector.analyze.return_value = forged_aadhaar
        mock_verifier.pan_detector.analyze.return_value = forged_pan
        
        result = mock_verifier.verify(sample_image, "PAN", dual_check=True)
        
        assert result.agreement is True
        assert result.consensus_verdict == "FORGED"
    
    def test_dual_check_disagreement(self, mock_verifier, sample_image):
        """Test dual-check when detectors disagree"""
        # Primary (Aadhaar) says AUTHENTIC
        authentic_result = AadhaarVerificationResult(
            document_type="AADHAAR",
            image_path=str(sample_image),
            verdict="AUTHENTIC",
            confidence=0.85,
            authentic_probability=0.85,
            forged_probability=0.15,
            threshold=0.5
        )
        
        # Secondary (PAN) says FORGED
        forged_result = PANVerificationResult(
            document_type="PAN",
            image_path=str(sample_image),
            verdict="FORGED",
            confidence=0.88,
            forgery_probability=0.88,
            clean_probability=0.12,
            threshold=0.49
        )
        
        mock_verifier.aadhaar_detector.analyze.return_value = authentic_result
        mock_verifier.pan_detector.analyze.return_value = forged_result
        
        result = mock_verifier.verify(sample_image, "AADHAAR", dual_check=True)
        
        assert result.agreement is False
        assert result.consensus_verdict == "SUSPICIOUS"
        assert result.verdict == "SUSPICIOUS"  # Overrides primary
    
    def test_batch_verification(self, mock_verifier, tmp_path):
        """Test batch document verification"""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color='red')
            img_path = tmp_path / f"doc_{i}.jpg"
            img.save(img_path)
            images.append(img_path)
        
        results = mock_verifier.verify_batch(images, "PAN", dual_check=False)
        
        assert len(results) == 3
        assert all(isinstance(r, DocumentVerificationResult) for r in results)
        assert all(r.document_type == "PAN" for r in results)
    
    def test_verify_as_dict(self, mock_verifier, sample_image):
        """Test dictionary output"""
        result_dict = mock_verifier.verify_as_dict(
            sample_image, "AADHAAR", dual_check=False
        )
        
        assert isinstance(result_dict, dict)
        assert "document_type" in result_dict
        assert "verdict" in result_dict
        assert "confidence" in result_dict
    
    def test_get_detector_info_aadhaar(self, mock_verifier):
        """Test getting Aadhaar detector info"""
        mock_verifier.aadhaar_detector.get_model_info.return_value = {
            "model_type": "AadhaarForgeryDetector"
        }
        
        info = mock_verifier.get_detector_info("AADHAAR")
        assert info["model_type"] == "AadhaarForgeryDetector"
    
    def test_get_detector_info_pan(self, mock_verifier):
        """Test getting PAN detector info"""
        mock_verifier.pan_detector.get_model_info.return_value = {
            "model_type": "PANForgeryDetector"
        }
        
        info = mock_verifier.get_detector_info("PAN")
        assert info["model_type"] == "PANForgeryDetector"
    
    def test_consensus_confidence_calculation(self, mock_verifier, sample_image):
        """Test consensus confidence is average when agreement"""
        result = mock_verifier.verify(sample_image, "PAN", dual_check=True)
        
        if result.agreement:
            # Should be average of both confidences
            expected = (0.94 + 0.96) / 2.0  # From fixtures
            assert abs(result.consensus_confidence - expected) < 0.01
    
    def test_case_insensitive_doc_type(self, mock_verifier, sample_image):
        """Test document type is case-insensitive"""
        result1 = mock_verifier.verify(sample_image, "pan", dual_check=False)
        result2 = mock_verifier.verify(sample_image, "PAN", dual_check=False)
        result3 = mock_verifier.verify(sample_image, "Pan", dual_check=False)
        
        assert result1.document_type == "PAN"
        assert result2.document_type == "PAN"
        assert result3.document_type == "PAN"
    
    def test_missing_aadhaar_detector(self, sample_image):
        """Test graceful handling when Aadhaar detector missing"""
        with patch('src.cv_engine.document_verifier.AadhaarForgeryDetector') as mock_a:
            with patch('src.cv_engine.document_verifier.PANForgeryDetector') as mock_p:
                # Aadhaar fails to load
                mock_a.side_effect = Exception("Model not found")
                
                # PAN loads successfully
                pan_detector = MagicMock()
                mock_p.return_value = pan_detector
                
                verifier = DocumentVerifier(device="cpu")
                
                assert verifier.aadhaar_detector is None
                assert verifier.pan_detector is not None
                
                available = verifier.get_available_detectors()
                assert "PAN" in available
                assert "AADHAAR" not in available
    
    def test_verify_unavailable_detector(self, sample_image):
        """Test error when trying to use unavailable detector"""
        with patch('src.cv_engine.document_verifier.AadhaarForgeryDetector') as mock_a:
            with patch('src.cv_engine.document_verifier.PANForgeryDetector') as mock_p:
                mock_a.side_effect = Exception("Model not found")
                pan_detector = MagicMock()
                mock_p.return_value = pan_detector
                
                verifier = DocumentVerifier(device="cpu")
                
                with pytest.raises(ValueError) as exc_info:
                    verifier.verify(sample_image, "AADHAAR")
                
                assert "not available" in str(exc_info.value)
    
    def test_dual_check_secondary_failure(self, mock_verifier, sample_image):
        """Test dual-check handles secondary detector failure gracefully"""
        # Make secondary detector fail
        mock_verifier.pan_detector.analyze.side_effect = Exception("Inference failed")
        
        result = mock_verifier.verify(sample_image, "AADHAAR", dual_check=True)
        
        # Should still return primary result
        assert result.verdict == "AUTHENTIC"
        assert "error" in result.secondary_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
