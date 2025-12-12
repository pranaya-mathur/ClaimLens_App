"""
Centralized Risk Categorization
Ensures consistent risk level thresholds across all ClaimLens modules
"""

from typing import Dict, Literal
import os
from loguru import logger


# Risk level type
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RiskCategories:
    """
    Centralized risk categorization with configurable thresholds.
    
    All modules (ML, CV, Graph) should use this for consistent risk levels.
    
    Default thresholds:
    - LOW: score < 0.3
    - MEDIUM: 0.3 <= score < 0.5  
    - HIGH: 0.5 <= score < 0.7
    - CRITICAL: score >= 0.7
    """
    
    # Configurable thresholds via environment variables
    THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.3"))
    THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.5"))
    THRESHOLD_CRITICAL = float(os.getenv("RISK_THRESHOLD_CRITICAL", "0.7"))
    
    @classmethod
    def categorize(cls, score: float) -> RiskLevel:
        """
        Categorize a fraud/risk score into risk level.
        
        Args:
            score: Fraud/risk score (0-1 range)
            
        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL
            
        Examples:
            >>> RiskCategories.categorize(0.2)
            'LOW'
            >>> RiskCategories.categorize(0.4)
            'MEDIUM'
            >>> RiskCategories.categorize(0.6)
            'HIGH'
            >>> RiskCategories.categorize(0.8)
            'CRITICAL'
        """
        # Normalize score to 0-1 if out of bounds
        score = max(0.0, min(1.0, score))
        
        if score < cls.THRESHOLD_MEDIUM:
            return "LOW"
        elif score < cls.THRESHOLD_HIGH:
            return "MEDIUM"
        elif score < cls.THRESHOLD_CRITICAL:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @classmethod
    def get_thresholds(cls) -> Dict[str, float]:
        """
        Get current threshold values.
        
        Returns:
            Dictionary of threshold names to values
        """
        return {
            "LOW_to_MEDIUM": cls.THRESHOLD_MEDIUM,
            "MEDIUM_to_HIGH": cls.THRESHOLD_HIGH,
            "HIGH_to_CRITICAL": cls.THRESHOLD_CRITICAL
        }
    
    @classmethod
    def get_score_range(cls, risk_level: RiskLevel) -> Dict[str, float]:
        """
        Get score range for a given risk level.
        
        Args:
            risk_level: Risk level name
            
        Returns:
            Dictionary with 'min' and 'max' score values
            
        Examples:
            >>> RiskCategories.get_score_range('MEDIUM')
            {'min': 0.3, 'max': 0.5}
        """
        ranges = {
            "LOW": {"min": 0.0, "max": cls.THRESHOLD_MEDIUM},
            "MEDIUM": {"min": cls.THRESHOLD_MEDIUM, "max": cls.THRESHOLD_HIGH},
            "HIGH": {"min": cls.THRESHOLD_HIGH, "max": cls.THRESHOLD_CRITICAL},
            "CRITICAL": {"min": cls.THRESHOLD_CRITICAL, "max": 1.0}
        }
        
        return ranges.get(risk_level, {"min": 0.0, "max": 1.0})
    
    @classmethod
    def validate_thresholds(cls) -> bool:
        """
        Validate that thresholds are in correct order.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not (0.0 < cls.THRESHOLD_MEDIUM < cls.THRESHOLD_HIGH < cls.THRESHOLD_CRITICAL <= 1.0):
            raise ValueError(
                f"Invalid risk thresholds! Must satisfy: "
                f"0 < MEDIUM < HIGH < CRITICAL <= 1.0. "
                f"Current: MEDIUM={cls.THRESHOLD_MEDIUM}, "
                f"HIGH={cls.THRESHOLD_HIGH}, CRITICAL={cls.THRESHOLD_CRITICAL}"
            )
        
        logger.info(
            f"Risk thresholds validated: LOW<{cls.THRESHOLD_MEDIUM} | "
            f"MEDIUM<{cls.THRESHOLD_HIGH} | HIGH<{cls.THRESHOLD_CRITICAL} | CRITICAL"
        )
        return True
    
    @classmethod
    def get_color_code(cls, risk_level: RiskLevel) -> str:
        """
        Get color code for UI display.
        
        Args:
            risk_level: Risk level
            
        Returns:
            Hex color code
        """
        colors = {
            "LOW": "#22c55e",      # Green
            "MEDIUM": "#f59e0b",   # Amber
            "HIGH": "#ef4444",     # Red
            "CRITICAL": "#991b1b"  # Dark Red
        }
        return colors.get(risk_level, "#6b7280")  # Gray fallback
    
    @classmethod
    def get_icon(cls, risk_level: RiskLevel) -> str:
        """
        Get emoji/icon for risk level.
        
        Args:
            risk_level: Risk level
            
        Returns:
            Emoji string
        """
        icons = {
            "LOW": "✅",
            "MEDIUM": "⚠️",
            "HIGH": "🚨",
            "CRITICAL": "🔴"
        }
        return icons.get(risk_level, "❓")
    
    @classmethod
    def get_description(cls, risk_level: RiskLevel) -> str:
        """
        Get human-readable description of risk level.
        
        Args:
            risk_level: Risk level
            
        Returns:
            Description string
        """
        descriptions = {
            "LOW": "Low fraud risk. Claim appears legitimate.",
            "MEDIUM": "Medium risk. Recommend manual review.",
            "HIGH": "High fraud risk. Requires detailed investigation.",
            "CRITICAL": "Critical fraud indicators. Reject or escalate immediately."
        }
        return descriptions.get(risk_level, "Unknown risk level")


# Validate thresholds on module import
try:
    RiskCategories.validate_thresholds()
except ValueError as e:
    logger.error(f"Risk threshold configuration error: {e}")
    raise


# Convenience function for backward compatibility
def categorize_risk(score: float) -> RiskLevel:
    """
    Convenience function to categorize risk score.
    
    Args:
        score: Risk score (0-1)
        
    Returns:
        Risk level string
    """
    return RiskCategories.categorize(score)
