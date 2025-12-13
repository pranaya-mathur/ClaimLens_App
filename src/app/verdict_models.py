"""
Semantic Verdict Models for ClaimLens Components
Provides structured, explainable results from each detection layer
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DocumentVerdict(str, Enum):
    """Document verification verdicts."""
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    FORGED = "FORGED"
    MISSING = "MISSING"


class DamageVerdict(str, Enum):
    """Damage assessment verdicts."""
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    EXCESSIVE = "EXCESSIVE"
    MISSING = "MISSING"


class RiskVerdict(str, Enum):
    """ML/Graph risk verdicts."""
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


class FraudRingVerdict(str, Enum):
    """Fraud ring detection verdicts."""
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    FRAUD_RING_DETECTED = "FRAUD_RING_DETECTED"


class FinalVerdict(str, Enum):
    """Final claim verdicts."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class ComponentResult:
    """
    Standard result structure for all detection components.
    
    This provides semantic meaning to scores and enables
    explainable AI through structured evidence.
    
    Attributes:
        component_name: Name of the detection component
        verdict: Semantic verdict (e.g., FORGED, HIGH_RISK)
        confidence: How confident we are in this verdict (0.0-1.0)
        score: Risk score where 0.0=safe, 1.0=fraudulent
        reason: Primary human-readable reason for verdict
        details: Component-specific additional information
        red_flags: List of specific concerns found
    """
    component_name: str
    verdict: str
    confidence: float
    score: float
    reason: str
    details: Dict[str, Any]
    red_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component_name,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 3),
            "reason": self.reason,
            "details": self.details,
            "red_flags": self.red_flags
        }


@dataclass
class CriticalFlag:
    """
    Critical red flag that may override normal scoring.
    
    These represent hard stops or mandatory manual reviews
    regardless of the aggregated score.
    """
    flag_type: str  # e.g., "HIGH_CONFIDENCE_FORGERY"
    action: str  # "REJECT", "REVIEW", or "FLAG"
    reason: str  # Human-readable explanation
    confidence: float  # Confidence in this flag (0.0-1.0)
    override: bool  # If True, overrides all other signals
    source_component: str  # Which component raised this flag
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.flag_type,
            "action": self.action,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "override": self.override,
            "source": self.source_component
        }


@dataclass
class ReasoningStep:
    """
    Single step in the decision-making reasoning chain.
    
    This provides full transparency for LLM explanation generation.
    """
    stage: str  # e.g., "document_verification", "adaptive_scoring"
    decision: Optional[str]  # Decision made at this stage
    reason: str  # Why this decision was made
    data: Dict[str, Any]  # Supporting data for this step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage,
            "decision": self.decision,
            "reason": self.reason,
            "data": self.data
        }


@dataclass
class FinalDecision:
    """
    Complete fraud detection decision with full reasoning chain.
    
    This structure contains everything needed for:
    - Returning API response to user
    - Generating LLM explanations
    - Audit trail logging
    - Performance monitoring
    """
    claim_id: str
    verdict: str  # APPROVE, REVIEW, REJECT
    confidence: float  # Overall confidence in verdict
    final_score: float  # Aggregated risk score
    
    # Component results
    component_results: Dict[str, ComponentResult]
    
    # Decision reasoning
    reasoning_chain: List[ReasoningStep]
    critical_flags: List[CriticalFlag]
    
    # Metadata
    primary_reason: str  # Main reason for verdict
    red_flags: List[str]  # All red flags found
    fallbacks_used: List[str]  # What fallback methods were used
    processing_notes: str  # Additional context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "final_score": round(self.final_score, 3),
            "primary_reason": self.primary_reason,
            "red_flags": self.red_flags,
            
            # Component details
            "component_results": {
                name: result.to_dict() 
                for name, result in self.component_results.items()
            },
            
            # Reasoning
            "reasoning_chain": [step.to_dict() for step in self.reasoning_chain],
            "critical_flags": [flag.to_dict() for flag in self.critical_flags],
            
            # Metadata
            "fallbacks_used": self.fallbacks_used,
            "processing_notes": self.processing_notes
        }
    
    def get_evidence_for_llm(self) -> Dict[str, Any]:
        """
        Get structured evidence for LLM explanation generation.
        
        This formats the decision data in a way that's easy for
        LLM to understand and explain.
        """
        return {
            "verdict": self.verdict,
            "confidence": f"{self.confidence:.0%}",
            "final_score": round(self.final_score, 2),
            "primary_reason": self.primary_reason,
            
            # Component verdicts in simple format
            "components": {
                name: {
                    "verdict": result.verdict,
                    "confidence": f"{result.confidence:.0%}",
                    "reason": result.reason,
                    "key_findings": result.red_flags[:3]  # Top 3 flags only
                }
                for name, result in self.component_results.items()
            },
            
            # Critical issues
            "critical_flags": [
                {
                    "issue": flag.reason,
                    "severity": flag.action,
                    "confidence": f"{flag.confidence:.0%}"
                }
                for flag in self.critical_flags
            ],
            
            # Decision path
            "decision_steps": [
                f"{step.stage}: {step.reason}"
                for step in self.reasoning_chain
            ],
            
            # All red flags
            "all_red_flags": self.red_flags
        }
