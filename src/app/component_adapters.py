"""
Component Adapters
Converts raw component outputs into semantic ComponentResult format
"""

from typing import Dict, Any
from loguru import logger

from .verdict_models import (
    ComponentResult,
    DocumentVerdict,
    DamageVerdict,
    RiskVerdict,
    FraudRingVerdict
)


class ComponentAdapter:
    """
    Adapts raw component outputs to semantic verdict format.
    
    This allows existing detection components to work without modification
    while providing structured, explainable results.
    """
    
    @staticmethod
    def adapt_document_verification(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert document verification output to semantic format.
        
        Expected raw_result keys:
        - method: str (actual or fallback)
        - average_confidence or confidence: float
        - verdict or forged: bool (optional)
        - document_count: int (optional)
        """
        method = raw_result.get("method", "actual")
        confidence = raw_result.get("average_confidence", raw_result.get("confidence", 0.5))
        
        # Determine verdict from raw result
        if "verdict" in raw_result:
            raw_verdict = raw_result["verdict"]
            if raw_verdict in ["FORGED", "SUSPICIOUS", "AUTHENTIC"]:
                verdict = raw_verdict
            else:
                verdict = DocumentVerdict.SUSPICIOUS
        elif raw_result.get("forged", False):
            verdict = DocumentVerdict.FORGED
        elif confidence > 0.8:
            verdict = DocumentVerdict.AUTHENTIC
        elif confidence > 0.5:
            verdict = DocumentVerdict.SUSPICIOUS
        else:
            verdict = DocumentVerdict.FORGED
        
        # Score is inverse of confidence for authentic docs
        if verdict == DocumentVerdict.AUTHENTIC:
            score = 1.0 - confidence  # High confidence authentic = low risk score
        else:
            score = confidence  # High confidence forged = high risk score
        
        # Build reason
        if method == "fallback":
            reason = f"Document check using fallback method (limited data)"
        elif verdict == DocumentVerdict.FORGED:
            reason = f"Document forgery detected with {confidence:.0%} confidence"
        elif verdict == DocumentVerdict.SUSPICIOUS:
            reason = f"Document shows suspicious patterns ({confidence:.0%} confidence)"
        else:
            reason = f"Documents appear authentic ({confidence:.0%} confidence)"
        
        # Extract red flags
        red_flags = []
        if raw_result.get("document_count", 2) < 2:
            red_flags.append("Insufficient supporting documents")
        if method == "fallback":
            red_flags.append("Document verification used fallback method")
        if verdict == DocumentVerdict.FORGED:
            red_flags.append(f"Document forgery detected ({confidence:.0%} confidence)")
        
        return ComponentResult(
            component_name="document_verification",
            verdict=verdict,
            confidence=confidence,
            score=score,
            reason=reason,
            details=raw_result,
            red_flags=red_flags
        )
    
    @staticmethod
    def adapt_damage_detection(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert damage detection output to semantic format.
        
        Expected raw_result keys:
        - method: str
        - risk_assessment: dict with risk_score
        - detected_damage: str
        - estimated_repair: float
        - claimed_amount: float
        """
        method = raw_result.get("method", "actual")
        risk_score = raw_result.get("risk_assessment", {}).get("risk_score", 0.3)
        
        detected_damage = raw_result.get("detected_damage", "Unknown")
        estimated = raw_result.get("estimated_repair", 0)
        claimed = raw_result.get("claimed_amount", 0)
        
        # Determine verdict
        if method == "fallback":
            verdict = DamageVerdict.MISSING
            confidence = 0.5
        elif estimated > 0 and claimed > estimated * 5:
            verdict = DamageVerdict.EXCESSIVE
            confidence = 0.85
        elif estimated > 0 and abs(claimed - estimated) / estimated > 0.5:
            verdict = DamageVerdict.INCONSISTENT
            confidence = 0.75
        else:
            verdict = DamageVerdict.CONSISTENT
            confidence = 0.8
        
        # Build reason
        if method == "fallback":
            reason = f"Damage assessment unavailable (subtype: {raw_result.get('subtype_analysis', 'N/A')})"
        elif verdict == DamageVerdict.EXCESSIVE:
            reason = f"Claimed amount (₹{claimed:,}) is {claimed/estimated:.1f}x estimated repair (₹{estimated:,})"
        elif verdict == DamageVerdict.INCONSISTENT:
            reason = f"Claimed amount (₹{claimed:,}) inconsistent with damage severity"
        else:
            reason = f"Damage assessment consistent with claim amount"
        
        # Red flags
        red_flags = raw_result.get("red_flags", [])
        if verdict == DamageVerdict.EXCESSIVE:
            red_flags.append(f"Claim amount {claimed/estimated:.1f}x damage estimate")
        
        return ComponentResult(
            component_name="damage_detection",
            verdict=verdict,
            confidence=confidence,
            score=risk_score,
            reason=reason,
            details=raw_result,
            red_flags=red_flags
        )
    
    @staticmethod
    def adapt_health_analysis(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert health claim analysis to semantic format.
        """
        medical_risk = raw_result.get("medical_risk", 0.5)
        fraud_ring_risk = raw_result.get("fraud_ring_risk", 0.0)
        
        # Combined risk
        combined_risk = max(medical_risk, fraud_ring_risk)
        
        # Verdict
        if combined_risk < 0.3:
            verdict = RiskVerdict.LOW_RISK
        elif combined_risk < 0.6:
            verdict = RiskVerdict.MEDIUM_RISK
        elif combined_risk < 0.8:
            verdict = RiskVerdict.HIGH_RISK
        else:
            verdict = RiskVerdict.CRITICAL
        
        confidence = 0.75 if raw_result.get("method") != "fallback" else 0.5
        
        # Reason
        if fraud_ring_risk > 0.7:
            reason = f"Fraud ring indicators detected (risk: {fraud_ring_risk:.0%})"
        elif medical_risk > 0.6:
            reason = f"Elevated medical claim risk (risk: {medical_risk:.0%})"
        else:
            reason = f"Health claim within normal risk parameters"
        
        return ComponentResult(
            component_name="health_analysis",
            verdict=verdict,
            confidence=confidence,
            score=combined_risk,
            reason=reason,
            details=raw_result,
            red_flags=raw_result.get("red_flags", [])
        )
    
    @staticmethod
    def adapt_ml_scoring(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert ML fraud score to semantic format.
        """
        fraud_probability = raw_result.get("fraud_probability", raw_result.get("score", 0.5))
        
        # Verdict based on probability
        if fraud_probability < 0.3:
            verdict = RiskVerdict.LOW_RISK
        elif fraud_probability < 0.6:
            verdict = RiskVerdict.MEDIUM_RISK
        elif fraud_probability < 0.8:
            verdict = RiskVerdict.HIGH_RISK
        else:
            verdict = RiskVerdict.CRITICAL
        
        confidence = 0.85 if raw_result.get("method") != "fallback" else 0.5
        
        # Reason with top features if available
        top_features = raw_result.get("top_features", [])
        if top_features and len(top_features) > 0:
            top_feature = top_features[0]
            reason = f"ML fraud probability {fraud_probability:.0%} (key factor: {top_feature.get('feature', 'N/A')})"
        else:
            reason = f"ML model predicts {fraud_probability:.0%} fraud probability"
        
        # Red flags from top risky features
        red_flags = []
        for feature in top_features[:3]:  # Top 3 features
            feature_name = feature.get("feature", "")
            importance = feature.get("importance", 0)
            if importance > 0.2:  # Significant importance
                red_flags.append(f"{feature_name} (importance: {importance:.0%})")
        
        return ComponentResult(
            component_name="ml_fraud_score",
            verdict=verdict,
            confidence=confidence,
            score=fraud_probability,
            reason=reason,
            details=raw_result,
            red_flags=red_flags
        )
    
    @staticmethod
    def adapt_graph_analysis(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert graph fraud ring analysis to semantic format.
        """
        final_risk = raw_result.get("final_risk_score", raw_result.get("score", 0.5))
        
        # Check for fraud ring indicators
        connected_claims = raw_result.get("connected_claims", 0)
        shared_entity = raw_result.get("shared_entity", "")
        
        # Verdict
        if connected_claims >= 3:
            verdict = FraudRingVerdict.FRAUD_RING_DETECTED
            confidence = 0.9
        elif final_risk > 0.6 or connected_claims >= 2:
            verdict = FraudRingVerdict.SUSPICIOUS
            confidence = 0.75
        else:
            verdict = FraudRingVerdict.CLEAN
            confidence = 0.8
        
        # Reason
        if verdict == FraudRingVerdict.FRAUD_RING_DETECTED:
            reason = f"Claimant linked to {connected_claims} suspicious claims via {shared_entity}"
        elif verdict == FraudRingVerdict.SUSPICIOUS:
            reason = f"Potentially suspicious network patterns detected (risk: {final_risk:.0%})"
        else:
            reason = f"No fraud ring indicators detected"
        
        # Red flags
        red_flags = []
        if connected_claims > 0:
            red_flags.append(f"Connected to {connected_claims} claims in network")
        if shared_entity:
            red_flags.append(f"Shared entity: {shared_entity}")
        
        confidence = 0.8 if raw_result.get("method") != "fallback" else 0.5
        
        return ComponentResult(
            component_name="graph_analysis",
            verdict=verdict,
            confidence=confidence,
            score=final_risk,
            reason=reason,
            details=raw_result,
            red_flags=red_flags
        )
    
    @staticmethod
    def adapt_life_analysis(raw_result: Dict[str, Any]) -> ComponentResult:
        """
        Convert life claim analysis to semantic format.
        """
        validity_score = raw_result.get("validity_score", 0.5)
        death_cert_verified = raw_result.get("death_certificate_verified", False)
        suspicious_timing = raw_result.get("policy_timing_suspicious", False)
        
        # Verdict
        if validity_score > 0.6:
            verdict = RiskVerdict.HIGH_RISK
        elif validity_score > 0.4:
            verdict = RiskVerdict.MEDIUM_RISK
        else:
            verdict = RiskVerdict.LOW_RISK
        
        confidence = 0.75
        
        # Reason
        if not death_cert_verified:
            reason = "Death certificate verification incomplete"
        elif suspicious_timing:
            reason = "Life claim timing raises concerns (filed within first policy year)"
        else:
            reason = "Life claim documentation and timing appear normal"
        
        # Red flags
        red_flags = raw_result.get("red_flags", [])
        
        return ComponentResult(
            component_name="life_analysis",
            verdict=verdict,
            confidence=confidence,
            score=validity_score,
            reason=reason,
            details=raw_result,
            red_flags=red_flags
        )
