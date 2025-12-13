"""
Semantic Aggregation Engine
Converts raw component scores into semantic verdicts with full reasoning chain
"""

from typing import Dict, Any, List, Tuple
from loguru import logger

from .verdict_models import (
    ComponentResult, CriticalFlag, ReasoningStep, FinalDecision,
    DocumentVerdict, DamageVerdict, RiskVerdict, FraudRingVerdict, FinalVerdict
)


class SemanticAggregator:
    """
    Aggregates component results into final fraud verdict with full explainability.
    
    Key Features:
    - Converts numeric scores to semantic verdicts
    - Implements critical flag gating logic
    - Adaptive weighting based on confidence
    - Generates complete reasoning chain
    """
    
    def __init__(self):
        logger.info("Initializing SemanticAggregator with critical flags engine")
    
    def aggregate(
        self,
        claim_id: str,
        component_results: Dict[str, ComponentResult],
        product_type: str,
        fallbacks_used: List[str]
    ) -> FinalDecision:
        """
        Aggregate component results into final decision.
        
        Args:
            claim_id: Unique claim identifier
            component_results: Results from all detection components
            product_type: motor/health/life/property
            fallbacks_used: List of components that used fallback logic
        
        Returns:
            FinalDecision with complete reasoning chain
        """
        logger.info(f"Aggregating results for claim {claim_id}")
        
        reasoning_chain = []
        all_red_flags = []
        
        # Collect all red flags from components
        for component_result in component_results.values():
            all_red_flags.extend(component_result.red_flags)
        
        # STAGE 1: Check for critical flags (hard stops)
        critical_flags = self._check_critical_flags(component_results)
        
        if critical_flags:
            logger.info(f"Critical flags detected: {len(critical_flags)}")
            reasoning_chain.append(ReasoningStep(
                stage="critical_flag_check",
                decision="GATING_TRIGGERED",
                reason=f"Found {len(critical_flags)} critical flag(s)",
                data={"flags": [f.flag_type for f in critical_flags]}
            ))
            
            # Check for override flags
            override_flags = [f for f in critical_flags if f.override]
            if override_flags:
                # Hard stop - immediate decision
                primary_flag = override_flags[0]
                return self._create_override_decision(
                    claim_id=claim_id,
                    override_flag=primary_flag,
                    component_results=component_results,
                    reasoning_chain=reasoning_chain,
                    all_red_flags=all_red_flags,
                    fallbacks_used=fallbacks_used,
                    critical_flags=critical_flags
                )
        
        # STAGE 2: No hard stops - calculate adaptive score
        scoring_result = self._calculate_adaptive_score(
            component_results, 
            product_type, 
            fallbacks_used
        )
        
        reasoning_chain.append(ReasoningStep(
            stage="adaptive_scoring",
            decision="SCORE_CALCULATED",
            reason=f"Aggregated score: {scoring_result['final_score']:.2f}",
            data={
                "final_score": scoring_result['final_score'],
                "weighting_breakdown": scoring_result['weighting_breakdown'],
                "dominant_factor": scoring_result['dominant_factor']
            }
        ))
        
        # STAGE 3: Score-based verdict with thresholds
        verdict, confidence = self._score_to_verdict(
            scoring_result['final_score'],
            scoring_result['dominant_factor']
        )
        
        reasoning_chain.append(ReasoningStep(
            stage="verdict_determination",
            decision=verdict,
            reason=f"Score {scoring_result['final_score']:.2f} maps to {verdict}",
            data={
                "threshold_used": self._get_threshold_info(scoring_result['final_score']),
                "confidence": confidence
            }
        ))
        
        # STAGE 4: Fallback quality check
        if len(fallbacks_used) >= 2:
            reasoning_chain.append(ReasoningStep(
                stage="data_quality_check",
                decision="REVIEW_RECOMMENDED",
                reason=f"Decision based on limited data ({len(fallbacks_used)} fallbacks)",
                data={"fallbacks": fallbacks_used}
            ))
            
            # Upgrade to review if too many fallbacks
            if verdict == FinalVerdict.APPROVE:
                logger.warning(f"Upgrading APPROVE to REVIEW due to {len(fallbacks_used)} fallbacks")
                verdict = FinalVerdict.REVIEW
                confidence *= 0.7  # Reduce confidence
        
        # Build primary reason
        primary_reason = self._generate_primary_reason(
            verdict,
            scoring_result['dominant_factor'],
            critical_flags
        )
        
        # Build processing notes
        processing_notes = self._generate_processing_notes(
            fallbacks_used,
            len(critical_flags)
        )
        
        # Create final decision
        return FinalDecision(
            claim_id=claim_id,
            verdict=verdict,
            confidence=confidence,
            final_score=scoring_result['final_score'],
            component_results=component_results,
            reasoning_chain=reasoning_chain,
            critical_flags=critical_flags,
            primary_reason=primary_reason,
            red_flags=all_red_flags,
            fallbacks_used=fallbacks_used,
            processing_notes=processing_notes
        )
    
    def _check_critical_flags(self, component_results: Dict[str, ComponentResult]) -> List[CriticalFlag]:
        """
        Check for critical red flags that may override scoring.
        
        Critical flags are hard constraints that represent:
        - High-confidence fraud detection
        - Fraud ring membership
        - Severe claim-damage mismatches
        - Missing mandatory documents
        """
        flags = []
        
        # Rule 1: High-confidence document forgery
        doc_result = component_results.get("document_verification")
        if doc_result:
            if doc_result.verdict == DocumentVerdict.FORGED and doc_result.confidence > 0.90:
                flags.append(CriticalFlag(
                    flag_type="HIGH_CONFIDENCE_FORGERY",
                    action="REJECT",
                    reason=f"Document forgery detected with {doc_result.confidence:.0%} confidence",
                    confidence=doc_result.confidence,
                    override=True,  # Automatic rejection
                    source_component="document_verification"
                ))
            elif doc_result.verdict == DocumentVerdict.FORGED and doc_result.confidence > 0.75:
                flags.append(CriticalFlag(
                    flag_type="LIKELY_FORGERY",
                    action="REVIEW",
                    reason=f"Likely document forgery ({doc_result.confidence:.0%} confidence)",
                    confidence=doc_result.confidence,
                    override=False,  # Strong signal but not auto-reject
                    source_component="document_verification"
                ))
        
        # Rule 2: Fraud ring membership
        graph_result = component_results.get("graph_analysis")
        if graph_result:
            if graph_result.verdict == FraudRingVerdict.FRAUD_RING_DETECTED:
                flags.append(CriticalFlag(
                    flag_type="FRAUD_RING_MEMBER",
                    action="REVIEW",
                    reason=f"Claimant linked to fraud ring: {graph_result.reason}",
                    confidence=graph_result.confidence,
                    override=True,  # Mandatory manual review
                    source_component="graph_analysis"
                ))
        
        # Rule 3: Excessive claim amount vs damage
        damage_result = component_results.get("damage_detection")
        if damage_result and damage_result.verdict == DamageVerdict.EXCESSIVE:
            damage_details = damage_result.details
            claimed = damage_details.get("claimed_amount", 0)
            estimated = damage_details.get("estimated_repair", 0)
            
            if estimated > 0 and claimed > estimated * 5:
                flags.append(CriticalFlag(
                    flag_type="CLAIM_DAMAGE_MISMATCH",
                    action="REVIEW",
                    reason=f"Claimed ₹{claimed:,} but damage estimated at ₹{estimated:,} ({claimed/estimated:.1f}x)",
                    confidence=0.85,
                    override=False,
                    source_component="damage_detection"
                ))
        
        # Rule 4: ML model very high fraud probability
        ml_result = component_results.get("ml_fraud_score")
        if ml_result:
            if ml_result.verdict == RiskVerdict.CRITICAL and ml_result.confidence > 0.90:
                flags.append(CriticalFlag(
                    flag_type="ML_HIGH_FRAUD_PROBABILITY",
                    action="REVIEW",
                    reason=f"ML model predicts {ml_result.score:.0%} fraud probability",
                    confidence=ml_result.confidence,
                    override=False,
                    source_component="ml_fraud_score"
                ))
        
        return flags
    
    def _calculate_adaptive_score(
        self,
        component_results: Dict[str, ComponentResult],
        product_type: str,
        fallbacks_used: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate final score with adaptive weighting.
        
        Weighting adapts based on:
        - Component confidence levels
        - Product type
        - Availability of data (fallbacks)
        """
        # Extract scores and confidences
        scores = {}
        confidences = {}
        
        for name, result in component_results.items():
            scores[name] = result.score
            confidences[name] = result.confidence
        
        # Base weights by product type
        if product_type == "motor":
            base_weights = {
                "document_verification": 0.20,
                "damage_detection": 0.25,
                "ml_fraud_score": 0.35,
                "graph_analysis": 0.20
            }
        elif product_type == "health":
            base_weights = {
                "document_verification": 0.15,
                "health_analysis": 0.30,
                "ml_fraud_score": 0.35,
                "graph_analysis": 0.20
            }
        elif product_type == "life":
            base_weights = {
                "document_verification": 0.40,
                "life_analysis": 0.20,
                "ml_fraud_score": 0.25,
                "graph_analysis": 0.15
            }
        else:
            # Generic/property
            base_weights = {
                "document_verification": 0.30,
                "ml_fraud_score": 0.40,
                "graph_analysis": 0.30
            }
        
        # Adjust weights based on confidence
        adjusted_weights = {}
        for component, base_weight in base_weights.items():
            if component in confidences:
                confidence = confidences[component]
                # High confidence gets boost, low confidence loses weight
                confidence_multiplier = 0.8 + (confidence * 0.4)
                adjusted_weights[component] = base_weight * confidence_multiplier
            else:
                # Component missing, redistribute its weight
                adjusted_weights[component] = 0.0
        
        # If component used fallback, reduce its weight
        for fallback in fallbacks_used:
            if fallback in adjusted_weights:
                adjusted_weights[fallback] *= 0.7
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        else:
            # All components missing - equal weights
            normalized_weights = {k: 1.0/len(base_weights) for k in base_weights.keys()}
        
        # Calculate final score
        final_score = sum(
            scores.get(comp, 0.5) * normalized_weights.get(comp, 0.0)
            for comp in normalized_weights.keys()
        )
        
        # Generate weighting breakdown for transparency
        weighting_breakdown = []
        for component in sorted(normalized_weights.keys(), key=lambda x: normalized_weights[x], reverse=True):
            weight = normalized_weights[component]
            score = scores.get(component, 0.5)
            contribution = score * weight
            
            weighting_breakdown.append({
                "component": component,
                "weight": f"{weight*100:.1f}%",
                "score": round(score, 3),
                "contribution": round(contribution, 3),
                "confidence": confidences.get(component, 0.5)
            })
        
        # Identify dominant factor
        dominant = max(weighting_breakdown, key=lambda x: x["contribution"])
        
        return {
            "final_score": final_score,
            "weighting_breakdown": weighting_breakdown,
            "dominant_factor": dominant
        }
    
    def _score_to_verdict(self, score: float, dominant_factor: Dict) -> Tuple[str, float]:
        """
        Convert numeric score to semantic verdict.
        
        Returns: (verdict, confidence)
        """
        if score < 0.30:
            verdict = FinalVerdict.APPROVE
            # Confidence is distance from decision boundary
            confidence = 1.0 - (score / 0.30)
        elif score < 0.60:
            verdict = FinalVerdict.REVIEW
            # Lower confidence in middle zone
            confidence = 0.7
        else:
            verdict = FinalVerdict.REJECT
            # Confidence increases with score
            confidence = min(0.6 + (score - 0.60) * 0.8, 0.95)
        
        return verdict, confidence
    
    def _get_threshold_info(self, score: float) -> Dict[str, Any]:
        """Get information about which threshold was used."""
        if score < 0.30:
            return {"threshold": "<0.30", "category": "low_risk"}
        elif score < 0.60:
            return {"threshold": "0.30-0.60", "category": "medium_risk"}
        else:
            return {"threshold": ">0.60", "category": "high_risk"}
    
    def _generate_primary_reason(
        self,
        verdict: str,
        dominant_factor: Dict,
        critical_flags: List[CriticalFlag]
    ) -> str:
        """
        Generate primary human-readable reason for verdict.
        """
        if critical_flags:
            # Primary reason is the critical flag
            primary_flag = critical_flags[0]
            return primary_flag.reason
        
        # Otherwise use dominant factor
        component = dominant_factor["component"].replace("_", " ").title()
        contribution = dominant_factor["contribution"]
        
        if verdict == FinalVerdict.APPROVE:
            return f"Low fraud risk - all checks passed"
        elif verdict == FinalVerdict.REVIEW:
            return f"Manual review recommended - primary concern: {component}"
        else:
            return f"High fraud risk detected - key issue: {component}"
    
    def _generate_processing_notes(self, fallbacks_used: List[str], critical_flag_count: int) -> str:
        """
        Generate processing notes for transparency.
        """
        notes = []
        
        if critical_flag_count > 0:
            notes.append(f"{critical_flag_count} critical flag(s) raised")
        
        if fallbacks_used:
            notes.append(f"Fallback methods used: {', '.join(fallbacks_used)}")
        
        if not notes:
            return "Claim processed with full analysis pipeline"
        
        return "; ".join(notes)
    
    def _create_override_decision(
        self,
        claim_id: str,
        override_flag: CriticalFlag,
        component_results: Dict[str, ComponentResult],
        reasoning_chain: List[ReasoningStep],
        all_red_flags: List[str],
        fallbacks_used: List[str],
        critical_flags: List[CriticalFlag]
    ) -> FinalDecision:
        """
        Create decision when critical flag overrides normal scoring.
        """
        reasoning_chain.append(ReasoningStep(
            stage="critical_override",
            decision=override_flag.action,
            reason=f"Critical flag overrides scoring: {override_flag.reason}",
            data={
                "flag_type": override_flag.flag_type,
                "confidence": override_flag.confidence
            }
        ))
        
        return FinalDecision(
            claim_id=claim_id,
            verdict=override_flag.action,
            confidence=override_flag.confidence,
            final_score=0.95 if override_flag.action == "REJECT" else 0.65,
            component_results=component_results,
            reasoning_chain=reasoning_chain,
            critical_flags=critical_flags,
            primary_reason=override_flag.reason,
            red_flags=all_red_flags,
            fallbacks_used=fallbacks_used,
            processing_notes=f"Decision overridden by critical flag: {override_flag.flag_type}"
        )
