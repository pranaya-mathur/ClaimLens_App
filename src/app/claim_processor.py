"""
Claim Processor - Product-Aware Routing
Routes claims to appropriate analysis pipeline based on product type
"""

from typing import Dict, Any, Optional
from loguru import logger

# Import all analysis engines
try:
    from src.cv_engine import DamageDetector, DocumentVerifier
    from src.ml_engine import FeatureEngineer, MLFraudScorer
    from src.fraud_engine import FraudDetector
    from .health_analyzer import HealthClaimAnalyzer
    CV_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some engines not available: {e}")
    CV_ENGINE_AVAILABLE = False


class ClaimProcessor:
    """
    Main claim processing orchestrator.
    
    Routes claims to product-specific analysis pipelines:
    - Motor: DamageDetector + DocumentVerifier + ML + Graph
    - Health: HealthClaimAnalyzer + DocumentVerifier + ML + Graph
    - Life: DocumentVerifier + ML + Graph (no damage detection)
    - Property: DamageDetector (future) + DocumentVerifier + ML + Graph
    
    All products get ML fraud scoring and graph analysis.
    Only motor/property get damage detection.
    Only health gets medical cost analysis.
    """
    
    def __init__(
        self,
        damage_detector: Optional[Any] = None,
        doc_verifier: Optional[Any] = None,
        ml_scorer: Optional[Any] = None,
        graph_analyzer: Optional[Any] = None,
        health_analyzer: Optional[Any] = None
    ):
        """
        Initialize claim processor.
        
        Args:
            damage_detector: DamageDetector instance (for motor/property)
            doc_verifier: DocumentVerifier instance (all products)
            ml_scorer: MLFraudScorer instance (all products)
            graph_analyzer: FraudDetector instance (all products)
            health_analyzer: HealthClaimAnalyzer instance (for health)
        """
        logger.info("Initializing ClaimProcessor...")
        
        # Common modules (used by all products)
        self.doc_verifier = doc_verifier
        self.ml_scorer = ml_scorer
        self.graph_analyzer = graph_analyzer
        
        # Product-specific modules
        self.damage_detector = damage_detector  # Motor/Property only
        self.health_analyzer = health_analyzer or HealthClaimAnalyzer()  # Health only
        
        logger.success("✓ ClaimProcessor ready")
    
    def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route claim to appropriate analysis pipeline based on product type.
        
        Args:
            claim_data: Claim data dictionary with at minimum:
                - product: str (motor/health/life/property)
                - claim_id: str
                - Other fields depending on product
        
        Returns:
            Complete analysis result dictionary with:
                - product: str
                - claim_id: str
                - document_verification: dict (if available)
                - product_specific_analysis: dict (motor/health/life)
                - ml_analysis: dict (if available)
                - graph_analysis: dict (if available)
                - scores: dict with component scores
                - final_score: float (0-1)
                - risk_level: str (LOW/MEDIUM/HIGH/CRITICAL)
                - verdict: str (APPROVE/REVIEW/REJECT)
                - red_flags: list of strings
        """
        product = claim_data.get("product", "").lower()
        claim_id = claim_data.get("claim_id", "unknown")
        
        logger.info(f"Processing {product} claim: {claim_id}")
        
        # Initialize result structure
        result = {
            "claim_id": claim_id,
            "product": product,
            "scores": {},
            "red_flags": [],
            "warnings": []
        }
        
        # Step 1: Document verification (COMMON - all products)
        if self.doc_verifier and claim_data.get("documents"):
            try:
                doc_results = self._verify_documents(claim_data)
                result["document_verification"] = doc_results
                result["scores"]["doc_score"] = doc_results.get("average_confidence", 0.5)
            except Exception as e:
                logger.warning(f"Document verification failed: {e}")
                result["warnings"].append(f"Document verification error: {str(e)}")
                result["scores"]["doc_score"] = 0.5  # Neutral
        else:
            logger.info("Skipping document verification (no docs or verifier unavailable)")
            result["scores"]["doc_score"] = 0.5  # Neutral
        
        # Step 2: Product-specific analysis
        if product == "motor":
            result["motor_analysis"] = self._analyze_motor_claim(claim_data)
            result["scores"]["damage_score"] = result["motor_analysis"].get("damage_risk", 0.5)
            result["red_flags"].extend(result["motor_analysis"].get("red_flags", []))
        
        elif product == "health":
            result["health_analysis"] = self._analyze_health_claim(claim_data)
            result["scores"]["medical_risk"] = result["health_analysis"].get("medical_risk", 0.5)
            result["scores"]["fraud_ring_risk"] = result["health_analysis"].get("fraud_ring_risk", 0.0)
            result["red_flags"].extend(result["health_analysis"].get("red_flags", []))
        
        elif product == "life":
            result["life_analysis"] = self._analyze_life_claim(claim_data)
            result["scores"]["life_risk"] = result["life_analysis"].get("validity_score", 0.5)
            result["red_flags"].extend(result["life_analysis"].get("red_flags", []))
        
        elif product == "property":
            result["property_analysis"] = self._analyze_property_claim(claim_data)
            result["scores"]["property_damage_score"] = result["property_analysis"].get("damage_risk", 0.5)
            result["red_flags"].extend(result["property_analysis"].get("red_flags", []))
        
        else:
            logger.warning(f"Unknown product type: {product}, using generic analysis")
            result["warnings"].append(f"Unknown product type: {product}")
        
        # Step 3: ML fraud scoring (COMMON - all products)
        if self.ml_scorer:
            try:
                ml_result = self._ml_fraud_scoring(claim_data)
                result["ml_analysis"] = ml_result
                result["scores"]["ml_fraud_score"] = ml_result.get("fraud_probability", 0.5)
            except Exception as e:
                logger.warning(f"ML scoring failed: {e}")
                result["warnings"].append(f"ML scoring error: {str(e)}")
                result["scores"]["ml_fraud_score"] = 0.5
        else:
            logger.info("Skipping ML scoring (scorer unavailable)")
            result["scores"]["ml_fraud_score"] = 0.5
        
        # Step 4: Graph analysis (COMMON - all products)
        if self.graph_analyzer:
            try:
                graph_result = self._graph_analysis(claim_data)
                result["graph_analysis"] = graph_result
                result["scores"]["graph_risk_score"] = graph_result.get("final_risk_score", 0.5)
            except Exception as e:
                logger.warning(f"Graph analysis failed: {e}")
                result["warnings"].append(f"Graph analysis error: {str(e)}")
                result["scores"]["graph_risk_score"] = 0.5
        else:
            logger.info("Skipping graph analysis (analyzer unavailable)")
            result["scores"]["graph_risk_score"] = 0.5
        
        # Step 5: Calculate final score with product-specific weights
        result["final_score"] = self._calculate_final_score(
            result["scores"], 
            product
        )
        
        # Step 6: Categorize risk level
        result["risk_level"] = self._categorize_risk(result["final_score"])
        
        # Step 7: Generate verdict
        result["verdict"] = self._generate_verdict(
            result["final_score"], 
            product,
            result["red_flags"]
        )
        
        logger.success(
            f"Claim {claim_id} processed: score={result['final_score']:.2f}, "
            f"risk={result['risk_level']}, verdict={result['verdict']}"
        )
        
        return result
    
    def _analyze_motor_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze motor claim (vehicle damage detection).
        
        Args:
            claim_data: Claim data
        
        Returns:
            Motor-specific analysis results
        """
        logger.info("Running motor claim analysis...")
        
        if not self.damage_detector:
            logger.warning("DamageDetector not available, skipping motor analysis")
            return {
                "error": "DamageDetector not initialized",
                "damage_risk": 0.5,
                "red_flags": []
            }
        
        damage_photos = claim_data.get("damage_photos", [])
        if not damage_photos:
            logger.warning("No damage photos provided for motor claim")
            return {
                "error": "No damage photos provided",
                "damage_risk": 0.7,  # High risk without photos
                "red_flags": ["No damage photos submitted for motor claim"]
            }
        
        # Run damage detection (would call actual detector)
        # For now, return placeholder
        return {
            "damages_detected": [],
            "damage_risk": 0.3,
            "estimated_cost": claim_data.get("claim_amount", 0) * 0.8,
            "inflation_ratio": 1.2,
            "red_flags": []
        }
    
    def _analyze_health_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze health claim (medical cost benchmarking, fraud rings).
        
        Args:
            claim_data: Claim data
        
        Returns:
            Health-specific analysis results
        """
        logger.info("Running health claim analysis...")
        
        # Use HealthClaimAnalyzer
        return self.health_analyzer.analyze(claim_data)
    
    def _analyze_life_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze life claim (death certificate verification, nominee checks).
        
        Args:
            claim_data: Claim data
        
        Returns:
            Life-specific analysis results
        """
        logger.info("Running life claim analysis...")
        
        # Placeholder for life claim analysis
        # In production, would verify death certificate, check nominee KYC, etc.
        
        red_flags = []
        validity_score = 0.2  # Default low risk
        
        # Check if death certificate provided
        docs = claim_data.get("documents_submitted", "").lower()
        if "death" not in docs:
            red_flags.append("Death certificate not provided")
            validity_score += 0.5
        
        # Check claim timing (suspicious if very recent policy)
        days_since_policy = claim_data.get("days_since_policy_start", 999)
        if days_since_policy < 365:  # Less than 1 year
            red_flags.append("Life claim within first year of policy (suspicious timing)")
            validity_score += 0.3
        
        return {
            "validity_score": min(validity_score, 1.0),
            "death_certificate_verified": "death" in docs,
            "policy_timing_suspicious": days_since_policy < 365,
            "red_flags": red_flags
        }
    
    def _analyze_property_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze property claim (property damage, fire/flood detection).
        
        Args:
            claim_data: Claim data
        
        Returns:
            Property-specific analysis results
        """
        logger.info("Running property claim analysis...")
        
        # Placeholder for property damage detection
        # Would use specialized property damage detector
        
        return {
            "damage_risk": 0.3,
            "estimated_loss": claim_data.get("claim_amount", 0) * 0.75,
            "red_flags": []
        }
    
    def _verify_documents(self, claim_data: Dict) -> Dict:
        """Document verification (placeholder)."""
        return {"average_confidence": 0.95}
    
    def _ml_fraud_scoring(self, claim_data: Dict) -> Dict:
        """ML fraud scoring (placeholder)."""
        return {"fraud_probability": 0.3}
    
    def _graph_analysis(self, claim_data: Dict) -> Dict:
        """Graph fraud analysis (placeholder)."""
        return {"final_risk_score": 0.25}
    
    def _calculate_final_score(self, scores: Dict, product: str) -> float:
        """
        Calculate final fraud score with product-specific weights.
        
        Args:
            scores: Dictionary of component scores
            product: Product type
        
        Returns:
            Final score (0-1)
        """
        if product == "motor":
            # Motor: damage + docs + ML + graph
            return (
                0.20 * scores.get("doc_score", 0.5) +
                0.25 * scores.get("damage_score", 0.5) +
                0.35 * scores.get("ml_fraud_score", 0.5) +
                0.20 * scores.get("graph_risk_score", 0.5)
            )
        
        elif product == "health":
            # Health: medical risk + fraud ring + ML + graph
            return (
                0.15 * scores.get("doc_score", 0.5) +
                0.25 * scores.get("medical_risk", 0.5) +
                0.15 * scores.get("fraud_ring_risk", 0.5) +
                0.30 * scores.get("ml_fraud_score", 0.5) +
                0.15 * scores.get("graph_risk_score", 0.5)
            )
        
        elif product == "life":
            # Life: strict KYC + verification focus
            return (
                0.40 * scores.get("doc_score", 0.5) +
                0.20 * scores.get("life_risk", 0.5) +
                0.25 * scores.get("ml_fraud_score", 0.5) +
                0.15 * scores.get("graph_risk_score", 0.5)
            )
        
        elif product == "property":
            # Property: damage + ML + graph
            return (
                0.25 * scores.get("doc_score", 0.5) +
                0.25 * scores.get("property_damage_score", 0.5) +
                0.30 * scores.get("ml_fraud_score", 0.5) +
                0.20 * scores.get("graph_risk_score", 0.5)
            )
        
        else:
            # Default: equal weighting
            return (
                0.30 * scores.get("doc_score", 0.5) +
                0.40 * scores.get("ml_fraud_score", 0.5) +
                0.30 * scores.get("graph_risk_score", 0.5)
            )
    
    def _categorize_risk(self, score: float) -> str:
        """
        Categorize risk level from score.
        
        Args:
            score: Fraud score (0-1)
        
        Returns:
            Risk level string
        """
        if score < 0.3:
            return "LOW"
        elif score < 0.5:
            return "MEDIUM"
        elif score < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_verdict(self, score: float, product: str, red_flags: List[str]) -> str:
        """
        Generate final verdict.
        
        Args:
            score: Final fraud score
            product: Product type
            red_flags: List of red flags
        
        Returns:
            Verdict string (APPROVE/REVIEW/REJECT)
        """
        # Strict rules for life claims
        if product == "life" and score > 0.4:
            return "REVIEW"
        
        # Critical red flags = manual review
        critical_flags = [
            "death certificate",
            "fraud ring",
            "collusion",
            "pre-existing condition"
        ]
        
        has_critical_flag = any(
            any(cf in flag.lower() for cf in critical_flags)
            for flag in red_flags
        )
        
        if has_critical_flag:
            return "REVIEW"
        
        # Score-based verdict
        if score < 0.3:
            return "APPROVE"
        elif score < 0.6:
            return "REVIEW"
        else:
            return "REJECT"
