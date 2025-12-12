"""
Claim Processor - Product-Aware Routing with Smart Fallbacks
Routes claims to appropriate analysis pipeline and handles missing data gracefully
"""

from typing import Dict, Any, Optional, List
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
    Main claim processing orchestrator with smart fallback handling.
    
    **Smart Fallback System:**
    - Missing damage photos → Skip damage detection, rely on ML/Graph
    - Theft/Fire claims → Document-focused analysis
    - Missing medical docs → Higher scrutiny, fraud ring checks
    - Any missing data → Graceful degradation, never crash
    
    Routes claims to product-specific analysis pipelines:
    - Motor: DamageDetector + DocumentVerifier + ML + Graph
    - Health: HealthClaimAnalyzer + DocumentVerifier + ML + Graph
    - Life: DocumentVerifier + ML + Graph (no damage detection)
    - Property: DamageDetector (future) + DocumentVerifier + ML + Graph
    
    All products get ML fraud scoring and graph analysis.
    """
    
    # Subtypes that don't require damage photos
    NO_DAMAGE_PHOTO_SUBTYPES = {
        "motor": ["theft", "total_loss", "fire", "vandalism"],
        "property": ["theft", "fire", "flood"],
        "health": ["ambulance", "consultation", "diagnostics"],
    }
    
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
        logger.info("Initializing ClaimProcessor with smart fallback system...")
        
        # Common modules (used by all products)
        self.doc_verifier = doc_verifier
        self.ml_scorer = ml_scorer
        self.graph_analyzer = graph_analyzer
        
        # Product-specific modules
        self.damage_detector = damage_detector  # Motor/Property only
        self.health_analyzer = health_analyzer or HealthClaimAnalyzer()  # Health only
        
        logger.success("✓ ClaimProcessor ready with fallback handling")
    
    def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route claim to appropriate analysis pipeline with smart fallback.
        
        Args:
            claim_data: Claim data dictionary with at minimum:
                - product: str (motor/health/life/property)
                - subtype: str (theft/accident/hospitalization/etc)
                - claim_id: str
                - Other fields depending on product
        
        Returns:
            Complete analysis result with graceful degradation for missing data
        """
        product = claim_data.get("product", "").lower()
        subtype = claim_data.get("subtype", "").lower()
        claim_id = claim_data.get("claim_id", "unknown")
        
        logger.info(f"Processing {product}/{subtype} claim: {claim_id}")
        
        # Initialize result structure
        result = {
            "claim_id": claim_id,
            "product": product,
            "subtype": subtype,
            "scores": {},
            "red_flags": [],
            "warnings": [],
            "fallbacks_used": []  # Track what fallbacks were triggered
        }
        
        # Step 1: Document verification (COMMON - all products)
        doc_result = self._safe_document_verification(claim_data)
        result["document_verification"] = doc_result["result"]
        result["scores"]["doc_score"] = doc_result["score"]
        if doc_result["fallback_used"]:
            result["fallbacks_used"].append("document_verification")
        
        # Step 2: Product-specific analysis with subtype awareness
        if product == "motor":
            motor_result = self._analyze_motor_claim_safe(claim_data)
            result["motor_analysis"] = motor_result["result"]
            result["scores"]["damage_score"] = motor_result["score"]
            result["red_flags"].extend(motor_result["red_flags"])
            if motor_result["fallback_used"]:
                result["fallbacks_used"].append("motor_damage_detection")
        
        elif product == "health":
            health_result = self._analyze_health_claim_safe(claim_data)
            result["health_analysis"] = health_result["result"]
            result["scores"]["medical_risk"] = health_result["score"]
            result["scores"]["fraud_ring_risk"] = health_result.get("fraud_ring_risk", 0.0)
            result["red_flags"].extend(health_result["red_flags"])
            if health_result["fallback_used"]:
                result["fallbacks_used"].append("health_analysis")
        
        elif product == "life":
            life_result = self._analyze_life_claim_safe(claim_data)
            result["life_analysis"] = life_result["result"]
            result["scores"]["life_risk"] = life_result["score"]
            result["red_flags"].extend(life_result["red_flags"])
            if life_result["fallback_used"]:
                result["fallbacks_used"].append("life_verification")
        
        elif product == "property":
            property_result = self._analyze_property_claim_safe(claim_data)
            result["property_analysis"] = property_result["result"]
            result["scores"]["property_damage_score"] = property_result["score"]
            result["red_flags"].extend(property_result["red_flags"])
            if property_result["fallback_used"]:
                result["fallbacks_used"].append("property_damage_detection")
        
        else:
            logger.warning(f"Unknown product type: {product}, using generic analysis")
            result["warnings"].append(f"Unknown product type: {product}")
            result["scores"]["generic_risk"] = 0.5  # Neutral
        
        # Step 3: ML fraud scoring (COMMON - always runs)
        ml_result = self._safe_ml_scoring(claim_data)
        result["ml_analysis"] = ml_result["result"]
        result["scores"]["ml_fraud_score"] = ml_result["score"]
        if ml_result["fallback_used"]:
            result["fallbacks_used"].append("ml_scoring")
        
        # Step 4: Graph analysis (COMMON - always runs)
        graph_result = self._safe_graph_analysis(claim_data)
        result["graph_analysis"] = graph_result["result"]
        result["scores"]["graph_risk_score"] = graph_result["score"]
        if graph_result["fallback_used"]:
            result["fallbacks_used"].append("graph_analysis")
        
        # Step 5: Calculate final score with adaptive weighting
        result["final_score"] = self._calculate_final_score(
            result["scores"], 
            product,
            result["fallbacks_used"]
        )
        
        # Step 6: Categorize risk level
        result["risk_level"] = self._categorize_risk(result["final_score"])
        
        # Step 7: Generate verdict
        result["verdict"] = self._generate_verdict(
            result["final_score"], 
            product,
            result["red_flags"],
            result["fallbacks_used"]
        )
        
        # Add metadata
        result["processing_notes"] = self._generate_processing_notes(
            result["fallbacks_used"],
            result["warnings"]
        )
        
        logger.success(
            f"Claim {claim_id} processed: score={result['final_score']:.2f}, "
            f"risk={result['risk_level']}, verdict={result['verdict']}, "
            f"fallbacks={len(result['fallbacks_used'])}"
        )
        
        return result
    
    def _safe_document_verification(self, claim_data: Dict) -> Dict:
        """Document verification with fallback."""
        try:
            if self.doc_verifier and claim_data.get("documents"):
                # Run actual verification
                doc_result = self.doc_verifier.verify(claim_data["documents"])
                return {
                    "result": doc_result,
                    "score": doc_result.get("average_confidence", 0.5),
                    "fallback_used": False
                }
        except Exception as e:
            logger.warning(f"Document verification failed: {e}, using fallback")
        
        # Fallback: Basic document count check
        docs = claim_data.get("documents_submitted", "")
        doc_count = len(docs.split(",")) if docs else 0
        
        return {
            "result": {"method": "fallback", "document_count": doc_count},
            "score": 0.5 if doc_count >= 2 else 0.7,  # Low docs = higher risk
            "fallback_used": True
        }
    
    def _analyze_motor_claim_safe(self, claim_data: Dict) -> Dict:
        """Motor analysis with subtype-aware fallback."""
        subtype = claim_data.get("subtype", "").lower()
        
        # Check if this subtype needs damage detection
        needs_damage_detection = subtype not in self.NO_DAMAGE_PHOTO_SUBTYPES.get("motor", [])
        
        if not needs_damage_detection:
            logger.info(f"Motor subtype '{subtype}' doesn't require damage photos - using document-based analysis")
            return self._motor_fallback_analysis(claim_data, reason="subtype_no_damage")
        
        # Check if damage photos provided
        damage_photos = claim_data.get("damage_photos", [])
        if not damage_photos:
            logger.warning(f"No damage photos for motor/{subtype} claim - using fallback")
            return self._motor_fallback_analysis(claim_data, reason="missing_photos")
        
        # Try damage detection
        try:
            if self.damage_detector:
                # Run actual damage detection
                damage_result = self.damage_detector.detect_damage(damage_photos[0])
                return {
                    "result": damage_result,
                    "score": damage_result.get("risk_assessment", {}).get("risk_score", 0.3),
                    "red_flags": damage_result.get("risk_assessment", {}).get("factors", []),
                    "fallback_used": False
                }
        except Exception as e:
            logger.error(f"Damage detection failed: {e}, using fallback")
        
        return self._motor_fallback_analysis(claim_data, reason="detection_failed")
    
    def _motor_fallback_analysis(self, claim_data: Dict, reason: str) -> Dict:
        """Fallback for motor claims without damage detection."""
        logger.info(f"Using motor fallback analysis (reason: {reason})")
        
        subtype = claim_data.get("subtype", "").lower()
        claim_amount = claim_data.get("claim_amount", 0)
        docs = claim_data.get("documents_submitted", "").lower()
        
        red_flags = []
        risk_score = 0.3  # Base risk
        
        # Theft-specific checks
        if subtype == "theft":
            if "fir" not in docs:
                red_flags.append("Theft claim missing FIR")
                risk_score += 0.3
            if "rc" not in docs:
                red_flags.append("Missing RC copy for theft claim")
                risk_score += 0.1
            if claim_amount > 500000:  # High value theft
                red_flags.append("High value theft claim requires extra scrutiny")
                risk_score += 0.2
        
        # Fire-specific checks
        elif subtype == "fire":
            if "fire_certificate" not in docs and "fir" not in docs:
                red_flags.append("Fire claim missing fire department certificate or FIR")
                risk_score += 0.3
        
        # Generic checks
        if not docs:
            red_flags.append("No supporting documents provided")
            risk_score += 0.4
        
        return {
            "result": {
                "method": "fallback",
                "reason": reason,
                "subtype_analysis": subtype,
                "document_based_risk": risk_score
            },
            "score": min(risk_score, 1.0),
            "red_flags": red_flags,
            "fallback_used": True
        }
    
    def _analyze_health_claim_safe(self, claim_data: Dict) -> Dict:
        """Health analysis with fallback."""
        try:
            result = self.health_analyzer.analyze(claim_data)
            return {
                "result": result,
                "score": result.get("medical_risk", 0.5),
                "fraud_ring_risk": result.get("fraud_ring_risk", 0.0),
                "red_flags": result.get("red_flags", []),
                "fallback_used": False
            }
        except Exception as e:
            logger.error(f"Health analysis failed: {e}, using fallback")
            return self._health_fallback_analysis(claim_data)
    
    def _health_fallback_analysis(self, claim_data: Dict) -> Dict:
        """Fallback for health claims."""
        logger.info("Using health fallback analysis")
        
        claim_amount = claim_data.get("claim_amount", 0)
        docs = claim_data.get("documents_submitted", "").lower()
        
        red_flags = []
        risk_score = 0.3
        
        # Document checks
        if "hospital" not in docs and "bill" not in docs:
            red_flags.append("Missing hospital bills")
            risk_score += 0.3
        
        if claim_amount > 100000:  # High value
            red_flags.append("High value health claim requires manual review")
            risk_score += 0.2
        
        return {
            "result": {"method": "fallback", "document_based_risk": risk_score},
            "score": min(risk_score, 1.0),
            "fraud_ring_risk": 0.0,
            "red_flags": red_flags,
            "fallback_used": True
        }
    
    def _analyze_life_claim_safe(self, claim_data: Dict) -> Dict:
        """Life claim analysis with fallback."""
        docs = claim_data.get("documents_submitted", "").lower()
        days_since_policy = claim_data.get("days_since_policy_start", 999)
        
        red_flags = []
        validity_score = 0.2
        
        if "death" not in docs:
            red_flags.append("Death certificate not provided")
            validity_score += 0.5
        
        if days_since_policy < 365:
            red_flags.append("Life claim within first year of policy (suspicious timing)")
            validity_score += 0.3
        
        return {
            "result": {
                "validity_score": min(validity_score, 1.0),
                "death_certificate_verified": "death" in docs,
                "policy_timing_suspicious": days_since_policy < 365
            },
            "score": min(validity_score, 1.0),
            "red_flags": red_flags,
            "fallback_used": False  # Life claims always use this method
        }
    
    def _analyze_property_claim_safe(self, claim_data: Dict) -> Dict:
        """Property analysis with fallback."""
        subtype = claim_data.get("subtype", "").lower()
        damage_photos = claim_data.get("damage_photos", [])
        
        if subtype in self.NO_DAMAGE_PHOTO_SUBTYPES.get("property", []):
            logger.info(f"Property subtype '{subtype}' doesn't require damage photos")
            return self._property_fallback_analysis(claim_data, reason="subtype_no_damage")
        
        if not damage_photos:
            return self._property_fallback_analysis(claim_data, reason="missing_photos")
        
        # Placeholder for actual property damage detection
        return {
            "result": {"damage_risk": 0.3},
            "score": 0.3,
            "red_flags": [],
            "fallback_used": False
        }
    
    def _property_fallback_analysis(self, claim_data: Dict, reason: str) -> Dict:
        """Fallback for property claims."""
        logger.info(f"Using property fallback analysis (reason: {reason})")
        
        subtype = claim_data.get("subtype", "").lower()
        docs = claim_data.get("documents_submitted", "").lower()
        
        red_flags = []
        risk_score = 0.3
        
        if subtype == "fire" and "fire_certificate" not in docs:
            red_flags.append("Fire claim missing fire department certificate")
            risk_score += 0.3
        
        return {
            "result": {"method": "fallback", "document_based_risk": risk_score},
            "score": min(risk_score, 1.0),
            "red_flags": red_flags,
            "fallback_used": True
        }
    
    def _safe_ml_scoring(self, claim_data: Dict) -> Dict:
        """ML scoring with fallback."""
        try:
            if self.ml_scorer:
                result = self.ml_scorer.score(claim_data)
                return {
                    "result": result,
                    "score": result.get("fraud_probability", 0.5),
                    "fallback_used": False
                }
        except Exception as e:
            logger.warning(f"ML scoring failed: {e}, using fallback")
        
        return {"result": {"method": "fallback"}, "score": 0.5, "fallback_used": True}
    
    def _safe_graph_analysis(self, claim_data: Dict) -> Dict:
        """Graph analysis with fallback."""
        try:
            if self.graph_analyzer:
                result = self.graph_analyzer.analyze(claim_data)
                return {
                    "result": result,
                    "score": result.get("final_risk_score", 0.5),
                    "fallback_used": False
                }
        except Exception as e:
            logger.warning(f"Graph analysis failed: {e}, using fallback")
        
        return {"result": {"method": "fallback"}, "score": 0.5, "fallback_used": True}
    
    def _calculate_final_score(
        self, 
        scores: Dict, 
        product: str,
        fallbacks_used: List[str]
    ) -> float:
        """Calculate final score with adaptive weighting based on available data."""
        
        # Adjust weights based on what fallbacks were used
        if product == "motor":
            if "motor_damage_detection" in fallbacks_used:
                # No damage detection → rely more on ML and Graph
                return (
                    0.10 * scores.get("doc_score", 0.5) +
                    0.15 * scores.get("damage_score", 0.5) +
                    0.45 * scores.get("ml_fraud_score", 0.5) +
                    0.30 * scores.get("graph_risk_score", 0.5)
                )
            else:
                # Full data available
                return (
                    0.20 * scores.get("doc_score", 0.5) +
                    0.25 * scores.get("damage_score", 0.5) +
                    0.35 * scores.get("ml_fraud_score", 0.5) +
                    0.20 * scores.get("graph_risk_score", 0.5)
                )
        
        elif product == "health":
            return (
                0.15 * scores.get("doc_score", 0.5) +
                0.25 * scores.get("medical_risk", 0.5) +
                0.15 * scores.get("fraud_ring_risk", 0.5) +
                0.30 * scores.get("ml_fraud_score", 0.5) +
                0.15 * scores.get("graph_risk_score", 0.5)
            )
        
        elif product == "life":
            return (
                0.40 * scores.get("doc_score", 0.5) +
                0.20 * scores.get("life_risk", 0.5) +
                0.25 * scores.get("ml_fraud_score", 0.5) +
                0.15 * scores.get("graph_risk_score", 0.5)
            )
        
        else:
            # Generic/fallback scoring
            return (
                0.30 * scores.get("doc_score", 0.5) +
                0.40 * scores.get("ml_fraud_score", 0.5) +
                0.30 * scores.get("graph_risk_score", 0.5)
            )
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level from score."""
        if score < 0.3:
            return "LOW"
        elif score < 0.5:
            return "MEDIUM"
        elif score < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_verdict(
        self, 
        score: float, 
        product: str, 
        red_flags: List[str],
        fallbacks_used: List[str]
    ) -> str:
        """Generate verdict with fallback awareness."""
        
        # If many fallbacks used, recommend review
        if len(fallbacks_used) >= 2:
            return "REVIEW"  # Too much missing data
        
        # Strict rules for life claims
        if product == "life" and score > 0.4:
            return "REVIEW"
        
        # Critical red flags
        critical_flags = [
            "death certificate",
            "fraud ring",
            "collusion",
            "missing fir"
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
    
    def _generate_processing_notes(
        self,
        fallbacks_used: List[str],
        warnings: List[str]
    ) -> str:
        """Generate human-readable processing notes."""
        if not fallbacks_used and not warnings:
            return "Claim processed with full analysis pipeline."
        
        notes = []
        
        if fallbacks_used:
            notes.append(
                f"Fallback methods used for: {', '.join(fallbacks_used)}. "
                f"Recommendation: Manual review for missing data."
            )
        
        if warnings:
            notes.append(f"Warnings: {'; '.join(warnings)}")
        
        return " ".join(notes)
