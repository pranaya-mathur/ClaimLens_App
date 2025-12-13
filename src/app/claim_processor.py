"""
Claim Processor - Product-Aware Routing with Smart Fallbacks, Semantic Aggregation & LLM Explanations
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
    from .semantic_aggregator import SemanticAggregator
    from .component_adapters import ComponentAdapter
    from src.explainability import FraudExplainer
    CV_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some engines not available: {e}")
    CV_ENGINE_AVAILABLE = False


class ClaimProcessor:
    """
    Main claim processing orchestrator with smart fallback handling.
    
    **New in v2.0: Semantic Aggregation + LLM Explanations**
    - Semantic verdicts (FORGED, HIGH_RISK, etc) instead of just scores
    - Critical flag gating logic
    - Adaptive weighting based on confidence
    - Full reasoning chain for explainability
    - LLM-powered natural language explanations (Groq + Llama-3.3-70B)
    - Backward compatible with old API
    
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
        health_analyzer: Optional[Any] = None,
        use_semantic_aggregation: bool = True,  # NEW: Enable semantic mode
        enable_llm_explanations: bool = True,  # NEW: Enable LLM explanations
        groq_api_key: Optional[str] = None  # NEW: Groq API key
    ):
        """
        Initialize claim processor.
        
        Args:
            damage_detector: DamageDetector instance (for motor/property)
            doc_verifier: DocumentVerifier instance (all products)
            ml_scorer: MLFraudScorer instance (all products)
            graph_analyzer: FraudDetector instance (all products)
            health_analyzer: HealthClaimAnalyzer instance (for health)
            use_semantic_aggregation: If True, uses new semantic system (default: True)
            enable_llm_explanations: If True, generates LLM explanations (default: True)
            groq_api_key: Groq API key for LLM explanations (or set GROQ_API_KEY env var)
        """
        logger.info("Initializing ClaimProcessor with smart fallback system...")
        
        # Common modules (used by all products)
        self.doc_verifier = doc_verifier
        self.ml_scorer = ml_scorer
        self.graph_analyzer = graph_analyzer
        
        # Product-specific modules
        self.damage_detector = damage_detector  # Motor/Property only
        self.health_analyzer = health_analyzer or HealthClaimAnalyzer()  # Health only
        
        # NEW: Semantic aggregation components
        self.use_semantic_aggregation = use_semantic_aggregation
        if use_semantic_aggregation:
            self.semantic_aggregator = SemanticAggregator()
            self.adapter = ComponentAdapter()
            logger.info("✓ Semantic aggregation enabled")
        else:
            self.semantic_aggregator = None
            self.adapter = None
            logger.info("✓ Using legacy aggregation")
        
        # NEW: LLM explainer
        self.enable_llm_explanations = enable_llm_explanations
        if enable_llm_explanations:
            try:
                self.explainer = FraudExplainer(api_key=groq_api_key)
                logger.info("✓ LLM explainer initialized (Groq + Llama-3.3-70B)")
            except Exception as e:
                logger.warning(f"LLM explainer initialization failed: {e}, using templates")
                self.explainer = FraudExplainer()  # Falls back to templates
        else:
            self.explainer = None
            logger.info("✓ LLM explanations disabled")
        
        logger.success("✓ ClaimProcessor ready with fallback handling")
    
    def process_claim(
        self, 
        claim_data: Dict[str, Any],
        generate_explanation: bool = True,
        explanation_audience: str = "adjuster"
    ) -> Dict[str, Any]:
        """
        Route claim to appropriate analysis pipeline with smart fallback.
        
        This method now supports:
        1. Semantic mode: Structured verdicts + reasoning chain
        2. Legacy mode: Numeric scores only
        3. LLM explanations: Human-readable natural language
        
        Args:
            claim_data: Claim data dictionary with at minimum:
                - product: str (motor/health/life/property)
                - subtype: str (theft/accident/hospitalization/etc)
                - claim_id: str
                - Other fields depending on product
            generate_explanation: If True, generates LLM explanation
            explanation_audience: "adjuster" (technical) or "customer" (friendly)
        
        Returns:
            Complete analysis result with optional explanation
        """
        product = claim_data.get("product", "").lower()
        subtype = claim_data.get("subtype", "").lower()
        claim_id = claim_data.get("claim_id", "unknown")
        
        logger.info(f"Processing {product}/{subtype} claim: {claim_id}")
        
        # Run component-level analysis (same for both modes)
        raw_results = self._run_component_analysis(claim_data)
        
        # NEW: If semantic aggregation enabled, use new path
        if self.use_semantic_aggregation and self.semantic_aggregator:
            try:
                semantic_result = self._process_with_semantic_aggregation(
                    claim_id, product, raw_results
                )
                
                # Add legacy fields for backward compatibility
                semantic_result["scores"] = raw_results["scores"]
                semantic_result["product"] = product
                semantic_result["subtype"] = subtype
                
                # NEW: Generate LLM explanation if requested
                if generate_explanation and self.explainer:
                    try:
                        explanation = self.explainer.explain_verdict(
                            decision=semantic_result,
                            audience=explanation_audience
                        )
                        semantic_result["explanation"] = explanation
                        semantic_result["explanation_audience"] = explanation_audience
                        logger.info("LLM explanation generated")
                    except Exception as e:
                        logger.warning(f"LLM explanation failed: {e}")
                        semantic_result["explanation"] = "Explanation unavailable"
                
                return semantic_result
                
            except Exception as e:
                logger.error(f"Semantic aggregation failed: {e}, falling back to legacy mode")
                # Fall through to legacy mode
        
        # Legacy aggregation (original logic)
        return self._process_with_legacy_aggregation(claim_id, product, subtype, raw_results)
    
    def _run_component_analysis(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all component-level analysis.
        Returns raw results from each component.
        """
        product = claim_data.get("product", "").lower()
        
        result = {
            "scores": {},
            "red_flags": [],
            "warnings": [],
            "fallbacks_used": [],
            "components": {}  # Raw component results
        }
        
        # Step 1: Document verification (COMMON - all products)
        doc_result = self._safe_document_verification(claim_data)
        result["components"]["document_verification"] = doc_result["result"]
        result["scores"]["doc_score"] = doc_result["score"]
        if doc_result["fallback_used"]:
            result["fallbacks_used"].append("document_verification")
        
        # Step 2: Product-specific analysis
        if product == "motor":
            motor_result = self._analyze_motor_claim_safe(claim_data)
            result["components"]["damage_detection"] = motor_result["result"]
            result["scores"]["damage_score"] = motor_result["score"]
            result["red_flags"].extend(motor_result["red_flags"])
            if motor_result["fallback_used"]:
                result["fallbacks_used"].append("damage_detection")
        
        elif product == "health":
            health_result = self._analyze_health_claim_safe(claim_data)
            result["components"]["health_analysis"] = health_result["result"]
            result["scores"]["medical_risk"] = health_result["score"]
            result["scores"]["fraud_ring_risk"] = health_result.get("fraud_ring_risk", 0.0)
            result["red_flags"].extend(health_result["red_flags"])
            if health_result["fallback_used"]:
                result["fallbacks_used"].append("health_analysis")
        
        elif product == "life":
            life_result = self._analyze_life_claim_safe(claim_data)
            result["components"]["life_analysis"] = life_result["result"]
            result["scores"]["life_risk"] = life_result["score"]
            result["red_flags"].extend(life_result["red_flags"])
            if life_result["fallback_used"]:
                result["fallbacks_used"].append("life_verification")
        
        elif product == "property":
            property_result = self._analyze_property_claim_safe(claim_data)
            result["components"]["property_analysis"] = property_result["result"]
            result["scores"]["property_damage_score"] = property_result["score"]
            result["red_flags"].extend(property_result["red_flags"])
            if property_result["fallback_used"]:
                result["fallbacks_used"].append("property_damage_detection")
        
        # Step 3: ML fraud scoring (COMMON)
        ml_result = self._safe_ml_scoring(claim_data)
        result["components"]["ml_fraud_score"] = ml_result["result"]
        result["scores"]["ml_fraud_score"] = ml_result["score"]
        if ml_result["fallback_used"]:
            result["fallbacks_used"].append("ml_scoring")
        
        # Step 4: Graph analysis (COMMON)
        graph_result = self._safe_graph_analysis(claim_data)
        result["components"]["graph_analysis"] = graph_result["result"]
        result["scores"]["graph_risk_score"] = graph_result["score"]
        if graph_result["fallback_used"]:
            result["fallbacks_used"].append("graph_analysis")
        
        return result
    
    def _process_with_semantic_aggregation(
        self, claim_id: str, product: str, raw_results: Dict
    ) -> Dict[str, Any]:
        """
        NEW: Process claim using semantic aggregation.
        
        Converts raw component results to semantic verdicts,
        applies critical flag logic, and generates reasoning chain.
        """
        logger.info(f"Using semantic aggregation for claim {claim_id}")
        
        # Convert raw results to semantic ComponentResult objects
        component_results = {}
        
        # Document verification
        if "document_verification" in raw_results["components"]:
            component_results["document_verification"] = self.adapter.adapt_document_verification(
                raw_results["components"]["document_verification"]
            )
        
        # Damage detection (motor/property)
        if "damage_detection" in raw_results["components"]:
            component_results["damage_detection"] = self.adapter.adapt_damage_detection(
                raw_results["components"]["damage_detection"]
            )
        
        # Health analysis
        if "health_analysis" in raw_results["components"]:
            component_results["health_analysis"] = self.adapter.adapt_health_analysis(
                raw_results["components"]["health_analysis"]
            )
        
        # Life analysis
        if "life_analysis" in raw_results["components"]:
            component_results["life_analysis"] = self.adapter.adapt_life_analysis(
                raw_results["components"]["life_analysis"]
            )
        
        # ML fraud scoring
        if "ml_fraud_score" in raw_results["components"]:
            component_results["ml_fraud_score"] = self.adapter.adapt_ml_scoring(
                raw_results["components"]["ml_fraud_score"]
            )
        
        # Graph analysis
        if "graph_analysis" in raw_results["components"]:
            component_results["graph_analysis"] = self.adapter.adapt_graph_analysis(
                raw_results["components"]["graph_analysis"]
            )
        
        # Run semantic aggregation
        final_decision = self.semantic_aggregator.aggregate(
            claim_id=claim_id,
            component_results=component_results,
            product_type=product,
            fallbacks_used=raw_results["fallbacks_used"]
        )
        
        # Convert to API response format
        return final_decision.to_dict()
    
    # ============================================================================
    # Legacy & Component Methods (unchanged - keeping original implementation)
    # ============================================================================
    
    def _process_with_legacy_aggregation(
        self, claim_id: str, product: str, subtype: str, raw_results: Dict
    ) -> Dict[str, Any]:
        """Legacy aggregation logic."""
        logger.info(f"Using legacy aggregation for claim {claim_id}")
        
        result = {
            "claim_id": claim_id,
            "product": product,
            "subtype": subtype,
            "scores": raw_results["scores"],
            "red_flags": raw_results["red_flags"],
            "warnings": raw_results["warnings"],
            "fallbacks_used": raw_results["fallbacks_used"]
        }
        
        result["final_score"] = self._calculate_final_score(
            result["scores"], product, result["fallbacks_used"]
        )
        result["risk_level"] = self._categorize_risk(result["final_score"])
        result["verdict"] = self._generate_verdict(
            result["final_score"], product, result["red_flags"], result["fallbacks_used"]
        )
        result["processing_notes"] = self._generate_processing_notes(
            result["fallbacks_used"], result["warnings"]
        )
        
        logger.success(
            f"Claim {claim_id} processed: score={result['final_score']:.2f}, "
            f"verdict={result['verdict']}, fallbacks={len(result['fallbacks_used'])}"
        )
        return result
    
    def _safe_document_verification(self, claim_data: Dict) -> Dict:
        """Document verification with fallback."""
        try:
            if self.doc_verifier and claim_data.get("documents"):
                doc_result = self.doc_verifier.verify(claim_data["documents"])
                return {
                    "result": doc_result,
                    "score": doc_result.get("average_confidence", 0.5),
                    "fallback_used": False
                }
        except Exception as e:
            logger.warning(f"Document verification failed: {e}, using fallback")
        docs = claim_data.get("documents_submitted", "")
        doc_count = len(docs.split(",")) if docs else 0
        return {
            "result": {"method": "fallback", "document_count": doc_count},
            "score": 0.5 if doc_count >= 2 else 0.7,
            "fallback_used": True
        }
    
    def _analyze_motor_claim_safe(self, claim_data: Dict) -> Dict:
        """Motor analysis with fallback."""
        subtype = claim_data.get("subtype", "").lower()
        needs_damage = subtype not in self.NO_DAMAGE_PHOTO_SUBTYPES.get("motor", [])
        if not needs_damage:
            return self._motor_fallback_analysis(claim_data, "subtype_no_damage")
        if not claim_data.get("damage_photos"):
            return self._motor_fallback_analysis(claim_data, "missing_photos")
        try:
            if self.damage_detector:
                result = self.damage_detector.detect_damage(claim_data["damage_photos"][0])
                return {
                    "result": result,
                    "score": result.get("risk_assessment", {}).get("risk_score", 0.3),
                    "red_flags": result.get("risk_assessment", {}).get("factors", []),
                    "fallback_used": False
                }
        except Exception:
            pass
        return self._motor_fallback_analysis(claim_data, "detection_failed")
    
    def _motor_fallback_analysis(self, claim_data: Dict, reason: str) -> Dict:
        """Motor fallback."""
        red_flags, risk = [], 0.3
        docs = claim_data.get("documents_submitted", "").lower()
        subtype = claim_data.get("subtype", "").lower()
        amount = claim_data.get("claim_amount", 0)
        
        if subtype == "theft":
            if "fir" not in docs:
                red_flags.append("Theft claim missing FIR")
                risk += 0.3
            if amount > 500000:
                red_flags.append("High value theft")
                risk += 0.2
        elif subtype == "fire" and "fire_certificate" not in docs:
            red_flags.append("Fire claim missing certificate")
            risk += 0.3
        
        if not docs:
            red_flags.append("No documents")
            risk += 0.4
        
        return {
            "result": {"method": "fallback", "reason": reason, "document_based_risk": risk},
            "score": min(risk, 1.0),
            "red_flags": red_flags,
            "fallback_used": True
        }
    
    def _analyze_health_claim_safe(self, claim_data: Dict) -> Dict:
        """Health analysis."""
        try:
            result = self.health_analyzer.analyze(claim_data)
            return {
                "result": result,
                "score": result.get("medical_risk", 0.5),
                "fraud_ring_risk": result.get("fraud_ring_risk", 0.0),
                "red_flags": result.get("red_flags", []),
                "fallback_used": False
            }
        except Exception:
            return self._health_fallback_analysis(claim_data)
    
    def _health_fallback_analysis(self, claim_data: Dict) -> Dict:
        """Health fallback."""
        red_flags, risk = [], 0.3
        docs = claim_data.get("documents_submitted", "").lower()
        if "hospital" not in docs and "bill" not in docs:
            red_flags.append("Missing hospital bills")
            risk += 0.3
        if claim_data.get("claim_amount", 0) > 100000:
            red_flags.append("High value claim")
            risk += 0.2
        return {
            "result": {"method": "fallback"},
            "score": min(risk, 1.0),
            "fraud_ring_risk": 0.0,
            "red_flags": red_flags,
            "fallback_used": True
        }
    
    def _analyze_life_claim_safe(self, claim_data: Dict) -> Dict:
        """Life analysis."""
        docs = claim_data.get("documents_submitted", "").lower()
        days = claim_data.get("days_since_policy_start", 999)
        red_flags, risk = [], 0.2
        if "death" not in docs:
            red_flags.append("Death certificate missing")
            risk += 0.5
        if days < 365:
            red_flags.append("Claim within first year")
            risk += 0.3
        return {
            "result": {"validity_score": min(risk, 1.0), "death_certificate_verified": "death" in docs},
            "score": min(risk, 1.0),
            "red_flags": red_flags,
            "fallback_used": False
        }
    
    def _analyze_property_claim_safe(self, claim_data: Dict) -> Dict:
        """Property analysis."""
        subtype = claim_data.get("subtype", "").lower()
        if subtype in self.NO_DAMAGE_PHOTO_SUBTYPES.get("property", []):
            return self._property_fallback_analysis(claim_data, "subtype_no_damage")
        if not claim_data.get("damage_photos"):
            return self._property_fallback_analysis(claim_data, "missing_photos")
        return {"result": {}, "score": 0.3, "red_flags": [], "fallback_used": False}
    
    def _property_fallback_analysis(self, claim_data: Dict, reason: str) -> Dict:
        """Property fallback."""
        docs = claim_data.get("documents_submitted", "").lower()
        risk, red_flags = 0.3, []
        if claim_data.get("subtype") == "fire" and "fire_certificate" not in docs:
            red_flags.append("Missing fire certificate")
            risk += 0.3
        return {"result": {"method": "fallback"}, "score": min(risk, 1.0), "red_flags": red_flags, "fallback_used": True}
    
    def _safe_ml_scoring(self, claim_data: Dict) -> Dict:
        """ML scoring."""
        try:
            if self.ml_scorer:
                result = self.ml_scorer.score(claim_data)
                return {"result": result, "score": result.get("fraud_probability", 0.5), "fallback_used": False}
        except Exception:
            pass
        return {"result": {"method": "fallback"}, "score": 0.5, "fallback_used": True}
    
    def _safe_graph_analysis(self, claim_data: Dict) -> Dict:
        """Graph analysis."""
        try:
            if self.graph_analyzer:
                result = self.graph_analyzer.analyze(claim_data)
                return {"result": result, "score": result.get("final_risk_score", 0.5), "fallback_used": False}
        except Exception:
            pass
        return {"result": {"method": "fallback"}, "score": 0.5, "fallback_used": True}
    
    def _calculate_final_score(self, scores: Dict, product: str, fallbacks: List[str]) -> float:
        """Calculate score."""
        if product == "motor":
            if "motor_damage_detection" in fallbacks:
                return 0.10*scores.get("doc_score",0.5) + 0.15*scores.get("damage_score",0.5) + 0.45*scores.get("ml_fraud_score",0.5) + 0.30*scores.get("graph_risk_score",0.5)
            return 0.20*scores.get("doc_score",0.5) + 0.25*scores.get("damage_score",0.5) + 0.35*scores.get("ml_fraud_score",0.5) + 0.20*scores.get("graph_risk_score",0.5)
        elif product == "health":
            return 0.15*scores.get("doc_score",0.5) + 0.25*scores.get("medical_risk",0.5) + 0.15*scores.get("fraud_ring_risk",0.5) + 0.30*scores.get("ml_fraud_score",0.5) + 0.15*scores.get("graph_risk_score",0.5)
        elif product == "life":
            return 0.40*scores.get("doc_score",0.5) + 0.20*scores.get("life_risk",0.5) + 0.25*scores.get("ml_fraud_score",0.5) + 0.15*scores.get("graph_risk_score",0.5)
        return 0.30*scores.get("doc_score",0.5) + 0.40*scores.get("ml_fraud_score",0.5) + 0.30*scores.get("graph_risk_score",0.5)
    
    def _categorize_risk(self, score: float) -> str:
        if score < 0.3: return "LOW"
        elif score < 0.5: return "MEDIUM"
        elif score < 0.7: return "HIGH"
        return "CRITICAL"
    
    def _generate_verdict(self, score: float, product: str, flags: List[str], fallbacks: List[str]) -> str:
        if len(fallbacks) >= 2: return "REVIEW"
        if product == "life" and score > 0.4: return "REVIEW"
        critical = ["death certificate", "fraud ring", "collusion", "missing fir"]
        if any(any(c in f.lower() for c in critical) for f in flags): return "REVIEW"
        if score < 0.3: return "APPROVE"
        elif score < 0.6: return "REVIEW"
        return "REJECT"
    
    def _generate_processing_notes(self, fallbacks: List[str], warnings: List[str]) -> str:
        if not fallbacks and not warnings: return "Claim processed with full pipeline."
        notes = []
        if fallbacks: notes.append(f"Fallbacks: {', '.join(fallbacks)}")
        if warnings: notes.append(f"Warnings: {'; '.join(warnings)}")
        return " ".join(notes)
