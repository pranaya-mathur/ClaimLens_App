"""
Health Claim Analyzer
Handles health-specific claim analysis (replaces vehicle damage detection for health claims)
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from loguru import logger
from collections import defaultdict
from datetime import datetime, timedelta


class HealthClaimAnalyzer:
    """
    Health-specific claim analysis pipeline.
    
    Provides equivalent analysis for health claims that motor claims get via DamageDetector.
    
    Features:
    - Hospital verification
    - Treatment cost benchmarking
    - Pre-existing condition detection
    - Medical document validation
    - Similar claims fraud detection (fraud rings)
    """
    
    def __init__(self):
        """Initialize health analyzer with reference data."""
        logger.info("Initializing HealthClaimAnalyzer...")
        
        # Load reference data (placeholder - would be from DB in production)
        self.hospital_registry = self._load_hospital_registry()
        self.treatment_benchmarks = self._load_treatment_benchmarks()
        
        # Fraud ring detection cache (in production, use Redis/Neo4j)
        self.recent_claims_cache = defaultdict(list)  # hospital -> claims
        self.claimant_history = defaultdict(list)  # claimant_id -> claims
        
        logger.success("✓ HealthClaimAnalyzer ready")
    
    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive health claim analysis.
        
        Args:
            claim_data: Dictionary containing claim information:
                - claim_id (required)
                - claimant_id (required)
                - hospital_name (optional)
                - treatment_type (optional)
                - claim_amount (required)
                - city (optional)
                - narrative (optional)
                - diagnosis (optional)
                - days_since_policy_start (optional)
                - incident_date (optional)
        
        Returns:
            Analysis results dictionary with:
                - hospital_verified: bool
                - cost_inflation_ratio: float
                - preexisting_condition_detected: bool
                - document_forgery_detected: bool
                - similar_claims_detected: bool
                - fraud_ring_risk: float (0-1)
                - medical_risk: float (0-1)
                - red_flags: list
        """
        logger.info(f"Analyzing health claim: {claim_data.get('claim_id', 'unknown')}")
        
        result = {
            "hospital_name": claim_data.get("hospital_name", "Not provided"),
            "treatment_type": claim_data.get("treatment_type", "Not specified"),
            "hospital_verified": False,
            "cost_benchmarking": {},
            "cost_inflation_ratio": 1.0,
            "preexisting_condition_detected": False,
            "document_forgery_detected": False,
            "similar_claims_detected": False,
            "similar_claims_count": 0,
            "fraud_ring_risk": 0.0,
            "red_flags": []
        }
        
        # 1. Hospital verification
        if claim_data.get("hospital_name"):
            result["hospital_verified"] = self._verify_hospital(
                claim_data.get("hospital_name"),
                claim_data.get("city")
            )
            if not result["hospital_verified"]:
                result["red_flags"].append("Hospital not verified in registry")
        
        # 2. Cost benchmarking
        if claim_data.get("treatment_type") and claim_data.get("claim_amount"):
            cost_analysis = self._benchmark_treatment_cost(
                treatment_type=claim_data.get("treatment_type"),
                claimed_amount=claim_data["claim_amount"],
                city=claim_data.get("city", "Unknown")
            )
            result["cost_benchmarking"] = cost_analysis
            result["cost_inflation_ratio"] = cost_analysis["inflation_ratio"]
            
            if cost_analysis["inflation_ratio"] > 1.5:
                result["red_flags"].append(
                    f"Treatment cost {cost_analysis['inflation_ratio']:.1f}x "
                    f"higher than average"
                )
        
        # 3. Pre-existing condition detection
        if claim_data.get("narrative"):
            result["preexisting_condition_detected"] = self._detect_preexisting_condition(
                narrative=claim_data.get("narrative", ""),
                days_since_policy_start=claim_data.get("days_since_policy_start", 999)
            )
            
            if result["preexisting_condition_detected"]:
                if claim_data.get("days_since_policy_start", 999) < 730:  # 2 years
                    result["red_flags"].append(
                        "Pre-existing condition claimed within waiting period"
                    )
        
        # 4. Medical document check (basic)
        if claim_data.get("documents_submitted"):
            result["document_forgery_detected"] = self._check_medical_documents(
                claim_data.get("documents_submitted", "")
            )
            
            if result["document_forgery_detected"]:
                result["red_flags"].append("Missing critical medical documents")
        
        # 5. Similar claims fraud detection (NEW!)
        similar_claims_analysis = self._detect_similar_claims(
            claim_data=claim_data,
            hospital_name=claim_data.get("hospital_name"),
            claimant_id=claim_data.get("claimant_id"),
            claim_amount=claim_data.get("claim_amount", 0),
            treatment_type=claim_data.get("treatment_type")
        )
        
        result["similar_claims_detected"] = similar_claims_analysis["detected"]
        result["similar_claims_count"] = similar_claims_analysis["count"]
        result["fraud_ring_risk"] = similar_claims_analysis["fraud_ring_risk"]
        
        if similar_claims_analysis["detected"]:
            result["red_flags"].extend(similar_claims_analysis["red_flags"])
        
        # 6. Update claim cache for future fraud detection
        self._update_claims_cache(claim_data)
        
        # 7. Calculate overall medical risk
        result["medical_risk"] = self._calculate_medical_risk(result)
        
        logger.info(
            f"Health analysis complete: risk={result['medical_risk']:.2f}, "
            f"fraud_ring_risk={result['fraud_ring_risk']:.2f}, "
            f"flags={len(result['red_flags'])}"
        )
        
        return result
    
    def _detect_similar_claims(
        self,
        claim_data: Dict[str, Any],
        hospital_name: Optional[str],
        claimant_id: Optional[str],
        claim_amount: float,
        treatment_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Detect similar claims that may indicate fraud rings.
        
        Checks for:
        - Multiple claims from same hospital with similar patterns
        - Same claimant filing rapid claims
        - Similar treatment amounts from same hospital
        - Suspicious hospital-claimant connections
        
        Args:
            claim_data: Full claim data dictionary
            hospital_name: Name of hospital
            claimant_id: Claimant identifier
            claim_amount: Claimed amount
            treatment_type: Type of treatment
        
        Returns:
            Dictionary with detection results
        """
        result = {
            "detected": False,
            "count": 0,
            "fraud_ring_risk": 0.0,
            "red_flags": [],
            "details": {}
        }
        
        if not hospital_name or not claimant_id:
            return result
        
        # Check 1: Similar claims from same hospital (hospital-based fraud ring)
        hospital_key = hospital_name.lower().strip()
        similar_hospital_claims = self.recent_claims_cache.get(hospital_key, [])
        
        # Filter claims from last 90 days with similar amounts
        recent_similar = [
            c for c in similar_hospital_claims
            if abs(c["claim_amount"] - claim_amount) / max(claim_amount, 1) < 0.15  # 15% similarity
        ]
        
        if len(recent_similar) >= 3:
            result["detected"] = True
            result["count"] = len(recent_similar)
            result["fraud_ring_risk"] += 0.4
            result["red_flags"].append(
                f"Found {len(recent_similar)} similar claims from {hospital_name} "
                f"with amounts within 15% (possible fraud ring)"
            )
            result["details"]["hospital_fraud_ring"] = {
                "hospital": hospital_name,
                "similar_claims_count": len(recent_similar)
            }
        
        # Check 2: Claimant rapid claims (serial fraudster)
        claimant_claims = self.claimant_history.get(claimant_id, [])
        
        if len(claimant_claims) >= 2:
            # Check if multiple claims within short period
            recent_claimant_claims = [
                c for c in claimant_claims
                if c.get("product") == "health"  # Only count health claims
            ]
            
            if len(recent_claimant_claims) >= 2:
                result["detected"] = True
                result["fraud_ring_risk"] += 0.3
                result["red_flags"].append(
                    f"Claimant has filed {len(recent_claimant_claims)} health claims "
                    f"recently (possible serial fraudster)"
                )
                result["details"]["rapid_claims"] = {
                    "claimant_id": claimant_id,
                    "recent_claims_count": len(recent_claimant_claims)
                }
        
        # Check 3: Hospital-claimant repeat connection (collusion)
        hospital_claimant_pairs = [
            c for c in similar_hospital_claims
            if c.get("claimant_id") == claimant_id
        ]
        
        if len(hospital_claimant_pairs) >= 2:
            result["detected"] = True
            result["fraud_ring_risk"] += 0.3
            result["red_flags"].append(
                f"Claimant has filed {len(hospital_claimant_pairs)} claims "
                f"at {hospital_name} (possible hospital-claimant collusion)"
            )
            result["details"]["hospital_claimant_collusion"] = {
                "hospital": hospital_name,
                "claimant_id": claimant_id,
                "claim_count": len(hospital_claimant_pairs)
            }
        
        # Normalize fraud ring risk to 0-1
        result["fraud_ring_risk"] = min(result["fraud_ring_risk"], 1.0)
        result["fraud_ring_risk"] = round(result["fraud_ring_risk"], 3)
        
        return result
    
    def _update_claims_cache(self, claim_data: Dict[str, Any]) -> None:
        """
        Update claims cache for fraud detection.
        
        Args:
            claim_data: Claim data to cache
        """
        hospital_name = claim_data.get("hospital_name")
        claimant_id = claim_data.get("claimant_id")
        
        # Cache by hospital
        if hospital_name:
            hospital_key = hospital_name.lower().strip()
            self.recent_claims_cache[hospital_key].append({
                "claim_id": claim_data.get("claim_id"),
                "claimant_id": claimant_id,
                "claim_amount": claim_data.get("claim_amount", 0),
                "treatment_type": claim_data.get("treatment_type"),
                "timestamp": datetime.now()
            })
            
            # Keep only last 90 days (memory optimization)
            cutoff = datetime.now() - timedelta(days=90)
            self.recent_claims_cache[hospital_key] = [
                c for c in self.recent_claims_cache[hospital_key]
                if c["timestamp"] > cutoff
            ]
        
        # Cache by claimant
        if claimant_id:
            self.claimant_history[claimant_id].append({
                "claim_id": claim_data.get("claim_id"),
                "hospital_name": hospital_name,
                "claim_amount": claim_data.get("claim_amount", 0),
                "product": "health",
                "timestamp": datetime.now()
            })
            
            # Keep only last 180 days
            cutoff = datetime.now() - timedelta(days=180)
            self.claimant_history[claimant_id] = [
                c for c in self.claimant_history[claimant_id]
                if c["timestamp"] > cutoff
            ]
    
    def _verify_hospital(self, hospital_name: str, city: Optional[str]) -> bool:
        """
        Verify hospital exists in registry.
        
        Args:
            hospital_name: Name of hospital
            city: City where hospital is located
        
        Returns:
            True if verified, False otherwise
        """
        if not hospital_name:
            return False
        
        # Normalize for lookup
        hospital_key = hospital_name.lower().strip()
        if city:
            lookup_key = f"{hospital_key}_{city.lower().strip()}"
        else:
            lookup_key = hospital_key
        
        # Check against registry (flexible matching)
        is_verified = (
            lookup_key in self.hospital_registry or
            hospital_key in self.hospital_registry
        )
        
        logger.debug(f"Hospital verification: {hospital_name} = {is_verified}")
        
        return is_verified
    
    def _benchmark_treatment_cost(
        self,
        treatment_type: str,
        claimed_amount: float,
        city: str
    ) -> Dict[str, Any]:
        """
        Compare claimed cost with treatment benchmarks.
        
        Args:
            treatment_type: Type of treatment (e.g., "surgery", "hospitalization")
            claimed_amount: Amount claimed
            city: City where treatment occurred
        
        Returns:
            Dictionary with benchmark data and inflation ratio
        """
        # Create lookup key
        benchmark_key = f"{treatment_type.lower()}_{city.lower()}"
        
        if benchmark_key in self.treatment_benchmarks:
            benchmark = self.treatment_benchmarks[benchmark_key]
            avg_cost = benchmark["average_cost"]
            
            inflation_ratio = claimed_amount / avg_cost if avg_cost > 0 else 1.0
            
            return {
                "average_cost": avg_cost,
                "claimed_cost": claimed_amount,
                "inflation_ratio": round(inflation_ratio, 2),
                "is_outlier": inflation_ratio > 2.0
            }
        
        # No benchmark data available - neutral assessment
        logger.warning(f"No benchmark data for {benchmark_key}")
        return {
            "average_cost": None,
            "claimed_cost": claimed_amount,
            "inflation_ratio": 1.0,
            "is_outlier": False
        }
    
    def _detect_preexisting_condition(
        self,
        narrative: str,
        days_since_policy_start: int
    ) -> bool:
        """
        Detect if condition appears to be pre-existing.
        
        Args:
            narrative: Claim narrative text
            days_since_policy_start: Days since policy started
        
        Returns:
            True if pre-existing condition detected
        """
        # Keywords indicating pre-existing conditions
        preexisting_keywords = [
            "chronic", "long-standing", "since childhood", "for years",
            "पुरानी बीमारी", "बचपन से", "कई साल से", "लंबे समय से",
            "purani bimari", "bachpan se", "kai saal se"
        ]
        
        narrative_lower = narrative.lower()
        
        for keyword in preexisting_keywords:
            if keyword in narrative_lower:
                logger.warning(f"Pre-existing condition keyword detected: {keyword}")
                return True
        
        return False
    
    def _check_medical_documents(self, documents_submitted: str) -> bool:
        """
        Check if required medical documents are present.
        
        Args:
            documents_submitted: Comma-separated list of submitted documents
        
        Returns:
            True if critical documents are missing (forgery suspected)
        """
        if not documents_submitted:
            return True  # No docs = suspicious
        
        docs_lower = documents_submitted.lower()
        
        # Required documents for health claims
        required_docs = ["hospital", "bill", "discharge"]
        
        missing_count = 0
        for req_doc in required_docs:
            if req_doc not in docs_lower:
                missing_count += 1
        
        # If 2+ required docs missing, flag as suspicious
        return missing_count >= 2
    
    def _calculate_medical_risk(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calculate overall medical risk score.
        
        Args:
            analysis_result: Dictionary with analysis components
        
        Returns:
            Risk score (0-1)
        """
        risk_score = 0.0
        
        # Hospital not verified
        if not analysis_result["hospital_verified"]:
            risk_score += 0.20
        
        # Cost inflation
        inflation = analysis_result["cost_inflation_ratio"]
        if inflation > 2.0:
            risk_score += 0.25
        elif inflation > 1.5:
            risk_score += 0.15
        
        # Pre-existing condition
        if analysis_result["preexisting_condition_detected"]:
            risk_score += 0.20
        
        # Document issues
        if analysis_result["document_forgery_detected"]:
            risk_score += 0.15
        
        # Fraud ring risk (most important)
        risk_score += analysis_result["fraud_ring_risk"] * 0.35
        
        # Normalize to 0-1
        risk_score = min(risk_score, 1.0)
        
        return round(risk_score, 3)
    
    def _load_hospital_registry(self) -> Dict[str, Dict]:
        """
        Load verified hospital registry.
        
        In production, this would load from database.
        Returns placeholder data for now.
        """
        # Placeholder registry (top hospitals in major cities)
        return {
            "apollo_mumbai": {"license": "MH-12345", "verified": True},
            "apollo": {"license": "MULTI", "verified": True},
            "fortis_delhi": {"license": "DL-67890", "verified": True},
            "fortis": {"license": "MULTI", "verified": True},
            "max_healthcare_delhi": {"license": "DL-11111", "verified": True},
            "max_healthcare": {"license": "MULTI", "verified": True},
            "manipal_bangalore": {"license": "KA-22222", "verified": True},
            "manipal": {"license": "MULTI", "verified": True},
            "aiims_delhi": {"license": "DL-00001", "verified": True},
            "aiims": {"license": "MULTI", "verified": True},
        }
    
    def _load_treatment_benchmarks(self) -> Dict[str, Dict]:
        """
        Load treatment cost benchmarks by city.
        
        In production, this would load from database.
        Returns placeholder data for now.
        """
        # Placeholder benchmarks (average costs in INR)
        return {
            "surgery_mumbai": {"average_cost": 50000, "std_dev": 10000},
            "surgery_delhi": {"average_cost": 45000, "std_dev": 9000},
            "surgery_bangalore": {"average_cost": 48000, "std_dev": 9500},
            "hospitalization_mumbai": {"average_cost": 3000, "std_dev": 500},
            "hospitalization_delhi": {"average_cost": 2800, "std_dev": 450},
            "hospitalization_bangalore": {"average_cost": 2900, "std_dev": 475},
            "diagnostics_mumbai": {"average_cost": 5000, "std_dev": 1000},
            "diagnostics_delhi": {"average_cost": 4500, "std_dev": 900},
            "diagnostics_bangalore": {"average_cost": 4800, "std_dev": 950},
        }
