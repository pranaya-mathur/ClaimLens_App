"""Unified Analysis API Route - Auto-Detection System

Intelligently detects claim type from narrative + files and runs
only relevant modules (ML, Documents, CV, Graph).

Features:
- Auto claim type detection (motor/health/life/property)
- Smart module selection based on claim type
- Hinglish keyword support
- Single endpoint for complete fraud analysis
- 30% ML + 40% Documents + 30% Graph risk formula
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
import httpx
import json
from loguru import logger
import re

router = APIRouter()


def detect_claim_type(
    narrative: str,
    claim_amount: float,
    files_uploaded: dict
) -> tuple[str, dict]:
    """
    🧠 Intelligent claim type detection from narrative + files.
    
    Args:
        narrative: Claim description text
        claim_amount: Claim amount in rupees
        files_uploaded: Dictionary of uploaded file names
    
    Returns:
        tuple: (detected_claim_type, detection_scores)
    """
    
    narrative_lower = narrative.lower() if narrative else ""
    
    # Score for each claim type
    scores = {
        "motor": 0,
        "health": 0,
        "life": 0,
        "property": 0
    }
    
    # ═══════════════════════════════════════
    # KEYWORD ANALYSIS (English + Hinglish)
    # ═══════════════════════════════════════
    
    # Motor keywords
    motor_keywords = [
        'car', 'vehicle', 'accident', 'collision', 'bike', 'motorcycle',
        'driving', 'road', 'driver', 'license', 'traffic', 'tire', 'tyre',
        'bumper', 'dent', 'scratch', 'windshield', 'dashboard', 'gadi',
        'gaadi', 'car ka', 'bike ka', 'scooter', 'auto', 'crash', 'hood',
        'steering', 'brake', 'engine', 'damage to vehicle', 'hit'
    ]
    
    # Health keywords
    health_keywords = [
        'hospital', 'surgery', 'doctor', 'medical', 'treatment', 'illness',
        'disease', 'medicine', 'prescription', 'diagnosis', 'patient',
        'clinic', 'healthcare', 'emergency', 'ambulance', 'operation',
        'bimari', 'dawai', 'ilaj', 'davai', 'hospital mein', 'admit',
        'admitted', 'ward', 'icu', 'health', 'injury', 'wound', 'fracture'
    ]
    
    # Life insurance keywords
    life_keywords = [
        'death', 'died', 'deceased', 'passed away', 'demise', 'funeral',
        'mortality', 'fatal', 'expire', 'mrityu', 'mar gaya', 'mar gayi',
        'nahi rahe', 'guzar gaye', 'death certificate', 'last rites',
        'obituary', 'cremation', 'burial', 'nominee'
    ]
    
    # Property keywords
    property_keywords = [
        'house', 'home', 'property', 'building', 'fire', 'flood', 'earthquake',
        'theft', 'burglary', 'damage', 'roof', 'wall', 'ceiling', 'water damage',
        'ghar', 'makan', 'imarat', 'aag', 'chori', 'toota', 'tutna',
        'broken', 'leakage', 'storm', 'hurricane', 'collapse'
    ]
    
    # Count keyword matches (with weight)
    for keyword in motor_keywords:
        if keyword in narrative_lower:
            scores["motor"] += 2
    
    for keyword in health_keywords:
        if keyword in narrative_lower:
            scores["health"] += 2
    
    for keyword in life_keywords:
        if keyword in narrative_lower:
            scores["life"] += 3  # Higher weight for life claims
    
    for keyword in property_keywords:
        if keyword in narrative_lower:
            scores["property"] += 2
    
    # ═══════════════════════════════════════
    # FILE TYPE ANALYSIS
    # ═══════════════════════════════════════
    
    # Vehicle-related files
    if files_uploaded.get('vehicle_file') or files_uploaded.get('damage_photo'):
        vehicle_name = (files_uploaded.get('vehicle_file', '') + 
                       files_uploaded.get('damage_photo', '')).lower()
        if any(word in vehicle_name for word in ['car', 'vehicle', 'bike', 'auto']):
            scores["motor"] += 5
        else:
            scores["motor"] += 3  # Generic damage photo
    
    # Death certificate
    if files_uploaded.get('supporting_file_2'):
        filename = files_uploaded['supporting_file_2'].lower()
        if 'death' in filename or 'certificate' in filename:
            scores["life"] += 10
    
    # Hospital/medical documents
    if files_uploaded.get('supporting_file_1'):
        filename = files_uploaded['supporting_file_1'].lower()
        if any(word in filename for word in ['hospital', 'medical', 'bill', 'prescription', 'report']):
            scores["health"] += 5
        elif any(word in filename for word in ['license', 'licence', 'dl']):
            scores["motor"] += 4
    
    # Property damage photos (damage_photo without vehicle context)
    if files_uploaded.get('damage_photo') and not files_uploaded.get('vehicle_file'):
        scores["property"] += 3
    
    # ═══════════════════════════════════════
    # CLAIM AMOUNT HINTS
    # ═══════════════════════════════════════
    
    # Very high amounts often indicate life/property claims
    if claim_amount > 1000000:  # > 10 lakhs
        scores["life"] += 1
        scores["property"] += 1
    
    # Medium amounts common in health
    if 50000 <= claim_amount <= 500000:
        scores["health"] += 1
    
    # ═══════════════════════════════════════
    # DECISION
    # ═══════════════════════════════════════
    
    detected_type = max(scores, key=scores.get)
    confidence = scores[detected_type]
    
    logger.info(f"🎯 Claim type scores: {scores}")
    logger.info(f"🎯 Detected: {detected_type} (confidence: {confidence})")
    
    # If no clear winner, default to health (most common)
    if confidence < 3:
        logger.warning("⚠️ Low confidence in detection, defaulting to 'health'")
        return "health", scores
    
    return detected_type, scores


@router.post(
    "/analyze-complete",
    summary="🤖 Complete AI-Powered Claim Analysis with Auto-Detection",
    description="""
    Intelligent unified analysis that:
    1. Auto-detects claim type from narrative + files
    2. Runs only relevant modules (ML, Identity Docs, Claim-specific verification, Graph)
    3. Returns comprehensive risk assessment
    
    **Supported Claim Types:**
    - 🚗 Motor: Vehicle damage detection + License verification
    - 🏥 Health: Hospital bill verification + Medical document checks
    - ⚰️ Life: Death certificate verification
    - 🏠 Property: Property damage assessment + Invoice verification
    
    **Risk Formula:** 30% ML + 40% Documents + 30% Graph
    """
)
async def analyze_complete_claim(
    claim_data: str = Form(..., description="JSON string with claim details"),
    
    # Identity documents (common for all)
    pan_file: Optional[UploadFile] = File(None, description="PAN card image"),
    aadhaar_file: Optional[UploadFile] = File(None, description="Aadhaar card image"),
    
    # Claim-specific documents
    vehicle_file: Optional[UploadFile] = File(None, description="Vehicle damage photo (motor claims)"),
    damage_photo: Optional[UploadFile] = File(None, description="Additional damage photo"),
    
    # Supporting documents (claim-type dependent)
    supporting_file_1: Optional[UploadFile] = File(None, description="Hospital bill, license, invoice, etc."),
    supporting_file_2: Optional[UploadFile] = File(None, description="Death certificate, medical reports, etc."),
):
    """
    🤖 Fully automatic claim analysis with intelligent type detection.
    
    **No manual claim type selection needed!**
    
    The system automatically:
    - Detects claim type from narrative keywords and uploaded files
    - Selects relevant verification modules
    - Runs complete fraud analysis
    - Returns unified risk assessment
    """
    
    # ═══════════════════════════════════════
    # STEP 1: PARSE & VALIDATE INPUT
    # ═══════════════════════════════════════
    try:
        claim_info = json.loads(claim_data)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in claim_data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid claim_data JSON: {str(e)}"
        )
    
    # Extract narrative and amount for detection
    narrative = claim_info.get('narrative', '')
    claim_amount = claim_info.get('claim_amount', 0)
    
    if not narrative:
        logger.warning("⚠️ No narrative provided, detection may be less accurate")
    
    # ═══════════════════════════════════════
    # STEP 2: AUTO-DETECT CLAIM TYPE
    # ═══════════════════════════════════════
    files_info = {
        'vehicle_file': vehicle_file.filename if vehicle_file else None,
        'damage_photo': damage_photo.filename if damage_photo else None,
        'supporting_file_1': supporting_file_1.filename if supporting_file_1 else None,
        'supporting_file_2': supporting_file_2.filename if supporting_file_2 else None,
    }
    
    claim_type, detection_scores = detect_claim_type(
        narrative=narrative,
        claim_amount=claim_amount,
        files_uploaded=files_info
    )
    
    logger.success(f"✅ Auto-detected claim type: {claim_type.upper()}")
    
    # Update claim_info with detected type
    claim_info['product_type'] = claim_type
    
    # ═══════════════════════════════════════
    # STEP 3: INITIALIZE RESULTS
    # ═══════════════════════════════════════
    BASE_URL = "http://localhost:8000"
    results = {
        "claim_type": claim_type,
        "claim_type_auto_detected": True,
        "detection_scores": detection_scores,
        "ml_score": None,
        "documents": {},
        "graph_score": None,
        "final_risk": 0,
        "modules_executed": [],
        "errors": []
    }
    
    doc_risks = []
    
    # ═══════════════════════════════════════
    # STEP 4: EXECUTE ANALYSIS MODULES
    # ═══════════════════════════════════════
    async with httpx.AsyncClient(timeout=90.0) as client:
        
        # ───────────────────────────────────
        # MODULE 1: ML FRAUD DETECTION (Always)
        # ───────────────────────────────────
        try:
            logger.info("🧠 Running ML fraud detection...")
            response = await client.post(
                f"{BASE_URL}/api/ml/score/detailed",
                json=claim_info,
                timeout=30.0
            )
            if response.status_code == 200:
                results["ml_score"] = response.json()
                results["modules_executed"].append("ml_model")
                logger.success(f"✅ ML scoring complete: {results['ml_score'].get('fraud_probability', 0):.2%}")
            else:
                raise Exception(f"ML API returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ ML scoring failed: {e}")
            results["ml_score"] = {"error": str(e), "fraud_probability": 0.5}
            results["errors"].append(f"ML Engine: {str(e)}")
        
        # ───────────────────────────────────
        # MODULE 2: PAN VERIFICATION (Always)
        # ───────────────────────────────────
        if pan_file:
            try:
                logger.info("🆔 Verifying PAN card...")
                pan_bytes = await pan_file.read()
                files = {"file": (pan_file.filename, pan_bytes, pan_file.content_type)}
                response = await client.post(
                    f"{BASE_URL}/api/documents/verify-pan",
                    files=files,
                    timeout=30.0
                )
                if response.status_code == 200:
                    results["documents"]["pan"] = response.json()
                    doc_risks.append(results["documents"]["pan"].get("risk_score", 0) * 100)
                    results["modules_executed"].append("pan_verification")
                    logger.success("✅ PAN verification complete")
                else:
                    raise Exception(f"PAN API returned {response.status_code}")
            except Exception as e:
                logger.error(f"❌ PAN verification failed: {e}")
                results["documents"]["pan"] = {"error": str(e)}
                results["errors"].append(f"PAN Verification: {str(e)}")
        
        # ───────────────────────────────────
        # MODULE 3: AADHAAR VERIFICATION (Always)
        # ───────────────────────────────────
        if aadhaar_file:
            try:
                logger.info("🆔 Verifying Aadhaar card...")
                aadhaar_bytes = await aadhaar_file.read()
                files = {"file": (aadhaar_file.filename, aadhaar_bytes, aadhaar_file.content_type)}
                response = await client.post(
                    f"{BASE_URL}/api/documents/verify-aadhaar",
                    files=files,
                    timeout=30.0
                )
                if response.status_code == 200:
                    results["documents"]["aadhaar"] = response.json()
                    doc_risks.append(results["documents"]["aadhaar"].get("risk_score", 0) * 100)
                    results["modules_executed"].append("aadhaar_verification")
                    logger.success("✅ Aadhaar verification complete")
                else:
                    raise Exception(f"Aadhaar API returned {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Aadhaar verification failed: {e}")
                results["documents"]["aadhaar"] = {"error": str(e)}
                results["errors"].append(f"Aadhaar Verification: {str(e)}")
        
        # ═══════════════════════════════════════
        # CLAIM-TYPE SPECIFIC MODULES
        # ═══════════════════════════════════════
        
        if claim_type == "motor":
            # 🚗 MOTOR: Vehicle damage detection
            if vehicle_file or damage_photo:
                try:
                    logger.info("🚗 Running vehicle damage detection...")
                    image = vehicle_file or damage_photo
                    image_bytes = await image.read()
                    files = {"file": (image.filename, image_bytes, image.content_type)}
                    response = await client.post(
                        f"{BASE_URL}/api/cv/detect",
                        files=files,
                        timeout=60.0
                    )
                    if response.status_code == 200:
                        damage_data = response.json()
                        results["documents"]["vehicle_damage"] = damage_data
                        risk = damage_data.get("risk_assessment", {}).get("risk_score", 0) * 100
                        doc_risks.append(risk)
                        results["modules_executed"].append("vehicle_damage_detection")
                        logger.success(f"✅ Vehicle damage detection complete: {risk:.1f}% risk")
                    else:
                        raise Exception(f"CV API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Vehicle damage detection failed: {e}")
                    results["documents"]["vehicle_damage"] = {"error": str(e)}
                    results["errors"].append(f"Vehicle Damage Detection: {str(e)}")
            
            # Driving license verification
            if supporting_file_1:
                try:
                    logger.info("🪪 Verifying driving license...")
                    sup1_bytes = await supporting_file_1.read()
                    files = {"file": (supporting_file_1.filename, sup1_bytes, supporting_file_1.content_type)}
                    data = {"document_type": "license"}
                    response = await client.post(
                        f"{BASE_URL}/api/documents/verify-document",
                        files=files,
                        data=data,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        results["documents"]["driving_license"] = response.json()
                        doc_risks.append(results["documents"]["driving_license"].get("risk_score", 0) * 100)
                        results["modules_executed"].append("license_verification")
                        logger.success("✅ License verification complete")
                    else:
                        raise Exception(f"Document API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ License verification failed: {e}")
                    results["documents"]["driving_license"] = {"error": str(e)}
                    results["errors"].append(f"License Verification: {str(e)}")
        
        elif claim_type in ["health", "life"]:
            # 🏥 HEALTH/LIFE: Hospital bill verification
            if supporting_file_1:
                try:
                    logger.info("🏥 Verifying hospital bill...")
                    sup1_bytes = await supporting_file_1.read()
                    files = {"file": (supporting_file_1.filename, sup1_bytes, supporting_file_1.content_type)}
                    data = {"document_type": "hospital_bill"}
                    response = await client.post(
                        f"{BASE_URL}/api/documents/verify-document",
                        files=files,
                        data=data,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        results["documents"]["hospital_bill"] = response.json()
                        doc_risks.append(results["documents"]["hospital_bill"].get("risk_score", 0) * 100)
                        results["modules_executed"].append("hospital_bill_verification")
                        logger.success("✅ Hospital bill verification complete")
                    else:
                        raise Exception(f"Document API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Hospital bill verification failed: {e}")
                    results["documents"]["hospital_bill"] = {"error": str(e)}
                    results["errors"].append(f"Hospital Bill Verification: {str(e)}")
            
            # ⚰️ LIFE: Death certificate verification
            if claim_type == "life" and supporting_file_2:
                try:
                    logger.info("📜 Verifying death certificate...")
                    sup2_bytes = await supporting_file_2.read()
                    files = {"file": (supporting_file_2.filename, sup2_bytes, supporting_file_2.content_type)}
                    data = {"document_type": "death_certificate"}
                    response = await client.post(
                        f"{BASE_URL}/api/documents/verify-document",
                        files=files,
                        data=data,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        results["documents"]["death_certificate"] = response.json()
                        doc_risks.append(results["documents"]["death_certificate"].get("risk_score", 0) * 100)
                        results["modules_executed"].append("death_cert_verification")
                        logger.success("✅ Death certificate verification complete")
                    else:
                        raise Exception(f"Document API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Death certificate verification failed: {e}")
                    results["documents"]["death_certificate"] = {"error": str(e)}
                    results["errors"].append(f"Death Certificate Verification: {str(e)}")
        
        elif claim_type == "property":
            # 🏠 PROPERTY: Damage assessment
            if damage_photo:
                try:
                    logger.info("🏚️ Analyzing property damage...")
                    damage_bytes = await damage_photo.read()
                    files = {"file": (damage_photo.filename, damage_bytes, damage_photo.content_type)}
                    response = await client.post(
                        f"{BASE_URL}/api/cv/detect",
                        files=files,
                        timeout=60.0
                    )
                    if response.status_code == 200:
                        damage_data = response.json()
                        results["documents"]["property_damage"] = damage_data
                        risk = damage_data.get("risk_assessment", {}).get("risk_score", 0) * 100
                        doc_risks.append(risk)
                        results["modules_executed"].append("property_damage_assessment")
                        logger.success(f"✅ Property damage assessment complete: {risk:.1f}% risk")
                    else:
                        raise Exception(f"CV API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Property damage assessment failed: {e}")
                    results["documents"]["property_damage"] = {"error": str(e)}
                    results["errors"].append(f"Property Damage Assessment: {str(e)}")
            
            # Property documents (invoices, etc.)
            if supporting_file_1:
                try:
                    logger.info("🧾 Verifying property documents...")
                    sup1_bytes = await supporting_file_1.read()
                    files = {"file": (supporting_file_1.filename, sup1_bytes, supporting_file_1.content_type)}
                    data = {"document_type": "other"}
                    response = await client.post(
                        f"{BASE_URL}/api/documents/verify-document",
                        files=files,
                        data=data,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        results["documents"]["property_docs"] = response.json()
                        doc_risks.append(results["documents"]["property_docs"].get("risk_score", 0) * 100)
                        results["modules_executed"].append("property_doc_verification")
                        logger.success("✅ Property document verification complete")
                    else:
                        raise Exception(f"Document API returned {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Property document verification failed: {e}")
                    results["documents"]["property_docs"] = {"error": str(e)}
                    results["errors"].append(f"Property Doc Verification: {str(e)}")
        
        # ───────────────────────────────────
        # MODULE: GRAPH ANALYSIS (Always)
        # ───────────────────────────────────
        try:
            logger.info("🕸️ Running fraud network analysis...")
            response = await client.post(
                f"{BASE_URL}/api/fraud/score",
                json=claim_info,
                timeout=30.0
            )
            if response.status_code == 200:
                results["graph_score"] = response.json()
                results["modules_executed"].append("graph_analysis")
                logger.success(f"✅ Graph analysis complete: {results['graph_score'].get('risk_score', 0):.2%}")
            else:
                raise Exception(f"Graph API returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Graph analysis failed: {e}")
            results["graph_score"] = {"error": str(e), "risk_score": 0.3}
            results["errors"].append(f"Graph Analysis: {str(e)}")
    
    # ═══════════════════════════════════════
    # STEP 5: CALCULATE FINAL RISK
    # ═══════════════════════════════════════
    ml_risk = results["ml_score"].get("fraud_probability", 0.5) * 100
    avg_doc_risk = sum(doc_risks) / len(doc_risks) if doc_risks else 50.0
    graph_risk = results["graph_score"].get("risk_score", 0.3) * 100
    
    # Weighted average: 30% ML + 40% Docs + 30% Graph
    final_risk = (ml_risk * 0.3) + (avg_doc_risk * 0.4) + (graph_risk * 0.3)
    
    results["final_risk"] = round(final_risk, 2)
    results["avg_document_risk"] = round(avg_doc_risk, 2)
    results["risk_breakdown"] = {
        "ml_risk": round(ml_risk, 2),
        "ml_contribution": round(ml_risk * 0.3, 2),
        "doc_risk": round(avg_doc_risk, 2),
        "doc_contribution": round(avg_doc_risk * 0.4, 2),
        "graph_risk": round(graph_risk, 2),
        "graph_contribution": round(graph_risk * 0.3, 2)
    }
    
    # Risk recommendation
    if final_risk >= 60:
        results["recommendation"] = "🔴 HIGH RISK - Reject/Manual Review Required"
        results["action"] = "REJECT"
    elif final_risk >= 40:
        results["recommendation"] = "🟡 MEDIUM RISK - Additional Verification Needed"
        results["action"] = "REVIEW"
    elif final_risk >= 20:
        results["recommendation"] = "🟢 LOW RISK - Proceed with Caution"
        results["action"] = "APPROVE_WITH_CAUTION"
    else:
        results["recommendation"] = "✅ VERY LOW RISK - Likely Legitimate"
        results["action"] = "APPROVE"
    
    logger.success(
        f"✅ Analysis complete for {claim_type.upper()} claim | "
        f"Final risk: {final_risk:.2f}% | "
        f"Modules: {', '.join(results['modules_executed'])}"
    )
    
    return results


@router.get(
    "/health",
    summary="Health check for unified analysis endpoint"
)
async def health_check():
    """Check if unified analysis endpoint is operational."""
    return {
        "status": "healthy",
        "service": "Unified Analysis with Auto-Detection",
        "version": "2.2.0",
        "features": [
            "Auto claim type detection",
            "Smart module selection",
            "Multi-product support (motor/health/life/property)",
            "Hinglish keyword recognition",
            "30-40-30 risk weighting"
        ]
    }
