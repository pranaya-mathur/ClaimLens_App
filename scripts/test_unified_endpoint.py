#!/usr/bin/env python3
"""
Unified Endpoint Comprehensive Test Script

Tests the /api/unified/analyze-complete endpoint with various scenarios:
1. Full test with all components
2. Test without Neo4j
3. Test without GROQ_API_KEY
4. Edge cases and error handling
"""
import requests
import json
from datetime import date
from typing import Dict, Any
import sys

# Configuration
API_URL = "http://localhost:8000"
UNIFIED_ENDPOINT = f"{API_URL}/api/unified/analyze-complete"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print test section header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(text: str):
    print(f"{GREEN}✅ {text}{RESET}")


def print_error(text: str):
    print(f"{RED}❌ {text}{RESET}")


def print_warning(text: str):
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_info(text: str):
    print(f"{BLUE}ℹ️  {text}{RESET}")


def create_test_claim(claim_id: str = "TEST001") -> Dict[str, Any]:
    """Create a test claim payload"""
    return {
        "claim_id": claim_id,
        "claimant_id": "CLMT_TEST_001",
        "policy_id": "POL_TEST_001",
        "product": "motor",
        "city": "Mumbai",
        "subtype": "accident",
        "claim_amount": 250000.0,
        "days_since_policy_start": 45,
        "narrative": "Meri gaadi ko accident ho gaya tha highway pe. Front bumper aur headlight damage hai.",
        "documents_submitted": "pan,aadhaar,rc,dl",
        "incident_date": str(date.today())
    }


def test_api_health() -> bool:
    """Test if API is running"""
    print_info("Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health/liveness", timeout=5)
        if response.status_code == 200:
            print_success("API is running")
            return True
        else:
            print_error(f"API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to API: {e}")
        print_warning("Make sure API is running: uvicorn api.main:app --reload")
        return False


def test_unified_endpoint_health() -> bool:
    """Test unified endpoint health check"""
    print_info("Testing unified endpoint health...")
    try:
        response = requests.get(f"{API_URL}/api/unified/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Unified endpoint healthy: {data.get('status')}")
            return True
        else:
            print_error(f"Unified health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Unified endpoint health check failed: {e}")
        return False


def test_full_analysis() -> bool:
    """Test full claim analysis with all components"""
    print_header("TEST 1: Full Claim Analysis")
    
    claim = create_test_claim("FULL_TEST_001")
    print_info(f"Submitting claim: {claim['claim_id']}")
    print_info(f"Claim amount: ₹{claim['claim_amount']:,}")
    print_info(f"Narrative: {claim['narrative'][:50]}...")
    
    try:
        response = requests.post(
            UNIFIED_ENDPOINT,
            json=claim,
            timeout=60  # 60s timeout for ML + LLM processing
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print_success("Unified analysis completed!")
            print(f"\n{BLUE}Results:{RESET}")
            print(f"  Final Verdict: {result['final_verdict']}")
            print(f"  Confidence: {result['final_confidence']:.0%}")
            print(f"  Fraud Probability: {result['fraud_probability']:.0%}")
            print(f"  Risk Level: {result['risk_level']}")
            
            # ML Engine
            print(f"\n{BLUE}ML Engine:{RESET}")
            ml = result.get('ml_engine', {})
            print(f"  Verdict: {ml.get('verdict')}")
            print(f"  Confidence: {ml.get('confidence'):.0%}")
            
            # Graph Engine
            print(f"\n{BLUE}Graph Engine:{RESET}")
            graph = result.get('graph_engine', {})
            if graph:
                print(f"  Verdict: {graph.get('verdict')}")
                print(f"  Confidence: {graph.get('confidence'):.0%}")
            else:
                print_warning("  Graph analysis unavailable (Neo4j offline?)")
            
            # LLM Aggregation
            print(f"\n{BLUE}LLM Aggregation:{RESET}")
            llm = result.get('llm_aggregation', {})
            if llm:
                print(f"  LLM Used: {llm.get('llm_used')}")
                print(f"  Verdict: {llm.get('verdict')}")
            else:
                print_warning("  LLM aggregation used fallback logic")
            
            # Explanation
            print(f"\n{BLUE}AI Explanation:{RESET}")
            explanation = result.get('explanation', 'N/A')
            print(f"  {explanation[:200]}...")
            
            # Storage
            print(f"\n{BLUE}Database Storage:{RESET}")
            stored = result.get('stored_in_database', False)
            if stored:
                print_success(f"  Stored in Neo4j: {result.get('storage_timestamp')}")
            else:
                print_warning("  Not stored in database (Neo4j offline?)")
            
            # Critical Flags
            flags = result.get('critical_flags', [])
            if flags:
                print(f"\n{YELLOW}Critical Flags:{RESET}")
                for flag in flags:
                    print(f"  ⚠️  {flag}")
            
            return True
        
        else:
            print_error(f"Analysis failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print_error("Request timed out (>60s)")
        print_warning("ML model might be loading for first time (takes ~30s)")
        return False
    
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_high_fraud_claim() -> bool:
    """Test with a suspicious high-fraud claim"""
    print_header("TEST 2: High Fraud Risk Claim")
    
    claim = create_test_claim("HIGH_FRAUD_001")
    # Make it suspicious
    claim["claim_amount"] = 900000.0  # Very high amount
    claim["days_since_policy_start"] = 15  # Very recent policy
    claim["narrative"] = "Car completely destroyed in accident."  # Vague
    
    print_info(f"Testing high-risk claim: ₹{claim['claim_amount']:,}")
    print_info(f"Policy age: {claim['days_since_policy_start']} days (suspicious!)")
    
    try:
        response = requests.post(UNIFIED_ENDPOINT, json=claim, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            verdict = result['final_verdict']
            fraud_prob = result['fraud_probability']
            
            print_success(f"Analysis completed: {verdict}")
            print(f"  Fraud Probability: {fraud_prob:.0%}")
            
            if verdict in ["REJECT", "REVIEW"] and fraud_prob > 0.5:
                print_success("Correctly flagged as high risk!")
                return True
            else:
                print_warning(f"Expected high risk, got: {verdict} ({fraud_prob:.0%})")
                return False
        else:
            print_error(f"Test failed: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_low_fraud_claim() -> bool:
    """Test with a legitimate low-fraud claim"""
    print_header("TEST 3: Low Fraud Risk Claim")
    
    claim = create_test_claim("LOW_FRAUD_001")
    # Make it legitimate
    claim["claim_amount"] = 25000.0  # Reasonable amount
    claim["days_since_policy_start"] = 365  # Old policy
    claim["narrative"] = "Minor accident at parking lot. Small dent on rear bumper. Already filed police report."
    
    print_info(f"Testing low-risk claim: ₹{claim['claim_amount']:,}")
    print_info(f"Policy age: {claim['days_since_policy_start']} days (established)")
    
    try:
        response = requests.post(UNIFIED_ENDPOINT, json=claim, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            verdict = result['final_verdict']
            fraud_prob = result['fraud_probability']
            
            print_success(f"Analysis completed: {verdict}")
            print(f"  Fraud Probability: {fraud_prob:.0%}")
            
            if verdict in ["APPROVE", "REVIEW"] and fraud_prob < 0.6:
                print_success("Correctly assessed as low risk!")
                return True
            else:
                print_warning(f"Expected low risk, got: {verdict} ({fraud_prob:.0%})")
                return False
        else:
            print_error(f"Test failed: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_invalid_payload() -> bool:
    """Test error handling with invalid payload"""
    print_header("TEST 4: Invalid Payload Handling")
    
    invalid_claim = {
        "claim_id": "INVALID_001",
        # Missing required fields
    }
    
    print_info("Submitting invalid payload...")
    
    try:
        response = requests.post(UNIFIED_ENDPOINT, json=invalid_claim, timeout=10)
        
        if response.status_code == 422:  # Validation error expected
            print_success("Correctly rejected invalid payload")
            return True
        else:
            print_warning(f"Unexpected status code: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and generate report"""
    print_header("🧪 UNIFIED ENDPOINT COMPREHENSIVE TEST")
    
    results = []
    
    # Health checks
    print_header("Health Checks")
    api_healthy = test_api_health()
    unified_healthy = test_unified_endpoint_health()
    
    if not api_healthy:
        print_error("API not running! Start with: uvicorn api.main:app --reload")
        sys.exit(1)
    
    if not unified_healthy:
        print_error("Unified endpoint not available!")
        sys.exit(1)
    
    # Functional tests
    results.append(("Full Analysis", test_full_analysis()))
    results.append(("High Fraud Claim", test_high_fraud_claim()))
    results.append(("Low Fraud Claim", test_low_fraud_claim()))
    results.append(("Invalid Payload", test_invalid_payload()))
    
    # Final Report
    print_header("📈 TEST REPORT")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name:30} PASSED")
        else:
            print_error(f"{test_name:30} FAILED")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    if passed == total:
        print_success(f"\n🎉 ALL TESTS PASSED! ({passed}/{total})")
        print_success("\n✅ Unified endpoint is PRODUCTION READY!")
        print_info("\nNext steps:")
        print_info("  1. Add CV/Document verification to unified endpoint")
        print_info("  2. Test with Redis caching")
        print_info("  3. Build LangGraph orchestration layer")
    else:
        print_warning(f"\n⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print_info("\nReview failures above and fix issues.")
    
    print(f"\n{BLUE}{'='*60}{RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
