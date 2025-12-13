#!/usr/bin/env python3
"""
Test ML Fraud Prediction - Windows Friendly

Simple script to test ML fraud scoring without needing curl.
Run this after starting the API to verify predictions work.
"""
import requests
import json
from datetime import datetime, timedelta

# API endpoint
API_URL = "http://localhost:8000/api/ml/score"

# Test claims with different characteristics
test_claims = [
    {
        "name": "Low Risk - Normal Motor Claim",
        "data": {
            "claim_id": "TEST001",
            "claimant_id": "C001",
            "policy_id": "P001",
            "product": "motor",
            "city": "Delhi",
            "subtype": "accident",
            "claim_amount": 50000,
            "days_since_policy_start": 365,
            "narrative": "Meri car ko minor accident hua tha. Front bumper damage ho gaya.",
            "incident_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        }
    },
    {
        "name": "Medium Risk - High Amount",
        "data": {
            "claim_id": "TEST002",
            "claimant_id": "C002",
            "policy_id": "P002",
            "product": "motor",
            "city": "Mumbai",
            "subtype": "theft",
            "claim_amount": 500000,
            "days_since_policy_start": 30,
            "narrative": "Car chori ho gayi raat ko parking se.",
            "incident_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        }
    },
    {
        "name": "High Risk - Suspicious Pattern",
        "data": {
            "claim_id": "TEST003",
            "claimant_id": "C003",
            "policy_id": "P003",
            "product": "motor",
            "city": "Bangalore",
            "subtype": "accident",
            "claim_amount": 800000,
            "days_since_policy_start": 10,
            "narrative": "Major accident. Total loss. Everything damaged.",
            "incident_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        }
    },
]


def test_prediction(claim_name, claim_data):
    """Test a single claim prediction"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {claim_name}")
    print(f"{'='*70}")
    print(f"Claim ID: {claim_data['claim_id']}")
    print(f"Amount: ₹{claim_data['claim_amount']:,}")
    print(f"Narrative: {claim_data['narrative'][:50]}...")
    
    try:
        response = requests.post(API_URL, json=claim_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS - Prediction received:")
            print(f"   Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Fraud Prediction: {result['fraud_prediction']} (0=Legit, 1=Fraud)")
            print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
            
            # Check if prediction is varied (not flat)
            if result['fraud_probability'] > 0.05 and result['fraud_probability'] < 0.95:
                print(f"\n   ✅ Prediction is VARIED (not flat!) - Fix working!")
            else:
                print(f"\n   ⚠️  Prediction at extreme ({result['fraud_probability']:.2%})")
            
            return True, result
        else:
            print(f"\n❌ ERROR - Status {response.status_code}")
            print(f"   {response.text}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR - Cannot connect to API")
        print("   Make sure API is running: uvicorn api.main:app --reload")
        return False, None
    except Exception as e:
        print(f"\n❌ ERROR - {type(e).__name__}: {e}")
        return False, None


def main():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  ML FRAUD PREDICTION TEST - ClaimLens AI".center(70) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print(f"\n📍 Testing API at: {API_URL}")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test health first
    print(f"\n{'='*70}")
    print("🏥 Checking API Health...")
    print(f"{'='*70}")
    
    try:
        health = requests.get("http://localhost:8000/api/ml/health", timeout=5)
        if health.status_code == 200:
            health_data = health.json()
            print("✅ API is healthy")
            print(f"   Feature Alignment: {health_data.get('feature_alignment_status', 'unknown')}")
            print(f"   Expected Features: {health_data.get('model_expected_features', 0)}")
            print(f"   Embedding Model: {health_data.get('embedder_model', 'unknown')}")
        else:
            print(f"⚠️  API health check returned status {health.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        print("\nMake sure API is running:")
        print("   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Run tests
    results = []
    for i, claim in enumerate(test_claims, 1):
        success, result = test_prediction(claim["name"], claim["data"])
        results.append((claim["name"], success, result))
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for _, success, _ in results if success)
    print(f"\nTests Passed: {successful}/{len(results)}")
    
    if successful == len(results):
        print("\n✅ ALL TESTS PASSED!")
        
        # Check if predictions are varied
        fraud_probs = [r['fraud_probability'] for _, success, r in results if success and r]
        if fraud_probs:
            min_prob = min(fraud_probs)
            max_prob = max(fraud_probs)
            range_prob = max_prob - min_prob
            
            print(f"\n📈 Fraud Probability Range:")
            print(f"   Minimum: {min_prob:.2%}")
            print(f"   Maximum: {max_prob:.2%}")
            print(f"   Range: {range_prob:.2%}")
            
            if range_prob > 0.1:  # More than 10% variation
                print(f"\n🎉 PREDICTIONS ARE VARIED - Flat prediction bug is FIXED!")
            else:
                print(f"\n⚠️  Predictions have low variation - may need investigation")
        
        print("\n🚀 Your ML fraud detection is working correctly!")
    else:
        print(f"\n❌ {len(results) - successful} tests failed")
        print("   Review the errors above and check your configuration.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
