"""
ClaimLens Diagnostic Tool
Check API and database connectivity
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

print("🔍 ClaimLens Diagnostic Tool")
print("=" * 50)

# 1. Check API
print("\n1. Checking API Connection...")
try:
    r = requests.get(f"{API_URL}/", timeout=3)
    if r.status_code == 200:
        print("✅ API is running")
        data = r.json()
        print(f"   Version: {data.get('version')}")
    else:
        print(f"❌ API Error: {r.status_code}")
except Exception as e:
    print(f"❌ Cannot reach API: {str(e)}")
    print("   Make sure API is running: python -m uvicorn api.main:app --reload")
    exit(1)

# 2. Check Ingest Health
print("\n2. Checking Ingest Service (Neo4j)...")
try:
    r = requests.get(f"{API_URL}/api/ingest/health", timeout=3)
    if r.status_code == 200:
        result = r.json()
        if result.get('neo4j_connected'):
            print("✅ Neo4j is connected")
        else:
            print("❌ Neo4j is NOT connected")
            print("   Make sure Neo4j is running!")
            print("   Docker: docker run -d -p 7687:7687 neo4j")
            print("   Or: neo4j start")
    else:
        print(f"❌ Ingest service error: {r.status_code}")
except Exception as e:
    print(f"❌ Error checking ingest: {str(e)}")

# 3. Test Ingest Flow
print("\n3. Testing Ingest Flow...")
test_claim = {
    "claim_id": "DIAG_TEST_001",
    "claimant_id": "DIAG_CLT_001",
    "policy_id": "DIAG_POL_001",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 100000.0,
    "days_since_policy_start": 30,
    "narrative": "Test claim for diagnostics",
    "documents_submitted": "pan,aadhaar",
    "incident_date": "2024-12-13",
    "premium": 10000.0
}

try:
    r = requests.post(
        f"{API_URL}/api/ingest/claim",
        json=test_claim,
        timeout=30
    )
    
    if r.status_code in [201, 409]:
        print("✅ Ingest successful")
        result = r.json()
        print(f"   Nodes created: {result.get('nodes_created')}")
        print(f"   Relationships: {result.get('relationships_created')}")
    else:
        print(f"❌ Ingest failed: {r.status_code}")
        print(f"   Response: {r.text}")
        exit(1)

except Exception as e:
    print(f"❌ Ingest error: {str(e)}")
    exit(1)

# 4. Test Fraud Scoring
print("\n4. Testing Fraud Scoring...")
try:
    r = requests.post(
        f"{API_URL}/api/fraud/score",
        json={"claim_id": "DIAG_TEST_001"},
        timeout=30
    )
    
    if r.status_code == 200:
        result = r.json()
        print("✅ Fraud scoring works")
        print(f"   Risk Level: {result.get('risk_level')}")
        print(f"   Final Score: {result.get('final_risk_score')}")
    else:
        print(f"❌ Fraud scoring failed: {r.status_code}")
        print(f"   Response: {r.text}")
        if r.status_code == 404:
            print("\n   ⚠️ PROBLEM: Claim was ingested but not found in database!")
            print("   This means Neo4j is not persisting data properly.")
            print("   Check: Is Neo4j actually running?")

except Exception as e:
    print(f"❌ Scoring error: {str(e)}")

print("\n" + "=" * 50)
print("🎯 Diagnostic Complete")
print("\nIf Neo4j connection failed:")
print("  1. Start Neo4j: neo4j start (or Docker)")
print("  2. Check .env file for correct credentials")
print("  3. Run this diagnostic again")
