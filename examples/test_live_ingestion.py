"""
Example: Test Live Claim Ingestion

This script demonstrates how to use the live ingestion API endpoint.
Run this after starting the FastAPI server.
"""
import requests
import json
from datetime import date, timedelta


# API endpoint (adjust if needed)
API_BASE_URL = "http://localhost:8000"
INGEST_URL = f"{API_BASE_URL}/api/ingest/claim"
FRAUD_SCORE_URL = f"{API_BASE_URL}/api/fraud/score"


def create_sample_claim():
    """
    Create a sample claim payload for testing.
    """
    today = date.today()
    incident_date = today - timedelta(days=5)
    policy_start = today - timedelta(days=180)
    
    claim_payload = {
        "claim_id": "CLM_TEST_2025_001",
        "claim_amount": 45000,
        "claim_date": str(today),
        "incident_date": str(incident_date),
        "product_type": "motor",
        "subtype": "collision",
        "claimant": {
            "claimant_id": "CLMT_TEST_001",
            "name": "Rahul Kumar",
            "phone": "+919876543210",
            "email": "rahul.kumar@example.com",
            "city": "Mumbai",
            "fraud_history": 0
        },
        "policy": {
            "policy_number": "POL_TEST_001",
            "product_type": "motor",
            "sum_insured": 500000,
            "start_date": str(policy_start)
        },
        "location": {
            "city_name": "Mumbai",
            "state": "Maharashtra"
        },
        "documents": [
            {
                "doc_id": "DOC_TEST_001",
                "doc_hash": "abc123def456789",
                "doc_type": "invoice"
            },
            {
                "doc_id": "DOC_TEST_002",
                "doc_hash": "xyz789abc123456",
                "doc_type": "estimate"
            }
        ]
    }
    
    return claim_payload


def create_suspicious_claim():
    """
    Create a potentially fraudulent claim for testing fraud detection.
    """
    today = date.today()
    incident_date = today - timedelta(days=15)  # Long delay
    policy_start = today - timedelta(days=10)  # Recently issued
    
    claim_payload = {
        "claim_id": "CLM_TEST_2025_002",
        "claim_amount": 450000,  # High amount
        "claim_date": str(today),
        "incident_date": str(incident_date),
        "product_type": "motor",
        "subtype": "total_loss",
        "claimant": {
            "claimant_id": "CLMT_TEST_002",
            "name": "Suspicious Person",
            "phone": "+919999999999",
            "email": "suspicious@example.com",
            "city": "Delhi",
            "fraud_history": 2  # Past fraud
        },
        "policy": {
            "policy_number": "POL_TEST_002",
            "product_type": "motor",
            "sum_insured": 500000,
            "start_date": str(policy_start)
        },
        "location": {
            "city_name": "Delhi",
            "state": "Delhi"
        },
        "documents": [
            {
                "doc_id": "DOC_TEST_003",
                "doc_hash": "abc123def456789",  # Same hash as previous claim (reused!)
                "doc_type": "invoice"
            }
        ]
    }
    
    return claim_payload


def ingest_claim(claim_data):
    """
    Send claim to ingestion endpoint.
    """
    print(f"\n{'='*60}")
    print(f"Ingesting Claim: {claim_data['claim_id']}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            INGEST_URL,
            json=claim_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            result = response.json()
            print("\n✓ Claim ingested successfully!")
            print(f"  - Nodes created: {result['nodes_created']}")
            print(f"  - Relationships created: {result['relationships_created']}")
            print(f"  - Status: {result['graph_status']}")
            print(f"  - Message: {result['message']}")
            return True
        else:
            print(f"\n✗ Ingestion failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


def check_fraud_score(claim_id):
    """
    Get fraud score for ingested claim.
    """
    print(f"\n{'='*60}")
    print(f"Checking Fraud Score: {claim_id}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            FRAUD_SCORE_URL,
            json={"claim_id": claim_id}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Fraud Analysis Complete")
            print(f"  - Risk Score: {result.get('final_risk_score', 'N/A')}")
            print(f"  - Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"  - Red Flags: {result.get('red_flags', [])}")
            return result
        else:
            print(f"\n✗ Scoring failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return None


def main():
    """
    Main test workflow
    """
    print("\n" + "#"*60)
    print("#  ClaimLens Live Ingestion Test")
    print("#" + "#"*60)
    
    # Test 1: Normal claim
    print("\n\n[TEST 1] Ingesting Normal Claim...")
    normal_claim = create_sample_claim()
    if ingest_claim(normal_claim):
        check_fraud_score(normal_claim['claim_id'])
    
    # Test 2: Suspicious claim
    print("\n\n[TEST 2] Ingesting Suspicious Claim...")
    suspicious_claim = create_suspicious_claim()
    if ingest_claim(suspicious_claim):
        check_fraud_score(suspicious_claim['claim_id'])
    
    print("\n\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Check Neo4j browser at http://localhost:7474")
    print("2. Run Cypher query: MATCH (c:Claim)-[r]->(n) WHERE c.claim_id STARTS WITH 'CLM_TEST' RETURN c, r, n")
    print("3. View Streamlit dashboard for fraud analysis")
    print("\n")


if __name__ == "__main__":
    main()
