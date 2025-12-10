#!/bin/bash

# ClaimLens Live Ingestion - cURL Examples
# Make this file executable: chmod +x examples/curl_examples.sh

API_BASE="http://localhost:8000"

echo "=========================================="
echo "ClaimLens API - cURL Test Examples"
echo "=========================================="

# 1. Health Check
echo -e "\n1. Testing Ingestion Service Health..."
curl -X GET "${API_BASE}/api/ingest/health" \
  -H "Content-Type: application/json" | jq

# 2. Ingest Normal Claim
echo -e "\n\n2. Ingesting Normal Claim..."
curl -X POST "${API_BASE}/api/ingest/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_CURL_001",
    "claim_amount": 35000,
    "claim_date": "2025-12-10",
    "incident_date": "2025-12-05",
    "product_type": "motor",
    "subtype": "collision",
    "claimant": {
      "claimant_id": "CLMT_CURL_001",
      "name": "Amit Patel",
      "phone": "+919876543210",
      "email": "amit@example.com",
      "city": "Bangalore",
      "fraud_history": 0
    },
    "policy": {
      "policy_number": "POL_CURL_001",
      "product_type": "motor",
      "sum_insured": 600000,
      "start_date": "2024-08-01"
    },
    "location": {
      "city_name": "Bangalore",
      "state": "Karnataka"
    },
    "documents": [
      {
        "doc_id": "DOC_CURL_001",
        "doc_hash": "curl123abc456",
        "doc_type": "invoice"
      }
    ]
  }' | jq

# 3. Check Claim Status
echo -e "\n\n3. Checking Claim Status..."
curl -X GET "${API_BASE}/api/ingest/status/CLM_CURL_001" \
  -H "Content-Type: application/json" | jq

# 4. Get Fraud Score
echo -e "\n\n4. Getting Fraud Score..."
curl -X POST "${API_BASE}/api/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_CURL_001"
  }' | jq

# 5. Ingest Suspicious Claim (with document reuse)
echo -e "\n\n5. Ingesting Suspicious Claim (Document Reuse)..."
curl -X POST "${API_BASE}/api/ingest/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_CURL_002",
    "claim_amount": 480000,
    "claim_date": "2025-12-10",
    "incident_date": "2025-11-20",
    "product_type": "motor",
    "subtype": "total_loss",
    "claimant": {
      "claimant_id": "CLMT_CURL_002",
      "name": "Fraud Suspect",
      "phone": "+919999999999",
      "city": "Delhi",
      "fraud_history": 3
    },
    "policy": {
      "policy_number": "POL_CURL_002",
      "product_type": "motor",
      "sum_insured": 500000,
      "start_date": "2025-11-15"
    },
    "location": {
      "city_name": "Delhi",
      "state": "Delhi"
    },
    "documents": [
      {
        "doc_id": "DOC_CURL_002",
        "doc_hash": "curl123abc456",
        "doc_type": "invoice"
      }
    ]
  }' | jq

# 6. Get Fraud Score for Suspicious Claim
echo -e "\n\n6. Getting Fraud Score for Suspicious Claim..."
curl -X POST "${API_BASE}/api/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_CURL_002"
  }' | jq

# 7. Test Duplicate Claim (should fail with 409)
echo -e "\n\n7. Testing Duplicate Claim Prevention..."
curl -X POST "${API_BASE}/api/ingest/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_CURL_001",
    "claim_amount": 1000,
    "claim_date": "2025-12-10",
    "incident_date": "2025-12-05",
    "product_type": "motor",
    "claimant": {
      "claimant_id": "CLMT_TEST",
      "name": "Test",
      "phone": "+911111111111",
      "city": "Mumbai"
    },
    "policy": {
      "policy_number": "POL_TEST",
      "product_type": "motor",
      "sum_insured": 100000,
      "start_date": "2024-01-01"
    },
    "location": {
      "city_name": "Mumbai",
      "state": "Maharashtra"
    },
    "documents": []
  }' | jq

echo -e "\n\n=========================================="
echo "Testing Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Check Neo4j browser: http://localhost:7474"
echo "2. View API docs: http://localhost:8000/docs"
echo "3. Query: MATCH (c:Claim)-[r]->(n) WHERE c.claim_id STARTS WITH 'CLM_CURL' RETURN c, r, n"
echo ""
