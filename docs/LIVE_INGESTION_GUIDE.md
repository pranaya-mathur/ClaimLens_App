# Live Claim Ingestion Guide

## Overview

ClaimLens now supports **real-time claim ingestion** directly into the Neo4j fraud graph. This eliminates the need for batch processing and enables immediate fraud detection on new claims.

## Features

✅ **Real-time Graph Updates** - Claims loaded instantly into Neo4j  
✅ **Duplicate Prevention** - MERGE operations prevent duplicate nodes  
✅ **Document Reuse Detection** - Automatically flags shared documents  
✅ **Fraud Statistics** - Auto-updates claimant fraud counts  
✅ **Immediate Scoring** - Ready for fraud analysis right after ingestion  
✅ **Transaction Safety** - Single atomic transaction per claim  

## Quick Start

### 1. Start Services

```bash
# Start Neo4j and API
docker-compose up -d

# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test Ingestion

```bash
# Run example test script
python examples/test_live_ingestion.py
```

### 3. API Documentation

Visit: `http://localhost:8000/docs`

## API Endpoints

### POST `/api/ingest/claim`

Ingest a new claim into the fraud graph.

**Request Body:**
```json
{
  "claim_id": "CLM_2025_123456",
  "claim_amount": 45000,
  "claim_date": "2025-12-10",
  "incident_date": "2025-12-05",
  "product_type": "motor",
  "subtype": "collision",
  "claimant": {
    "claimant_id": "CLMT_98765",
    "name": "Rahul Sharma",
    "phone": "+919876543210",
    "email": "rahul@example.com",
    "city": "Mumbai",
    "fraud_history": 0
  },
  "policy": {
    "policy_number": "POL_45678",
    "product_type": "motor",
    "sum_insured": 500000,
    "start_date": "2024-06-15"
  },
  "location": {
    "city_name": "Mumbai",
    "state": "Maharashtra"
  },
  "documents": [
    {
      "doc_id": "DOC_111",
      "doc_hash": "abc123def456",
      "doc_type": "invoice"
    }
  ]
}
```

**Response (201 Created):**
```json
{
  "status": "success",
  "claim_id": "CLM_2025_123456",
  "graph_status": "loaded",
  "nodes_created": 5,
  "relationships_created": 4,
  "timestamp": "2025-12-10T09:10:00Z",
  "message": "Claim successfully ingested. Ready for fraud scoring."
}
```

### GET `/api/ingest/status/{claim_id}`

Check if a claim exists in the graph.

**Response:**
```json
{
  "claim_id": "CLM_2025_123456",
  "exists": true,
  "status": "found"
}
```

### GET `/api/ingest/health`

Check ingestion service health.

**Response:**
```json
{
  "status": "healthy",
  "service": "claim_ingestion",
  "neo4j_connected": true,
  "message": "Ingestion service is operational"
}
```

## Integration Workflow

### Complete Claim Processing Flow

```python
import requests

API_BASE = "http://localhost:8000"

# Step 1: Ingest claim
response = requests.post(
    f"{API_BASE}/api/ingest/claim",
    json=claim_data
)

if response.status_code == 201:
    result = response.json()
    claim_id = result['claim_id']
    
    # Step 2: Get fraud score
    score_response = requests.post(
        f"{API_BASE}/api/fraud/score",
        json={"claim_id": claim_id}
    )
    
    if score_response.status_code == 200:
        fraud_data = score_response.json()
        risk_score = fraud_data['final_risk_score']
        recommendation = fraud_data['recommendation']
        
        # Step 3: Make decision
        if risk_score < 0.3:
            decision = "AUTO_APPROVE"
        elif risk_score < 0.7:
            decision = "MANUAL_REVIEW"
        else:
            decision = "REJECT"
        
        print(f"Decision: {decision}")
        print(f"Risk Score: {risk_score}")
        print(f"Recommendation: {recommendation}")
```

## Data Validation

All fields are validated using Pydantic:

- **claim_id**: Required, unique identifier
- **claim_amount**: Required, must be > 0
- **claim_date**: Required, cannot be in future
- **incident_date**: Required, must be before claim_date
- **claimant.phone**: Required phone number
- **policy.sum_insured**: Required, must be > 0

## Graph Operations

### Nodes Created/Updated

1. **Claimant** - MERGE by `claimant_id`
   - Updates `total_claims` counter
   - Updates `fraud_count` if fraud_history > 0
   - Sets `is_high_risk` flag

2. **Policy** - MERGE by `policy_number`
   - Updates `claim_count` counter
   - Updates `total_claimed` amount

3. **City** - MERGE by `city_name`
   - Updates `claim_count` counter

4. **Claim** - CREATE (always new)
   - Stores all claim details
   - Calculates `filing_delay_days`
   - Calculates `days_since_policy_start`

5. **Document** - MERGE by `doc_hash`
   - Detects document reuse across claims
   - Sets `is_suspicious` if reused
   - Tracks `usage_count`

### Relationships Created

- `(Claim)-[:FILED_BY]->(Claimant)`
- `(Claim)-[:ON_POLICY]->(Policy)`
- `(Claim)-[:IN_LOCATION]->(City)`
- `(Claim)-[:HAS_DOCUMENT]->(Document)`
- `(Claimant)-[:HAS_POLICY]->(Policy)`

## Error Handling

### 409 Conflict
Claim already exists in graph.

```json
{
  "detail": "Claim CLM_2025_123456 already exists. Use update endpoint to modify."
}
```

### 422 Validation Error
Invalid request data.

```json
{
  "detail": [
    {
      "loc": ["body", "claim_amount"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

### 500 Internal Server Error
Neo4j connection or query failure.

## Performance

- **Ingestion Time**: ~50-100ms per claim
- **Graph Update**: Single atomic transaction
- **Connection Pooling**: Reuses Neo4j driver connection
- **Concurrent Requests**: Supported via FastAPI async

## Monitoring

### Check Neo4j Browser

Visit: `http://localhost:7474`

**Query recent claims:**
```cypher
MATCH (c:Claim)
WHERE c.created_at > datetime() - duration('P1D')
RETURN c
ORDER BY c.created_at DESC
LIMIT 10
```

**View claim graph:**
```cypher
MATCH (c:Claim {claim_id: 'CLM_2025_123456'})-[r]->(n)
RETURN c, r, n
```

**Check document reuse:**
```cypher
MATCH (d:Document)
WHERE d.usage_count > 1
RETURN d.doc_hash, d.usage_count, d.is_suspicious
ORDER BY d.usage_count DESC
```

## Troubleshooting

### Neo4j Connection Failed

```bash
# Check Neo4j is running
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### Ingestion Timeout

- Check Neo4j memory settings in `docker-compose.yml`
- Increase heap size: `NEO4J_dbms_memory_heap_max__size=2G`
- Check for index creation on frequently queried properties

## Next Steps

After successful ingestion:

1. **Run Fraud Analysis**: Use `/api/fraud/score` endpoint
2. **Check Fraud Rings**: Use `/api/fraud/rings` endpoint
3. **View Analytics**: Use `/api/analytics/*` endpoints
4. **Streamlit Dashboard**: Visualize fraud patterns

## Production Considerations

- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Set up monitoring/alerting
- [ ] Configure CORS properly
- [ ] Enable HTTPS
- [ ] Add request logging
- [ ] Implement claim update endpoint
- [ ] Add batch ingestion endpoint

## Support

For issues or questions:
- Check API docs: `http://localhost:8000/docs`
- View logs: `docker-compose logs api`
- Test endpoint: `/api/ingest/health`
