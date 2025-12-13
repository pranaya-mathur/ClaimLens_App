# 🚀 Unified Fraud Analysis Endpoint - Complete Guide

## Overview

ClaimLens v2.0 now features a **unified fraud analysis endpoint** that integrates ALL 4 fraud detection modules in a single API call:

1. **🤖 ML Engine** - Feature engineering + CatBoost fraud scoring
2. **📷 CV Engine** - Document verification (PAN/Aadhaar forgery detection)
3. **🕸️ Graph Engine** - Neo4j fraud network analysis
4. **🧠 LLM Engine** - Groq/Llama semantic aggregation + AI explanations

---

## 🎯 Endpoint Details

### **POST `/api/unified/analyze-complete`**

**Purpose:** Analyze NEW claims in real-time with all fraud detection modules, then store in Neo4j for future graph queries.

**Request Body:**
```json
{
  "claim_id": "CLM-2025-001",
  "claimant_id": "CLMT-12345",
  "policy_id": "POL-12345",
  "product": "motor",
  "city": "Mumbai",
  "subtype": "accident",
  "claim_amount": 250000.0,
  "days_since_policy_start": 45,
  "narrative": "Meri gaadi ko accident ho gaya tha highway pe.",
  "documents_submitted": "pan,aadhaar,rc,dl",
  "incident_date": "2025-12-13"
}
```

**Response:**
```json
{
  "claim_id": "CLM-2025-001",
  "final_verdict": "REVIEW",
  "final_confidence": 0.75,
  "fraud_probability": 0.45,
  "risk_level": "MEDIUM",
  
  "ml_engine": {
    "verdict": "MEDIUM_RISK",
    "confidence": 0.45,
    "score": 0.45,
    "reason": "ML fraud probability 45%",
    "red_flags": ["Risk level: MEDIUM"]
  },
  
  "cv_engine": null,
  
  "graph_engine": {
    "verdict": "NEW_CLAIMANT",
    "confidence": 0.85,
    "score": 0.0,
    "reason": "First claim from this claimant",
    "red_flags": []
  },
  
  "llm_aggregation": {
    "verdict": "REVIEW",
    "confidence": 0.75,
    "llm_used": true
  },
  
  "explanation": "This claim requires manual review based on multi-modal analysis. The ML model detected 45% fraud probability...",
  
  "reasoning_chain": [
    {
      "stage": "ml_fraud_scoring",
      "decision": "MEDIUM_RISK",
      "confidence": 0.45,
      "reason": "ML model processed 145 features"
    },
    {
      "stage": "graph_analysis",
      "decision": "NEW_CLAIMANT",
      "confidence": 0.85,
      "reason": "0 previous claims found"
    },
    {
      "stage": "llm_aggregation",
      "decision": "REVIEW",
      "confidence": 0.75,
      "reason": "LLM analyzed all component signals"
    },
    {
      "stage": "final_decision",
      "decision": "REVIEW",
      "confidence": 0.75,
      "reason": "Verdict based on 2 component analyses"
    }
  ],
  
  "critical_flags": [],
  
  "stored_in_database": true,
  "storage_timestamp": "2025-12-13T20:45:30.123456"
}
```

---

## 🔍 How It Works

### Step 1: Real-Time Analysis

When you submit a new claim, the endpoint:

1. **ML Scoring** (always runs):
   - Engineers 145+ features from claim data
   - Generates Hinglish narrative embeddings
   - Runs CatBoost model prediction
   - Returns fraud probability (0-1)

2. **Graph Analysis** (if Neo4j available):
   - Checks if claimant exists in database
   - Finds previous claim history
   - Detects fraud connections
   - Calculates network risk score

3. **CV Verification** (if documents uploaded):
   - PAN/Aadhaar forgery detection
   - OCR text extraction
   - Authenticity scoring
   - Red flag identification

4. **LLM Aggregation** (if Groq API key configured):
   - Combines all component signals
   - Generates semantic verdict
   - Creates human-readable explanation
   - Provides confidence score

### Step 2: Database Storage

After analysis, the claim is automatically stored in Neo4j:

```cypher
CREATE (c:Claim {
  claim_id: "CLM-2025-001",
  ml_fraud_score: 0.45,
  final_verdict: "REVIEW",
  claim_amount: 250000,
  incident_date: "2025-12-13"
})

MATCH (claimant:Claimant {claimant_id: "CLMT-12345"})
MERGE (claimant)-[:FILED]->(c)
```

This enables:
- Future fraud ring detection
- Serial fraudster identification
- Network-based risk scoring
- Pattern analysis

---

## 🧪 Testing the Endpoint

### Test 1: Basic Claim (No documents)

```bash
curl -X POST "http://localhost:8000/api/unified/analyze-complete" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "TEST-001",
    "claimant_id": "CLMT-TEST",
    "policy_id": "POL-TEST",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 100000,
    "days_since_policy_start": 30,
    "narrative": "Car accident on highway",
    "documents_submitted": "pan,aadhaar",
    "incident_date": "2025-12-13"
  }'
```

**Expected:** ML + Graph + LLM analysis, stored in Neo4j

### Test 2: High Fraud Claim

```bash
curl -X POST "http://localhost:8000/api/unified/analyze-complete" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "TEST-FRAUD-001",
    "claimant_id": "CLMT-FRAUD",
    "policy_id": "POL-FRAUD",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "theft",
    "claim_amount": 5000000,
    "days_since_policy_start": 2,
    "narrative": "Car stolen",
    "documents_submitted": "pan",
    "incident_date": "2025-12-13"
  }'
```

**Expected:** High fraud probability, REJECT verdict, critical flags

### Test 3: Repeat Claimant

Submit the same claim twice - the second time should detect previous history!

---

## ⚙️ Configuration

### Required Environment Variables

```bash
# Neo4j (for graph analysis + storage)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=claimlens123

# Groq LLM (for semantic aggregation)
GROQ_API_KEY=your_groq_api_key_here
LLM_ENABLED=true
LLM_EXPLANATION_ENABLED=true

# ML Model
ML_MODEL_PATH=models/catboost_fraud_v1.cbm
ML_METADATA_PATH=models/model_metadata.json
```

### Module Fallbacks

The endpoint gracefully handles missing modules:

- **No Neo4j**: Skips graph analysis, uses ML score only
- **No Groq API**: Uses rule-based verdict logic
- **No Documents**: Skips CV verification

**✅ All modules are OPTIONAL - ML Engine always works!**

---

## 📦 Database Persistence

### Why Store Claims?

Every analyzed claim is stored in Neo4j to enable:

1. **Fraud Ring Detection**: Find groups sharing documents
2. **Serial Fraudsters**: Identify repeat offenders
3. **Network Analysis**: Visualize fraud connections
4. **Pattern Recognition**: Detect emerging fraud trends
5. **Historical Context**: Future claims benefit from past data

### Example: Growing Graph

**Claim 1 (New claimant):**
```
Analysis: NEW_CLAIMANT, 0 connections
Stored: Claim node created
```

**Claim 2 (Same claimant):**
```
Analysis: REPEAT_CLAIMANT, 1 previous claim
Stored: Linked to existing claimant
```

**Claim 3 (Different claimant, same PAN):**
```
Analysis: FRAUD_RING_DETECTED, document sharing
Stored: Creates fraud connection
```

---

## 🔄 Migration from Old Endpoints

### Before (Multiple API Calls):
```python
# 1. ML scoring
ml_response = requests.post("/api/ml/score/detailed", json=claim_data)

# 2. Document verification
doc_response = requests.post("/api/documents/verify-pan", files=files)

# 3. Graph analysis
graph_response = requests.post("/api/fraud/score", json={"claim_id": claim_id})

# 4. Combine results manually
final_verdict = combine_results(ml_response, doc_response, graph_response)
```

### After (Single Unified Call):
```python
# ONE call does everything!
unified_response = requests.post(
    "/api/unified/analyze-complete",
    json=claim_data
)

# All results included
final_verdict = unified_response.json()["final_verdict"]
explanation = unified_response.json()["explanation"]
stored = unified_response.json()["stored_in_database"]
```

---

## 🛠️ Troubleshooting

### Error: "Neo4j connection failed"

**Solution:**
```bash
# Start Neo4j
docker-compose up -d neo4j

# Verify it's running
docker ps | grep neo4j

# Check logs
docker logs claimlens-neo4j
```

### Error: "LLM API call failed"

**Solution:**
1. Check Groq API key in `.env`
2. Verify `LLM_ENABLED=true`
3. Test API key: https://console.groq.com/
4. Check rate limits

### Error: "Feature alignment failed"

**Solution:**
- This should NOT happen with unified endpoint
- Feature alignment is automatic
- If it occurs, check ML model metadata

---

## 🚀 Next Steps

1. **Pull latest code:**
   ```bash
   git pull origin main
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Update Streamlit** (if needed):
   ```bash
   python scripts/update_streamlit_unified.py
   ```

4. **Restart API:**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Test endpoint:**
   ```bash
   curl -X GET http://localhost:8000/api/unified/health
   ```

6. **Launch Streamlit:**
   ```bash
   streamlit run frontend/streamlit_app_v2_sota.py
   ```

---

## ✅ Success Checklist

- [ ] Neo4j running (`docker ps`)
- [ ] Groq API key configured
- [ ] ML model loaded
- [ ] Unified endpoint returns 200
- [ ] Claims stored in Neo4j
- [ ] LLM explanations generated
- [ ] Streamlit displays all modules
- [ ] Graph grows with each claim

---

**🎉 Congratulations! You now have a complete production-ready fraud detection system with ALL modules integrated!**
