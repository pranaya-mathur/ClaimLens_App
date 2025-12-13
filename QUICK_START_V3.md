# 🚀 ClaimLens v3.0 - Quick Start Guide

## What's New? ✨

✅ **Unified Endpoint** → Single API call processes ALL 4 engines
✅ **ML + CV + Graph + LLM** → All results in one response
✅ **Auto Storage** → Claims automatically saved to Neo4j
✅ **LLM Explanations** → Groq Llama-3.3-70B generates explanations
✅ **Reasoning Chain** → Transparent decision-making process

---

## 🔥 30-Second Setup

### Step 1: Start the API Server
```bash
# Terminal 1 - Start FastAPI
python -m uvicorn api.main:app --reload
# API runs on http://localhost:8000
```

### Step 2: Start Streamlit Frontend
```bash
# Terminal 2 - Start Streamlit
streamlit run frontend/streamlit_app_unified.py
# Opens at http://localhost:8501
```

### Step 3: Run Tests (Optional)
```bash
# Terminal 3 - Run test suite
python tests/test_unified_endpoint.py
```

---

## 📋 What to Expect

### Streamlit UI (3 Pages)

#### 🎯 Page 1: Claim Analysis
- Fill in claim details
- Click "RUN UNIFIED ANALYSIS"
- Get:
  - ✅ Final Verdict (APPROVE/REVIEW/REJECT)
  - ✅ Fraud Probability %
  - ✅ ML Engine results
  - ✅ Graph analysis (fraud rings)
  - ✅ LLM explanation from Groq
  - ✅ Reasoning chain
  - ✅ Database storage confirmation

#### 📊 Page 2: Test Multiple Claims
- Pre-configured test claims
  - 🟢 LOW RISK (straightforward)
  - 🟡 MEDIUM RISK (early claim + high amount)
  - 🔴 HIGH RISK (very early + theft)
- Click to test and see all 4 engines in action

#### 📈 Page 3: Analytics
- Overall statistics
- Fraud rate trends
- Claims summary

---

## 🔬 Test the Unified Endpoint

### Option A: Using Python Requests (Quick)
```python
import requests
from datetime import date

API_URL = "http://localhost:8000"

claim = {
    "claim_id": "CLM-001",
    "claimant_id": "CLMT-001",
    "policy_id": "POL-001",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 250000,
    "days_since_policy_start": 45,
    "narrative": "Minor accident on highway. Documents verified.",
    "documents_submitted": "pan,aadhaar,rc",
    "incident_date": str(date.today())
}

response = requests.post(
    f"{API_URL}/api/unified/analyze-complete",
    json=claim
)

result = response.json()
print(f"Verdict: {result['final_verdict']}")
print(f"Fraud Probability: {result['fraud_probability']*100:.1f}%")
print(f"Stored: {result['stored_in_database']}")
```

### Option B: Using Test Script (Comprehensive)
```bash
python tests/test_unified_endpoint.py
```

This runs:
- ✅ API health check
- ✅ Unified endpoint health
- ✅ LOW RISK claim test
- ✅ MEDIUM RISK claim test
- ✅ HIGH RISK claim test
- ✅ Fraud ring detection test
- ✅ Summary report

---

## 🎯 Expected Response Structure

```json
{
  "claim_id": "CLM-001",
  "final_verdict": "APPROVE",
  "final_confidence": 0.85,
  "fraud_probability": 0.25,
  "risk_level": "LOW",
  
  "ml_engine": {
    "verdict": "LOW",
    "confidence": 0.25,
    "score": 0.25,
    "reason": "...",
    "red_flags": []
  },
  
  "graph_engine": {
    "verdict": "NEW_CLAIMANT",
    "confidence": 0.85,
    "score": 0,
    "reason": "First claim from this claimant",
    "red_flags": []
  },
  
  "llm_aggregation": {
    "verdict": "APPROVE",
    "confidence": 0.85,
    "llm_used": true
  },
  
  "explanation": "AI-generated explanation from Groq Llama...",
  
  "reasoning_chain": [
    {
      "stage": "ml_fraud_scoring",
      "decision": "LOW",
      "confidence": 0.25,
      "reason": "..."
    },
    {
      "stage": "graph_analysis",
      "decision": "NEW_CLAIMANT",
      "confidence": 0.85,
      "reason": "..."
    },
    {
      "stage": "llm_aggregation",
      "decision": "APPROVE",
      "confidence": 0.85,
      "reason": "LLM analyzed all component signals"
    }
  ],
  
  "critical_flags": [],
  "stored_in_database": true,
  "storage_timestamp": "2025-12-13T15:50:00Z"
}
```

---

## 🚨 Troubleshooting

### ❌ "Cannot connect to API"
```bash
# Make sure API is running
python -m uvicorn api.main:app --reload
```

### ❌ "Neo4j not available"
- Graph engine will still work (shows "NEW_CLAIMANT" by default)
- Claims won't be stored in database
- Other engines continue normally

### ❌ "LLM not available"
- Check GROQ_API_KEY in .env
- Explanation will use fallback logic
- Verdict still generated from ML + Graph

### ❌ "Timeout after 60s"
- API taking too long
- Check if all dependencies loaded
- Try restarting server

---

## 📊 Architecture Overview

```
Streamlit UI (frontend/streamlit_app_unified.py)
        ↓
    User Input (Claim Details)
        ↓
POST /api/unified/analyze-complete
        ↓
┌───────────────────────────────────┐
│   Unified Analysis Engine         │
├───────────────────────────────────┤
│ 🤖 ML Engine                      │
│    ├─ Feature Engineering         │
│    ├─ CatBoost Scoring            │
│    └─ Risk Level Calculation      │
│                                   │
│ 🕸️ Graph Engine                   │
│    ├─ Query Claimant History      │
│    ├─ Detect Fraud Rings          │
│    └─ Serial Fraudster Check      │
│                                   │
│ 🧠 LLM Engine                     │
│    ├─ Semantic Aggregation        │
│    ├─ Groq Llama Call             │
│    └─ Explanation Generation      │
│                                   │
│ 💾 Storage Layer                  │
│    └─ Neo4j Persistence           │
└───────────────────────────────────┘
        ↓
JSON Response (All Results)
        ↓
Streamlit Display
    - Verdict Card
    - Component Results
    - Reasoning Chain
    - LLM Explanation
```

---

## 🎓 Sample Claims to Try

### Low Risk ✅
```json
{
  "claim_id": "CLM-LOW-001",
  "claimant_id": "CLMT-LOW",
  "policy_id": "POL-LOW",
  "product": "motor",
  "city": "Delhi",
  "subtype": "accident",
  "claim_amount": 50000,
  "days_since_policy_start": 365,
  "narrative": "Small accident after 1 year. All documents verified.",
  "documents_submitted": "pan,aadhaar,rc,dl",
  "incident_date": "2025-12-13"
}
```
**Expected:** APPROVE (365 days, low amount, good docs)

### Medium Risk ⚠️
```json
{
  "claim_id": "CLM-MED-001",
  "claimant_id": "CLMT-MED",
  "policy_id": "POL-MED",
  "product": "health",
  "city": "Mumbai",
  "subtype": "medical",
  "claim_amount": 500000,
  "days_since_policy_start": 30,
  "narrative": "Hospitalization claim. Early filing. Moderate amount.",
  "documents_submitted": "pan,discharge,bills",
  "incident_date": "2025-12-13"
}
```
**Expected:** REVIEW (early claim + high amount = needs verification)

### High Risk 🚩
```json
{
  "claim_id": "CLM-HIGH-001",
  "claimant_id": "CLMT-HIGH",
  "policy_id": "POL-HIGH",
  "product": "motor",
  "city": "Bangalore",
  "subtype": "theft",
  "claim_amount": 2000000,
  "days_since_policy_start": 10,
  "narrative": "Vehicle theft 10 days after policy. Very early. High amount.",
  "documents_submitted": "pan,aadhaar",
  "incident_date": "2025-12-13"
}
```
**Expected:** REJECT (very early + theft + high amount = red flags)

---

## 🎯 For Interviews/Demos

### Quick Demo Flow (5 minutes)
1. Start API + Streamlit
2. Show Streamlit UI
3. Go to "Test Multiple Claims" page
4. Test HIGH RISK claim
5. Show:
   - Final Verdict (🔴 REJECT)
   - ML Score (high fraud probability)
   - Graph Analysis (no network)
   - LLM Explanation (Groq generated)
   - Reasoning Chain (transparent)
   - Database Storage (stored confirmation)

### What Impresses Interviewers
✅ **Single unified endpoint** (efficiency)
✅ **All 4 engines in one call** (system integration)
✅ **LLM explanations** (AI transparency)
✅ **Reasoning chain** (interpretability)
✅ **Auto database storage** (data persistence)
✅ **Production-ready code** (quality)

---

## 📞 API Endpoints

### Health Checks
```
GET /health/liveness          → Is API running?
GET /api/unified/health       → Are all modules ready?
```

### Main Unified Endpoint
```
POST /api/unified/analyze-complete
  → Input: Claim details
  → Output: Complete analysis (ML + Graph + LLM)
```

### Individual Components (if needed)
```
POST /api/ml/score/detailed              → ML scoring only
POST /api/fraud/score                    → Graph analysis only
GET  /api/analytics/overview             → Summary stats
```

---

## 📦 What's Included

✅ **frontend/streamlit_app_unified.py** - Complete Streamlit UI
✅ **tests/test_unified_endpoint.py** - Comprehensive test suite
✅ **api/routes/unified_fraud.py** - Unified endpoint (already exists)
✅ **All 4 Engines** - ML, CV, Graph, LLM (fully integrated)
✅ **Neo4j Storage** - Auto-persist claims to database
✅ **Documentation** - This guide

---

## 🎬 Next Steps

1. **Run it:** Start API + Streamlit
2. **Test it:** Use test script or Streamlit UI
3. **Demo it:** Show to interviews/stakeholders
4. **Deploy it:** Docker support already in repo

---

## 💡 Pro Tips

- **For Quick Testing:** Use "Test Multiple Claims" page (pre-configured)
- **For Production:** API already handles multiple concurrent requests
- **For Monitoring:** Check `/api/unified/health` for module status
- **For Debugging:** Full reasoning chain shows decision logic

---

## 🎊 You're All Set!

**Everything is ready to go. Just:**
1. Start API
2. Start Streamlit
3. Click "RUN UNIFIED ANALYSIS"
4. Watch magic happen! ✨

---

**Built with ❤️ | Production Ready | Ready for Interviews**
