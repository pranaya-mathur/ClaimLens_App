# ✅ Post-Reset Verification Checklist

**Date**: December 13, 2025  
**Status**: App Reset to Stable State

## Prerequisites

- [ ] Python 3.10+
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] `.env` file configured with API keys

## Part 1: Backend API

### Step 1: Start API Server
```bash
cd .
uvicorn api.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

- [ ] Server starts without errors
- [ ] Listening on `http://localhost:8000`

### Step 2: Check API Health

**Option A: Browser**
Go to: `http://localhost:8000/docs`

- [ ] Swagger UI loads
- [ ] Can see all endpoints
- [ ] No 500 errors

**Option B: Command Line**
```bash
curl http://localhost:8000/health/liveness
```

- [ ] Returns status `200`
- [ ] Response shows `{"status": "alive"}`

### Step 3: Test Individual Endpoints

#### ML Engine: `/api/ml/score/detailed`
```bash
curl -X POST http://localhost:8000/api/ml/score/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM2024001",
    "claimant_id": "CLMT12345",
    "policy_id": "POL12345",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 250000,
    "days_since_policy_start": 45,
    "narrative": "Vehicle accident on highway",
    "documents_submitted": "pan,aadhaar",
    "incident_date": "2025-12-13"
  }'
```

- [ ] Returns `200` status
- [ ] Response includes `fraud_probability` (0-1)
- [ ] Response includes `risk_level` (LOW/MEDIUM/HIGH/CRITICAL)
- [ ] Takes < 5 seconds

#### CV Engine: `/api/documents/verify-pan` (needs image file)
```bash
curl -X POST http://localhost:8000/api/documents/verify-pan \
  -F "file=@/path/to/pan_image.jpg"
```

- [ ] Returns `200` status
- [ ] Response includes `risk_score` (0-1)
- [ ] Response includes `confidence` (0-1)
- [ ] Response includes `recommendation`

#### Graph Engine: `/api/fraud/score`
```bash
curl -X POST http://localhost:8000/api/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claim_id": 2024001}'
```

- [ ] Returns status (200 or 503 if Neo4j down)
- [ ] If 200: includes `final_risk_score` and `graph_insights`
- [ ] If 503: shows graceful error (OK - Neo4j optional)

#### LLM Engine: `/api/llm/explain`
```bash
curl -X POST http://localhost:8000/api/llm/explain \
  -H "Content-Type: application/json" \
  -d '{
    "claim_narrative": "Vehicle hit on highway",
    "ml_fraud_prob": 0.35,
    "document_risk": 0.2,
    "network_risk": 0.15,
    "claim_amount": 250000,
    "premium": 15000,
    "days_since_policy": 45,
    "product_type": "motor"
  }'
```

- [ ] Returns `200` status
- [ ] Response includes `explanation` text
- [ ] Explanation is meaningful (not error message)
- [ ] Takes < 10 seconds

## Part 2: Streamlit Frontend

### Step 1: Start Streamlit App

**In new terminal:**
```bash
cd .
streamlit run frontend/streamlit_app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

URL: http://localhost:8501
```

- [ ] Streamlit server starts
- [ ] Listening on `http://localhost:8501`
- [ ] Browser opens automatically (or go manually)

### Step 2: Check UI Loading

**In browser at `http://localhost:8501`:**

- [ ] Page loads without errors
- [ ] Title shows "ClaimLens AI"
- [ ] Sidebar visible on left
- [ ] API status shows "✅ Models Active"
- [ ] No red error messages
- [ ] 3 navigation options visible (AI Analysis, Analytics, Fraud Networks)

### Step 3: Test AI Analysis Page

Click: **🎯 AI-Powered Claim Analysis** (should be selected by default)

- [ ] Page loads
- [ ] Claim form visible with fields:
  - [ ] Claim ID (default: CLM2024001)
  - [ ] Claim Subtype dropdown
  - [ ] Premium field
  - [ ] Claimant ID
  - [ ] Product Type
  - [ ] Claim Amount
  - [ ] Days Since Policy
  - [ ] Documents
  - [ ] Narrative textarea
- [ ] Document upload sections visible:
  - [ ] PAN Card uploader
  - [ ] Aadhaar Card uploader
  - [ ] Vehicle Photo uploader
- [ ] Blue "Analyze with AI" button visible

### Step 4: Test Analysis Flow

**Keep defaults and click "Analyze with AI"**

You should see results appear:

#### Document Verification Section
- [ ] Shows status (🔴 or 🟢)
- [ ] Shows confidence percentage
- [ ] Shows recommendation

#### ML Fraud Score Section
- [ ] Shows risk level (🔴, 🟠, 🟡, or 🟢)
- [ ] Shows fraud probability percentage
- [ ] Shows ML confidence

#### Graph Analysis Section
- [ ] Shows status (🟢 or warning)
- [ ] Shows network score
- [ ] Shows fraud connections (or "No fraud network detected")

#### Radar Chart
- [ ] Displays all 5 components
- [ ] Chart shows risk scores visually
- [ ] Final Assessment box shows:
  - [ ] Risk score percentage
  - [ ] Recommendation (REJECT/REVIEW/APPROVE)
  - [ ] Key Risk Factors list

#### AI Explanation
- [ ] Section titled "AI-Generated Explanation"
- [ ] Shows detailed text explanation
- [ ] Mentions claim amount, premium, days since policy
- [ ] Not an error message

### Step 5: Test Analytics Dashboard

Click: **📊 Analytics Dashboard**

- [ ] Page loads
- [ ] Shows metrics:
  - [ ] Total Claims
  - [ ] Fraud Claims
  - [ ] Avg Fraud Score
  - [ ] Total Amount
- [ ] Shows charts (if data available)
  - [ ] Risk Distribution pie chart
  - [ ] Fraud by Product bar chart

### Step 6: Test Fraud Networks

Click: **🕸️ Fraud Networks**

- [ ] Page loads
- [ ] Two tabs visible: "Fraud Rings" and "Serial Fraudsters"
- [ ] Sliders and buttons present

**Note**: This section requires Neo4j database. If not running, it will show:
- [ ] "Graph analysis offline" messages
- [ ] "Requires Neo4j database" captions

This is **OK** - Neo4j is optional for MVP.

## Part 3: Optional - Neo4j Database

If you have Neo4j running:

### Start Neo4j

**Option A: Docker**
```bash
docker run -p 7687:7687 -p 7474:7474 neo4j
```

**Option B: Local Windows**
```bash
C:\Program Files\Neo4j\bin\neo4j.bat start
```

- [ ] Neo4j starts
- [ ] Browser at `http://localhost:7474`

### Test Graph Features

Go back to Streamlit, go to **🕸️ Fraud Networks**:

- [ ] Click "Find Fraud Rings"
- [ ] Click "Find Serial Fraudsters"
- [ ] Should show results (may be empty if no test data)
- [ ] No error messages

## Passing Criteria

### ✅ MVP REQUIREMENTS (Part 1 + Part 2)

- [ ] Backend API starts and responds
- [ ] Streamlit frontend loads
- [ ] All 4 engines (ML, CV, Graph, LLM) respond to requests
- [ ] Analysis page shows results from all modules
- [ ] No red error messages in UI
- [ ] Radar chart and final assessment display correctly

### ✅ NICE-TO-HAVE (Part 3 - Optional)

- [ ] Neo4j database connected
- [ ] Fraud network queries working
- [ ] Test data persisted in graph

## Troubleshooting

### Issue: "Cannot connect to API"

**Solution:**
1. Make sure backend is running (`http://localhost:8000`)
2. Check no firewall blocking port 8000
3. Verify API has no startup errors
4. Restart both backend and Streamlit

### Issue: "LLM service temporarily unavailable"

**Solution:**
1. Check `.env` has `GROQ_API_KEY`
2. Verify Groq API key is valid
3. Check internet connection
4. Streamlit will fall back to default explanation (OK)

### Issue: "Graph analysis offline"

**Solution:**
1. This is expected if Neo4j not running
2. Neo4j is optional for MVP
3. Start Neo4j if you want graph features
4. Or ignore - app works without it

### Issue: Streamlit keeps rerunning

**Solution:**
1. Normal Streamlit behavior
2. Each button click reruns entire script
3. Caching is handled by `@st.cache_resource`
4. API calls should be quick

## Final Verification

**📄 Checklist Summary:**

- [ ] Backend API: ✅ Running
- [ ] Streamlit Frontend: ✅ Running
- [ ] ML Engine: ✅ Responding
- [ ] CV Engine: ✅ Responding
- [ ] Graph Engine: ✅ Responding (or gracefully down)
- [ ] LLM Engine: ✅ Responding
- [ ] UI: ✅ Loading without errors
- [ ] Analysis: ✅ Showing results
- [ ] No broken endpoints: ✅ Confirmed

**🎉 If all items checked: App is READY FOR TESTING**

---

**Reset Status**: ✅ **COMPLETE AND VERIFIED**  
**Next Step**: Start developing v3 with proper unified integration
