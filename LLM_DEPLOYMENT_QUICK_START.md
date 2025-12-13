# ⚡ LLM Deployment - Quick Start Guide

**Time to Deploy:** 5 minutes  
**Status:** ✅ Ready to Go

---

## 🚀 5-Minute Deployment

### Step 1: Get API Key (2 min)
1. Go to https://console.groq.com/
2. Sign up (Free)
3. Create new API key
4. Copy the key (looks like: `gsk_xxxxx...`)

### Step 2: Update .env (1 min)
```bash
# Edit .env in repo root
GROQ_API_KEY=gsk_your_actual_key_here
ENABLE_SEMANTIC_AGGREGATION=true
ENABLE_LLM_EXPLANATIONS=true
```

### Step 3: Restart Backend (1 min)
```bash
pkill -f uvicorn || true
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
✅ GROQ_API_KEY configured
✅ LLM Model: llama-3.3-70b-versatile
✅ Semantic Aggregation: True
✅ LLM Explanations: True
```

### Step 4: Test in Streamlit (1 min)
1. Submit a health insurance claim
2. Check response for `llm_verdict` and `llm_explanation`
3. Done! 🎉

---

## 🧪 Quick Test

### Via cURL
```bash
curl -X POST http://localhost:8000/api/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claim_id": "CLM-2025-TEST"}'
```

### Expected Response
```json
{
  "claim_id": "CLM-2025-TEST",
  "base_fraud_score": 0.22,
  "final_risk_score": 0.22,
  "llm_verdict": "APPROVE",
  "llm_confidence": 0.82,
  "llm_explanation": "[AI-generated explanation from Groq Llama-3.3-70B]",
  "llm_used": true
}
```

---

## ✅ Verification Checklist

- [ ] API key added to .env
- [ ] Backend restarted
- [ ] Startup logs show "✅ GROQ_API_KEY configured"
- [ ] Can submit claim in Streamlit
- [ ] Response includes llm_verdict
- [ ] llm_explanation is 200+ words
- [ ] llm_used: true

---

## 🐛 Troubleshooting

### Problem: "llm_verdict is None"
- [ ] Check if GROQ_API_KEY is in .env
- [ ] Check if key is valid (not truncated)
- [ ] Check API logs for errors
- [ ] Restart backend

### Problem: "langchain-groq not found"
```bash
pip install langchain-groq
```

### Problem: "Connection refused"
- [ ] Check if API is running: `ps aux | grep uvicorn`
- [ ] Check if port 8000 is open: `lsof -i :8000`
- [ ] Restart API

---

## 📊 What's New

### Files Added/Updated
1. ✅ `src/llm_engine/semantic_aggregator.py` - Verdict synthesis
2. ✅ `src/llm_engine/explanation_generator.py` - AI explanations
3. ✅ `config/settings.py` - LLM configuration
4. ✅ `api/routes/fraud.py` - Backend wiring
5. ✅ `requirements.txt` - Added langchain-groq

### New Response Fields
```json
"llm_verdict": "APPROVE|REVIEW|REJECT",
"llm_confidence": 0.82,
"llm_explanation": "Human-readable explanation",
"llm_used": true
```

---

## 🎯 Features

✅ Intelligent verdict synthesis using Llama-3.3-70B  
✅ AI-generated human-friendly explanations  
✅ Streaming support for real-time display  
✅ Graceful fallback when LLM unavailable  
✅ Configurable feature flags  
✅ Comprehensive error handling  
✅ Full logging for debugging  

---

## 📖 Full Documentation

See `LLM_INTEGRATION_REPORT.md` for:
- Detailed implementation info
- Architecture diagrams
- Error handling strategies
- Performance metrics
- Advanced configuration

---

**Ready to deploy? Follow the 4 steps above and you're done!** 🚀
