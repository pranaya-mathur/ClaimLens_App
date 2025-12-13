# Feature Alignment Fix - Quick Reference

## Problem
ML model was returning flat ~3% fraud predictions due to feature schema mismatch between training and live inference.

## Solution
Updated `api/routes/ml_engine.py` to pass expected feature names from MLFraudScorer to FeatureEngineer, enabling automatic feature alignment.

## Verify Fix is Working

### 1. Check Health Endpoint
```bash
curl http://localhost:8000/ml/health
```

Look for:
- `"feature_alignment_status": "enabled"`
- `"model_expected_features": 145` (or your feature count)
- `"feature_alignment_match": true`

### 2. Check Server Logs
On startup, you should see:
```
✅ MLFraudScorer loaded successfully with 145 expected features
🎯 Passing 145 expected features to FeatureEngineer for schema alignment
✅ FeatureEngineer loaded successfully with feature alignment enabled (145 features)
```

### 3. Test a Prediction
```bash
curl -X POST http://localhost:8000/ml/score \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "TEST_001",
    "claimant_id": "C12345",
    "policy_id": "P67890",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 250000,
    "days_since_policy_start": 45,
    "narrative": "Gadi ka accident hua tha. Heavy damage",
    "documents_submitted": "FIR,photos",
    "incident_date": "2025-12-01"
  }'
```

Fraud probabilities should now vary based on content, not stay flat at ~3%.

## What Changed

**File**: `api/routes/ml_engine.py`
- `get_feature_engineer()` now calls `get_ml_scorer()` first to get expected features
- Expected features passed to FeatureEngineer for schema alignment
- Enhanced logging to track feature alignment status
- Health check now verifies feature alignment is enabled

**How It Works**:
1. MLFraudScorer loads model and extracts feature names
2. FeatureEngineer receives these expected features
3. During prediction, `_align_features_with_model()` ensures:
   - Missing features added with zeros
   - Extra features removed
   - Columns reordered to match training
4. Model receives correctly aligned features

## Troubleshooting

**⚠️ If feature alignment shows "DISABLED":**
- Check model file exists at path in ML_MODEL_PATH env var
- Verify model was saved with feature names during training
- Restart server after code changes

**Still getting flat predictions?**
- Check logs for "Feature alignment complete: X features (Y added, Z removed)"
- Verify model path is correct
- Ensure PCA dimensions match training (default: 100)

---
**Commit**: [7a62ae5](https://github.com/pranaya-mathur/ClaimLens_App/commit/7a62ae51903ea36a3e1603c062bed53a290f48c1)
