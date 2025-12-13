"""Quick diagnostic to find the feature mismatch issue.

Run this to see exactly what's wrong:
    python diagnose_features.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml_engine.feature_engineer import FeatureEngineer
from src.ml_engine.ml_scorer import MLFraudScorer

print("="*80)
print("🔍 FEATURE MISMATCH DIAGNOSTIC")
print("="*80)

# 1. Load model and check expected features
print("\n📦 Step 1: Loading trained model...")
model_path = "models/ml_engine/claimlens_catboost_hinglish.cbm"

if not Path(model_path).exists():
    print(f"❌ Model not found at: {model_path}")
    sys.exit(1)

model = CatBoostClassifier()
model.load_model(model_path)

expected_features = list(model.feature_names_)
print(f"✅ Model loaded: {len(expected_features)} features expected")
print(f"\n🔍 First 20 expected features:")
for i, feat in enumerate(expected_features[:20]):
    print(f"  {i+1:2d}. {feat}")

print(f"\n🔍 Last 10 expected features:")
for i, feat in enumerate(expected_features[-10:], start=len(expected_features)-10):
    print(f"  {i+1:2d}. {feat}")

# Check embedding format
emb_features = [f for f in expected_features if 'emb' in f.lower()]
print(f"\n🎯 Embedding features: {len(emb_features)} found")
if emb_features:
    print(f"   Sample: {emb_features[:5]}")
    print(f"   Format: {'WITH underscores (emb_X)' if '_' in emb_features[0] else 'WITHOUT underscores (embX)'}")

# 2. Create test claim and engineer features
print("\n📦 Step 2: Engineering features for test claim...")

test_claim = pd.DataFrame([{
    "claim_id": "TEST_001",
    "claimant_id": "CLM_TEST",
    "policy_id": "POL_TEST",
    "product": "motor",
    "city": "Mumbai",
    "subtype": "accident",
    "claim_amount": 250000.0,
    "days_since_policy_start": 45,
    "narrative": "Gadi ka accident hua tha. Bahut damage hai.",
    "documents_submitted": "FIR,photos,estimate",
    "incident_date": "2025-12-14"
}])

engineer = FeatureEngineer(
    pca_dims=100,
    model_name="AkshitaS/bhasha-embed-v0",
    expected_features=expected_features
)

features_df = engineer.engineer_features(test_claim, keep_ids=True)

print(f"✅ Features engineered: {features_df.shape[1]} columns")

# Remove ID columns for comparison
id_cols = ['claim_id', 'claimant_id', 'policy_id']
feature_cols = [c for c in features_df.columns if c not in id_cols]

print(f"\n🔍 First 20 generated features:")
for i, feat in enumerate(feature_cols[:20]):
    print(f"  {i+1:2d}. {feat}")

print(f"\n🔍 Last 10 generated features:")
for i, feat in enumerate(feature_cols[-10:], start=len(feature_cols)-10):
    print(f"  {i+1:2d}. {feat}")

# Check generated embeddings
generated_emb = [f for f in feature_cols if 'emb' in f.lower()]
print(f"\n🎯 Generated embedding features: {len(generated_emb)} found")
if generated_emb:
    print(f"   Sample: {generated_emb[:5]}")
    print(f"   Format: {'WITH underscores (emb_X)' if '_' in generated_emb[0] else 'WITHOUT underscores (embX)'}")

# 3. Compare features
print("\n📦 Step 3: Comparing expected vs generated...")

expected_set = set(expected_features)
generated_set = set(feature_cols)

missing = expected_set - generated_set
extra = generated_set - expected_set

print(f"\n✅ Matching features: {len(expected_set & generated_set)}/{len(expected_features)}")

if missing:
    print(f"\n⚠️  MISSING {len(missing)} features from model expectations:")
    print(f"   Sample (first 10): {list(missing)[:10]}")
    
    # Categorize missing features
    missing_emb = [f for f in missing if 'emb' in f.lower()]
    missing_cat = [f for f in missing if f.startswith(('product_', 'city_', 'subtype_'))]
    missing_num = [f for f in missing if f not in missing_emb and f not in missing_cat]
    
    print(f"   - Embeddings: {len(missing_emb)}")
    print(f"   - Categorical: {len(missing_cat)}")
    print(f"   - Numeric: {len(missing_num)}")

if extra:
    print(f"\n⚠️  EXTRA {len(extra)} features not in model:")
    print(f"   Sample (first 10): {list(extra)[:10]}")
    
    # Categorize extra features
    extra_emb = [f for f in extra if 'emb' in f.lower()]
    extra_cat = [f for f in extra if f.startswith(('product_', 'city_', 'subtype_'))]
    extra_num = [f for f in extra if f not in extra_emb and f not in extra_cat]
    
    print(f"   - Embeddings: {len(extra_emb)}")
    print(f"   - Categorical: {len(extra_cat)}")
    print(f"   - Numeric: {len(extra_num)}")

if not missing and not extra:
    print("\n🎉 PERFECT MATCH! All features align correctly!")
else:
    print("\n❌ MISMATCH DETECTED! Features don't align.")

# 4. Test prediction
print("\n📦 Step 4: Testing prediction...")

scorer = MLFraudScorer(model_path=model_path, threshold=0.5)
features_for_pred = features_df.drop(columns=id_cols, errors='ignore')

try:
    fraud_prob = scorer.predict_proba(features_for_pred)[0]
    print(f"✅ Prediction successful: {fraud_prob:.4f}")
    
    if 0.02 <= fraud_prob <= 0.04:
        print("⚠️  WARNING: Fraud probability is ~3% (flat prediction issue!)")
        print("   This suggests embeddings are still being replaced with zeros.")
    else:
        print("✅ Fraud probability looks varied (good!)")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# 5. Final diagnosis
print("\n" + "="*80)
print("🎯 DIAGNOSIS SUMMARY")
print("="*80)

if missing or extra:
    print("\n❌ ROOT CAUSE: Feature name mismatch")
    
    if missing_emb or extra_emb:
        print("\n🔥 CRITICAL: Embedding column format mismatch!")
        if missing_emb:
            print(f"   Model expects: {missing_emb[:3]}")
        if extra_emb:
            print(f"   Engineer creates: {extra_emb[:3]}")
        print("\n   FIX: Update _normalize_column_names() to preserve embedding format.")
    
    if missing_cat:
        print(f"\n⚠️  Missing categorical dummies: {len(missing_cat)}")
        print(f"   Sample: {list(missing_cat)[:5]}")
        print("   FIX: Check if training used different product/city/subtype values.")
    
    if missing_num:
        print(f"\n⚠️  Missing numeric features: {len(missing_num)}")
        print(f"   Features: {list(missing_num)}")
        print("   FIX: Check if feature engineering pipeline changed.")
else:
    print("\n✅ ALL GOOD! Features align perfectly.")
    print("   If still getting flat 3%, check:")
    print("   1. Model file is correct version")
    print("   2. PCA is fitted properly")
    print("   3. Embeddings have actual values (not zeros)")

print("\n" + "="*80)
print("Run complete!")
print("="*80)
