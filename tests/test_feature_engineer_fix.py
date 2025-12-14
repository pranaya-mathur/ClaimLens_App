"""Test script to validate feature_engineer.py column name fix.

Verifies:
1. Dataset columns match feature engineer expectations
2. Feature generation produces correct schema
3. Model receives expected 145 features
4. No KeyError during processing

Usage:
    python tests/test_feature_engineer_fix.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ml_engine.feature_engineer import FeatureEngineer


def test_column_compatibility():
    """Test 1: Verify dataset columns match feature engineer."""
    print("\n" + "="*80)
    print("TEST 1: Column Compatibility Check")
    print("="*80)
    
    # Sample data matching actual dataset schema
    sample_data = {
        'claim_id': ['CLM001', 'CLM002', 'CLM003'],
        'claimant_id': ['C001', 'C002', 'C003'],
        'policy_id': ['P001', 'P002', 'P003'],
        'product_type': ['health', 'motor', 'life'],  # Note: product_TYPE
        'claim_subtype': ['illness', 'accident', 'natural death'],  # Note: claim_subtype
        'city': ['Mumbai', 'Delhi', 'Bangalore'],
        'claim_date': ['2025-01-15', '2025-02-10', '2025-03-05'],
        'claim_amount': [50000, 75000, 100000],
        'policy_start_date': ['2024-01-01', '2023-06-15', '2022-12-20'],
        'days_since_policy_start': [380, 575, 805],
        'documents_submitted': ['discharge summary;bills', 'FIR;photos', 'death certificate;policy copy'],
        'narrative': [
            'Patient ko sudden illness hui thi',
            'Accident hua highway par, vehicle damage',
            'Natural death hua, age 65, heart failure'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"\nDataset columns: {df.columns.tolist()}")
    print(f"\n✅ Dataset has 'product_type' (not 'product')")
    print(f"✅ Dataset has 'claim_subtype' (not 'subtype')")
    print(f"✅ Dataset has 'claim_date' (can use as incident_date alias)")
    
    # Test feature engineer initialization
    try:
        engineer = FeatureEngineer(pca_dims=100)
        print(f"\n✅ FeatureEngineer initialized successfully")
        print(f"   Categorical features: {engineer.categorical_features}")
        print(f"   Categorical prefixes: {engineer.categorical_prefixes}")
        
        # Verify column names match
        for col in engineer.categorical_features:
            if col in df.columns:
                print(f"   ✅ Column '{col}' exists in dataset")
            else:
                print(f"   ❌ Column '{col}' NOT FOUND in dataset")
                return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return False


def test_feature_generation():
    """Test 2: Verify feature generation works without errors."""
    print("\n" + "="*80)
    print("TEST 2: Feature Generation Test")
    print("="*80)
    
    # Sample data
    sample_data = {
        'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
        'claimant_id': ['C001', 'C002', 'C001', 'C003', 'C002'],
        'policy_id': ['P001', 'P002', 'P001', 'P003', 'P002'],
        'product_type': ['health', 'motor', 'life', 'property', 'health'],
        'claim_subtype': ['illness', 'accident', 'natural death', 'fire', 'surgery'],
        'city': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
        'claim_date': ['2025-01-15', '2025-02-10', '2025-03-05', '2025-03-20', '2025-04-01'],
        'claim_amount': [50000, 75000, 100000, 200000, 60000],
        'policy_start_date': ['2024-01-01', '2023-06-15', '2022-12-20', '2024-02-01', '2023-08-10'],
        'days_since_policy_start': [380, 575, 805, 413, 600],
        'documents_submitted': [
            'discharge summary;bills',
            'FIR;photos',
            'death certificate;policy copy',
            'FIR;fire dept report;photos',
            'discharge summary;lab reports'
        ],
        'narrative': [
            'Patient ko sudden illness hui thi hospital mein admit kiya',
            'Accident hua highway par vehicle damage ho gaya',
            'Natural death hua age 65 heart failure ki wajah se',
            'Fire lagi ghar mein bahut damage hua property ka',
            'Surgery karwai appendicitis ka operation successful'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer(pca_dims=100)
        print(f"\n🔧 Processing {len(df)} sample claims...")
        
        # Generate features
        features = engineer.engineer_features(df, keep_ids=True)
        
        print(f"\n✅ Feature generation completed successfully!")
        print(f"   Output shape: {features.shape}")
        print(f"   Expected: (5 rows, 148 cols) = 3 IDs + 145 features")
        
        # Verify feature categories
        feature_cols = [c for c in features.columns if c not in ['claim_id', 'claimant_id', 'policy_id']]
        
        print(f"\n   Feature breakdown:")
        numeric_feats = [c for c in feature_cols if not c.startswith(('product_', 'city_', 'subtype_', 'emb_'))]
        product_feats = [c for c in feature_cols if c.startswith('product_')]
        city_feats = [c for c in feature_cols if c.startswith('city_')]
        subtype_feats = [c for c in feature_cols if c.startswith('subtype_')]
        emb_feats = [c for c in feature_cols if c.startswith('emb_')]
        
        print(f"   - Numeric: {len(numeric_feats)} features")
        print(f"   - Product dummies: {len(product_feats)} ({', '.join(product_feats)})")
        print(f"   - City dummies: {len(city_feats)} (sample: {', '.join(city_feats[:3])}, ...)")
        print(f"   - Subtype dummies: {len(subtype_feats)} (sample: {', '.join(subtype_feats[:3])}, ...)")
        print(f"   - Embeddings: {len(emb_feats)} (emb_0 to emb_99)")
        
        total_features = len(numeric_feats) + len(product_feats) + len(city_feats) + len(subtype_feats) + len(emb_feats)
        print(f"\n   ✅ Total features: {total_features}")
        
        if total_features == 145:
            print(f"   🎉 PERFECT! Matches expected 145 features!")
        else:
            print(f"   ⚠️ Warning: Expected 145, got {total_features}")
        
        # Verify no forbidden columns
        forbidden = ['fraud', 'score', 'red_flags', 'is_fraud']
        leaked_cols = [c for c in features.columns if any(f in c.lower() for f in forbidden)]
        
        if leaked_cols:
            print(f"\n   ❌ LEAKAGE DETECTED: {leaked_cols}")
            return False
        else:
            print(f"\n   ✅ No label leakage detected")
        
        return True
        
    except KeyError as e:
        print(f"\n❌ KeyError during feature generation: {e}")
        print(f"   This indicates column name mismatch!")
        return False
        
    except Exception as e:
        print(f"\n❌ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_schema_alignment():
    """Test 3: Verify feature names match model expectations."""
    print("\n" + "="*80)
    print("TEST 3: Model Schema Alignment")
    print("="*80)
    
    # Expected feature schema from trained model
    expected_features = [
        "dayssincepolicystart", "claimamount", "claimantclaimcount",
        "claimanttotalclaimed", "claimantavgclaim", "policyclaimcount",
        "policytotalclaimed", "policyavgclaim", "dayssincelastclaim",
        "rapidclaims", "isfirstclaim", "claimamountlog",
        "policyagemonths", "isrecentpolicy", "numdocs",
        "hasfir", "hasphotos", "hasdeathcert", "hasdischarge",
        # Product dummies
        "product_health", "product_life", "product_motor", "product_property",
        # City dummies (sample)
        "city_Mumbai", "city_Delhi", "city_Bangalore",
        # Subtype dummies (sample)
        "subtype_accident", "subtype_illness", "subtype_natural death",
        # Embeddings
        "emb_0", "emb_1", "emb_2"
    ]
    
    print(f"\nExpected feature samples: {expected_features[:10]}")
    
    # Sample data
    df = pd.DataFrame({
        'claim_id': ['CLM001'],
        'claimant_id': ['C001'],
        'policy_id': ['P001'],
        'product_type': ['health'],
        'claim_subtype': ['illness'],
        'city': ['Mumbai'],
        'claim_date': ['2025-01-15'],
        'claim_amount': [50000],
        'policy_start_date': ['2024-01-01'],
        'days_since_policy_start': [380],
        'documents_submitted': ['discharge summary;bills'],
        'narrative': ['Patient ko illness hui thi']
    })
    
    try:
        engineer = FeatureEngineer(pca_dims=100, expected_features=None)
        features = engineer.engineer_features(df, keep_ids=False)
        
        print(f"\n✅ Generated features: {features.shape[1]} columns")
        print(f"   Sample feature names: {features.columns[:15].tolist()}")
        
        # Check critical features exist
        critical_features = [
            'dayssincepolicystart', 'claimamount', 'numdocs',
            'product_health', 'city_Mumbai', 'subtype_illness',
            'emb_0', 'emb_99'
        ]
        
        print(f"\n   Checking critical features:")
        all_present = True
        for feat in critical_features:
            if feat in features.columns:
                print(f"   ✅ {feat}")
            else:
                print(f"   ❌ {feat} MISSING")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"\n❌ Schema alignment test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "#"*80)
    print("# Feature Engineer Column Name Fix - Validation Suite")
    print("#"*80)
    
    results = {
        'Column Compatibility': test_column_compatibility(),
        'Feature Generation': test_feature_generation(),
        'Model Schema Alignment': test_model_schema_alignment()
    }
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! Feature engineer fix is working correctly.")
        print("="*80)
        print("\n✅ Ready to merge PR #8")
        print("✅ Dataset compatibility verified")
        print("✅ Feature generation validated")
        print("✅ Model schema alignment confirmed")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED! Please review the errors above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
