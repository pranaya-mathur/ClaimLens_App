#!/usr/bin/env python3
"""
Critical Fixes Verification Script

Verifies that all 3 critical configuration bugs have been fixed:
1. ML embedding model uses correct model (AkshitaS/bhasha-embed-v0)
2. CV model paths use correct file extensions (.pt for YOLO models)
3. .gitignore excludes .cbm files (CatBoost models)

Run this after pulling latest changes to verify your local setup.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings


def check_embedding_model():
    """Verify ML embedding model is correct"""
    print("\n" + "="*70)
    print("🔍 CHECK #1: ML Embedding Model")
    print("="*70)
    
    settings = get_settings()
    expected = "AkshitaS/bhasha-embed-v0"
    actual = settings.ML_EMBEDDING_MODEL
    
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    
    if actual == expected:
        print("✅ PASS: Embedding model is correct (768 dims)")
        return True
    else:
        print("❌ FAIL: Wrong embedding model!")
        print("   This will cause dimension mismatch and flat predictions.")
        print(f"   Update config/settings.py line 33 to: {expected}")
        return False


def check_cv_model_extensions():
    """Verify CV model file extensions are correct"""
    print("\n" + "="*70)
    print("🔍 CHECK #2: CV Model File Extensions")
    print("="*70)
    
    settings = get_settings()
    checks = [
        ("PARTS_MODEL_PATH", settings.PARTS_MODEL_PATH, ".pt"),
        ("DAMAGE_MODEL_PATH", settings.DAMAGE_MODEL_PATH, ".pt"),
    ]
    
    all_passed = True
    for name, path, expected_ext in checks:
        actual_ext = Path(path).suffix
        status = "✅ PASS" if actual_ext == expected_ext else "❌ FAIL"
        print(f"  {status} {name}: {path}")
        print(f"         Extension: {actual_ext} (expected: {expected_ext})")
        
        if actual_ext != expected_ext:
            all_passed = False
            print(f"         ⚠️ Update config/settings.py to use {expected_ext} extension")
    
    if all_passed:
        print("\n✅ PASS: All CV model extensions are correct")
    else:
        print("\n❌ FAIL: Some CV model extensions are wrong")
        print("   This will cause FileNotFoundError when loading models")
    
    return all_passed


def check_gitignore():
    """Verify .gitignore includes .cbm extension"""
    print("\n" + "="*70)
    print("🔍 CHECK #3: .gitignore includes .cbm")
    print("="*70)
    
    gitignore_path = PROJECT_ROOT / ".gitignore"
    
    if not gitignore_path.exists():
        print("❌ FAIL: .gitignore not found!")
        return False
    
    content = gitignore_path.read_text()
    
    # Check for .cbm pattern
    if "models/**/*.cbm" in content or "*.cbm" in content:
        print("✅ PASS: .gitignore excludes .cbm files")
        print("   CatBoost models won't be accidentally committed")
        return True
    else:
        print("❌ FAIL: .gitignore missing .cbm exclusion")
        print("   Add 'models/**/*.cbm' to .gitignore")
        print("   This prevents committing large CatBoost model files")
        return False


def check_model_files_exist():
    """Bonus check: Verify model files exist locally"""
    print("\n" + "="*70)
    print("💾 BONUS CHECK: Local Model Files")
    print("="*70)
    
    settings = get_settings()
    
    model_files = [
        ("ML CatBoost Model", settings.ML_MODEL_PATH),
        ("ML Metadata", settings.ML_METADATA_PATH),
        ("YOLO Parts Model", settings.PARTS_MODEL_PATH),
        ("YOLO Damage Model", settings.DAMAGE_MODEL_PATH),
        ("EfficientNet Model", settings.SEVERITY_MODEL_PATH),
    ]
    
    all_exist = True
    for name, path in model_files:
        full_path = PROJECT_ROOT / path
        exists = full_path.exists()
        status = "✅ EXISTS" if exists else "❌ MISSING"
        
        print(f"  {status} {name}")
        print(f"          {path}")
        
        if exists:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"          Size: {size_mb:.1f} MB")
        
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✅ All model files found locally!")
    else:
        print("\n⚠️  Some model files are missing")
        print("   Copy them from your local training directory")
        print("   See models/README.md for instructions")
    
    return all_exist


def main():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  CLAIMLENS CRITICAL FIXES VERIFICATION".center(70) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    results = [
        check_embedding_model(),
        check_cv_model_extensions(),
        check_gitignore(),
    ]
    
    # Bonus check (doesn't affect pass/fail)
    models_exist = check_model_files_exist()
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL SUMMARY")
    print("="*70)
    
    all_passed = all(results)
    
    if all_passed:
        print("✅ ALL CRITICAL FIXES VERIFIED!")
        print("\n🎉 Your configuration is correct!")
        print("\nNext steps:")
        print("  1. Start API: uvicorn api.main:app --reload")
        print("  2. Test health: curl http://localhost:8000/api/ml/health")
        print("  3. Test prediction: curl -X POST http://localhost:8000/api/ml/score ...")
        
        if not models_exist:
            print("\n⚠️  WARNING: Some model files are missing locally")
            print("   API will fail to start until models are copied")
        
        return 0
    else:
        print("❌ SOME FIXES FAILED!")
        print("\n⚠️  Your configuration has issues that need to be fixed.")
        print("   Review the failed checks above and update the files.")
        print("\n   After fixing, run this script again to verify.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
