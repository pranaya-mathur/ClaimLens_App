"""
Test script for Forgery Detection Module
Verifies ForgeryDetector can be imported and initialized
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import():
    """Test that forgery detector can be imported"""
    print("Test 1: Importing ForgeryDetector...")
    try:
        from src.cv_engine import ForgeryDetector
        print("✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_initialization():
    """Test that ForgeryDetector can be initialized (without model files)"""
    print("\nTest 2: Checking initialization requirements...")
    try:
        from src.cv_engine.forgery_detector import (
            DEFAULT_MODEL_PATH,
            DEFAULT_CONFIG_PATH,
        )

        print(f"Expected model path: {DEFAULT_MODEL_PATH}")
        print(f"Expected config path: {DEFAULT_CONFIG_PATH}")

        if DEFAULT_MODEL_PATH.exists():
            print("✓ Model file found")
        else:
            print("⚠ Model file not found (expected - needs to be added manually)")

        if DEFAULT_CONFIG_PATH.exists():
            print("✓ Config file found")
        else:
            print("⚠ Config file not found (expected - needs to be added manually)")

        print("\n📝 Note: Model and config files should be placed in models/ directory")
        print("   - models/forgery_detector_latest_run.pth")
        print("   - models/forgery_detector_latest_run_config.json")
        return True
    except Exception as e:
        print(f"✗ Initialization check failed: {e}")
        return False


def test_class_structure():
    """Test that ForgeryDetector has expected methods"""
    print("\nTest 3: Checking class structure...")
    try:
        from src.cv_engine import ForgeryDetector

        expected_methods = [
            "analyze_image",
            "analyze_batch",
            "analyze_image_as_dict",
            "analyze_batch_as_dicts",
        ]

        for method in expected_methods:
            if hasattr(ForgeryDetector, method):
                print(f"✓ Method '{method}' exists")
            else:
                print(f"✗ Method '{method}' missing")
                return False

        return True
    except Exception as e:
        print(f"✗ Structure check failed: {e}")
        return False


def test_utilities():
    """Test forgery utilities (ELA and noise)"""
    print("\nTest 4: Testing forgery utilities...")
    try:
        from src.cv_engine.forgery_utils import ForgeryUtils
        import numpy as np

        utils = ForgeryUtils()

        # Test noise variation on dummy image
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        noise_var = utils.compute_noise_variation(dummy_img)
        print(f"✓ Noise variation computed: {noise_var:.2f}")

        # Test ELA intensity score on dummy array
        dummy_ela = np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
        ela_score = utils.ela_intensity_score(dummy_ela)
        print(f"✓ ELA intensity score: {ela_score:.3f}")

        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("FORGERY DETECTION MODULE TEST SUITE")
    print("="*60)

    results = []
    results.append(("Import Test", test_import()))
    results.append(("Initialization Check", test_initialization()))
    results.append(("Class Structure", test_class_structure()))
    results.append(("Utilities Test", test_utilities()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! Forgery detection module is ready.")
        print("\n📝 Next steps:")
        print("   1. Add model files to models/ directory")
        print("   2. Test with real images")
        print("   3. Integrate with API endpoints")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
