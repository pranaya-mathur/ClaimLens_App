"""
Test script for CV Detection Integration
Run this after placing model files to verify setup
"""
import sys
from pathlib import Path
import requests
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_files_exist():
    """Check if all required model files are in place"""
    print("\n=== Testing Model Files ===")
    
    required_files = [
        "models/parts_segmentation/yolo11n_best.pt",
        "models/damage_detection/yolo11m_best.pt",
        "models/severity_classification/efficientnet_b0_best.pth"
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_present = False
    
    return all_present


def test_detector_loading():
    """Test loading DamageDetector directly"""
    print("\n=== Testing DamageDetector Loading ===")
    
    try:
        from src.cv_engine.damage_detector import DamageDetector
        from config.settings import get_settings
        
        settings = get_settings()
        
        print("Loading models...")
        detector = DamageDetector(
            parts_model_path=settings.PARTS_MODEL_PATH,
            damage_model_path=settings.DAMAGE_MODEL_PATH,
            severity_model_path=settings.SEVERITY_MODEL_PATH,
            device="cpu"  # Use CPU for testing
        )
        
        print(f"✓ All models loaded successfully")
        print(f"  Device: {detector.device}")
        print(f"  Parts classes: {len(detector.PARTS_CLASSES)}")
        print(f"  Damage classes: {len(detector.DAMAGE_CLASSES)}")
        print(f"  Severity classes: {len(detector.SEVERITY_CLASSES)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return False


def test_api_health(base_url="http://localhost:8000"):
    """Test API health endpoint"""
    print("\n=== Testing API Health ===")
    
    try:
        response = requests.get(f"{base_url}/api/cv/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Health Check Passed")
            print(f"  Status: {data['status']}")
            print(f"  Models Loaded: {data['models_loaded']}")
            print(f"  Device: {data['device']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure server is running:")
        print("  uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_api_model_info(base_url="http://localhost:8000"):
    """Test API model info endpoint"""
    print("\n=== Testing API Model Info ===")
    
    try:
        response = requests.get(f"{base_url}/api/cv/info", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Model Info Retrieved")
            print(f"  Parts Model: {data['models']['parts_segmentation']['architecture']}")
            print(f"  Damage Model: {data['models']['damage_detection']['architecture']}")
            print(f"  Severity Model: {data['models']['severity_classification']['architecture']}")
            return True
        else:
            print(f"✗ Info request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_sample_detection(image_path: str, base_url="http://localhost:8000"):
    """Test damage detection with a sample image"""
    print("\n=== Testing Sample Detection ===")
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        print("  Skipping detection test")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(
                f"{base_url}/api/cv/detect",
                files=files,
                timeout=60
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Detection Successful")
            print(f"  Parts Detected: {len(data['parts_detected'])}")
            print(f"  Damages Detected: {len(data['damages_detected'])}")
            print(f"  Risk Level: {data['risk_assessment']['risk_level']}")
            
            if data['damages_detected']:
                print("\n  Damage Details:")
                for i, damage in enumerate(data['damages_detected'][:3], 1):
                    print(f"    {i}. {damage['damage_type']} - {damage['severity']} (conf: {damage['confidence']:.2f})")
            
            return True
        else:
            print(f"✗ Detection failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """
    Run all tests
    """
    print("""
╭──────────────────────────────────────╮
│   ClaimLens CV Integration Test   │
╰──────────────────────────────────────╯
    """)
    
    results = []
    
    # Test 1: Model files
    results.append(("Model Files", test_model_files_exist()))
    
    # Test 2: Direct loading
    if results[0][1]:  # Only if files exist
        results.append(("Model Loading", test_detector_loading()))
    
    # Test 3: API tests (if API is running)
    print("\n--- API Tests (requires running server) ---")
    results.append(("API Health", test_api_health()))
    results.append(("API Model Info", test_api_model_info()))
    
    # Test 4: Sample detection (optional)
    # Uncomment and provide image path to test
    # results.append(("Sample Detection", test_sample_detection("path/to/test/image.jpg")))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! CV integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")


if __name__ == "__main__":
    main()
