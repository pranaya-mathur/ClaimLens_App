# ClaimLens API Testing Guide

Complete guide for testing the Computer Vision endpoints including damage detection and forgery detection.

## Prerequisites

1. **API Server Running:**
   ```bash
   cd ClaimLens_App
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Model Files in Place:**
   - Damage detection models in `models/` directory
   - Forgery detection models: `forgery_detector_latest_run.pth` and `forgery_detector_latest_run_config.json`

3. **Test Image:**
   - Prepare a car damage image (JPG/PNG)

---

## API Endpoints Overview

| Endpoint | Method | Purpose |
|----------|--------|----------|
| `/api/cv/detect` | POST | Damage detection only |
| `/api/cv/detect-forgery` | POST | Forgery detection only |
| `/api/cv/analyze-complete` | POST | Complete analysis (damage + forgery) |
| `/api/cv/health` | GET | Check service health |
| `/api/cv/info` | GET | Model information |

---

## Testing with cURL

### 1. Health Check
```bash
curl -X GET http://localhost:8000/api/cv/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "damage_detector": true,
    "forgery_detector": true
  },
  "device": "cpu",
  "message": "CV detection service is operational"
}
```

### 2. Model Info
```bash
curl -X GET http://localhost:8000/api/cv/info
```

### 3. Damage Detection Only
```bash
curl -X POST http://localhost:8000/api/cv/detect \
  -F "file=@/path/to/car_damage.jpg" \
  -F "parts_conf=0.25" \
  -F "damage_conf=0.25"
```

**Expected Response:**
```json
{
  "status": "success",
  "parts_detected": [
    {
      "class": "hood",
      "confidence": 0.92,
      "bbox": [120, 45, 380, 290]
    }
  ],
  "damages_detected": [
    {
      "damage_type": "dent",
      "confidence": 0.87,
      "bbox": [150, 80, 220, 140],
      "severity": "moderate",
      "severity_confidence": 0.91
    }
  ],
  "summary": {
    "total_damages": 1,
    "damage_types": {"dent": 1},
    "severity_distribution": {"moderate": 1}
  },
  "risk_assessment": {
    "risk_level": "MEDIUM",
    "risk_score": 0.45,
    "factors": ["1 moderate damage(s)"]
  }
}
```

### 4. Forgery Detection Only
```bash
curl -X POST http://localhost:8000/api/cv/detect-forgery \
  -F "file=@/path/to/car_damage.jpg"
```

**Expected Response:**
```json
{
  "status": "success",
  "image_path": "car_damage.jpg",
  "is_forged": false,
  "forgery_probability": 0.23,
  "threshold": 0.55,
  "ela_score": 0.34,
  "noise_variation": 42.5,
  "confidence_level": "high",
  "recommendation": "APPROVE - Image appears authentic"
}
```

### 5. Complete Analysis (Damage + Forgery)
```bash
curl -X POST http://localhost:8000/api/cv/analyze-complete \
  -F "file=@/path/to/car_damage.jpg" \
  -F "parts_conf=0.25" \
  -F "damage_conf=0.25"
```

**Expected Response:**
```json
{
  "status": "success",
  "forgery_analysis": {
    "image_path": "car_damage.jpg",
    "is_forged": false,
    "forgery_prob": 0.23,
    "threshold": 0.55,
    "ela_score": 0.34,
    "noise_variation": 42.5
  },
  "damage_analysis": {
    "parts_detected": [...],
    "damages_detected": [...],
    "summary": {...},
    "risk_assessment": {...}
  },
  "final_risk_assessment": {
    "final_risk_level": "LOW",
    "final_risk_score": 0.25,
    "risk_factors": [],
    "recommendation": "APPROVE - Low fraud risk"
  }
}
```

---

## Testing with Python

### Using `requests` Library

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# 1. Health check
response = requests.get(f"{BASE_URL}/api/cv/health")
print("Health:", response.json())

# 2. Test forgery detection
with open("car_damage.jpg", "rb") as f:
    files = {"file": ("car_damage.jpg", f, "image/jpeg")}
    response = requests.post(f"{BASE_URL}/api/cv/detect-forgery", files=files)
    print("\nForgery Detection:")
    print(response.json())

# 3. Test complete analysis
with open("car_damage.jpg", "rb") as f:
    files = {"file": ("car_damage.jpg", f, "image/jpeg")}
    data = {"parts_conf": 0.25, "damage_conf": 0.25}
    response = requests.post(
        f"{BASE_URL}/api/cv/analyze-complete", 
        files=files,
        data=data
    )
    print("\nComplete Analysis:")
    result = response.json()
    print(f"Forgery: {result['forgery_analysis']['is_forged']}")
    print(f"Risk Level: {result['final_risk_assessment']['final_risk_level']}")
    print(f"Recommendation: {result['final_risk_assessment']['recommendation']}")
```

---

## Testing with Postman

### Setup
1. Open Postman
2. Create new collection: "ClaimLens CV API"

### Request 1: Forgery Detection
- **Method:** POST
- **URL:** `http://localhost:8000/api/cv/detect-forgery`
- **Body:** 
  - Type: `form-data`
  - Key: `file` (File type)
  - Value: Select your image file
- **Send** and verify response

### Request 2: Complete Analysis
- **Method:** POST
- **URL:** `http://localhost:8000/api/cv/analyze-complete`
- **Body:** 
  - Type: `form-data`
  - `file`: (File) your image
  - `parts_conf`: (Text) 0.25
  - `damage_conf`: (Text) 0.25
- **Send** and verify response

---

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

### Swagger UI
1. Navigate to: `http://localhost:8000/docs`
2. Find "Computer Vision" section
3. Click on any endpoint to expand
4. Click "Try it out"
5. Upload image and adjust parameters
6. Click "Execute" to test

### ReDoc
Alternative documentation: `http://localhost:8000/redoc`

---

## Response Status Codes

| Code | Meaning | Scenario |
|------|---------|----------|
| 200 | Success | Detection completed successfully |
| 400 | Bad Request | Invalid file type or missing parameters |
| 413 | Payload Too Large | Image exceeds size limit (10MB) |
| 500 | Internal Server Error | Model loading failed or processing error |

---

## Common Issues & Solutions

### Issue 1: "Model file not found"
**Solution:** Ensure model files are in the correct paths:
```
models/
├── forgery_detector_latest_run.pth
└── forgery_detector_latest_run_config.json
```

### Issue 2: "Module 'src.cv_engine.forgery_detector' not found"
**Solution:** Run API from project root directory:
```bash
cd ClaimLens_App
uvicorn api.main:app --reload
```

### Issue 3: CUDA out of memory
**Solution:** Set `CV_DEVICE=cpu` in `.env` file

### Issue 4: Slow inference
**Solution:** 
- Use GPU if available: `CV_DEVICE=cuda`
- Reduce image size before upload
- Check if models are properly loaded (health endpoint)

---

## Performance Benchmarks

| Operation | CPU (Intel i5) | GPU (NVIDIA 1660) |
|-----------|----------------|-------------------|
| Forgery Detection | ~150ms | ~50ms |
| Damage Detection | ~800ms | ~200ms |
| Complete Analysis | ~1000ms | ~300ms |

---

## Next Steps

1. **Test with Multiple Images:** Try authentic vs forged images
2. **Batch Testing:** Create script to test multiple files
3. **Integration:** Connect to frontend/Streamlit
4. **Monitoring:** Add logging and performance tracking
5. **Production:** Deploy with proper error handling and retries

---

## Sample Test Images

Create a test dataset:
```
test_images/
├── authentic/
│   ├── car1.jpg
│   └── car2.jpg
└── forged/
    ├── edited1.jpg
    └── edited2.jpg
```

Run batch test:
```python
import os
import requests

for category in ["authentic", "forged"]:
    for img in os.listdir(f"test_images/{category}"):
        with open(f"test_images/{category}/{img}", "rb") as f:
            files = {"file": f}
            r = requests.post("http://localhost:8000/api/cv/detect-forgery", files=files)
            result = r.json()
            print(f"{category}/{img}: Forged={result['is_forged']}, Prob={result['forgery_probability']:.3f}")
```

---

**Happy Testing! 🚀**
