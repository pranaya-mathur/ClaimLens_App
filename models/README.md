# ClaimLens Model Files

All trained models for CV, document verification, and ML fraud detection.

## Directory Structure

```
models/
├── parts_segmentation/          # Car parts detection
│   └── yolo11n_best.pt
├── damage_detection/            # Damage detection  
│   └── yolo11m_best.pt
├── severity_classification/     # Damage severity
│   └── efficientnet_b0_best.pth
├── forgery_detection/           # Document forgery
│   ├── resnet50_finetuned.pth
│   └── aadhaar_balanced_model.pth
└── ml_engine/                   # ML fraud scoring
    ├── claimlens_catboost_hinglish.cbm
    ├── claimlens_model_metadata.json
    └── claimlens_feature_importance.csv
```

## Quick Setup

### Step 1: Create directories

```bash
cd ClaimLens_App
mkdir -p models/{parts_segmentation,damage_detection,severity_classification,forgery_detection,ml_engine}
```

### Step 2: Copy your 8 model files

```bash
# CV Models (rename .pth to .pt for YOLO models)
cp ~/Downloads/yolo11n_best.pth models/parts_segmentation/yolo11n_best.pt
cp ~/Downloads/yolo11m_best.pth models/damage_detection/yolo11m_best.pt
cp ~/Downloads/efficientnet_b0_best.pth models/severity_classification/

# Forgery Detection Models
cp ~/Downloads/resnet50_finetuned_after_strong_forgeries.pth models/forgery_detection/resnet50_finetuned.pth
cp ~/Downloads/aadhaar_balanced_model.pth models/forgery_detection/

# ML Engine Models  
cp ~/Downloads/claimlens_catboost_hinglish.cbm models/ml_engine/
cp ~/Downloads/claimlens_model_metadata.json models/ml_engine/
cp ~/Downloads/claimlens_feature_importance.csv models/ml_engine/
```

### Step 3: Verify

```bash
# Check all files are in place
find models -type f -name "*.pt" -o -name "*.pth" -o -name "*.cbm" | sort

# Expected output (8 files):
# models/damage_detection/yolo11m_best.pt
# models/forgery_detection/aadhaar_balanced_model.pth
# models/forgery_detection/resnet50_finetuned.pth
# models/ml_engine/claimlens_catboost_hinglish.cbm
# models/parts_segmentation/yolo11n_best.pt
# models/severity_classification/efficientnet_b0_best.pth
# (+ 2 CSV/JSON files)
```

## Model Sizes

| Model | Size | Used By |
|-------|------|----------|
| yolo11n_best.pt | ~6 MB | DamageDetector |
| yolo11m_best.pt | ~40 MB | DamageDetector |
| efficientnet_b0_best.pth | ~17 MB | DamageDetector |
| resnet50_finetuned.pth | ~100 MB | ForgeryDetector |
| aadhaar_balanced_model.pth | ~100 MB | AadhaarForgeryDetector |
| claimlens_catboost_hinglish.cbm | 50-100 MB | MLFraudScorer |
| claimlens_model_metadata.json | <1 KB | MLFraudScorer |
| claimlens_feature_importance.csv | ~10 KB | MLFraudScorer |
| **Total** | **~260-280 MB** | |

## Testing

```bash
# Start API
uvicorn api.main:app --reload

# Check all models loaded
curl http://localhost:8000/api/cv/health
curl http://localhost:8000/api/ml/health
curl http://localhost:8000/api/documents/health

# All should return {"status": "healthy", "models_loaded": true}
```

## Important Notes

- ❌ Model files are **NOT committed to Git** (too large)
- ✅ Each developer must copy files locally
- ✅ Files are in `.gitignore` to prevent accidental commits
- ✅ For team sharing, use Google Drive / Hugging Face / AWS S3

## Troubleshooting

### FileNotFoundError
Ensure filenames match exactly (check .pt vs .pth extension)

### Model loading fails
Verify PyTorch version: `pip install torch==2.0.0 torchvision`

### Out of memory
Models load lazily on first API call. Total RAM needed: ~1.5-2 GB

See individual README files in each subdirectory for more details.
