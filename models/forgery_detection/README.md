# Forgery Detection Models

Document forgery detection models for PAN and Aadhaar cards.

## Required Files

```
models/forgery_detection/
├── resnet50_finetuned.pth           (~100 MB) - Generic forgery detection
└── aadhaar_balanced_model.pth       (~100 MB) - Aadhaar-specific detector
```

## Setup

```bash
# Rename and copy your trained models
cp ~/Downloads/resnet50_finetuned_after_strong_forgeries.pth \
   models/forgery_detection/resnet50_finetuned.pth

cp ~/Downloads/aadhaar_balanced_model.pth \
   models/forgery_detection/
```

## Models

### ResNet50 Forgery Detector
- Detects tampering in PAN cards and generic documents
- Uses ELA (Error Level Analysis) + CNN
- Threshold: 0.55

### Aadhaar Balanced Model
- Specialized for Aadhaar card forgery
- Trained on balanced authentic/forged dataset
- Higher accuracy on Aadhaar-specific patterns

## Verify

```bash
ls -lh models/forgery_detection/

# Test document verification
curl -X POST http://localhost:8000/api/documents/health
```

These files are excluded from git due to size limits.
