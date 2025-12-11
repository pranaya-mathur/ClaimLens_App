# ClaimLens CV Models

This directory contains the trained computer vision models for damage detection.

## 📦 Required Model Files

You need to place the following trained model files in their respective directories:

```
models/
├── parts_segmentation/
│   └── yolo11n_best.pt          (YOLO11n-seg, ~5-6 MB)
├── damage_detection/
│   └── yolo11m_best.pt          (YOLO11m, ~38-40 MB)
└── severity_classification/
    └── efficientnet_b0_best.pth (EfficientNet-B0, ~17 MB)
```

## 🔧 Setup Instructions

### 1. Create Directory Structure

```bash
mkdir -p models/parts_segmentation
mkdir -p models/damage_detection
mkdir -p models/severity_classification
```

### 2. Download Model Files from Google Drive

From your training runs on Google Colab:

**Model 1 - Parts Segmentation:**
```
Source: /content/drive/MyDrive/carparts-seg/yolo11/runs/carparts-seg-yolo11n/weights/best.pt
Destination: models/parts_segmentation/yolo11n_best.pt
```

**Model 3 - Damage Detection:**
```
Source: /content/drive/MyDrive/cardd-model3-yolo11m/runs/cardd-y11m-seg/weights/best.pt
Destination: models/damage_detection/yolo11m_best.pt
```

**Model 4 - Severity Classification:**
```
Source: /content/drive/MyDrive/severity_classification/checkpoints/best_model.pth
Destination: models/severity_classification/efficientnet_b0_best.pth
```

### 3. Verify Files

Check that all files are in place:

```bash
ls -lh models/parts_segmentation/
ls -lh models/damage_detection/
ls -lh models/severity_classification/
```

## 📊 Model Details

### Model 1: Parts Segmentation
- **Architecture**: YOLO11n-seg
- **Classes**: 23 car parts
- **Performance**: 0.70 mAP50 (mask)
- **Input Size**: 640x640
- **Purpose**: Detect and segment car parts

### Model 3: Damage Detection
- **Architecture**: YOLO11m
- **Classes**: 6 damage types (dent, scratch, crack, glass-shatter, tire-flat, lamp-broken)
- **Performance**: 0.654 mAP50
- **Input Size**: 640x640
- **Purpose**: Detect damage locations on car

### Model 4: Severity Classification
- **Architecture**: EfficientNet-B0
- **Classes**: 3 severity levels (minor, moderate, severe)
- **Performance**: 69.7% accuracy
- **Input Size**: 224x224
- **Purpose**: Classify severity of detected damage

## 🚫 Note

Model files are **excluded from git** (see `.gitignore`) due to their large size. Each developer must download and place them manually.

## 🔗 Alternative: Model Registry

For production deployment, consider using:
- **Hugging Face Hub**: Upload models to HF for easy sharing
- **DVC (Data Version Control)**: Track model versions
- **MLflow Model Registry**: Centralized model management
- **AWS S3 / GCS**: Cloud storage with versioning

## ✅ Verify Setup

After placing the files, test the API:

```bash
# Start the API
uvicorn api.main:app --reload

# Check model health
curl http://localhost:8000/api/cv/health

# Expected response:
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda",
  "message": "CV detection service is operational"
}
```

## 📝 Training Notebooks

Refer to these notebooks for training details:
- `CarDD-Detection-Training-Done.ipynb` - Damage detection training
- `UltraAnalytics_Carparts_Model_Training_Done.ipynb` - Parts segmentation training
- `Efficient_IMage_Classification.ipynb` - Severity classification training
