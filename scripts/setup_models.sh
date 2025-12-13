#!/bin/bash
# Model Setup Script for ClaimLens
# Usage: ./scripts/setup_models.sh /path/to/your/models/

set -e

MODEL_SOURCE_DIR="${1:-.}"

echo "=================================="
echo "ClaimLens Model Setup"
echo "=================================="
echo ""

# Create directories
echo "[1/3] Creating model directories..."
mkdir -p models/parts_segmentation
mkdir -p models/damage_detection
mkdir -p models/severity_classification
mkdir -p models/forgery_detection
mkdir -p models/ml_engine
echo "✓ Directories created"
echo ""

# Copy CV models
echo "[2/3] Copying model files..."

if [ -f "$MODEL_SOURCE_DIR/yolo11n_best.pth" ]; then
    cp "$MODEL_SOURCE_DIR/yolo11n_best.pth" models/parts_segmentation/yolo11n_best.pt
    echo "✓ yolo11n_best.pt"
else
    echo "⚠ yolo11n_best.pth not found"
fi

if [ -f "$MODEL_SOURCE_DIR/yolo11m_best.pth" ]; then
    cp "$MODEL_SOURCE_DIR/yolo11m_best.pth" models/damage_detection/yolo11m_best.pt
    echo "✓ yolo11m_best.pt"
else
    echo "⚠ yolo11m_best.pth not found"
fi

if [ -f "$MODEL_SOURCE_DIR/efficientnet_b0_best.pth" ]; then
    cp "$MODEL_SOURCE_DIR/efficientnet_b0_best.pth" models/severity_classification/
    echo "✓ efficientnet_b0_best.pth"
else
    echo "⚠ efficientnet_b0_best.pth not found"
fi

if [ -f "$MODEL_SOURCE_DIR/resnet50_finetuned_after_strong_forgeries.pth" ]; then
    cp "$MODEL_SOURCE_DIR/resnet50_finetuned_after_strong_forgeries.pth" models/forgery_detection/resnet50_finetuned.pth
    echo "✓ resnet50_finetuned.pth"
else
    echo "⚠ resnet50_finetuned_after_strong_forgeries.pth not found"
fi

if [ -f "$MODEL_SOURCE_DIR/aadhaar_balanced_model.pth" ]; then
    cp "$MODEL_SOURCE_DIR/aadhaar_balanced_model.pth" models/forgery_detection/
    echo "✓ aadhaar_balanced_model.pth"
else
    echo "⚠ aadhaar_balanced_model.pth not found"
fi

if [ -f "$MODEL_SOURCE_DIR/claimlens_catboost_hinglish.cbm" ]; then
    cp "$MODEL_SOURCE_DIR/claimlens_catboost_hinglish.cbm" models/ml_engine/
    echo "✓ claimlens_catboost_hinglish.cbm"
else
    echo "⚠ claimlens_catboost_hinglish.cbm not found"
fi

if [ -f "$MODEL_SOURCE_DIR/claimlens_model_metadata.json" ]; then
    cp "$MODEL_SOURCE_DIR/claimlens_model_metadata.json" models/ml_engine/
    echo "✓ claimlens_model_metadata.json"
else
    echo "⚠ claimlens_model_metadata.json not found (optional)"
fi

if [ -f "$MODEL_SOURCE_DIR/claimlens_feature_importance.csv" ]; then
    cp "$MODEL_SOURCE_DIR/claimlens_feature_importance.csv" models/ml_engine/
    echo "✓ claimlens_feature_importance.csv"
else
    echo "⚠ claimlens_feature_importance.csv not found (optional)"
fi

echo ""

# Verify
echo "[3/3] Verifying setup..."
COUNT=$(find models -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.cbm" \) | wc -l)
echo "✓ Found $COUNT model files"

if [ $COUNT -ge 6 ]; then
    echo ""
    echo "=================================="
    echo "✅ Setup complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Start API: uvicorn api.main:app --reload"
    echo "2. Test health: curl http://localhost:8000/api/cv/health"
    echo "3. Start UI: streamlit run frontend/streamlit_app_v2_sota.py"
else
    echo ""
    echo "=================================="
    echo "⚠ Some models missing"
    echo "=================================="
    echo ""
    echo "Please check that all 8 model files are in: $MODEL_SOURCE_DIR"
    echo "Run: ./scripts/setup_models.sh /path/to/models/"
fi
