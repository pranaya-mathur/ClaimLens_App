# ML Engine Models

Place your trained CatBoost model and metadata files here.

## Required Files

```
models/ml_engine/
├── claimlens_catboost_hinglish.cbm       (50-100 MB)
├── claimlens_model_metadata.json         (~1 KB)
└── claimlens_feature_importance.csv      (~10 KB)
```

## Setup

```bash
# Copy your trained models
cp ~/path/to/claimlens_catboost_hinglish.cbm models/ml_engine/
cp ~/path/to/claimlens_model_metadata.json models/ml_engine/
cp ~/path/to/claimlens_feature_importance.csv models/ml_engine/
```

## Verification

```bash
# Check files are in place
ls -lh models/ml_engine/

# Start API and test
uvicorn api.main:app --reload
curl http://localhost:8000/api/ml/health
```

## Model Info

- **Type**: CatBoost Classifier
- **Features**: 150+ engineered features (claim ratios, embeddings, etc.)
- **Performance**: AUC 0.87, F1 0.82
- **Language**: Hinglish narrative support via bhasha-embed-v0

## Environment Variables (Optional)

```bash
ML_MODEL_PATH=models/ml_engine/claimlens_catboost_hinglish.cbm
ML_METADATA_PATH=models/ml_engine/claimlens_model_metadata.json
ML_THRESHOLD=0.5
```

Model files are excluded from git (.gitignore) due to size.
