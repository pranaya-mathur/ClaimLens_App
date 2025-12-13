# ClaimLens - Insurance Fraud Detection System

AI-powered fraud detection for insurance claims using computer vision, machine learning, and graph analytics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

ClaimLens is a fraud detection system that combines multiple AI techniques to analyze insurance claims. It verifies identity documents, assesses vehicle damage, scores fraud risk using machine learning, and detects fraud networks through graph analysis.

**Key Components:**
- Document verification for Aadhaar and PAN cards using ResNet50
- Vehicle damage detection with YOLO11 and EfficientNet
- CatBoost-based fraud scoring with Hinglish text support
- Neo4j graph database for fraud network detection
- FastAPI backend with Streamlit dashboard

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 8GB+ RAM

### Installation

```bash
git clone https://github.com/pranaya-mathur/ClaimLens_App
cd ClaimLens_App

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Model Setup

Place the following model files in the `models/` directory:

**Computer Vision Models:**
- `yolo11n-seg-car-parts.pt` - Vehicle parts segmentation
- `yolo11m-damage.pt` - Damage detection
- `efficientnet-b0-severity.pth` - Damage severity classification
- `aadhaar_balanced_model.pth` - Aadhaar forgery detection
- `resnet50_finetuned_after_strong_forgeries.pth` - PAN forgery detection

**ML Models:**
- `claimlens_catboost_hinglish.cbm` - Fraud scoring model
- `claimlens_model_metadata.json` - Model metadata

### Running

```bash
# Start Neo4j and Redis
docker-compose up -d

# Load data into graph database
python scripts/01_data_preparation.py
python scripts/02_load_graph.py

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start dashboard (in new terminal)
streamlit run frontend/streamlit_app.py
```

**Access:**
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Neo4j: http://localhost:7474

## Features

### Document Verification

Detects forged Aadhaar and PAN cards using deep learning models.

**Aadhaar Detection:**
- ResNet50 classifier trained on balanced dataset (2,116 real + 2,116 synthetic forgeries)
- 224x224 RGB input with ImageNet normalization
- 99.6% validation accuracy

**PAN Detection:**
- ResNet50 with 4-channel input (RGB + Error Level Analysis)
- Detects copy-move forgeries, text overlays, compression artifacts
- Multiple threshold modes: f1_optimal, precision_oriented, balanced

```python
from src.cv_engine import DocumentVerifier

verifier = DocumentVerifier()
result = verifier.verify("aadhaar.jpg", doc_type="AADHAAR")
print(f"{result.verdict}: {result.confidence:.2%}")
```

### Vehicle Damage Detection

Three-stage pipeline for damage assessment:

1. Parts Segmentation (YOLO11n-seg) - Identifies 23 vehicle components
2. Damage Detection (YOLO11m) - Classifies 6 damage types
3. Severity Classification (EfficientNet-B0) - Minor/moderate/severe

### Fraud Scoring

CatBoost model trained on 50,000 Hinglish claims with 145 features including:
- Bhasha-Embed narrative embeddings (100-dim PCA)
- Claimant/policy aggregations
- Document presence indicators
- Behavioral patterns

```python
from src.ml_engine import MLFraudScorer

scorer = MLFraudScorer(
    model_path="models/claimlens_catboost_hinglish.cbm",
    metadata_path="models/claimlens_model_metadata.json"
)
result = scorer.score_claim(features, return_details=True)
```

### Fraud Graph Analytics

Neo4j-based network analysis to detect:
- Fraud rings (shared documents across claims)
- Serial fraudsters (multiple claims from same claimant)
- Document reuse patterns
- Policy abuse

## Architecture

```
Claim Input
    |
    v
CV Engine (Document + Damage Detection)
    |
    v
ML Engine (Fraud Scoring)
    |
    v
Fraud Graph (Network Analysis)
    |
    v
Decision Engine (Semantic Aggregation + LLM Explanation)
    |
    v
Verdict (APPROVE / REVIEW / REJECT)
```

## Project Structure

```
ClaimLens_App/
├── src/
│   ├── cv_engine/          # Document and damage detection
│   ├── ml_engine/          # Fraud scoring
│   ├── fraud_engine/       # Graph analytics
│   ├── explainability/     # LLM explanations
│   └── app/                # Core logic
├── api/                    # FastAPI routes
├── frontend/               # Streamlit dashboard
├── tests/                  # Test suite
├── scripts/                # Data preparation
└── models/                 # Model files
```

## API Reference

### Document Verification

```http
POST /verify/aadhaar
{
  "image": "base64_encoded_image",
  "dual_check": false
}
```

### Damage Detection

```http
POST /detect/damage
{
  "image": "base64_encoded_image"
}
```

### Fraud Scoring

```http
POST /score/fraud
{
  "claim_data": {...}
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific modules
pytest tests/test_aadhaar_detector.py -v
pytest tests/test_cv_integration.py -v
```

## Technology Stack

**Computer Vision:** YOLOv11, ResNet50, EfficientNet, Error Level Analysis  
**Machine Learning:** CatBoost, Sentence Transformers (Bhasha-Embed)  
**Graph Database:** Neo4j  
**Backend:** FastAPI, Pydantic  
**Frontend:** Streamlit, Plotly  
**Infrastructure:** Docker, Redis

## Current Status

**Working:**
- Aadhaar and PAN forgery detection
- Vehicle damage detection pipeline
- CatBoost fraud scoring
- Neo4j graph database integration
- API endpoints and Streamlit dashboard

**In Progress:**
- EXIF metadata verification
- Multi-image consistency checks
- Additional document types

## License

MIT License - see [LICENSE](LICENSE) file

## Contact

Pranaya Mathur  
GitHub: [@pranaya-mathur](https://github.com/pranaya-mathur)
