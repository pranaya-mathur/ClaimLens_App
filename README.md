# ClaimLens - AI-Powered Insurance Fraud Detection

> Enterprise-grade fraud detection system for Indian insurance claims combining computer vision, graph analytics, and machine learning for real-time decision-making.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

## Overview

ClaimLens delivers production-ready fraud detection for insurance claims through a multi-layered AI architecture. The system processes vehicle damage assessments, verifies identity documents, analyzes fraud networks, and provides explainable decisions in under 2 seconds.

### Core Capabilities

- **Document Verification**: Deep learning-based forgery detection for Aadhaar and PAN cards with high accuracy
- **Vehicle Damage Assessment**: Multi-model pipeline for parts segmentation, damage detection, and severity classification
- **Fraud Network Analysis**: Graph-based detection of fraud rings, serial fraudsters, and document reuse patterns
- **Explainable AI**: LLM-powered explanations tailored for adjusters and customers
- **Real-time Processing**: End-to-end claim adjudication with semantic aggregation

## Architecture

```
┌─────────────────┐
│  Claim Upload   │
└────────┬────────┘
         │
    ┌────▼─────────────────────────┐
    │    Computer Vision Engine    │
    │  • Document Verification     │
    │  • Damage Detection          │
    └────────┬─────────────────────┘
             │
    ┌────────▼─────────────────────┐
    │   Machine Learning Engine    │
    │  • XGBoost Risk Scoring      │
    │  • NLP Narrative Analysis    │
    └────────┬─────────────────────┘
             │
    ┌────────▼─────────────────────┐
    │    Fraud Graph Analytics     │
    │  • Neo4j Network Analysis    │
    │  • Pattern Detection         │
    └────────┬─────────────────────┘
             │
    ┌────────▼─────────────────────┐
    │    Decision Engine + LLM     │
    │  • Semantic Aggregation      │
    │  • Rule-based Logic          │
    │  • Explainable Decisions     │
    └────────┬─────────────────────┘
             │
    ┌────────▼─────────────────────┐
    │  APPROVE / REVIEW / REJECT   │
    └──────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/pranaya-mathur/ClaimLens_App
cd ClaimLens_App

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OpenAI, Groq, Neo4j credentials)
```

### Model Setup

Download required model files and place them in the `models/` directory:

**Damage Detection Models:**
- `yolo11n-seg-car-parts.pt` - Parts segmentation
- `yolo11m-damage.pt` - Damage detection
- `efficientnet-b0-severity.pth` - Severity classifier

**Document Verification Models:**
- `aadhaar_balanced_model.pth` - Aadhaar forgery detection
- `resnet50_finetuned_after_strong_forgeries.pth` - PAN forgery detection

**Legacy Models:**
- `forgery_detector_latest_run.pth` - Generic forgery detection
- `forgery_detector_latest_run_config.json` - Model configuration

### Running the System

```bash
# Start services (Neo4j + Redis)
docker-compose up -d

# Load initial data
python scripts/01_data_preparation.py
python scripts/02_load_graph.py

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Launch dashboard (new terminal)
streamlit run frontend/streamlit_app.py
```

**Access Points:**
- API Documentation: http://localhost:8000/docs
- Streamlit Dashboard: http://localhost:8501
- Neo4j Browser: http://localhost:7474

## Key Features

### 1. Document Verification System

Production-ready identity document verification with enterprise-grade accuracy.

#### Aadhaar Card Detection
- **Architecture**: ResNet50 backbone with custom classification head
- **Input**: 224×224 RGB images (ImageNet normalized)
- **Training**: Balanced dataset with real and synthetic forgeries
- **Deployment**: Sub-150ms inference time

#### PAN Card Detection
- **Architecture**: ResNet50 with 4-channel input (RGB + Error Level Analysis)
- **Input**: 320×320 images with ELA forensics layer
- **Detections**: Copy-move forgeries, text overlays, print-scan artifacts, compression manipulation
- **Modes**: F1-optimal, precision-oriented, and balanced thresholds

#### Unified API with Dual-Check
```python
from src.cv_engine import DocumentVerifier

verifier = DocumentVerifier()

# Single document verification
result = verifier.verify("aadhaar.jpg", doc_type="AADHAAR")

# High-stakes dual-check mode
result = verifier.verify(
    "pan_card.jpg", 
    doc_type="PAN", 
    dual_check=True  # Cross-validates with both detectors
)

# Batch processing
results = verifier.verify_batch(
    ["doc1.jpg", "doc2.jpg", "doc3.jpg"],
    doc_type="PAN"
)
```

**Consensus Logic:**
- ✅ Both detectors agree on CLEAN → High confidence authentic
- ✅ Both detectors agree on FORGED → High confidence forgery
- ⚠️ Disagreement → Flagged as SUSPICIOUS for manual review

### 2. Vehicle Damage Detection

Multi-stage computer vision pipeline for comprehensive damage assessment.

**Pipeline Stages:**
1. **Parts Segmentation** (YOLO11n-seg) - Identifies 23 vehicle component classes
2. **Damage Detection** (YOLO11m) - Classifies 6 damage types (dent, scratch, crack, shatter, flat, broken)
3. **Severity Assessment** (EfficientNet-B0) - Categorizes as minor, moderate, or severe

**Capabilities:**
- Automated cost estimation based on damage severity
- Sub-second inference per image
- Detailed damage localization with bounding boxes

### 3. Generic Forgery Detection

Hybrid deep learning and forensics approach for image manipulation detection.

**Techniques:**
- **Deep Learning**: ResNet50 binary classifier trained on forgery datasets
- **Error Level Analysis (ELA)**: Detects JPEG compression inconsistencies
- **Noise Variation Analysis**: Identifies spliced or pasted regions

```python
from src.cv_engine import ForgeryDetector

detector = ForgeryDetector(
    model_path="models/forgery_detector_latest_run.pth",
    config_path="models/forgery_detector_latest_run_config.json"
)

result = detector.analyze_image("claim_photo.jpg")
print(f"Forgery Detected: {result.is_forged}")
print(f"Confidence: {result.forgery_prob:.2%}")
```

### 4. Fraud Graph Analytics

Neo4j-powered network analysis for detecting organized fraud patterns.

**Detection Capabilities:**
- Fraud ring identification through shared documents
- Serial fraudster tracking across multiple claims
- Document reuse pattern recognition
- Policy abuse detection
- Community detection algorithms

**Performance:**
- Sub-100ms query execution
- Real-time graph updates
- Scalable to millions of nodes

### 5. Semantic Decision Engine

Advanced decision-making with LLM-powered explainability.

**Features:**
- **Semantic Aggregation**: Converts raw model outputs to human-interpretable verdicts
- **Rule-based Logic**: Configurable business rules for approval thresholds
- **LLM Explanations**: Dual-audience prompts (technical for adjusters, simplified for customers)
- **Audit Trail**: Complete decision provenance tracking

### 6. Modern Streamlit Dashboard

State-of-the-art interface with live streaming and interactive visualizations.

**Components:**
- Real-time confidence meters and risk gauges
- Interactive fraud network graphs
- Live claim processing streams
- Document verification results with visual overlays
- Damage detection annotations

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Computer Vision** | YOLOv11, ResNet50, EfficientNet, Error Level Analysis |
| **Machine Learning** | XGBoost, CatBoost, Sentence Transformers |
| **Graph Database** | Neo4j, NetworkX |
| **Backend API** | FastAPI, Pydantic, Uvicorn |
| **LLM Integration** | LangChain, OpenAI, Groq |
| **Frontend** | Streamlit, Plotly |
| **Infrastructure** | Docker, Docker Compose, Redis |
| **Testing** | Pytest, unittest, mock |

## Project Structure

```
ClaimLens_App/
├── src/
│   ├── cv_engine/              # Computer vision modules
│   │   ├── damage_detector.py
│   │   ├── aadhaar_detector.py
│   │   ├── pan_detector.py
│   │   ├── document_verifier.py
│   │   ├── forgery_detector.py
│   │   └── forgery_utils.py
│   ├── fraud_engine/           # Graph analytics
│   ├── ml_engine/              # ML scoring models
│   ├── explainability/         # LLM explainer
│   └── app/                    # Core application logic
├── api/                        # FastAPI routes
├── frontend/                   # Streamlit dashboard
├── scripts/                    # Data preparation & ingestion
├── tests/                      # Comprehensive test suite
├── models/                     # Trained model files
├── config/                     # Configuration files
├── docs/                       # Additional documentation
└── docker-compose.yml          # Service orchestration
```

## API Reference

### Document Verification

```http
POST /verify/aadhaar
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "dual_check": false
}
```

**Response:**
```json
{
  "document_type": "AADHAAR",
  "verdict": "AUTHENTIC",
  "confidence": 0.9845,
  "authentic_probability": 0.9845,
  "forged_probability": 0.0155
}
```

### PAN Verification

```http
POST /verify/pan
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "threshold_mode": "f1_optimal"
}
```

**Threshold Modes:**
- `f1_optimal` - Balanced performance (default)
- `precision_oriented` - Minimizes false positives
- `balanced` - Standard threshold

### Unified Verification with Dual-Check

```http
POST /verify/document
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "doc_type": "PAN",
  "dual_check": true
}
```

**Response (Dual-Check Enabled):**
```json
{
  "document_type": "PAN",
  "verdict": "CLEAN",
  "confidence": 0.94,
  "dual_check_enabled": true,
  "agreement": true,
  "consensus_verdict": "CLEAN",
  "consensus_confidence": 0.95
}
```

## Testing

Comprehensive test suite with unit, integration, and end-to-end tests.

```bash
# Test document verification
pytest tests/test_aadhaar_detector.py -v
pytest tests/test_pan_detector.py -v
pytest tests/test_document_verifier.py -v

# Test CV components
pytest tests/test_cv_integration.py -v

# Run all tests
pytest tests/ -v --cov=src
```

**Test Coverage:**
- ✅ Model initialization and loading
- ✅ Inference accuracy validation
- ✅ Dual-check consensus logic
- ✅ Batch processing workflows
- ✅ Error handling edge cases
- ✅ Mock-based tests for CI/CD

## Roadmap

### Recently Completed
- ✅ Document verification system (Aadhaar & PAN)
- ✅ Vehicle damage detection pipeline
- ✅ Fraud graph database integration
- ✅ LLM-powered explainability
- ✅ Semantic aggregation engine
- ✅ Modern Streamlit dashboard
- ✅ Docker containerization

### In Progress
- 🚧 XGBoost fraud scoring engine
- 🚧 Duplicate image detection
- 🚧 EXIF metadata verification
- 🚧 Multi-image consistency checks

### Planned
- 📋 Passport and driver's license verification
- 📋 GAN-generated image detection
- 📋 Real-time monitoring dashboard
- 📋 Model serving optimization (ONNX, TensorRT)
- 📋 A/B testing framework
- 📋 Kubernetes deployment configs

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Pranaya Mathur**

- GitHub: [@pranaya-mathur](https://github.com/pranaya-mathur)
- Issues: [GitHub Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues)

---

**Built with ❤️ using cutting-edge AI • Optimized for fraud detection at scale**
