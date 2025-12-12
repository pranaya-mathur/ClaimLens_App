# 🔍 ClaimLens - AI-Powered Fraud Detection System

Lemonade-level fraud detection for Indian insurance claims using:
- 🖼️ **Computer Vision** (damage detection, document forgery detection)
- 🕸️ **Graph Analytics** (fraud rings, document reuse)
- 🤖 **Machine Learning** (XGBoost fraud scorer)
- 💬 **NLP** (narrative analysis)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repo
git clone https://github.com/pranaya-mathur/ClaimLens_App
cd ClaimLens_App

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Files
```bash
# Place these files in models/ directory:

# Damage Detection Models
# - yolo11n-seg-car-parts.pt (parts segmentation)
# - yolo11m-damage.pt (damage detection)
# - efficientnet-b0-severity.pth (severity classifier)

# Document Verification Models (NEW!)
# - aadhaar_balanced_model.pth (Aadhaar forgery - 99.62% accuracy)
# - resnet50_finetuned_after_strong_forgeries.pth (PAN forgery - 99.19% accuracy)

# Legacy Forgery Models
# - forgery_detector_latest_run.pth (generic forgery detection)
# - forgery_detector_latest_run_config.json
```

### 3. Start Services (Docker)
```bash
# Start Neo4j + API
docker-compose up -d

# Check services
docker-compose ps
```

### 4. Load Data
```bash
# Prepare data
python scripts/01_data_preparation.py

# Load fraud graph
python scripts/02_load_graph.py
```

### 5. Run API
```bash
# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API docs: http://localhost:8000/docs
```

### 6. Launch Dashboard
```bash
streamlit run frontend/streamlit_app.py
```

## 📊 Architecture

```
User Upload
    ↓
[CV Engine] → Damage Detection + Document Verification
    ↓
[ML Engine] → XGBoost Fraud Score + Narrative NLP
    ↓
[Fraud Graph] → Graph Risk Score (Neo4j)
    ↓
[Decision Engine] → Rules + LLM Explanation
    ↓
APPROVE / REVIEW / REJECT
```

## 🔧 Tech Stack

- **CV**: YOLOv11, ResNet50 (4-channel), EfficientNet, ELA
- **ML**: XGBoost, CatBoost, Transformers
- **Graph**: Neo4j, NetworkX
- **API**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Infra**: Docker, Redis

## 📁 Project Structure

```
ClaimLens/
├── src/
│   ├── cv_engine/              # Computer Vision
│   │   ├── damage_detector.py         # YOLO-based damage detection
│   │   ├── forgery_detector.py        # Generic forgery detection
│   │   ├── aadhaar_detector.py        # Aadhaar card verification (NEW!)
│   │   ├── pan_detector.py            # PAN card verification (NEW!)
│   │   ├── document_verifier.py       # Unified document API (NEW!)
│   │   ├── forgery_models.py          # CNN architectures (3ch + 4ch)
│   │   └── forgery_utils.py           # ELA & noise analysis
│   ├── fraud_engine/           # Graph analytics
│   └── app/                    # Core application
├── api/                        # FastAPI backend
├── frontend/                   # Streamlit UI
├── scripts/                    # Data pipelines
├── models/                     # Trained models (.pth, .pt)
├── tests/                      # Test suites (75+ tests)
└── data/                       # Datasets
```

## 🎯 Key Features

### 1️⃣ Vehicle Damage Detection
**Multi-Model Pipeline:**
- **Parts Segmentation** (YOLO11n-seg) - 23 car part classes
- **Damage Detection** (YOLO11m) - 6 damage types (dent, scratch, crack, etc.)
- **Severity Classification** (EfficientNet-B0) - 3 levels (minor, moderate, severe)

**Capabilities:**
- Detect dents, scratches, cracks, glass shatters, tire flats, broken lamps
- Auto cost estimation based on damage severity
- Sub-second inference per image

### 2️⃣ Document Verification System **NEW! ✨**

#### **Aadhaar Card Forgery Detection**
**Performance Metrics:**
- ✅ **99.62% Validation Accuracy**
- ✅ **99.80% Balanced Accuracy** (critical for imbalanced data)
- ✅ **AUC: 0.9999**
- ✅ **0 False Negatives** in validation (530 forged samples)
- ✅ **7 False Positives** out of 1,060 total samples

**Model Architecture:**
- Backbone: ResNet50 (ImageNet pretrained)
- Input: 224×224 RGB images
- Head: Simple Linear(2048 → 2)
- Training: 2,116 real + 2,116 synthetic forgeries
- Threshold: 0.5 (balanced)

**Usage:**
```python
from src.cv_engine import AadhaarForgeryDetector

detector = AadhaarForgeryDetector()
result = detector.analyze("aadhaar_card.jpg")

print(f"Verdict: {result.verdict}")  # "AUTHENTIC" or "FORGED"
print(f"Confidence: {result.confidence:.2%}")
print(f"Authentic Probability: {result.authentic_probability:.2%}")
```

#### **PAN Card Forgery Detection**
**Performance Metrics:**
- ✅ **99.19% Accuracy @ threshold 0.5**
- ✅ **AUC: 0.9996**
- ✅ **F1 Score: 0.9942 @ threshold 0.49** (optimal)
- ✅ **95%+ Precision @ threshold 0.48** (precision mode)
- ✅ **Only 11 errors** out of 1,350 samples (9 FP, 2 FN)

**Model Architecture:**
- Backbone: ResNet50 with **4-channel input (RGB + ELA)**
- Input: 320×320 (RGB + Error Level Analysis)
- Output: Single logit (BCEWithLogitsLoss)
- Training: 640 initial + 550 strong forgeries (fine-tuned)
- Threshold Modes:
  - **0.49** = F1-optimal (balanced)
  - **0.48** = Precision-oriented (minimize false accusations)
  - **0.50** = Standard balanced

**Detects:**
- Copy-move forgeries (patch duplication)
- Text overlays (DOB/name alterations)
- Print-scan artifacts
- JPEG compression manipulation
- Frequency domain tampering

**Usage:**
```python
from src.cv_engine import PANForgeryDetector

# F1-optimal mode (default)
detector = PANForgeryDetector(threshold=0.49)
result = detector.analyze("pan_card.jpg")

print(f"Verdict: {result.verdict}")  # "CLEAN" or "FORGED"
print(f"Forgery Probability: {result.forgery_probability:.2%}")

# Switch to precision mode for critical cases
detector.set_threshold(mode="precision_oriented")
result = detector.analyze("high_stakes_pan.jpg")
```

#### **Unified Document Verification API**
**Dual-Check Mode for Cross-Validation:**
```python
from src.cv_engine import DocumentVerifier

verifier = DocumentVerifier()

# Single-check (standard)
result = verifier.verify("aadhaar.jpg", doc_type="AADHAAR")

# Dual-check (high-stakes fraud investigation)
result = verifier.verify(
    "suspicious_pan.jpg", 
    doc_type="PAN", 
    dual_check=True  # Runs both PAN and Aadhaar detectors
)

if result.agreement:
    print(f"Both detectors agree: {result.consensus_verdict}")
    print(f"Consensus confidence: {result.consensus_confidence:.2%}")
else:
    print("⚠️ SUSPICIOUS - Detectors disagree, manual review required")

# Batch processing
results = verifier.verify_batch(
    ["doc1.jpg", "doc2.jpg", "doc3.jpg"],
    doc_type="PAN",
    dual_check=False
)
```

**Consensus Logic:**
- ✅ Both agree **CLEAN** → High confidence authentic
- ✅ Both agree **FORGED** → High confidence forgery
- ⚠️ **Disagree** → Flagged as **SUSPICIOUS** for manual review

### 3️⃣ Generic Image Forgery Detection
**Hybrid CNN + Forensics Approach:**
- **Deep Learning** - ResNet50 binary classifier (83.6% validation accuracy)
- **Error Level Analysis (ELA)** - Detects JPEG compression inconsistencies
- **Noise Variation** - Identifies spliced/pasted regions

**Training Details:**
- Model: ResNet50 with custom classification head
- Epochs: 15 | Learning Rate: 0.0001 | Threshold: 0.55
- Input: 224×224 RGB images with ImageNet normalization
- Output: Forgery probability (0-1) + ELA score + noise metrics

**Usage:**
```python
from src.cv_engine import ForgeryDetector

detector = ForgeryDetector(
    model_path="models/forgery_detector_latest_run.pth",
    config_path="models/forgery_detector_latest_run_config.json"
)

result = detector.analyze_image("claim_photo.jpg")
print(f"Is Forged: {result.is_forged}")
print(f"Confidence: {result.forgery_prob:.2%}")
print(f"ELA Score: {result.ela_score:.3f}")
```

### 4️⃣ Fraud Graph Engine
**Network Analysis:**
- Find fraud rings (shared docs/images)
- Serial fraudster detection
- Policy abuse patterns
- Community detection algorithms

**Graph Queries:**
- Sub-100ms query performance
- Neo4j integration
- Real-time fraud network updates

### 5️⃣ ML Risk Scoring
**Planned Features:**
- 90%+ AUC fraud classifier (XGBoost/CatBoost)
- Narrative embedding + red flags (NLP)
- Time-delay risk analysis
- Hospital/vendor anomaly detection

### 6️⃣ Fast Decision Engine
**Decision Framework:**
- Sub-2-second end-to-end processing
- Auto-approve low risk claims
- LLM-powered explanations
- Complete audit trail

## 📈 Performance Metrics

| Component | Metric | Performance |
|-----------|--------|-------------|
| **Aadhaar Verification** | Validation Accuracy | **99.62%** |
| **Aadhaar Verification** | Balanced Accuracy | **99.80%** |
| **Aadhaar Verification** | AUC | **0.9999** |
| **PAN Verification** | Accuracy @ 0.5 | **99.19%** |
| **PAN Verification** | AUC | **0.9996** |
| **PAN Verification** | F1 Score @ 0.49 | **0.9942** |
| **Document Verification** | Inference Time | <150ms |
| **Generic Forgery** | Validation Accuracy | **83.6%** |
| **Damage Detection** | Parts Detection | 23 classes |
| **Damage Detection** | Damage Types | 6 categories |
| **Graph Queries** | Query Speed | <100ms |
| **Overall System** | Processing Time | <2s per claim |
| **Overall System** | Target FPR | <5% |

## 🧪 Testing

**Test Coverage: 75+ comprehensive tests**

```bash
# Test document verification modules
pytest tests/test_aadhaar_detector.py -v
pytest tests/test_pan_detector.py -v
pytest tests/test_document_verifier.py -v

# Test generic forgery detection
python tests/test_forgery_detector.py

# Test CV integration
python tests/test_cv_integration.py

# Run all tests
pytest tests/ -v
```

**Test Categories:**
- ✅ Unit tests (model initialization, inference)
- ✅ Integration tests (dual-check, consensus)
- ✅ Error handling (missing files, invalid images)
- ✅ Batch processing validation
- ✅ Threshold boundary testing
- ✅ Mock-based (CI/CD compatible)

## 🔄 Development Roadmap

### ✅ Completed
- [x] Vehicle damage detection pipeline (YOLO + EfficientNet)
- [x] **Aadhaar card forgery detection (99.62% accuracy)**
- [x] **PAN card forgery detection (99.19% accuracy)**
- [x] **Unified document verification API with dual-check**
- [x] **75+ unit and integration tests**
- [x] Generic forgery detection system (ResNet50 + ELA)
- [x] Fraud graph database (Neo4j)
- [x] API endpoints (FastAPI)
- [x] Docker containerization

### 🚧 In Progress
- [ ] ML risk scoring engine (XGBoost)
- [ ] Duplicate image detection
- [ ] Metadata verification (EXIF)
- [ ] Multi-image consistency checks

### 📋 Planned
- [ ] Passport verification
- [ ] Driver's license verification
- [ ] GAN-generated image detection
- [ ] Real-time monitoring dashboard
- [ ] Model serving optimization
- [ ] A/B testing framework

## 📚 API Documentation

### Document Verification Endpoints

```python
# POST /verify/aadhaar
{
  "image": "base64_encoded_image",
  "dual_check": false
}

# Response
{
  "document_type": "AADHAAR",
  "verdict": "AUTHENTIC",
  "confidence": 0.9845,
  "authentic_probability": 0.9845,
  "forged_probability": 0.0155
}

# POST /verify/pan
{
  "image": "base64_encoded_image",
  "threshold_mode": "f1_optimal"  # or "precision_oriented", "balanced"
}

# Response
{
  "document_type": "PAN",
  "verdict": "CLEAN",
  "confidence": 0.9612,
  "forgery_probability": 0.0388,
  "clean_probability": 0.9612
}

# POST /verify/document (unified)
{
  "image": "base64_encoded_image",
  "doc_type": "PAN",  # or "AADHAAR"
  "dual_check": true
}

# Response (with dual-check)
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

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 👨‍💻 Team

Built by **Pranaya Mathur** & Team

## 📄 License

MIT License

## 📞 Contact

For questions or collaboration: [GitHub Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues)

---

**⚡ Built with AI, optimized for fraud detection at scale**

**Latest Update:** Document verification system with 99%+ accuracy for Indian identity documents (Aadhaar & PAN)
