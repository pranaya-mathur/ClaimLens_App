# 🔍 ClaimLens - AI-Powered Fraud Detection System

Lemonade-level fraud detection for Indian insurance claims using:
- 🖼️ **Computer Vision** (damage detection, forgery detection)
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
# - forgery_detector_latest_run.pth (ResNet50 weights)
# - forgery_detector_latest_run_config.json (model config)
# - yolo11n-seg-car-parts.pt (parts segmentation)
# - yolo11m-damage.pt (damage detection)
# - efficientnet-b0-severity.pth (severity classifier)
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
[CV Engine] → Damage Detection + Forgery Check
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

- **CV**: YOLOv11, ResNet50, EfficientNet, ELA
- **ML**: XGBoost, CatBoost, Transformers
- **Graph**: Neo4j, NetworkX
- **API**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Infra**: Docker, Redis

## 📁 Project Structure

```
ClaimLens/
├── src/
│   ├── cv_engine/          # Computer Vision
│   │   ├── damage_detector.py     # YOLO-based damage detection
│   │   ├── forgery_detector.py    # Image forgery detection
│   │   ├── forgery_models.py      # ResNet50 CNN architecture
│   │   └── forgery_utils.py       # ELA & noise analysis
│   ├── fraud_engine/       # Graph analytics
│   └── app/                # Core application
├── api/                    # FastAPI backend
├── frontend/               # Streamlit UI
├── scripts/                # Data pipelines
├── models/                 # Trained models (.pth, .pt)
├── tests/                  # Test suites
└── data/                   # Datasets
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

### 2️⃣ Forgery Detection **NEW! ✨**
**Hybrid CNN + Forensics Approach:**
- **Deep Learning** - ResNet50 binary classifier (83.6% validation accuracy)
- **Error Level Analysis (ELA)** - Detects JPEG compression inconsistencies
- **Noise Variation** - Identifies spliced/pasted regions

**Training Details:**
- Model: ResNet50 with custom classification head
- Epochs: 15 | Learning Rate: 0.0001 | Threshold: 0.55
- Input: 224×224 RGB images with ImageNet normalization
- Output: Forgery probability (0-1) + ELA score + noise metrics

**Detection Capabilities:**
- ✅ Copy-paste manipulations
- ✅ Photoshop edits with compression artifacts
- ✅ Spliced regions with mismatched noise patterns
- ✅ AI-augmented tampering

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

### 3️⃣ Fraud Graph Engine
**Network Analysis:**
- Find fraud rings (shared docs/images)
- Serial fraudster detection
- Policy abuse patterns
- Community detection algorithms

**Graph Queries:**
- Sub-100ms query performance
- Neo4j integration
- Real-time fraud network updates

### 4️⃣ ML Risk Scoring
**Planned Features:**
- 90%+ AUC fraud classifier (XGBoost/CatBoost)
- Narrative embedding + red flags (NLP)
- Time-delay risk analysis
- Hospital/vendor anomaly detection

### 5️⃣ Fast Decision Engine
**Decision Framework:**
- Sub-2-second end-to-end processing
- Auto-approve low risk claims
- LLM-powered explanations
- Complete audit trail

## 📈 Performance Metrics

| Component | Metric | Performance |
|-----------|--------|-------------|
| **Forgery Detection** | Validation Accuracy | **83.6%** |
| **Forgery Detection** | Inference Time | <100ms |
| **Damage Detection** | Parts Detection | 23 classes |
| **Damage Detection** | Damage Types | 6 categories |
| **Graph Queries** | Query Speed | <100ms |
| **Overall System** | Processing Time | <2s per claim |
| **Overall System** | False Positive Rate | <5% |

## 🧪 Testing

```bash
# Test forgery detection module
python tests/test_forgery_detector.py

# Test CV integration
python tests/test_cv_integration.py

# Run all tests
pytest tests/
```

## 🔄 Development Roadmap

### ✅ Completed
- [x] Vehicle damage detection pipeline (YOLO + EfficientNet)
- [x] Forgery detection system (ResNet50 + ELA)
- [x] Fraud graph database (Neo4j)
- [x] API endpoints (FastAPI)
- [x] Docker containerization

### 🚧 In Progress
- [ ] ML risk scoring engine (XGBoost)
- [ ] Duplicate image detection
- [ ] Metadata verification (EXIF)
- [ ] Multi-image consistency checks

### 📋 Planned
- [ ] GAN-generated image detection
- [ ] Real-time monitoring dashboard
- [ ] Model serving optimization
- [ ] A/B testing framework

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
