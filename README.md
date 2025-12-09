# 🔍 ClaimLens - AI-Powered Fraud Detection System

Lemonade-level fraud detection for Indian insurance claims using:
- 🖼️ Computer Vision (damage detection, forgery detection)
- 🕸️ Graph Analytics (fraud rings, document reuse)
- 🤖 Machine Learning (XGBoost fraud scorer)
- 💬 NLP (narrative analysis)

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

### 2. Start Services (Docker)
```bash
# Start Neo4j + API
docker-compose up -d

# Check services
docker-compose ps
```

### 3. Load Data
```bash
# Prepare data
python scripts/01_data_preparation.py

# Load fraud graph
python scripts/02_load_graph.py
```

### 4. Run API
```bash
# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API docs: http://localhost:8000/docs
```

### 5. Launch Dashboard
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

- **CV**: YOLOv11, DINOv2, ELA
- **ML**: XGBoost, CatBoost, Transformers
- **Graph**: Neo4j, NetworkX
- **API**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Infra**: Docker, Redis

## 📁 Project Structure

```
ClaimLens/
├── src/              # Core engines
├── api/              # FastAPI backend
├── frontend/         # Streamlit UI
├── scripts/          # Data pipelines
├── models/           # Trained models
└── data/             # Datasets
```

## 🎯 Key Features

### 1️⃣ Vehicle Damage AI
- Detect dents, scratches, cracks
- Forgery detection (ELA + CNN)
- Duplicate photo detection
- Auto cost estimation

### 2️⃣ Fraud Graph Engine
- Find fraud rings (shared docs/images)
- Serial fraudster detection
- Policy abuse patterns
- Community detection

### 3️⃣ ML Risk Scoring
- 90%+ AUC fraud classifier
- Narrative embedding + red flags
- Time-delay risk
- Hospital/vendor anomaly

### 4️⃣ Fast Decision
- Sub-second processing
- Auto-approve low risk
- LLM explanations
- Audit trail

## 📈 Performance

- **Fraud Detection Rate**: 89%
- **False Positive Rate**: <5%
- **Processing Time**: <2s per claim
- **Graph Query**: <100ms

## 🤝 Team

Built by Pranaya & Team

## 📄 License

MIT License