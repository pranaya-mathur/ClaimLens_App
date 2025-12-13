# 🤖 ClaimLens AI

**AI-Powered Insurance Fraud Detection System**

> Multi-modal fraud detection combining Machine Learning, Computer Vision, Graph Analytics, and Large Language Models

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-darkgreen.svg)](https://neo4j.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](CHANGELOG.md)

---

## 🎯 **Overview**

ClaimLens AI is an enterprise-grade fraud detection system that analyzes insurance claims using multiple AI modalities:

- **🧠 Machine Learning**: CatBoost model with 145 features for fraud probability scoring
- **👁️ Computer Vision**: Document forgery detection using deep learning (PAN, Aadhaar, + Generic)
- **🕸️ Graph Analytics**: Network fraud ring detection via Neo4j
- **💬 LLM Integration**: Natural language explanations powered by Groq's Llama-3.3-70B

### **Key Features**

✅ **Multi-Modal Analysis** - Combines 4 AI engines for comprehensive fraud detection  
✅ **Explainable AI** - Human-readable explanations for every decision  
✅ **Real-Time Scoring** - Sub-second fraud probability predictions  
✅ **Document Verification** - PAN/Aadhaar/License/Passport forgery detection with OCR  
✅ **Generic Document Detector** - 🆕 NEW! Verify driving licenses, passports, bank statements, and more  
✅ **Network Detection** - Identifies fraud rings through graph relationships  
✅ **Hinglish Support** - Processes claims in English and Hindi-English mix  
✅ **Production Ready** - RESTful APIs with rate limiting and monitoring  

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- Docker (for Neo4j)
- Groq API Key ([Get free key](https://console.groq.com/))

### **Installation**

```bash
# Clone repository
git clone https://github.com/pranaya-mathur/ClaimLens_App.git
cd ClaimLens_App

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### **Run the Application**

```bash
# Terminal 1: Start FastAPI backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/streamlit_app.py

# Optional: Start Neo4j (for graph analysis)
docker-compose up neo4j -d
```

### **Verify Installation**

```bash
# Run diagnostic script
python scripts/diagnose_app.py
```

**Expected Output:**
```
✓ Environment Variables: OK
✓ FastAPI Server: OK
✓ LLM Engine: OK
✓ ML Engine: OK
✓ ALL SYSTEMS OPERATIONAL
```

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│            (Interactive Claim Analysis UI)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│                  (REST API Gateway)                      │
└──┬────────────┬────────────┬────────────┬───────────────┘
   │             │            │            │
   ▼             ▼            ▼            ▼
┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│ML      │  │CV       │  │Graph     │  │LLM       │
│Engine  │  │Engine   │  │Engine    │  │Engine    │
└────────┘  └─────────┘  └──────────┘  └──────────┘
   │             │            │            │
   ▼             ▼            ▼            ▼
CatBoost      YOLO/OCR      Neo4j      Groq API
              +ResNet50
```

### **Technology Stack**

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **Backend** | FastAPI | RESTful API framework |
| **Frontend** | Streamlit | Interactive dashboard |
| **ML Model** | CatBoost | Fraud probability scoring |
| **CV Models** | YOLO, ResNet50, Tesseract | Document forgery detection |
| **Graph DB** | Neo4j | Fraud network analysis |
| **LLM** | Groq Llama-3.3-70B | Natural language explanations |
| **Orchestration** | LangChain | LLM workflow management |
| **Monitoring** | Loguru | Structured logging |

---

## 📊 **API Endpoints**

### **Core Endpoints**

```bash
# Unified Analysis (All modules in one call)
POST /api/unified/analyze-complete

# ML Fraud Scoring
POST /api/ml/score
POST /api/ml/score/detailed

# Document Verification
POST /api/documents/verify-pan
POST /api/documents/verify-aadhaar
POST /api/documents/verify-document  # 🆕 NEW! Generic documents

# LLM Explanations
POST /api/llm/explain
GET  /api/llm/health

# Graph Analysis
POST /api/fraud/score
GET  /api/fraud/rings
GET  /api/fraud/serial-fraudsters

# Health Checks
GET  /health/liveness
GET  /health/readiness
```

**Interactive API Docs:** http://localhost:8000/docs

---

## 🎨 **Features Demo**

### **1. Multi-Modal Fraud Analysis**
Analyze claims using ML, CV, Graph, and LLM engines simultaneously

### **2. Document Verification**
Upload PAN/Aadhaar cards for forgery detection with real-time OCR

### **3. Generic Document Verification** 🆕
Verify:
- 🚗 Driving Licenses
- ✈️ Passports
- 🗳️ Voter IDs
- 🏦 Bank Statements
- 🏥 Hospital Bills
- ⚰️ Death Certificates

### **4. AI-Generated Explanations**
Get human-readable explanations in both technical and customer-friendly language

### **5. Fraud Network Detection**
Visualize fraud rings through shared documents and suspicious connections

### **6. Real-Time Dashboards**
Monitor fraud trends, risk distributions, and analytics

---

## 📚 **Documentation**

Detailed documentation available in `/docs`:

- **[Setup Guide](docs/SETUP.md)** - Complete installation instructions
- **[API Documentation](docs/API.md)** - Endpoint references and examples
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[Document Verification Guide](docs/guides/document_verification.md)** - CV engine usage
- **[Deployment](docs/DEPLOYMENT.md)** - Production deployment guide
- **[CHANGELOG](CHANGELOG.md)** - Version history and updates
- **[CONTRIBUTING](CONTRIBUTING.md)** - Contribution guidelines

---

## 🧪 **Testing**

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ml_engine.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
# LLM Configuration
GROQ_API_KEY=your_groq_api_key
EXPLANATION_MODEL=llama-3.3-70b-versatile

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=claimlens123

# Feature Flags
ENABLE_LLM_EXPLANATIONS=true
ENABLE_SEMANTIC_AGGREGATION=true
```

---

## 📊 **Performance**

- **ML Inference**: <100ms per claim
- **Document Analysis**: <2s per image
- **LLM Explanations**: <3s per explanation
- **Graph Queries**: <500ms for network analysis
- **API Throughput**: 100 requests/minute (rate limited)

---

## 🛣️ **Roadmap**

### v2.2.0 (Coming Soon)
- [ ] Batch processing API
- [ ] PDF document support
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Enhanced graph visualizations

### v3.0.0 (Future)
- [ ] **Agentic Architecture** - LangGraph-based autonomous fraud investigation
- [ ] **Web Intelligence** - Real-time fraud pattern search via Tavily/Perplexity
- [ ] **Advanced Analytics** - Time-series fraud trend prediction
- [ ] **Mobile App** - React Native mobile interface

See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## 🤝 **Contributing**

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📜 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👤 **Author**

**Pranay Mathur**
- GitHub: [@pranaya-mathur](https://github.com/pranaya-mathur)
- LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/your-profile)
- Email: pranaya.mathur@yahoo.com

---

## 🙏 **Acknowledgments**

- **Groq** - Ultra-fast LLM inference
- **LangChain** - LLM orchestration framework  
- **Neo4j** - Graph database platform
- **CatBoost** - Gradient boosting library

---

## 📞 **Support**

- 📧 Email: pranaya.mathur@yahoo.com
- 🐛 Issues: [GitHub Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/pranaya-mathur/ClaimLens_App/discussions)

---

<div align="center">

**Built with ❤️ for the Insurance Industry**

⭐ Star this repo if you find it useful!

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=pranaya-mathur.ClaimLens_App)

</div>
