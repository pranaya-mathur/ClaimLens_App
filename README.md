# ClaimLens AI

**AI-Powered Insurance Fraud Detection**

Multi-modal fraud detection combining Machine Learning, Computer Vision, Graph Analytics, and LLMs for insurance claim analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

ClaimLens AI analyzes insurance claims using multiple AI techniques to detect fraud:

- **Machine Learning**: CatBoost model with 145 features for fraud scoring
- **Computer Vision**: Document forgery detection (PAN, Aadhaar, generic documents)
- **Graph Analytics**: Network fraud ring detection using Neo4j
- **LLM Integration**: Natural language explanations via Groq's Llama-3.3-70B

### Key Features

- Multi-modal analysis combining 4 AI engines
- Explainable AI with human-readable explanations
- Real-time fraud probability predictions
- Document verification with OCR (PAN/Aadhaar/licenses/passports)
- Generic document detector for various document types
- Fraud network detection through graph relationships
- Hinglish language support
- RESTful APIs with rate limiting

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- Groq API Key ([get free key](https://console.groq.com/))

### Installation

```bash
git clone https://github.com/pranaya-mathur/ClaimLens_App.git
cd ClaimLens_App

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running the App

```bash
# Terminal 1: Backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py

# Optional: Neo4j
docker-compose up neo4j -d
```

Check installation:
```bash
python scripts/diagnose_app.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
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

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| Frontend | Streamlit |
| ML Model | CatBoost |
| CV Models | YOLO, ResNet50, Tesseract |
| Graph DB | Neo4j |
| LLM | Groq Llama-3.3-70B |
| Orchestration | LangChain |

---

## API Endpoints

### Main Endpoints

```bash
# Complete analysis (all modules)
POST /api/unified/analyze-complete

# ML scoring
POST /api/ml/score
POST /api/ml/score/detailed

# Document verification
POST /api/documents/verify-pan
POST /api/documents/verify-aadhaar
POST /api/documents/verify-document  # generic documents

# LLM explanations
POST /api/llm/explain
GET  /api/llm/health

# Graph analysis
POST /api/fraud/score
GET  /api/fraud/rings
GET  /api/fraud/serial-fraudsters

# Health
GET  /health/liveness
GET  /health/readiness
```

Interactive docs: http://localhost:8000/docs

---

## Features

### 1. Multi-Modal Fraud Analysis
Analyzes claims using ML, CV, Graph, and LLM engines simultaneously.

### 2. Document Verification
Upload PAN/Aadhaar cards for forgery detection with OCR.

### 3. Generic Document Verification
Supports driving licenses, passports, voter IDs, bank statements, hospital bills, death certificates.

### 4. AI Explanations
Human-readable explanations in technical and customer-friendly formats.

### 5. Network Detection
Visualizes fraud rings through shared documents and connections.

### 6. Analytics Dashboard
Real-time monitoring of fraud trends and risk distributions.

---

## Documentation

Available in `/docs`:

- [Setup Guide](docs/SETUP.md) - Installation details
- [API Documentation](docs/API.md) - Endpoint reference
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Document Verification](docs/guides/document_verification.md) - CV engine
- [Deployment](docs/DEPLOYMENT.md) - Production setup
- [CHANGELOG](CHANGELOG.md) - Version history
- [CONTRIBUTING](CONTRIBUTING.md) - How to contribute

---

## Testing

```bash
pytest tests/
pytest tests/test_ml_engine.py -v
pytest --cov=src tests/
```

---

## Configuration

Key environment variables:

```bash
GROQ_API_KEY=your_groq_api_key
EXPLANATION_MODEL=llama-3.3-70b-versatile

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=claimlens123

ENABLE_LLM_EXPLANATIONS=true
ENABLE_SEMANTIC_AGGREGATION=true
```

---

## Performance

- ML Inference: <100ms per claim
- Document Analysis: <2s per image
- LLM Explanations: <3s
- Graph Queries: <500ms
- API Throughput: 100 req/min (rate limited)

---

## Roadmap

### v2.2.0 (In Progress)
- [ ] Batch processing API
- [ ] PDF document support
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Enhanced graph visualizations

### v3.0.0 (Future)
- [ ] LangGraph-based agentic architecture
- [ ] Real-time web intelligence integration
- [ ] Time-series fraud prediction
- [ ] Mobile app (React Native)

See [CHANGELOG.md](CHANGELOG.md) for details.

---

## Known Issues

- Neo4j connection sometimes fails on first startup (restart fixes it)
- Large PDF files (>10MB) may timeout on document verification
- Hinglish embeddings work best with mixed English-Hindi text

See [Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues) for tracking.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repo
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push branch (`git push origin feature/NewFeature`)
5. Open Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Author

**Pranay Mathur**
- GitHub: [@pranaya-mathur](https://github.com/pranaya-mathur)
- Email: pranaya.mathur@yahoo.com

---

## Acknowledgments

Thanks to Groq, LangChain, Neo4j, and CatBoost teams for their excellent tools.

---

## Support

- Email: pranaya.mathur@yahoo.com
- Issues: [GitHub Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues)
- Discussions: [GitHub Discussions](https://github.com/pranaya-mathur/ClaimLens_App/discussions)

---

*Built for the insurance industry*
