# ClaimLens AI

AI-powered insurance fraud detection using ML, computer vision, graph analytics, and LLMs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is this?

ClaimLens analyzes insurance claims to detect fraud using multiple AI techniques:

- **ML Engine**: CatBoost classifier with 145 features
- **Computer Vision**: Document forgery detection (PAN, Aadhaar, licenses, etc.)
- **Graph Analytics**: Fraud network detection using Neo4j
- **LLM**: Natural language explanations powered by Groq

The system can process claims in English and Hinglish (Hindi-English mix).

## Quick Start

**Requirements:**
- Python 3.10+
- Docker (optional, for Neo4j)
- Groq API key (free at console.groq.com)

**Setup:**
```bash
git clone https://github.com/pranaya-mathur/ClaimLens_App.git
cd ClaimLens_App

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**Run:**
```bash
# Start backend
uvicorn api.main:app --reload

# Start frontend (new terminal)
streamlit run frontend/streamlit_app.py

# Optional: Neo4j for graph features
docker-compose up neo4j -d
```

Visit http://localhost:8501 for the UI or http://localhost:8000/docs for API docs.

## How it Works

The system processes claims through four engines:

1. **ML Engine** extracts 145 features (text embeddings, behavioral patterns, document flags) and runs CatBoost classification
2. **CV Engine** verifies document authenticity using ResNet50 + Error Level Analysis
3. **Graph Engine** checks for fraud networks (shared documents, suspicious connections)
4. **LLM Engine** aggregates results and generates explanations

All results are combined using semantic aggregation to produce a final verdict.

## API Examples

**Score a claim:**
```bash
POST /api/ml/score
{
  "claim_id": "CLM001",
  "narrative": "Car accident on highway",
  "claim_amount": 50000,
  "product": "motor",
  ...
}
```

**Verify a document:**
```bash
POST /api/documents/verify-pan
(upload PAN card image)
```

**Complete analysis:**
```bash
POST /api/unified/analyze-complete
(runs all engines)
```

See `/docs` folder for detailed API documentation.

## Features

- Multi-modal fraud detection
- Document forgery detection with OCR
- Fraud network visualization
- AI-generated explanations (technical & customer-friendly)
- Real-time processing (<2s per claim with caching)
- Hinglish support

## Project Structure

```
ClaimLens_App/
├── api/              # FastAPI backend
│   ├── routes/       # API endpoints
│   └── schemas/      # Pydantic models
├── src/              # Core engines
│   ├── ml_engine/    # CatBoost fraud scoring
│   ├── cv_engine/    # Document verification
│   ├── llm_engine/   # LLM explanations
│   └── fraud_engine/ # Graph analytics
├── frontend/         # Streamlit UI
├── models/           # Trained models
├── docs/             # Documentation
└── tests/            # Test suite
```

## Configuration

Key environment variables (see `.env.example`):

```bash
GROQ_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password

ENABLE_LLM_EXPLANATIONS=true
ML_THRESHOLD=0.5
```

## Performance

Based on testing with synthetic claims:

- ML inference: ~80ms
- Document analysis: 1-2s
- LLM explanations: 2-3s
- End-to-end (cached): <500ms

Rate limited to 100 requests/min by default.

## Known Issues

- Neo4j sometimes needs a restart on first launch
- Large PDFs (>10MB) may timeout
- Hinglish embeddings need mixed English-Hindi text to work well
- Generic document detector accuracy varies by document quality

See [Issues](https://github.com/pranaya-mathur/ClaimLens_App/issues) for more.

## TODO

- [ ] Add batch processing endpoint
- [ ] Support PDF documents
- [ ] Add more languages (Hindi, Tamil, Telugu)
- [ ] Improve graph visualizations
- [ ] Add model retraining pipeline

See full roadmap in [CHANGELOG.md](CHANGELOG.md).

## Contributing

PRs welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Submit a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE)

## Contact

Pranay Mathur
- GitHub: [@pranaya-mathur](https://github.com/pranaya-mathur)
- Email: pranaya.mathur@yahoo.com

Built with help from Groq, LangChain, Neo4j, and CatBoost.
