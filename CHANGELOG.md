# Changelog

All notable changes to ClaimLens AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2025-12-14

### ✨ Added
- **Generic Document Verification** tab in Streamlit dashboard
  - Support for Driving License, Passport, Voter ID
  - Support for Bank Statements, Hospital Bills, Death Certificates
  - Real-time forgery detection using ResNet50 + ELA analysis
- MIT License for open-source distribution
- CHANGELOG.md for version tracking
- Professional repository structure

### 🔧 Changed
- Cleaned up frontend directory (removed 5 old Streamlit files)
- Consolidated all working features into single `streamlit_app.py`
- Updated documentation structure

### 🐛 Fixed
- Generic Forgery Detector now accessible via UI (was backend-only)
- Frontend file confusion resolved

---

## [2.0.0] - 2025-12-13

### ✨ Added
- **Multi-Modal Fraud Detection System**
  - ML Engine: CatBoost-based fraud scoring
  - CV Engine: Document verification (PAN, Aadhaar)
  - Graph Engine: Neo4j network analysis
  - LLM Engine: Groq-powered explainable AI
- **Streamlit Dashboard** with 4 tabs:
  - AI-Powered Claim Analysis
  - Analytics Dashboard
  - Fraud Network Analysis
  - Document Upload & Verification
- **FastAPI Backend** with REST API endpoints
- **Hinglish Support** for narratives and explanations
- **Adaptive Risk Weighting** based on component confidence
- **Critical Flags Detection** for high-risk indicators

### 🔧 Changed
- Migrated from monolithic to modular engine architecture
- Enhanced ML model with PCA embeddings (768 → 100 dims)
- Improved document verification with 3-model ensemble

### 🐛 Fixed
- **Critical Bug #1**: Fixed ML embedding model (wrong dimensions)
- **Critical Bug #2**: Fixed CV model file extensions (.pth → .pt)
- **Critical Bug #3**: Added .cbm to .gitignore
- Feature alignment between ML training and inference
- Flat fraud predictions (was ~3% for all claims)

---

## [1.0.0] - 2025-11-30

### ✨ Added
- Initial release of ClaimLens AI
- Basic fraud detection using CatBoost
- Simple document verification
- Command-line interface
- Docker support

---

## Version History Summary

| Version | Release Date | Highlights |
|---------|-------------|------------|
| **2.1.0** | 2025-12-14 | Generic document verification, repo cleanup |
| **2.0.0** | 2025-12-13 | Multi-modal system, Streamlit UI, critical fixes |
| **1.0.0** | 2025-11-30 | Initial release |

---

## Upcoming Features (Roadmap)

### v2.2.0 (Planned)
- [ ] Batch processing API
- [ ] PDF document support
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Enhanced graph visualizations
- [ ] Model performance monitoring dashboard

### v3.0.0 (Future)
- [ ] Real-time fraud detection
- [ ] Mobile app integration
- [ ] Advanced ML models (ensemble methods)
- [ ] Automated retraining pipeline
- [ ] Cloud deployment (AWS/Azure)

---

## Support

For issues and feature requests, visit:
- GitHub Issues: https://github.com/pranaya-mathur/ClaimLens_App/issues
- Email: pranaya.mathur@yahoo.com

---

**Last Updated:** December 14, 2025
