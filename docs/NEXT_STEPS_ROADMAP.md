# ClaimLens Development Roadmap - Next Steps

**Last Updated:** December 11, 2025, 5:50 PM IST  
**Current Status:** Forgery Detection Module Integrated ✅

---

## 📊 Current Status Summary

### ✅ Completed Features (As of Dec 11, 2025)

1. **Vehicle Damage Detection Pipeline**
   - Parts Segmentation: YOLO11n-seg (23 car parts)
   - Damage Detection: YOLO11m (6 damage types)
   - Severity Classification: EfficientNet-B0 (3 levels)
   - API Endpoint: `/api/cv/detect`

2. **Forgery Detection System**
   - ResNet50 CNN Classifier (83.6% validation accuracy)
   - Error Level Analysis (ELA) for compression artifacts
   - Noise Variation Analysis for spliced regions
   - API Endpoints: `/api/cv/detect-forgery`, `/api/cv/analyze-complete`

3. **Fraud Graph Engine**
   - Neo4j integration for network analysis
   - Fraud ring detection
   - Document reuse patterns

4. **API Infrastructure**
   - FastAPI with 5 CV endpoints
   - Health checks and monitoring
   - Interactive documentation (Swagger)
   - Complete testing guide

---

## 🎯 Strategic Decision: Focus on Advanced Forgery Features

**Decision Made:** Option C - Defer image type classification  
**Rationale:**
- Complete forgery detection suite before moving to other modules
- Build comprehensive fraud detection capabilities
- Image classification can be added later when document pipeline is ready

---

## 🚀 Phase 3: Complete Forgery Detection Suite

### Priority 1: Duplicate Image Detection ⭐ (HIGH PRIORITY)
**Timeline:** December 12-13, 2025 (2 days)  
**Estimated Effort:** 8-12 hours

#### Objectives:
- Detect if same photo was submitted across multiple claims
- Identify fraud rings using shared images
- Integrate with Neo4j fraud graph for pattern analysis

#### Technical Approach:
```python
Methods:
1. Perceptual Hashing (pHash/dHash)
   - Fast, rotation-invariant
   - Hamming distance for similarity
   
2. Deep Feature Embeddings (Optional)
   - ResNet/MobileNet features
   - Cosine similarity
   
3. Neo4j Integration
   - Store image hashes in graph DB
   - Link claims with shared images
```

#### Deliverables:
- [ ] `src/cv_engine/duplicate_detector.py`
- [ ] API endpoint: `POST /api/cv/check-duplicate`
- [ ] Neo4j schema update for image hashes
- [ ] Test suite with sample duplicate images
- [ ] Documentation and usage examples

#### Expected Output:
```json
{
  "is_duplicate": true,
  "similarity_score": 0.95,
  "matching_claims": [
    {
      "claim_id": "CLM-2024-1234",
      "timestamp": "2024-11-20",
      "similarity": 0.95
    }
  ],
  "recommendation": "REJECT - Image reused from previous claim"
}
```

---

### Priority 2: Metadata Analysis ⭐ (QUICK WIN)
**Timeline:** December 14-15, 2025 (2 days)  
**Estimated Effort:** 4-6 hours

#### Objectives:
- Extract and verify EXIF metadata from images
- Detect metadata tampering or removal
- Validate GPS coordinates, timestamps, camera info

#### Technical Approach:
```python
Libraries:
- Pillow (PIL) for basic EXIF
- exifread for detailed metadata
- GPSPhoto for GPS validation

Checks:
1. Metadata presence (suspicious if stripped)
2. GPS coordinates (match claim location?)
3. Timestamp consistency (match claim date?)
4. Camera model (consistent across claim photos?)
5. Photo editing software traces
```

#### Deliverables:
- [ ] `src/cv_engine/metadata_analyzer.py`
- [ ] API endpoint: `POST /api/cv/analyze-metadata`
- [ ] Risk scoring based on metadata inconsistencies
- [ ] Integration with complete analysis endpoint
- [ ] Documentation with examples

#### Expected Output:
```json
{
  "metadata_present": true,
  "gps_coordinates": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "location_match": true
  },
  "timestamp": "2024-12-10 14:30:22",
  "camera_info": {
    "make": "Apple",
    "model": "iPhone 13",
    "software": "iOS 17.1"
  },
  "editing_traces": ["Adobe Photoshop"],
  "risk_flags": [
    "Image edited with Photoshop",
    "GPS coordinates 50km from claim location"
  ],
  "recommendation": "REVIEW - Metadata inconsistencies detected"
}
```

---

### Priority 3: Unified Forgery API Endpoint
**Timeline:** December 16, 2025 (1 day)  
**Estimated Effort:** 3-4 hours

#### Objectives:
- Create single endpoint combining all forgery checks
- Unified risk scoring across all forgery signals
- Clean API response structure

#### Endpoint Design:
```python
POST /api/cv/forgery-complete

Input:
- image: uploaded file
- check_duplicate: bool (default true)
- check_metadata: bool (default true)
- check_manipulation: bool (default true)

Output:
{
  "forgery_analysis": {
    "manipulation_check": {...},  # CNN + ELA + Noise
    "duplicate_check": {...},      # pHash similarity
    "metadata_check": {...}        # EXIF analysis
  },
  "unified_risk": {
    "risk_score": 0.85,
    "risk_level": "HIGH",
    "risk_factors": [...],
    "recommendation": "REJECT"
  }
}
```

#### Deliverables:
- [ ] Update `api/routes/cv_detection.py`
- [ ] Risk fusion logic across all forgery signals
- [ ] Updated API documentation
- [ ] Postman collection update

---

### Priority 4: Testing & Documentation
**Timeline:** December 17-18, 2025 (2 days)  
**Estimated Effort:** 6-8 hours

#### Tasks:
- [ ] End-to-end integration tests
- [ ] Performance benchmarking
- [ ] Create test dataset (authentic vs forged)
- [ ] Update README with all forgery features
- [ ] Create video demo for portfolio
- [ ] Update API testing guide

---

## 🎯 Phase 4: ML Risk Scoring Engine

**Timeline:** December 19-25, 2025 (1 week)  
**Status:** Not Started

### Objectives:
- Build XGBoost/CatBoost fraud classifier
- Combine CV signals with claim metadata
- Achieve 90%+ AUC on fraud detection

### Feature Engineering:
```python
Features to Include:
1. CV Signals:
   - Forgery probability
   - Duplicate score
   - Metadata risk
   - Damage severity
   - Number of damages

2. Claim Metadata:
   - Time since policy start
   - Claim amount
   - Claimant history
   - Policy type
   
3. Graph Features:
   - Fraud ring membership
   - Shared document count
   - Network centrality
```

### Deliverables:
- [ ] `src/ml_engine/fraud_classifier.py`
- [ ] Feature extraction pipeline
- [ ] Model training scripts
- [ ] API endpoint: `POST /api/ml/score-fraud`
- [ ] Model performance report

---

## 📅 2-Week Sprint Plan

### Week 1: Complete Forgery Module
```
Dec 12 (Thu): Duplicate Detection - Implementation
Dec 13 (Fri): Duplicate Detection - Neo4j Integration
Dec 14 (Sat): Metadata Analysis - Implementation
Dec 15 (Sun): Metadata Analysis - API Integration
Dec 16 (Mon): Unified Forgery Endpoint
Dec 17 (Tue): Testing & Bug Fixes
Dec 18 (Wed): Documentation & Portfolio Update
```

### Week 2: ML Risk Scoring
```
Dec 19 (Thu): Feature Engineering
Dec 20 (Fri): Dataset Preparation
Dec 21 (Sat): Model Training (XGBoost)
Dec 22 (Sun): Model Evaluation & Tuning
Dec 23 (Mon): API Integration
Dec 24 (Tue): End-to-End Testing
Dec 25 (Wed): Documentation & Demo Video
```

---

## 🎯 Deferred Features (Phase 5+)

### Image Type Classification
**When to Add:** After document processing pipeline is ready  
**Estimated Time:** 1 week

**Approach:**
- Train MobileNetV3 classifier
- Classes: vehicle, document, invoice, ID card, other
- Auto-route to correct pipeline

### Multi-Image Consistency Check
**When to Add:** After basic forgery suite is complete  
**Estimated Time:** 2-3 days

**Checks:**
- Lighting consistency across claim photos
- Weather conditions match
- Temporal consistency (timestamps)
- Same vehicle in all photos

### GAN-Generated Image Detection
**When to Add:** Research phase, after MVP complete  
**Estimated Time:** 3-5 days

**Approach:**
- Frequency domain analysis
- Specialized GAN detector models
- Integration with existing forgery pipeline

### Document Processing Pipeline
**When to Add:** Phase 6 (January 2026)  
**Estimated Time:** 2 weeks

**Features:**
- OCR with Tesseract/PaddleOCR
- Invoice validation
- Policy document verification
- ID card authenticity check

---

## 📦 Model Files Checklist

### Already Available:
- [x] `forgery_detector_latest_run.pth` (ResNet50 weights)
- [x] `forgery_detector_latest_run_config.json` (config)

### Need to Place in `models/` Directory:
- [ ] `yolo11n_best.pt` (parts segmentation)
- [ ] `yolo11m_best.pt` (damage detection)
- [ ] `efficientnet_b0_best.pth` (severity classifier)

### To Be Created:
- [ ] `fraud_xgboost.pkl` (ML risk scorer - Week 2)
- [ ] `scaler.pkl` (feature scaler - Week 2)

---

## 🎓 Learning Resources

### For Duplicate Detection:
- Perceptual Hashing: https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
- ImageHash library: https://github.com/JohannesBuchner/imagehash

### For Metadata Analysis:
- EXIF documentation: https://exiftool.org/
- Python exifread: https://pypi.org/project/ExifRead/

### For ML Risk Scoring:
- XGBoost docs: https://xgboost.readthedocs.io/
- Feature engineering: https://www.kaggle.com/learn/feature-engineering

---

## 🎯 Success Metrics

### By December 18, 2025:
- [ ] Duplicate detection working with >95% accuracy
- [ ] Metadata analysis extracting all key fields
- [ ] Unified forgery API returning consistent results
- [ ] Complete test suite with 20+ test cases
- [ ] Updated documentation and demos

### By December 25, 2025:
- [ ] ML fraud classifier trained with >85% AUC
- [ ] End-to-end fraud detection pipeline functional
- [ ] API handling 10+ requests/second
- [ ] Complete portfolio project with video demo

---

## 💼 Portfolio Impact

### Current State (Dec 11):
```
ClaimLens - 3 AI Systems
├─ Damage Detection ✅
├─ Basic Forgery Detection ✅
└─ Fraud Graph ✅
```

### After Week 1 (Dec 18):
```
ClaimLens - 3 Complete AI Systems
├─ Damage Detection ✅
├─ Complete Forgery Suite ✅✅
│   ├─ Manipulation Detection
│   ├─ Duplicate Detection
│   └─ Metadata Analysis
└─ Fraud Graph ✅
```

### After Week 2 (Dec 25):
```
ClaimLens - 4 Complete AI Systems
├─ Damage Detection ✅
├─ Complete Forgery Suite ✅
├─ ML Risk Scoring ✅ (NEW!)
└─ Fraud Graph ✅
```

---

## 📞 Daily Standup Template

### What I Did Yesterday:
- 

### What I'm Doing Today:
- 

### Blockers:
- 

### Notes:
- 

---

## 🔗 Quick Links

- **GitHub Repo:** https://github.com/pranaya-mathur/ClaimLens_App
- **API Docs:** http://localhost:8000/docs
- **Testing Guide:** `docs/API_TESTING_GUIDE.md`
- **Project README:** `README.md`

---

## 📝 Notes

### Decision Log:
- **Dec 11, 2025:** Decided to defer image type classification (Option C)
- **Dec 11, 2025:** Prioritize advanced forgery features over ML scoring
- **Dec 11, 2025:** Keep separate API endpoints for vehicle vs document

### Future Considerations:
- Consider GPU optimization for faster inference
- Evaluate model quantization for deployment
- Plan for horizontal scaling with multiple workers
- Consider adding Redis caching for duplicate checks

---

**Last Updated:** December 11, 2025, 5:50 PM IST  
**Next Review:** December 18, 2025 (After Week 1 completion)

**Status:** 🚀 Ready to Start Phase 3!
