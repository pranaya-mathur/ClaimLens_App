# 🤖 ML Engine - Fraud Detection with Hinglish NLP

**CatBoost-based fraud scoring for Indian insurance claims**

---

## 🎯 Overview

The ML Engine provides production-ready fraud detection using:

- **CatBoost Classifier**: 84.8% AUC, 42.8% F1 score on 50K claims
- **Hinglish Embeddings**: `AkshitaS/bhasha-embed-v0` with PCA (100 dims)
- **Behavioral Features**: Time-aware claimant/policy history
- **Leakage-Free Design**: Strict validation to prevent label contamination

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.8480 |
| **F1 Score** | 0.4282 @ threshold 0.5 |
| **Precision** | 0.428 @ threshold 0.5 |
| **Recall** | 0.428 @ threshold 0.5 |
| **False Positive Rate** | 17.5% @ threshold 0.5 |
| **Training Data** | 50,000 Hinglish claims |
| **Features** | 145 (100 embeddings + 45 engineered) |

---

## 🏗️ Architecture

```
Raw Claim
    ↓
[FeatureEngineer]
    ├─ Hinglish Embeddings (Bhasha-Embed)
    ├─ Time-aware Aggregations
    ├─ Document Indicators
    └─ Categorical Encoding
    ↓
[145 Features]
    ↓
[MLFraudScorer - CatBoost]
    ↓
Fraud Probability (0-1)
```

---

## 🚀 Quick Start

### 1. Feature Engineering

```python
from src.ml_engine import FeatureEngineer
import pandas as pd

# Load raw claims
df = pd.read_csv("claims.csv")

# Initialize feature engineer
fe = FeatureEngineer(pca_dims=100)

# Engineer features
features = fe.engineer_features(
    df,
    narrative_col="narrative",
    drop_ids=True,
)

# Validate no leakage
fe.validate_no_leakage(features)

print(f"Features shape: {features.shape}")
```

### 2. Fraud Scoring

```python
from src.ml_engine import MLFraudScorer

# Load trained model
scorer = MLFraudScorer(
    model_path="models/claimlens_catboost_hinglish.cbm",
    metadata_path="models/claimlens_model_metadata.json",
    threshold=0.5,
)

# Score single claim
fraud_prob = scorer.score_claim(features.iloc[[0]], return_details=False)
print(f"Fraud probability: {fraud_prob:.2%}")

# Get detailed explanation
details = scorer.score_claim(features.iloc[[0]], return_details=True)
print(details)
```

### 3. Batch Scoring

```python
# Score multiple claims
results = scorer.score_batch(features, return_dataframe=True)
print(results.head())

# Output:
#    fraud_probability  fraud_prediction  risk_level
# 0              0.234                 0         LOW
# 1              0.678                 1        HIGH
# 2              0.123                 0         LOW
```

---

## 📁 Module Structure

```
src/ml_engine/
├── __init__.py              # Module exports
├── feature_engineer.py      # Feature engineering pipeline
├── ml_scorer.py            # CatBoost fraud scorer
└── README.md               # This file
```

---

## 🔧 API Reference

### FeatureEngineer

**Class: `FeatureEngineer(pca_dims=100, model_name="AkshitaS/bhasha-embed-v0")`**

#### Methods

**`engineer_features(df, narrative_col='narrative', drop_ids=True)`**
- Complete feature engineering pipeline
- Returns: Feature DataFrame with embeddings + engineered features

**`create_embeddings(narratives, batch_size=64)`**
- Generate Hinglish embeddings with PCA
- Returns: PCA-reduced embeddings (n_samples, pca_dims)

**`validate_no_leakage(feature_df)`**
- Validate no label-leakage columns exist
- Raises ValueError if leakage detected

#### Feature Categories

1. **Time-aware Aggregations** (9 features)
   - Claimant claim count/history
   - Policy claim patterns
   - Days since last claim
   - Rapid claims indicator

2. **Numeric Features** (10 features)
   - Log-transformed amounts
   - Policy age in months
   - Recent policy indicator

3. **Document Features** (5 features)
   - Number of documents
   - FIR presence
   - Photo presence
   - Death certificate presence
   - Discharge summary presence

4. **Categorical Encodings** (21 features)
   - Product type (motor, health, life, property)
   - City (Mumbai, Delhi, Bangalore, etc.)
   - Claim subtype

5. **Narrative Embeddings** (100 features)
   - PCA-reduced Bhasha-Embed vectors
   - Captures Hinglish semantic meaning

---

### MLFraudScorer

**Class: `MLFraudScorer(model_path=None, metadata_path=None, threshold=0.5)`**

#### Methods

**`score_claim(features, return_details=False)`**
- Score single claim for fraud risk
- Returns: float (fraud probability) or detailed dict

**`score_batch(features, return_dataframe=True)`**
- Score multiple claims
- Returns: DataFrame with scores or numpy array

**`predict_proba(features)`**
- Get fraud probabilities
- Returns: numpy array of probabilities (0-1)

**`predict(features)`**
- Get binary fraud predictions
- Returns: numpy array (0=legitimate, 1=fraud)

**`explain_prediction(features, top_n=10)`**
- Explain fraud prediction with top features
- Returns: dict with prediction + feature contributions

**`get_feature_importance(top_n=20)`**
- Get top N most important features
- Returns: DataFrame with feature names + importance

**`analyze_threshold(y_true, y_proba, thresholds=None)`**
- Analyze performance across thresholds
- Returns: DataFrame with precision/recall/F1 for each threshold

**`update_threshold(new_threshold)`**
- Update fraud classification threshold

---

## 📈 Usage Examples

### End-to-End Workflow

```python
import pandas as pd
from src.ml_engine import FeatureEngineer, MLFraudScorer

# 1. Load raw claims
df = pd.read_csv("claimlens_robust_dataset_50k_hinglish.csv")

# 2. Feature engineering
fe = FeatureEngineer(pca_dims=100)
features = fe.engineer_features(df, narrative_col="narrative")
fe.validate_no_leakage(features)

# 3. Load trained model
scorer = MLFraudScorer(
    model_path="models/claimlens_catboost_hinglish.cbm",
    metadata_path="models/claimlens_model_metadata.json",
)

# 4. Score claims
results = scorer.score_batch(features)

# 5. Filter high-risk claims
high_risk = results[results["risk_level"].isin(["HIGH", "CRITICAL"])]
print(f"High-risk claims: {len(high_risk)}")
```

### Threshold Optimization

```python
# Load validation data
y_true = df["fraud_label"].values
features = fe.engineer_features(df.drop(columns=["fraud_label"]))
y_proba = scorer.predict_proba(features)

# Analyze thresholds
threshold_analysis = scorer.analyze_threshold(
    y_true, y_proba,
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
)

print(threshold_analysis)

# Update to optimal threshold
scorer.update_threshold(0.6)  # Example: prioritize precision
```

### Feature Importance Analysis

```python
# Get top contributing features
top_features = scorer.get_feature_importance(top_n=20)
print(top_features)

# Explain single prediction
explanation = scorer.explain_prediction(features.iloc[[100]])
print(f"Fraud probability: {explanation['fraud_probability']:.2%}")
print("\nTop contributing features:")
for feat in explanation['top_contributing_features'][:5]:
    print(f"  - {feat['feature']}: {feat['value']} (importance: {feat['importance']:.3f})")
```

---

## 🔒 Leakage Prevention

The ML Engine implements strict leakage prevention:

### Forbidden Columns
- Any column containing: `fraud`, `score`, `redflag`, `flag`
- Target variable: `fraud_label`, `is_fraud`

### Validation Process
```python
# Automatic validation
fe.validate_no_leakage(features)

# Manual check
forbidden = ["fraud", "score", "redflag"]
for col in features.columns:
    if any(sub in col.lower() for sub in forbidden):
        raise ValueError(f"Leakage detected: {col}")
```

---

## 🎓 Training Pipeline

To retrain the model:

1. **Data Preparation**: Use `claimlens_robust_dataset_50k_hinglish.csv`
2. **Feature Engineering**: Run feature engineering notebook
3. **Model Training**: Execute CatBoost training script
4. **Validation**: Verify AUC > 0.80, F1 > 0.40
5. **Save Artifacts**:
   - `models/claimlens_catboost_hinglish.cbm`
   - `models/claimlens_model_metadata.json`
   - `models/claimlens_feature_importance.csv`

See [`notebooks/ClaimLens_ML_Engine.ipynb`](../../notebooks/) for complete training pipeline.

---

## 🐛 Troubleshooting

### Feature Mismatch Error
```python
# Error: Missing features
ValueError: Missing features: {'emb_99', 'policy_claim_count'}

# Solution: Ensure feature engineering matches training
fe = FeatureEngineer(pca_dims=100)  # Same as training
features = fe.engineer_features(df)
```

### Embedding Model Download
```python
# If Bhasha-Embed download fails
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("AkshitaS/bhasha-embed-v0", cache_folder="./models/cache")
```

### Threshold Tuning
```python
# Adjust for precision vs recall
scorer.update_threshold(0.6)  # Higher precision, lower recall
scorer.update_threshold(0.4)  # Higher recall, lower precision
```

---

## 📚 References

- **Bhasha-Embed Model**: [AkshitaS/bhasha-embed-v0](https://huggingface.co/AkshitaS/bhasha-embed-v0)
- **CatBoost Documentation**: [catboost.ai](https://catboost.ai/)
- **Training Notebook**: `notebooks/ClaimLens_ML_Engine.ipynb`
- **Dataset**: `data/claimlens_robust_dataset_50k_hinglish.csv`

---

## 🤝 Integration with Other Engines

The ML Engine works alongside:

- **CV Engine**: Vehicle damage + document verification (99%+ accuracy)
- **Fraud Graph**: Neo4j network analysis for fraud rings
- **Decision Engine**: Final fraud verdict with LLM explanations

```python
# Combined workflow
from src.cv_engine import DocumentVerifier
from src.ml_engine import MLFraudScorer
from src.fraud_engine import FraudGraphAnalyzer

# 1. Document verification
doc_result = doc_verifier.verify(aadhaar_image, doc_type="AADHAAR")

# 2. ML fraud score
ml_score = ml_scorer.score_claim(features)

# 3. Graph analysis
graph_risk = graph_analyzer.analyze_claim(claim_id)

# 4. Final decision
final_score = 0.4 * ml_score + 0.3 * graph_risk + 0.3 * (1 - doc_result.confidence)
```

---

**Built with ❤️ for Indian Insurance Fraud Detection**

**Model trained on**: Dec 12, 2025  
**Version**: 1.0.0  
**Status**: Production-ready ✅
