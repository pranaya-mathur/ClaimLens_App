# 🚀 ClaimLens Setup Guide

Complete setup instructions for the Fraud Graph Engine.

## Quick Start (5 Minutes)

### Step 1: Clone & Install
```bash
git clone https://github.com/pranaya-mathur/ClaimLens_App
cd ClaimLens_App
git checkout fresh-fraud-engine

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Services
```bash
docker-compose up -d
```

### Step 3: Prepare & Load Data
```bash
# Place CSVs in data/raw/
python scripts/01_data_preparation.py
python scripts/02_load_graph.py
```

### Step 4: Run Application
```bash
# Terminal 1
uvicorn api.main:app --reload

# Terminal 2  
streamlit run frontend/streamlit_app.py
```

## Access
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Neo4j: http://localhost:7474 (neo4j/claimlens123)

See full guide at: SETUP_GUIDE.md