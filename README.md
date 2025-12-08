# ClaimLens AI - Clean Production Repo

This repo is a cleaner, well-structured production starter for ClaimLens AI evaluator.

Structure:
- src/app/config.py       # configuration loader (.env)
- src/app/models.py       # pydantic models
- src/app/evaluator.py    # robust evaluator (LLM wrapper + fallback + parsing)
- src/app/api.py          # FastAPI application (endpoints)
- data/sample5.jsonl      # small sample dataset
- Dockerfile.api + docker-compose.api.yml for running the API

Defaults: USE_GROQ=False to avoid token usage during local dev.

Run locally (venv):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.app.api:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs to test the evaluate endpoint.
