# src/app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # loads .env from repo root if present (do NOT commit your .env)

class Settings:
    USE_GROQ = os.getenv("USE_GROQ", "False").lower() in ("1", "true", "yes")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    RUN_COUNT = int(os.getenv("RUN_COUNT", "5"))
    GROQ_MAX_ATTEMPTS = int(os.getenv("GROQ_MAX_ATTEMPTS", "1"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "6.0"))
    DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
    PARTIAL_JSONL = DATA_DIR / "eval_results_partial.jsonl"
    FINAL_JSONL = DATA_DIR / "eval_results_final.jsonl"
    SUMMARY_CSV = DATA_DIR / "eval_summary.csv"

settings = Settings()
