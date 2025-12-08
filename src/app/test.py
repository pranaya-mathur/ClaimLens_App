# debug_llm_call.py
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import os, json

MODEL = os.getenv("GROQ_MODEL","llama-3.3-70b-versatile")
API_KEY = os.getenv("GROQ_API_KEY")
assert API_KEY, "GROQ_API_KEY must be set in env or .env"

llm = ChatGroq(model=MODEL, temperature=0)

prompt = (
    "You are an insurance claim evaluator. RETURN ONLY JSON. Exact keys:\n"
    '{"clarity":int,"clarity_explanation":str,"completeness":int,"completeness_explanation":str,'
    '"timeline_consistency":int,"timeline_explanation":str,"fraud_risk":float,"red_flags":list,"fraud_explanation":str}\n\n'
    "Example narrative:\nKal raat MI Road, Jaipur - minor rear bumper scratch, FIR filed, photos uploaded.\n\n"
    "Return JSON only:"
)

resp = llm.invoke(prompt)
print("=== RAW RESPONSE ===")
try:
    print(type(resp))
    # Some SDKs return an object; print .content if present
    print(getattr(resp, "content", resp))
except Exception as e:
    print("Error printing response:", e)
