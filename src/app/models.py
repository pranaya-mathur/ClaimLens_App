from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

class ClaimRequest(BaseModel):
    claim_id: Optional[str] = None
    narrative: str
    metadata: Optional[Dict[str, Any]] = {}

class CombinedOut(BaseModel):
    clarity: int = Field(..., ge=0, le=10)
    clarity_explanation: str
    completeness: int = Field(..., ge=0, le=10)
    completeness_explanation: str
    timeline_consistency: int = Field(..., ge=0, le=10)
    timeline_explanation: str
    fraud_risk: float = Field(..., ge=0.0, le=1.0)
    red_flags: List[str]
    fraud_explanation: str

class ClaimResponse(BaseModel):
    claim_id: Optional[str]
    outputs: Dict[str, Any]
    used_fallback: bool
    errors: List[str] = []
