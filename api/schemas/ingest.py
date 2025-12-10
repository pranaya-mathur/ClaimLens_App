"""
Pydantic schemas for live claim ingestion
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date, datetime


class DocumentMetadata(BaseModel):
    """Document metadata"""
    doc_id: str = Field(..., description="Unique document identifier")
    doc_hash: str = Field(..., description="Document hash for duplicate detection")
    doc_type: str = Field(..., description="Type of document (invoice, estimate, police_report, etc.)")
    
    class Config:
        schema_extra = {
            "example": {
                "doc_id": "DOC_12345",
                "doc_hash": "abc123def456",
                "doc_type": "invoice"
            }
        }


class LocationInfo(BaseModel):
    """Location/City information"""
    city_name: str = Field(..., description="City name")
    state: str = Field(..., description="State name")
    
    class Config:
        schema_extra = {
            "example": {
                "city_name": "Mumbai",
                "state": "Maharashtra"
            }
        }


class PolicyInfo(BaseModel):
    """Policy metadata"""
    policy_number: str = Field(..., description="Policy number")
    product_type: str = Field(..., description="Product type (motor, health, property, etc.)")
    sum_insured: float = Field(..., gt=0, description="Sum insured amount")
    start_date: date = Field(..., description="Policy start date")
    
    class Config:
        schema_extra = {
            "example": {
                "policy_number": "POL_45678",
                "product_type": "motor",
                "sum_insured": 500000,
                "start_date": "2024-06-15"
            }
        }


class ClaimantInfo(BaseModel):
    """Claimant information"""
    claimant_id: str = Field(..., description="Unique claimant identifier")
    name: str = Field(..., min_length=2, description="Claimant full name")
    phone: str = Field(..., description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    city: str = Field(..., description="Claimant city")
    fraud_history: Optional[int] = Field(0, ge=0, description="Number of past fraud claims")
    
    class Config:
        schema_extra = {
            "example": {
                "claimant_id": "CLMT_98765",
                "name": "Rahul Sharma",
                "phone": "+919876543210",
                "email": "rahul@example.com",
                "city": "Mumbai",
                "fraud_history": 0
            }
        }


class ClaimIngestRequest(BaseModel):
    """Complete claim ingestion request"""
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_amount: float = Field(..., gt=0, description="Claim amount")
    claim_date: date = Field(..., description="Claim filing date")
    incident_date: date = Field(..., description="Incident date")
    product_type: str = Field(..., description="Product type")
    subtype: Optional[str] = Field(None, description="Claim subtype")
    
    claimant: ClaimantInfo = Field(..., description="Claimant details")
    policy: PolicyInfo = Field(..., description="Policy details")
    location: LocationInfo = Field(..., description="Location details")
    documents: List[DocumentMetadata] = Field(default=[], description="Associated documents")
    
    @validator('claim_date')
    def claim_date_not_future(cls, v):
        if v > date.today():
            raise ValueError('Claim date cannot be in the future')
        return v
    
    @validator('incident_date')
    def incident_date_valid(cls, v, values):
        if v > date.today():
            raise ValueError('Incident date cannot be in the future')
        if 'claim_date' in values and v > values['claim_date']:
            raise ValueError('Incident date cannot be after claim date')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "claim_id": "CLM_2025_123456",
                "claim_amount": 45000,
                "claim_date": "2025-12-10",
                "incident_date": "2025-12-05",
                "product_type": "motor",
                "subtype": "collision",
                "claimant": {
                    "claimant_id": "CLMT_98765",
                    "name": "Rahul Sharma",
                    "phone": "+919876543210",
                    "city": "Mumbai"
                },
                "policy": {
                    "policy_number": "POL_45678",
                    "product_type": "motor",
                    "sum_insured": 500000,
                    "start_date": "2024-06-15"
                },
                "location": {
                    "city_name": "Mumbai",
                    "state": "Maharashtra"
                },
                "documents": [
                    {
                        "doc_id": "DOC_111",
                        "doc_hash": "abc123def456",
                        "doc_type": "invoice"
                    }
                ]
            }
        }


class ClaimIngestResponse(BaseModel):
    """Response for claim ingestion"""
    status: str = Field(..., description="Status (success/error)")
    claim_id: str = Field(..., description="Claim ID")
    graph_status: str = Field(..., description="Graph load status")
    nodes_created: int = Field(..., description="Number of nodes created")
    relationships_created: int = Field(..., description="Number of relationships created")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "claim_id": "CLM_2025_123456",
                "graph_status": "loaded",
                "nodes_created": 5,
                "relationships_created": 4,
                "timestamp": "2025-12-10T09:10:00Z",
                "message": "Claim successfully ingested into fraud graph"
            }
        }
