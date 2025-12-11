"""
Configuration settings for ClaimLens
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "claimlens123"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # CV Model Paths (Trained Models)
    PARTS_MODEL_PATH: str = "models/parts_segmentation/yolo11n_best.pt"
    DAMAGE_MODEL_PATH: str = "models/damage_detection/yolo11m_best.pt"
    SEVERITY_MODEL_PATH: str = "models/severity_classification/efficientnet_b0_best.pth"
    
    # Legacy Model Paths (for backward compatibility)
    DAMAGE_DETECTOR_PATH: str = "models/cv_models/damage_detector.pt"
    FRAUD_MODEL_PATH: str = "models/ml_models/fraud_xgboost.pkl"
    SCALER_PATH: str = "models/ml_models/scaler.pkl"
    
    # CV Detection Thresholds
    PARTS_CONFIDENCE_THRESHOLD: float = 0.25
    DAMAGE_CONFIDENCE_THRESHOLD: float = 0.25
    SEVERITY_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Fraud Thresholds
    FRAUD_THRESHOLD_HIGH: float = 0.8
    FRAUD_THRESHOLD_MEDIUM: float = 0.5
    AUTO_APPROVE_THRESHOLD: float = 0.3
    
    # Processing
    MAX_IMAGE_SIZE_MB: int = 10
    BATCH_SIZE: int = 32
    CV_DEVICE: str = "cuda"  # or "cpu"
    
    # LLM Configuration (Optional)
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    
    # Logging (Optional)
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/claimlens.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache()
def get_settings() -> Settings:
    return Settings()
