"""
Configuration settings for ClaimLens
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


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
    
    # Model Paths
    DAMAGE_DETECTOR_PATH: str = "models/cv_models/damage_detector.pt"
    FRAUD_MODEL_PATH: str = "models/ml_models/fraud_xgboost.pkl"
    SCALER_PATH: str = "models/ml_models/scaler.pkl"
    
    # Thresholds
    FRAUD_THRESHOLD_HIGH: float = 0.8
    FRAUD_THRESHOLD_MEDIUM: float = 0.5
    AUTO_APPROVE_THRESHOLD: float = 0.3
    
    # Processing
    MAX_IMAGE_SIZE_MB: int = 10
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()