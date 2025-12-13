"""
Configuration settings for ClaimLens
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
from loguru import logger


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
    
    # ========================================
    # ML Engine Configuration (CatBoost Fraud Detection)
    # ========================================
    ML_MODEL_PATH: str = "models/ml_engine/claimlens_catboost_hinglish.cbm"
    ML_METADATA_PATH: str = "models/ml_engine/claimlens_model_metadata.json"
    ML_THRESHOLD: float = 0.5  # Fraud classification threshold
    ML_PCA_DIMS: int = 100  # PCA dimensions for narrative embeddings
    ML_EMBEDDING_MODEL: str = "l3cube-pune/indic-sentence-similarity-sbert"  # Hinglish embeddings
    
    # CV Model Paths - Damage Detection
    PARTS_MODEL_PATH: str = "models/parts_segmentation/yolo11n_best.pth"
    DAMAGE_MODEL_PATH: str = "models/damage_detection/yolo11m_best.pth"
    SEVERITY_MODEL_PATH: str = "models/severity_classification/efficientnet_b0_best.pth"
    
    # CV Model Paths - Forgery Detection
    FORGERY_MODEL_PATH: str = "models/forgery_detection/resnet50_finetuned.pth"
    FORGERY_CONFIG_PATH: str = "models/forgery_detection/forgery_config.json"
    
    # Legacy Model Paths (for backward compatibility)
    DAMAGE_DETECTOR_PATH: str = "models/cv_models/damage_detector.pt"
    FRAUD_MODEL_PATH: str = "models/ml_models/fraud_xgboost.pkl"
    SCALER_PATH: str = "models/ml_models/scaler.pkl"
    
    # CV Detection Thresholds
    PARTS_CONFIDENCE_THRESHOLD: float = 0.25
    DAMAGE_CONFIDENCE_THRESHOLD: float = 0.25
    SEVERITY_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Forgery Detection Thresholds
    FORGERY_THRESHOLD: float = 0.55  # Threshold for binary classification
    
    # Fraud Thresholds
    FRAUD_THRESHOLD_HIGH: float = 0.8
    FRAUD_THRESHOLD_MEDIUM: float = 0.5
    AUTO_APPROVE_THRESHOLD: float = 0.3
    
    # Processing
    MAX_IMAGE_SIZE_MB: int = 10
    BATCH_SIZE: int = 32
    CV_DEVICE: str = "cpu"  # Default to CPU, set to 'cuda' in .env if GPU available
    
    # LLM Configuration
    GROQ_API_KEY: Optional[str] = None  # Get from https://console.groq.com/
    OPENAI_API_KEY: Optional[str] = None
    
    # LLM Feature Flags
    ENABLE_SEMANTIC_AGGREGATION: bool = True
    ENABLE_LLM_EXPLANATIONS: bool = True
    
    # LLM Model Configuration
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    
    # Explanation Generator Configuration
    EXPLANATION_MODEL: str = "llama-3.3-70b-versatile"
    EXPLANATION_TEMPERATURE: float = 0.3
    EXPLANATION_MAX_TOKENS: int = 512
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/claimlens.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env
    
    def get_llm_enabled(self) -> bool:
        """
        Check if LLM semantic aggregation is enabled and configured.
        
        Returns:
            True if both flag and API key are set
        """
        return self.ENABLE_SEMANTIC_AGGREGATION and bool(self.GROQ_API_KEY)
    
    def get_llm_explanation_enabled(self) -> bool:
        """
        Check if LLM explanations are enabled and configured.
        
        Returns:
            True if both flag and API key are set
        """
        return self.ENABLE_LLM_EXPLANATIONS and bool(self.GROQ_API_KEY)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    
    # Validate LLM configuration on first load
    if settings.ENABLE_SEMANTIC_AGGREGATION or settings.ENABLE_LLM_EXPLANATIONS:
        if settings.GROQ_API_KEY:
            logger.success(f"✅ LLM configured: {settings.LLM_MODEL}")
            logger.success(f"✅ Semantic Aggregation: {settings.ENABLE_SEMANTIC_AGGREGATION}")
            logger.success(f"✅ LLM Explanations: {settings.ENABLE_LLM_EXPLANATIONS}")
        else:
            logger.warning(
                "⚠️ LLM features enabled but GROQ_API_KEY not set. "
                "Will use fallback logic. Get key from https://console.groq.com/"
            )
    
    return settings
