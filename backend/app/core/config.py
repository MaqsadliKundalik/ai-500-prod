"""
Sentinel-RX Application Settings
================================
Centralized configuration management using Pydantic Settings
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===========================================
    # APPLICATION
    # ===========================================
    app_name: str = "Sentinel-RX"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"
    disable_auth: bool = False  # Authentication enabled for production
    
    # ===========================================
    # SERVER
    # ===========================================
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    
    # ===========================================
    # DATABASE
    # ===========================================
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sentinel_rx"
    
    @property
    def sync_database_url(self) -> str:
        """Sync database URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "")
    
    # ===========================================
    # REDIS
    # ===========================================
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    
    # ===========================================
    # SECURITY
    # ===========================================
    secret_key: str = "your-super-secret-key-change-in-production-minimum-32-characters"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # ===========================================
    # CORS
    # ===========================================
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:19006"]
    cors_allow_credentials: bool = True
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    # ===========================================
    # AI/ML SERVICES
    # ===========================================
    openai_api_key: Optional[str] = None
    pill_recognition_model_path: str = "./models/pill_recognition.pt"
    drug_interaction_model_path: str = "./models/drug_interaction.pt"
    
    # ===========================================
    # EXTERNAL APIS
    # ===========================================
    openfda_api_url: str = "https://api.fda.gov/drug/"
    drugbank_api_url: str = "https://api.drugbank.com/v1/"
    google_maps_api_key: Optional[str] = None
    exchange_rate_api_key: Optional[str] = None
    
    # ===========================================
    # SMS VERIFICATION
    # ===========================================
    sms_api_url: Optional[str] = None
    sms_api_key: Optional[str] = None
    sms_sender_id: str = "SentinelRX"
    
    # ===========================================
    # MONITORING
    # ===========================================
    sentry_dsn: Optional[str] = None
    prometheus_enabled: bool = True
    
    # ===========================================
    # FILE STORAGE
    # ===========================================
    upload_dir: str = "./uploads"
    max_upload_size: int = 10485760  # 10MB
    
    # ===========================================
    # LOGGING
    # ===========================================
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
