# app/config.py
"""Configuración centralizada del servicio."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración centralizada del servicio."""
    
    # Servicio
    SERVICE_NAME: str = "clustering-service-aura"
    SERVICE_PORT: int = 8001
    DEBUG: bool = False
    
    # Base de Datos Analítica (Target)
    DATABASE_URL_ANALYTICS: str
    
    # Bases de Datos Fuente (Source)
    DATABASE_URL_AUTH: str
    DATABASE_URL_SOCIAL: str
    DATABASE_URL_MESSAGING: str
    
    # NLP
    NLP_MODEL_NAME: str = "UMUTeam/roberta-spanish-sentiment-analysis"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton para obtener la configuración."""
    return Settings()


settings = get_settings()
