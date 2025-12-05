# app/database/__init__.py
"""Database module - connections and models."""

from app.database.connection import (
    analytics_engine,
    auth_engine,
    social_engine,
    messaging_engine,
    get_analytics_session
)
from app.database.models import Base, UserFeatureVector

__all__ = [
    "analytics_engine",
    "auth_engine", 
    "social_engine",
    "messaging_engine",
    "get_analytics_session",
    "Base",
    "UserFeatureVector"
]
