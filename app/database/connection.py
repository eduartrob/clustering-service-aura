# app/database/connection.py
"""Conexiones SQLAlchemy a las bases de datos Source y Target."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings

# ===== CONEXIÓN A LA DB ANALÍTICA (Target) =====
analytics_engine = create_engine(
    settings.DATABASE_URL_ANALYTICS,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)
AnalyticsSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=analytics_engine
)


# ===== CONEXIONES A LAS DBs FUENTE (Source) =====
auth_engine = create_engine(settings.DATABASE_URL_AUTH, pool_pre_ping=True)
social_engine = create_engine(settings.DATABASE_URL_SOCIAL, pool_pre_ping=True)
messaging_engine = create_engine(settings.DATABASE_URL_MESSAGING, pool_pre_ping=True)


def get_analytics_session() -> Session:
    """Dependencia para obtener sesión de DB Analítica."""
    db = AnalyticsSessionLocal()
    try:
        yield db
    finally:
        db.close()
