# app/database/models.py
"""Modelos SQLAlchemy para la base de datos analítica."""

from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class UserFeatureVector(Base):
    """
    Modelo de la tabla de vectores de características para clustering.
    Cada registro representa un usuario con sus KPIs normalizados.
    """
    __tablename__ = "user_feature_vector"
    
    # === Identificadores ===
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_raiz = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    extraction_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # === KPIs Normalizados (Feature Vector) ===
    # KPI 1: Ratio de Reciprocidad Social (normalizado 0-1)
    reciprocity_ratio_norm = Column(Float, nullable=True, default=0.0)
    
    # KPI 2: Días desde última conexión (normalizado 0-1)
    days_since_last_seen_norm = Column(Float, nullable=True, default=0.0)
    
    # KPI 3: Ratio de mensajes nocturnos (ya es ratio 0-1)
    ratio_night_messages = Column(Float, nullable=True, default=0.0)
    
    # KPI 4: Perfil incompleto (binario)
    is_profile_incomplete = Column(Boolean, nullable=True, default=False)
    
    # KPI 5: Índice de negatividad NLP (0-1)
    sentiment_negativity_index = Column(Float, nullable=True, default=0.0)
    
    # KPI 6: Densidad de participación comunitaria (normalizado)
    num_community_categories_norm = Column(Float, nullable=True, default=0.0)
    
    # === Variables One-Hot (Ejemplos) ===
    int_gaming_one_hot = Column(Boolean, default=False)
    comm_voluntariado_one_hot = Column(Boolean, default=False)
    
    # === Resultado del Clustering ===
    cluster_label = Column(String(50), nullable=True)
    
    # === Índices compuestos para consultas eficientes ===
    __table_args__ = (
        Index('idx_extraction_cluster', 'extraction_date', 'cluster_label'),
    )
    
    def __repr__(self):
        return f"<UserFeatureVector(user_id={self.user_id_raiz}, cluster={self.cluster_label})>"
