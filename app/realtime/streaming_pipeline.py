# app/realtime/streaming_pipeline.py
"""
Pipeline de ETL incremental para procesamiento en tiempo real.
Procesa actualizaciones de usuarios individuales sin re-ejecutar todo el ETL.
"""

import asyncio
import pandas as pd
import numpy as np
from uuid import UUID
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy import text

from app.database.connection import (
    social_engine, 
    messaging_engine, 
    analytics_engine
)
from app.database.models import UserFeatureVector
from app.realtime.websocket_manager import (
    manager, 
    create_user_risk_message,
    create_distribution_message,
    create_critical_alert_message,
    MessageType
)


class StreamingETLPipeline:
    """
    Pipeline ETL incremental para procesamiento en tiempo real.
    Extrae, transforma y carga datos de un usuario específico.
    """
    
    # Columnas de features esperadas
    FEATURE_COLUMNS = [
        'reciprocity_ratio_norm',
        'days_since_last_seen_norm', 
        'ratio_night_messages',
        'is_profile_incomplete',
        'sentiment_negativity_index',
        'num_community_categories_norm'
    ]
    
    def __init__(self):
        self._ensemble = None
        self._last_distribution: Dict[str, int] = {}
    
    def set_ensemble(self, ensemble) -> None:
        """Configura el ensamble de clustering para predicciones."""
        self._ensemble = ensemble
    
    async def process_notification(self, notification: Dict[str, Any]) -> None:
        """
        Procesa una notificación de cambio de datos.
        
        Args:
            notification: Diccionario con source, channel, payload
        """
        payload = notification.get("payload", {})
        user_id = payload.get("user_id")
        table = payload.get("table", "unknown")
        
        if not user_id:
            print(f"⚠️ Notificación sin user_id: {notification}")
            return
        
        print(f"🔄 Procesando actualización: {table} -> {user_id}")
        
        try:
            # 1. Extraer métricas actualizadas del usuario
            user_data = await self._extract_user_metrics(user_id)
            
            if user_data is None:
                print(f"   ⚠️ No se encontraron datos para usuario {user_id}")
                return
            
            # 2. Transformar y calcular KPIs
            features = self._transform_user_data(user_data)
            
            # 3. Obtener riesgo anterior (si existe)
            previous_risk = await self._get_previous_risk(user_id)
            
            # 4. Predecir nuevo nivel de riesgo
            risk_result = self._predict_risk(features)
            
            # 5. Actualizar base de datos analítica
            await self._update_analytics_db(user_id, features, risk_result)
            
            # 6. Notificar a clientes WebSocket
            await self._broadcast_update(
                user_id=str(user_id),
                risk_result=risk_result,
                previous_risk=previous_risk,
                triggering_table=table
            )
            
            print(f"   ✅ Usuario {user_id}: {risk_result['risk_level']} (severity: {risk_result['severity_index']:.1f})")
            
        except Exception as e:
            print(f"   ❌ Error procesando usuario {user_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _extract_user_metrics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Extrae las métricas de un usuario específico desde las DBs fuente.
        """
        # Query para métricas sociales
        social_query = text("""
            SELECT 
                up.user_id AS user_id_raiz,
                up.id AS profile_id_social,
                up.followers_count,
                up.following_count,
                up.bio IS NULL AS profile_bio_missing,
                cp.user_id IS NULL AS complete_profile_missing,
                COALESCE(cm_stats.distinct_categories, 0) AS num_community_categories
            FROM user_profiles up
            LEFT JOIN complete_profiles cp ON up.id = cp.user_id
            LEFT JOIN (
                SELECT 
                    cm.user_id,
                    COUNT(DISTINCT c.category) AS distinct_categories
                FROM community_members cm
                INNER JOIN communities c ON cm.community_id = c.id
                GROUP BY cm.user_id
            ) cm_stats ON up.id = cm_stats.user_id
            WHERE up.user_id = :user_id AND up.is_active = TRUE
        """)
        
        # Query para métricas de mensajería
        messaging_query = text("""
            SELECT 
                u.profile_id,
                u.last_seen_at,
                COALESCE(msg_stats.total_messages, 0) AS total_messages,
                COALESCE(msg_stats.night_messages, 0) AS night_messages
            FROM users u
            LEFT JOIN (
                SELECT 
                    sender_profile_id,
                    COUNT(*) AS total_messages,
                    SUM(CASE 
                        WHEN EXTRACT(HOUR FROM created_at) BETWEEN 1 AND 5 THEN 1 
                        ELSE 0 
                    END) AS night_messages
                FROM messages
                WHERE is_deleted = FALSE
                GROUP BY sender_profile_id
            ) msg_stats ON u.id = msg_stats.sender_profile_id
            WHERE u.profile_id = :user_id AND u.is_active = TRUE
        """)
        
        try:
            # Extraer datos sociales
            with social_engine.connect() as conn:
                result = conn.execute(social_query, {"user_id": user_id})
                social_row = result.fetchone()
            
            if social_row is None:
                return None
            
            social_data = dict(social_row._mapping)
            
            # Extraer datos de mensajería
            with messaging_engine.connect() as conn:
                result = conn.execute(messaging_query, {"user_id": social_data.get("profile_id_social")})
                msg_row = result.fetchone()
            
            if msg_row:
                social_data.update(dict(msg_row._mapping))
            
            return social_data
            
        except Exception as e:
            print(f"Error extrayendo métricas: {e}")
            return None
    
    def _transform_user_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Transforma los datos crudos en KPIs normalizados.
        """
        now = datetime.utcnow()
        
        # KPI 1: Ratio de Reciprocidad
        followers = data.get('followers_count', 0) or 0
        following = data.get('following_count', 0) or 0
        reciprocity = following / followers if followers > 0 else 0.0
        
        # KPI 2: Días desde última conexión
        last_seen = data.get('last_seen_at')
        if last_seen:
            days_since = (now - last_seen).total_seconds() / 86400
        else:
            days_since = 365.0  # Si nunca se conectó
        
        # KPI 3: Ratio de mensajes nocturnos
        total_msg = data.get('total_messages', 0) or 0
        night_msg = data.get('night_messages', 0) or 0
        night_ratio = night_msg / total_msg if total_msg > 0 else 0.0
        
        # KPI 4: Perfil incompleto
        is_incomplete = (
            data.get('profile_bio_missing', True) and 
            data.get('complete_profile_missing', True)
        )
        
        # KPI 5: Índice de negatividad (por ahora 0, requiere NLP)
        # TODO: Implementar análisis NLP incremental
        negativity = 0.0
        
        # KPI 6: Categorías de comunidad
        community_cats = data.get('num_community_categories', 0) or 0
        
        # Normalización simple (en producción usar los scalers del ETL principal)
        return {
            'reciprocity_ratio_norm': min(reciprocity / 10.0, 1.0),  # Cap at 10
            'days_since_last_seen_norm': min(days_since / 365.0, 1.0),  # Cap at 1 year
            'ratio_night_messages': night_ratio,
            'is_profile_incomplete': 1.0 if is_incomplete else 0.0,
            'sentiment_negativity_index': negativity,
            'num_community_categories_norm': min(community_cats / 12.0, 1.0)  # Cap at 12 categories
        }
    
    def _predict_risk(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predice el nivel de riesgo basado en las features.
        Usa una versión simplificada si el ensamble no está cargado.
        """
        # Si tenemos el ensamble completo, usarlo
        if self._ensemble is not None and hasattr(self._ensemble, 'predict_single'):
            return self._ensemble.predict_single(features)
        
        # Predicción heurística simple como fallback
        risk_score = (
            features['days_since_last_seen_norm'] * 0.25 +
            features['ratio_night_messages'] * 0.20 +
            features['is_profile_incomplete'] * 0.15 +
            features['sentiment_negativity_index'] * 0.25 +
            (1 - features['num_community_categories_norm']) * 0.15
        )
        
        severity = risk_score * 100
        
        if risk_score >= 0.6:
            risk_level = 'ALTO_RIESGO'
        elif risk_score >= 0.3:
            risk_level = 'RIESGO_MODERADO'
        else:
            risk_level = 'BAJO_RIESGO'
        
        return {
            'risk_level': risk_level,
            'severity_index': severity,
            'risk_score': risk_score
        }
    
    async def _get_previous_risk(self, user_id: str) -> Optional[str]:
        """Obtiene el nivel de riesgo anterior del usuario."""
        query = text("""
            SELECT cluster_label 
            FROM user_feature_vector 
            WHERE user_id_raiz = :user_id
        """)
        
        try:
            with analytics_engine.connect() as conn:
                result = conn.execute(query, {"user_id": user_id})
                row = result.fetchone()
                return row[0] if row else None
        except:
            return None
    
    async def _update_analytics_db(
        self, 
        user_id: str, 
        features: Dict[str, float], 
        risk_result: Dict[str, Any]
    ) -> None:
        """Actualiza o inserta el registro del usuario en la DB analítica."""
        upsert_query = text("""
            INSERT INTO user_feature_vector (
                user_id_raiz, extraction_date,
                reciprocity_ratio_norm, days_since_last_seen_norm,
                ratio_night_messages, is_profile_incomplete,
                sentiment_negativity_index, num_community_categories_norm,
                cluster_label
            ) VALUES (
                :user_id, :extraction_date,
                :reciprocity, :days_since,
                :night_ratio, :incomplete,
                :negativity, :community,
                :cluster_label
            )
            ON CONFLICT (user_id_raiz) DO UPDATE SET
                extraction_date = EXCLUDED.extraction_date,
                reciprocity_ratio_norm = EXCLUDED.reciprocity_ratio_norm,
                days_since_last_seen_norm = EXCLUDED.days_since_last_seen_norm,
                ratio_night_messages = EXCLUDED.ratio_night_messages,
                is_profile_incomplete = EXCLUDED.is_profile_incomplete,
                sentiment_negativity_index = EXCLUDED.sentiment_negativity_index,
                num_community_categories_norm = EXCLUDED.num_community_categories_norm,
                cluster_label = EXCLUDED.cluster_label
        """)
        
        try:
            with analytics_engine.begin() as conn:
                conn.execute(upsert_query, {
                    "user_id": user_id,
                    "extraction_date": datetime.utcnow(),
                    "reciprocity": features['reciprocity_ratio_norm'],
                    "days_since": features['days_since_last_seen_norm'],
                    "night_ratio": features['ratio_night_messages'],
                    "incomplete": features['is_profile_incomplete'] > 0.5,
                    "negativity": features['sentiment_negativity_index'],
                    "community": features['num_community_categories_norm'],
                    "cluster_label": risk_result['risk_level']
                })
        except Exception as e:
            print(f"Error actualizando DB analítica: {e}")
    
    async def _broadcast_update(
        self,
        user_id: str,
        risk_result: Dict[str, Any],
        previous_risk: Optional[str],
        triggering_table: str
    ) -> None:
        """Envía actualizaciones a los clientes WebSocket."""
        
        # Mensaje de actualización de usuario
        user_message = create_user_risk_message(
            user_id=user_id,
            risk_level=risk_result['risk_level'],
            severity_index=risk_result['severity_index'],
            previous_risk=previous_risk,
            triggering_factors=[{"source": triggering_table, "action": "update"}]
        )
        
        await manager.broadcast(user_message, channel="clustering")
        
        # Si cambió a alto riesgo, enviar alerta crítica
        if (risk_result['risk_level'] == 'ALTO_RIESGO' and 
            previous_risk != 'ALTO_RIESGO'):
            
            alert = create_critical_alert_message(
                user_id=user_id,
                severity_index=risk_result['severity_index'],
                factors=[
                    f"Cambio detectado en: {triggering_table}",
                    f"Riesgo anterior: {previous_risk or 'N/A'}",
                    f"Riesgo actual: {risk_result['risk_level']}"
                ],
                suggested_action="Revisar perfil del usuario y considerar contacto"
            )
            
            await manager.broadcast(alert, channel="alerts")
            await manager.broadcast(alert, channel="clustering")


# Instancia global del pipeline
streaming_pipeline = StreamingETLPipeline()
