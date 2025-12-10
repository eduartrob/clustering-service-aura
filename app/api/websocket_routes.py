# app/api/websocket_routes.py
"""Endpoints WebSocket y API JSON v2 para clientes React/ChartJS."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text

from app.realtime.websocket_manager import manager, MessageType
from app.database.connection import analytics_engine
from app.database.models import UserFeatureVector

router = APIRouter(prefix="/api/v2/clustering", tags=["Real-Time Clustering"])


# ===== WebSocket Endpoints =====

@router.websocket("/ws/live")
async def websocket_clustering_live(websocket: WebSocket):
    """
    WebSocket para actualizaciones en tiempo real del clustering.
    
    Canales disponibles al conectar:
    - clustering: Actualizaciones de riesgo de usuarios
    - alerts: Alertas críticas de usuarios en alto riesgo
    
    Mensajes enviados:
    - INITIAL_STATE: Estado inicial al conectar
    - USER_RISK_UPDATE: Cambio de riesgo de un usuario
    - DISTRIBUTION_UPDATE: Cambio en la distribución general
    - CRITICAL_ALERT: Alerta de usuario en alto riesgo
    """
    await manager.connect(websocket, channel="clustering")
    try:
        while True:
            # Mantener conexión activa, los mensajes se envían por broadcast
            data = await websocket.receive_text()
            # Opcionalmente procesar mensajes del cliente (ej: suscripciones)
            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel="clustering")


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket exclusivo para alertas críticas.
    Ideal para paneles de psicólogos que necesitan notificaciones inmediatas.
    """
    await manager.connect(websocket, channel="alerts")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel="alerts")


# ===== API JSON v2 para ChartJS =====

@router.get("/data/distribution")
def get_distribution_data():
    """
    Retorna datos de distribución de riesgo formateados para ChartJS.
    
    Formato compatible con Chart.js Bar/Pie chart.
    """
    query = text("""
        SELECT cluster_label, COUNT(*) as count
        FROM user_feature_vector
        WHERE cluster_label IS NOT NULL
        GROUP BY cluster_label
    """)
    
    try:
        with analytics_engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
        
        distribution = {row[0]: row[1] for row in rows}
        total = sum(distribution.values())
        
        # Ordenar por nivel de riesgo
        ordered_labels = ["BAJO_RIESGO", "RIESGO_MODERADO", "ALTO_RIESGO"]
        colors = {
            "BAJO_RIESGO": "#10B981",
            "RIESGO_MODERADO": "#F59E0B", 
            "ALTO_RIESGO": "#EF4444"
        }
        
        return {
            "labels": [label.replace("_", " ") for label in ordered_labels],
            "datasets": [{
                "label": "Usuarios por Nivel de Riesgo",
                "data": [distribution.get(label, 0) for label in ordered_labels],
                "backgroundColor": [colors[label] for label in ordered_labels],
                "borderColor": [colors[label] for label in ordered_labels],
                "borderWidth": 1
            }],
            "meta": {
                "total_users": total,
                "high_risk_percentage": round(distribution.get("ALTO_RIESGO", 0) / total * 100, 1) if total > 0 else 0,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
        }
    except Exception as e:
        return {"error": str(e), "labels": [], "datasets": []}


@router.get("/data/scatter")
def get_scatter_data():
    """
    Retorna datos PCA para scatter plot formateado para ChartJS.
    
    Formato compatible con Chart.js Scatter chart.
    """
    query = text(f"""
        SELECT user_id_raiz, cluster_label, 
               reciprocity_ratio_norm as pca_x,
               days_since_last_seen_norm as pca_y,
               sentiment_negativity_index,
               ratio_night_messages
        FROM {UserFeatureVector.__tablename__}
        WHERE cluster_label IS NOT NULL
    """)
    
    try:
        df = pd.read_sql(query, analytics_engine)
        
        if df.empty:
            return {"datasets": [], "meta": {"total_points": 0}}
        
        colors = {
            "BAJO_RIESGO": "rgba(16, 185, 129, 0.7)",
            "RIESGO_MODERADO": "rgba(245, 158, 11, 0.7)",
            "ALTO_RIESGO": "rgba(239, 68, 68, 0.7)"
        }
        
        datasets = []
        for risk_level in ["BAJO_RIESGO", "RIESGO_MODERADO", "ALTO_RIESGO"]:
            mask = df['cluster_label'] == risk_level
            if mask.sum() > 0:
                subset = df[mask]
                datasets.append({
                    "label": risk_level.replace("_", " "),
                    "data": [
                        {"x": round(row['pca_x'], 3), "y": round(row['pca_y'], 3)}
                        for _, row in subset.iterrows()
                    ],
                    "backgroundColor": colors[risk_level],
                    "pointRadius": 6,
                    "pointHoverRadius": 8
                })
        
        return {
            "datasets": datasets,
            "meta": {
                "total_points": len(df),
                "axis_labels": {
                    "x": "Ratio de Reciprocidad (normalizado)",
                    "y": "Días de Inactividad (normalizado)"
                }
            }
        }
    except Exception as e:
        return {"error": str(e), "datasets": []}


@router.get("/data/radar")
def get_radar_data():
    """
    Retorna datos de perfil de KPIs para radar chart.
    
    Formato compatible con Chart.js Radar chart.
    """
    query = text(f"""
        SELECT cluster_label,
               AVG(reciprocity_ratio_norm) as reciprocity,
               AVG(days_since_last_seen_norm) as inactivity,
               AVG(ratio_night_messages) as night_messages,
               AVG(CASE WHEN is_profile_incomplete THEN 1.0 ELSE 0.0 END) as profile_incomplete,
               AVG(sentiment_negativity_index) as negativity,
               AVG(num_community_categories_norm) as community
        FROM {UserFeatureVector.__tablename__}
        WHERE cluster_label IS NOT NULL
        GROUP BY cluster_label
    """)
    
    try:
        df = pd.read_sql(query, analytics_engine)
        
        labels = [
            "Reciprocidad Social",
            "Días Inactivo",
            "Mensajes Nocturnos",
            "Perfil Incompleto",
            "Negatividad NLP",
            "Participación Comunitaria"
        ]
        
        colors = {
            "BAJO_RIESGO": {"bg": "rgba(16, 185, 129, 0.2)", "border": "rgb(16, 185, 129)"},
            "RIESGO_MODERADO": {"bg": "rgba(245, 158, 11, 0.2)", "border": "rgb(245, 158, 11)"},
            "ALTO_RIESGO": {"bg": "rgba(239, 68, 68, 0.2)", "border": "rgb(239, 68, 68)"}
        }
        
        datasets = []
        for _, row in df.iterrows():
            risk_level = row['cluster_label']
            datasets.append({
                "label": risk_level.replace("_", " "),
                "data": [
                    round(row['reciprocity'], 3),
                    round(row['inactivity'], 3),
                    round(row['night_messages'], 3),
                    round(row['profile_incomplete'], 3),
                    round(row['negativity'], 3),
                    round(row['community'], 3)
                ],
                "fill": True,
                "backgroundColor": colors.get(risk_level, {}).get("bg", "rgba(128,128,128,0.2)"),
                "borderColor": colors.get(risk_level, {}).get("border", "rgb(128,128,128)"),
                "pointBackgroundColor": colors.get(risk_level, {}).get("border", "rgb(128,128,128)")
            })
        
        return {
            "labels": labels,
            "datasets": datasets,
            "meta": {
                "description": "Perfil promedio de KPIs por nivel de riesgo"
            }
        }
    except Exception as e:
        return {"error": str(e), "labels": [], "datasets": []}


@router.get("/data/severity-histogram")
def get_severity_histogram():
    """
    Retorna datos para histograma del índice de severidad.
    
    Formato compatible con Chart.js Bar chart.
    """
    # Como no tenemos severity_index persistido, lo calculamos
    query = text(f"""
        SELECT 
            cluster_label,
            days_since_last_seen_norm * 0.25 +
            ratio_night_messages * 0.20 +
            CASE WHEN is_profile_incomplete THEN 0.15 ELSE 0.0 END +
            sentiment_negativity_index * 0.25 +
            (1 - num_community_categories_norm) * 0.15 as severity_score
        FROM {UserFeatureVector.__tablename__}
        WHERE cluster_label IS NOT NULL
    """)
    
    try:
        df = pd.read_sql(query, analytics_engine)
        
        if df.empty:
            return {"labels": [], "datasets": []}
        
        df['severity_index'] = df['severity_score'] * 100
        
        # Crear bins para histograma
        bins = [0, 20, 40, 60, 80, 100]
        labels_bins = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        colors = ["#10B981", "#84CC16", "#F59E0B", "#F97316", "#EF4444"]
        
        counts = pd.cut(df['severity_index'], bins=bins, labels=labels_bins).value_counts().sort_index()
        
        return {
            "labels": labels_bins,
            "datasets": [{
                "label": "Distribución de Severidad",
                "data": counts.tolist(),
                "backgroundColor": colors,
                "borderColor": colors,
                "borderWidth": 1
            }],
            "meta": {
                "mean_severity": round(df['severity_index'].mean(), 1),
                "max_severity": round(df['severity_index'].max(), 1),
                "above_60_percent": round((df['severity_index'] >= 60).sum() / len(df) * 100, 1)
            }
        }
    except Exception as e:
        return {"error": str(e), "labels": [], "datasets": []}


@router.get("/data/kpi-trends")
def get_kpi_trends(hours: int = Query(default=24, ge=1, le=168)):
    """
    Retorna tendencias de KPIs en las últimas N horas.
    
    Formato compatible con Chart.js Line chart.
    
    Args:
        hours: Número de horas hacia atrás (default 24, máx 168 = 1 semana)
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    query = text(f"""
        SELECT 
            DATE_TRUNC('hour', extraction_date) as time_bucket,
            AVG(days_since_last_seen_norm) as avg_inactivity,
            AVG(sentiment_negativity_index) as avg_negativity,
            AVG(ratio_night_messages) as avg_night_ratio,
            COUNT(*) as user_count
        FROM {UserFeatureVector.__tablename__}
        WHERE extraction_date >= :cutoff
        GROUP BY DATE_TRUNC('hour', extraction_date)
        ORDER BY time_bucket
    """)
    
    try:
        with analytics_engine.connect() as conn:
            result = conn.execute(query, {"cutoff": cutoff})
            rows = result.fetchall()
        
        if not rows:
            return {"labels": [], "datasets": []}
        
        labels = [row[0].strftime("%H:%M") for row in rows]
        
        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Inactividad Promedio",
                    "data": [round(row[1] or 0, 3) for row in rows],
                    "borderColor": "rgb(239, 68, 68)",
                    "tension": 0.4
                },
                {
                    "label": "Negatividad NLP",
                    "data": [round(row[2] or 0, 3) for row in rows],
                    "borderColor": "rgb(245, 158, 11)",
                    "tension": 0.4
                },
                {
                    "label": "Ratio Nocturno",
                    "data": [round(row[3] or 0, 3) for row in rows],
                    "borderColor": "rgb(139, 92, 246)",
                    "tension": 0.4
                }
            ],
            "meta": {
                "time_range_hours": hours,
                "data_points": len(rows)
            }
        }
    except Exception as e:
        return {"error": str(e), "labels": [], "datasets": []}


@router.get("/data/high-risk-users")
def get_high_risk_users(limit: int = Query(default=10, ge=1, le=50)):
    """
    Retorna los usuarios de mayor riesgo con detalles para intervención.
    """
    query = text(f"""
        SELECT 
            user_id_raiz,
            cluster_label,
            days_since_last_seen_norm,
            ratio_night_messages,
            sentiment_negativity_index,
            num_community_categories_norm,
            extraction_date
        FROM {UserFeatureVector.__tablename__}
        WHERE cluster_label = 'ALTO_RIESGO'
        ORDER BY 
            days_since_last_seen_norm DESC,
            sentiment_negativity_index DESC
        LIMIT :limit
    """)
    
    try:
        with analytics_engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = result.fetchall()
        
        users = []
        for row in rows:
            # Calcular severidad
            severity = (
                row[2] * 0.25 +  # days_since_last_seen
                row[3] * 0.20 +  # night_messages
                row[4] * 0.25 +  # negativity
                (1 - row[5]) * 0.15  # community
            ) * 100
            
            users.append({
                "user_id": str(row[0]),
                "risk_level": row[1],
                "severity_index": round(severity, 1),
                "factors": {
                    "inactivity": round(row[2] * 100, 1),
                    "night_activity": round(row[3] * 100, 1),
                    "negativity": round(row[4] * 100, 1),
                    "community_engagement": round(row[5] * 100, 1)
                },
                "last_updated": row[6].isoformat() if row[6] else None
            })
        
        return {
            "users": users,
            "total_high_risk": len(users),
            "meta": {
                "query_limit": limit,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        }
    except Exception as e:
        return {"error": str(e), "users": []}


@router.get("/status")
def get_realtime_status():
    """Retorna el estado del sistema de tiempo real."""
    return {
        "websocket_connections": manager.get_connection_stats(),
        "available_channels": ["clustering", "alerts"],
        "api_version": "v2",
        "features": [
            "WebSocket live updates",
            "ChartJS-compatible JSON endpoints",
            "Real-time risk detection"
        ]
    }
