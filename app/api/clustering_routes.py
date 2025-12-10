# app/api/clustering_routes.py
"""Endpoints para clustering y visualización."""

from fastapi import APIRouter, HTTPException, status, Query, Response, Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from app.clustering.ensemble import RiskClusteringEnsemble
from app.clustering.visualizer import ClusterVisualizer
from app.database.connection import analytics_engine
from app.database.models import UserFeatureVector

router = APIRouter(prefix="/api/v1/clustering", tags=["Clustering"])

# Estado global del clustering (en producción usar Redis o DB)
_clustering_state = {
    "ensemble": None,
    "visualizer": ClusterVisualizer(),
    "last_execution": None
}


# ===== Modelos de Respuesta =====

class ClusteringExecuteResponse(BaseModel):
    status: str
    message: str
    execution_date: datetime
    total_users: int
    risk_distribution: Dict[str, int]
    metrics: Dict


class ClusteringResultsResponse(BaseModel):
    execution_date: Optional[datetime]
    metrics: Dict
    risk_distribution: Dict[str, int]
    cluster_centers: Optional[List]
    risk_cluster_id: Optional[int]


class UserRiskResponse(BaseModel):
    user_id_raiz: str
    risk_level: str
    severity_index: float
    total_votes: int


# ===== Endpoints =====

@router.post("/execute", response_model=ClusteringExecuteResponse)
def execute_clustering(
    n_clusters: int = Query(default=4, ge=2, le=10, description="Número de clusters K-Means"),
    contamination: float = Query(default=0.1, ge=0.01, le=0.5, description="Proporción de anomalías esperadas")
):
    """
    Ejecuta el ensamble de clustering sobre los datos del ETL.
    
    1. Lee los feature vectors de la base de datos analítica
    2. Ejecuta K-Means, DBSCAN e Isolation Forest
    3. Calcula votación y nivel de riesgo
    4. Genera visualizaciones
    """
    try:
        # Leer datos de la DB analítica
        query = f"SELECT * FROM {UserFeatureVector.__tablename__}"
        df = pd.read_sql(query, analytics_engine)
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No hay datos en la tabla user_feature_vector. Ejecuta primero el ETL (/api/v1/data-miner/execute-etl)"
            )
        
        # Crear y ejecutar ensamble
        ensemble = RiskClusteringEnsemble(n_clusters=n_clusters, contamination=contamination)
        results = ensemble.fit_predict(df)
        
        # Guardar estado
        _clustering_state["ensemble"] = ensemble
        _clustering_state["last_execution"] = datetime.utcnow()
        
        # Actualizar tabla con resultados
        _update_cluster_labels(results)
        
        return ClusteringExecuteResponse(
            status="success",
            message="Clustering ejecutado exitosamente",
            execution_date=_clustering_state["last_execution"],
            total_users=len(results),
            risk_distribution=ensemble.metrics.get('risk_distribution', {}),
            metrics={
                "silhouette_score": ensemble.metrics.get('silhouette_score', 0),
                "calinski_harabasz": ensemble.metrics.get('calinski_harabasz', 0),
                "high_risk_percentage": ensemble.metrics.get('high_risk_percentage', 0)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando clustering: {str(e)}"
        )


@router.get("/results", response_model=ClusteringResultsResponse)
def get_clustering_results():
    """Obtiene los resultados del último clustering ejecutado."""
    ensemble = _clustering_state.get("ensemble")
    
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se ha ejecutado ningún clustering. Usa POST /execute primero."
        )
    
    results = ensemble.get_results()
    return ClusteringResultsResponse(
        execution_date=_clustering_state.get("last_execution"),
        metrics=results.get("metrics", {}),
        risk_distribution=results.get("risk_distribution", {}),
        cluster_centers=results.get("cluster_centers"),
        risk_cluster_id=results.get("risk_cluster_id")
    )


@router.get("/users/{risk_level}", response_model=List[UserRiskResponse])
def get_users_by_risk(
    risk_level: str = Path(..., description="Nivel de riesgo: ALTO_RIESGO, RIESGO_MODERADO, BAJO_RIESGO")
):
    """Obtiene la lista de usuarios para un nivel de riesgo específico."""
    ensemble = _clustering_state.get("ensemble")
    
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se ha ejecutado ningún clustering."
        )
    
    valid_levels = ['ALTO_RIESGO', 'RIESGO_MODERADO', 'BAJO_RIESGO']
    if risk_level not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Nivel inválido. Usar: {valid_levels}"
        )
    
    users = ensemble.get_users_by_risk(risk_level)
    return [
        UserRiskResponse(
            user_id_raiz=str(u['user_id_raiz']),
            risk_level=u['risk_level'],
            severity_index=u['severity_index'],
            total_votes=u['total_votes']
        ) for u in users
    ]


@router.get("/profiles")
def get_cluster_profiles():
    """Obtiene el perfil promedio de KPIs para cada cluster."""
    ensemble = _clustering_state.get("ensemble")
    
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se ha ejecutado ningún clustering."
        )
    
    return ensemble.get_cluster_profiles()


# ===== Endpoints de Visualización =====

@router.get("/visualize/scatter", response_class=HTMLResponse)
def visualize_scatter():
    """
    Genera gráfico de dispersión PCA coloreado por nivel de riesgo.
    Retorna SVG embebido en HTML para visualización en navegador.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None or ensemble.results_df is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    svg = visualizer.scatter_plot_pca(ensemble.results_df)
    return _wrap_svg_in_html(svg, "Scatter Plot PCA - Nivel de Riesgo")


@router.get("/visualize/distribution", response_class=HTMLResponse)
def visualize_distribution():
    """
    Genera gráfico de barras con distribución de niveles de riesgo.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None or ensemble.results_df is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    svg = visualizer.distribution_bar_chart(ensemble.results_df)
    return _wrap_svg_in_html(svg, "Distribución de Riesgo")


@router.get("/visualize/radar", response_class=HTMLResponse)
def visualize_radar():
    """
    Genera radar chart con perfil de KPIs por cluster.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    profiles = ensemble.get_cluster_profiles()
    svg = visualizer.radar_chart_clusters(profiles)
    return _wrap_svg_in_html(svg, "Radar Chart - Perfil de Clusters")


@router.get("/visualize/severity", response_class=HTMLResponse)
def visualize_severity():
    """
    Genera histograma del índice de severidad.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None or ensemble.results_df is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    svg = visualizer.severity_histogram(ensemble.results_df)
    return _wrap_svg_in_html(svg, "Distribución de Severidad")


@router.get("/visualize/kmeans", response_class=HTMLResponse)
def visualize_kmeans_clusters():
    """
    Genera scatter plot coloreado por clusters K-Means.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None or ensemble.results_df is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    svg = visualizer.kmeans_clusters_scatter(ensemble.results_df, ensemble.n_clusters)
    return _wrap_svg_in_html(svg, "Clusters K-Means")


@router.get("/visualize/metrics", response_class=HTMLResponse)
def visualize_metrics():
    """
    Genera gráfico resumen de métricas del clustering.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    svg = visualizer.metrics_summary_chart(ensemble.metrics)
    return _wrap_svg_in_html(svg, "Métricas de Clustering")


@router.get("/visualize/dashboard", response_class=HTMLResponse)
def visualize_dashboard():
    """
    Genera un dashboard completo con todas las visualizaciones.
    """
    ensemble = _clustering_state.get("ensemble")
    visualizer = _clustering_state.get("visualizer")
    
    if ensemble is None or ensemble.results_df is None:
        raise HTTPException(status_code=404, detail="Ejecuta el clustering primero")
    
    # Generar todos los gráficos
    scatter_svg = visualizer.scatter_plot_pca(ensemble.results_df)
    distribution_svg = visualizer.distribution_bar_chart(ensemble.results_df)
    profiles = ensemble.get_cluster_profiles()
    radar_svg = visualizer.radar_chart_clusters(profiles)
    severity_svg = visualizer.severity_histogram(ensemble.results_df)
    
    # Crear dashboard HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AURA Clustering Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #eee;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #00d9ff;
                text-shadow: 0 0 10px rgba(0,217,255,0.5);
            }}
            .dashboard {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .card h2 {{
                color: #00d9ff;
                margin-top: 0;
                font-size: 1.2em;
            }}
            .card svg {{
                width: 100%;
                height: auto;
                border-radius: 10px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .metric {{
                background: rgba(0,217,255,0.1);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #00d9ff;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #888;
            }}
        </style>
    </head>
    <body>
        <h1>🔮 AURA Clustering Dashboard</h1>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{ensemble.metrics.get('total_users', 0)}</div>
                <div class="metric-label">Total Usuarios</div>
            </div>
            <div class="metric">
                <div class="metric-value">{ensemble.metrics.get('high_risk_percentage', 0):.1f}%</div>
                <div class="metric-label">Alto Riesgo</div>
            </div>
            <div class="metric">
                <div class="metric-value">{ensemble.metrics.get('silhouette_score', 0):.3f}</div>
                <div class="metric-label">Silhouette Score</div>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h2>📊 Distribución de Riesgo</h2>
                {distribution_svg}
            </div>
            <div class="card">
                <h2>🎯 Proyección PCA</h2>
                {scatter_svg}
            </div>
            <div class="card">
                <h2>📈 Índice de Severidad</h2>
                {severity_svg}
            </div>
            <div class="card">
                <h2>🕸️ Perfil de Clusters</h2>
                {radar_svg}
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


# ===== Funciones Auxiliares =====

def _wrap_svg_in_html(svg: str, title: str) -> str:
    """Envuelve SVG en HTML con estilos."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title} - AURA Clustering</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: #1a1a2e;
                color: #eee;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }}
            h1 {{ color: #00d9ff; }}
            .chart-container {{
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                max-width: 900px;
            }}
            svg {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="chart-container">
            {svg}
        </div>
    </body>
    </html>
    """


def _update_cluster_labels(results_df: pd.DataFrame):
    """Actualiza la columna cluster_label en la base de datos."""
    from sqlalchemy import text
    
    try:
        with analytics_engine.begin() as conn:
            for _, row in results_df.iterrows():
                conn.execute(
                    text(f"""
                        UPDATE {UserFeatureVector.__tablename__}
                        SET cluster_label = :label
                        WHERE user_id_raiz = :user_id
                    """),
                    {"label": row['risk_level'], "user_id": str(row['user_id_raiz'])}
                )
    except Exception as e:
        print(f"⚠️ Error actualizando cluster_label: {e}")
