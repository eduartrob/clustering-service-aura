# app/clustering/ensemble.py
"""Ensamble de modelos de clustering para detección de riesgo AURA."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RiskClusteringEnsemble:
    """
    Ensamble de modelos de clustering para detección de usuarios en riesgo.
    Combina K-Means, DBSCAN e Isolation Forest mediante votación.
    """
    
    # Columnas de features (KPIs normalizados)
    FEATURE_COLUMNS = [
        'reciprocity_ratio_norm',
        'days_since_last_seen_norm',
        'ratio_night_messages',
        'is_profile_incomplete',
        'sentiment_negativity_index',
        'num_community_categories_norm'
    ]
    
    # Pesos para el cálculo de severidad
    WEIGHTS = {
        'isolation_forest': 0.5,
        'dbscan': 0.3,
        'kmeans': 0.2
    }
    
    def __init__(self, n_clusters: int = 4, contamination: float = 0.1):
        """
        Inicializa el ensamble de clustering.
        
        Args:
            n_clusters: Número de clusters para K-Means
            contamination: Proporción esperada de anomalías para Isolation Forest
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=2)
        
        # Modelos
        self.kmeans = None
        self.dbscan = None
        self.isolation_forest = None
        
        # Resultados
        self.results_df: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.X_pca: Optional[np.ndarray] = None
        self.metrics: Dict = {}
        self.execution_date: Optional[datetime] = None
        self.risk_cluster_idx: Optional[int] = None
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de clustering.
        
        Args:
            df: DataFrame con user_id_raiz y las columnas de features
            
        Returns:
            DataFrame con resultados de clustering y nivel de riesgo
        """
        self.execution_date = datetime.utcnow()
        print("\n" + "="*60)
        print("🔮 INICIANDO CLUSTERING - DETECCIÓN DE RIESGOS")
        print("="*60 + "\n")
        
        # Preparar datos
        self._prepare_data(df)
        
        # Ejecutar modelos
        self._run_kmeans()
        self._run_dbscan()
        self._run_isolation_forest()
        
        # Calcular votación y nivel de riesgo
        self._calculate_ensemble_risk()
        
        # Calcular índice de severidad
        self._calculate_severity_index()
        
        # Calcular métricas de evaluación
        self._calculate_metrics()
        
        # Agregar proyección PCA para visualización
        self._calculate_pca()
        
        print("\n" + "="*60)
        print("✅ CLUSTERING COMPLETADO")
        print("="*60 + "\n")
        
        return self.results_df
    
    def _prepare_data(self, df: pd.DataFrame):
        """Prepara y normaliza los datos."""
        print("📊 Preparando datos...")
        
        # Verificar columnas necesarias
        missing_cols = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Copiar dataframe con user_id
        self.results_df = df[['user_id_raiz']].copy()
        
        # Extraer features
        X = df[self.FEATURE_COLUMNS].copy()
        
        # Convertir booleanos a int
        if 'is_profile_incomplete' in X.columns:
            X['is_profile_incomplete'] = X['is_profile_incomplete'].astype(int)
        
        # Rellenar NaN con 0
        X = X.fillna(0)
        
        # Ya están normalizados del ETL, pero aseguramos escala uniforme
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"   ✅ {len(df)} usuarios preparados con {len(self.FEATURE_COLUMNS)} features")
    
    def _run_kmeans(self):
        """Ejecuta K-Means clustering."""
        print(f"🎯 Ejecutando K-Means (k={self.n_clusters})...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = self.kmeans.fit_predict(self.X_scaled)
        
        # Identificar cluster de riesgo (menor promedio en features positivas)
        centers = self.kmeans.cluster_centers_
        # El cluster con mayor days_since_last_seen y menor community_categories es riesgoso
        # Usamos una combinación: alto en col 1 (días inactivo) y bajo en col 5 (comunidades)
        risk_scores = centers[:, 1] - centers[:, 5]  # días inactivo - comunidades
        self.risk_cluster_idx = np.argmax(risk_scores)
        
        self.results_df['kmeans_cluster'] = clusters
        self.results_df['vote_kmeans'] = (clusters == self.risk_cluster_idx).astype(int)
        
        print(f"   ✅ Cluster de riesgo identificado: {self.risk_cluster_idx}")
    
    def _run_dbscan(self):
        """Ejecuta DBSCAN para detección de outliers."""
        print("🔍 Ejecutando DBSCAN...")
        
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = self.dbscan.fit_predict(self.X_scaled)
        
        # -1 indica outlier en DBSCAN
        outliers = (clusters == -1).sum()
        
        self.results_df['dbscan_cluster'] = clusters
        self.results_df['vote_dbscan'] = (clusters == -1).astype(int)
        
        print(f"   ✅ Outliers detectados: {outliers}")
    
    def _run_isolation_forest(self):
        """Ejecuta Isolation Forest para detección de anomalías."""
        print(f"🌲 Ejecutando Isolation Forest (contamination={self.contamination})...")
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = self.isolation_forest.fit_predict(self.X_scaled)
        scores = self.isolation_forest.decision_function(self.X_scaled)
        
        # -1 es anomalía, 1 es normal
        anomalies = (predictions == -1).sum()
        
        self.results_df['iso_prediction'] = predictions
        self.results_df['iso_score'] = scores
        self.results_df['vote_iso'] = (predictions == -1).astype(int)
        
        print(f"   ✅ Anomalías detectadas: {anomalies}")
    
    def _calculate_ensemble_risk(self):
        """Calcula el nivel de riesgo basado en votación."""
        print("🗳️ Calculando votación del ensamble...")
        
        # Suma de votos
        self.results_df['total_votes'] = (
            self.results_df['vote_kmeans'] +
            self.results_df['vote_dbscan'] +
            self.results_df['vote_iso']
        )
        
        # Clasificación final
        conditions = [
            (self.results_df['total_votes'] >= 2),
            (self.results_df['total_votes'] == 1)
        ]
        choices = ['ALTO_RIESGO', 'RIESGO_MODERADO']
        self.results_df['risk_level'] = np.select(conditions, choices, default='BAJO_RIESGO')
        
        # Contar por nivel
        counts = self.results_df['risk_level'].value_counts()
        print(f"   🔴 ALTO RIESGO: {counts.get('ALTO_RIESGO', 0)}")
        print(f"   🟡 RIESGO MODERADO: {counts.get('RIESGO_MODERADO', 0)}")
        print(f"   🟢 BAJO RIESGO: {counts.get('BAJO_RIESGO', 0)}")
    
    def _calculate_severity_index(self):
        """Calcula el Índice de Severidad de Anomalía (ASI)."""
        print("📈 Calculando Índice de Severidad...")
        
        # Normalizar score de Isolation Forest a [0, 1]
        iso_score = self.results_df['iso_score']
        iso_normalized = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-10)
        
        # Calcular distancia al centroide de riesgo (normalizada)
        if self.risk_cluster_idx is not None:
            risk_centroid = self.kmeans.cluster_centers_[self.risk_cluster_idx]
            distances = np.linalg.norm(self.X_scaled - risk_centroid, axis=1)
            dist_normalized = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        else:
            dist_normalized = np.zeros(len(self.results_df))
        
        # ASI = w1*(1-S_iso) + w2*I_dbscan + w3*(1-D_kmeans)
        # Invertimos porque menor iso_score y menor distancia = más riesgo
        severity = (
            self.WEIGHTS['isolation_forest'] * (1 - iso_normalized) +
            self.WEIGHTS['dbscan'] * self.results_df['vote_dbscan'] +
            self.WEIGHTS['kmeans'] * (1 - dist_normalized)
        )
        
        # Escalar a 0-100
        self.results_df['severity_index'] = np.clip(severity * 100, 0, 100)
    
    def _calculate_metrics(self):
        """Calcula métricas de evaluación del clustering."""
        print("📊 Calculando métricas...")
        
        # Solo si hay más de 1 cluster
        if len(np.unique(self.results_df['kmeans_cluster'])) > 1:
            self.metrics['silhouette_score'] = silhouette_score(
                self.X_scaled, 
                self.results_df['kmeans_cluster']
            )
            self.metrics['calinski_harabasz'] = calinski_harabasz_score(
                self.X_scaled,
                self.results_df['kmeans_cluster']
            )
        else:
            self.metrics['silhouette_score'] = 0.0
            self.metrics['calinski_harabasz'] = 0.0
        
        # Estadísticas de riesgo
        risk_counts = self.results_df['risk_level'].value_counts().to_dict()
        self.metrics['risk_distribution'] = risk_counts
        self.metrics['total_users'] = len(self.results_df)
        self.metrics['high_risk_percentage'] = (
            risk_counts.get('ALTO_RIESGO', 0) / len(self.results_df) * 100
        )
        
        print(f"   ✅ Silhouette Score: {self.metrics['silhouette_score']:.3f}")
        print(f"   ✅ Calinski-Harabasz: {self.metrics['calinski_harabasz']:.2f}")
    
    def _calculate_pca(self):
        """Calcula proyección PCA para visualización 2D."""
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.results_df['pca_x'] = self.X_pca[:, 0]
        self.results_df['pca_y'] = self.X_pca[:, 1]
    
    def get_results(self) -> Dict:
        """Retorna los resultados del clustering."""
        if self.results_df is None:
            return {"error": "No se ha ejecutado el clustering"}
        
        return {
            "execution_date": self.execution_date.isoformat() if self.execution_date else None,
            "metrics": self.metrics,
            "risk_distribution": self.results_df['risk_level'].value_counts().to_dict(),
            "cluster_centers": self.kmeans.cluster_centers_.tolist() if self.kmeans else None,
            "risk_cluster_id": int(self.risk_cluster_idx) if self.risk_cluster_idx is not None else None
        }
    
    def get_users_by_risk(self, risk_level: str) -> List[Dict]:
        """Retorna usuarios filtrados por nivel de riesgo."""
        if self.results_df is None:
            return []
        
        filtered = self.results_df[self.results_df['risk_level'] == risk_level]
        return filtered[['user_id_raiz', 'risk_level', 'severity_index', 'total_votes']].to_dict('records')
    
    def get_cluster_profiles(self) -> Dict:
        """Retorna el perfil promedio de KPIs para cada cluster."""
        if self.results_df is None or self.X_scaled is None:
            return {}
        
        profiles = {}
        for cluster_id in range(self.n_clusters):
            mask = self.results_df['kmeans_cluster'] == cluster_id
            if mask.sum() > 0:
                cluster_data = self.X_scaled[mask]
                profiles[f"cluster_{cluster_id}"] = {
                    self.FEATURE_COLUMNS[i]: float(cluster_data[:, i].mean())
                    for i in range(len(self.FEATURE_COLUMNS))
                }
                profiles[f"cluster_{cluster_id}"]['count'] = int(mask.sum())
                profiles[f"cluster_{cluster_id}"]['is_risk_cluster'] = (cluster_id == self.risk_cluster_idx)
        
        return profiles
