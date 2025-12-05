# app/clustering/visualizer.py
"""Generador de visualizaciones SVG para resultados de clustering."""

import io
import base64
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


class ClusterVisualizer:
    """Generador de gráficos SVG para visualización de clustering."""
    
    # Colores por nivel de riesgo
    RISK_COLORS = {
        'ALTO_RIESGO': '#EF4444',      # Rojo
        'RIESGO_MODERADO': '#F59E0B',  # Amarillo/Naranja
        'BAJO_RIESGO': '#10B981'       # Verde
    }
    
    # Colores para clusters K-Means
    CLUSTER_COLORS = ['#3B82F6', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16']
    
    FEATURE_LABELS = {
        'reciprocity_ratio_norm': 'Reciprocidad Social',
        'days_since_last_seen_norm': 'Días Inactivo',
        'ratio_night_messages': 'Mensajes Nocturnos',
        'is_profile_incomplete': 'Perfil Incompleto',
        'sentiment_negativity_index': 'Negatividad NLP',
        'num_community_categories_norm': 'Participación Comunitaria'
    }
    
    def __init__(self, figsize: tuple = (10, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def _fig_to_svg(self, fig) -> str:
        """Convierte figura matplotlib a string SVG."""
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        svg_str = buf.getvalue().decode('utf-8')
        plt.close(fig)
        return svg_str
    
    def scatter_plot_pca(self, results_df: pd.DataFrame) -> str:
        """
        Genera scatter plot 2D usando proyección PCA.
        Coloreado por nivel de riesgo.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for risk_level, color in self.RISK_COLORS.items():
            mask = results_df['risk_level'] == risk_level
            if mask.sum() > 0:
                ax.scatter(
                    results_df.loc[mask, 'pca_x'],
                    results_df.loc[mask, 'pca_y'],
                    c=color,
                    label=risk_level.replace('_', ' '),
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        ax.set_xlabel('Componente Principal 1', fontsize=12)
        ax.set_ylabel('Componente Principal 2', fontsize=12)
        ax.set_title('Proyección PCA de Usuarios por Nivel de Riesgo', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_svg(fig)
    
    def distribution_bar_chart(self, results_df: pd.DataFrame) -> str:
        """
        Genera gráfico de barras con distribución de niveles de riesgo.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        counts = results_df['risk_level'].value_counts()
        risk_levels = ['BAJO_RIESGO', 'RIESGO_MODERADO', 'ALTO_RIESGO']
        values = [counts.get(level, 0) for level in risk_levels]
        colors = [self.RISK_COLORS[level] for level in risk_levels]
        labels = [level.replace('_', ' ') for level in risk_levels]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
        
        # Añadir etiquetas de valor
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val}\n({val/len(results_df)*100:.1f}%)',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )
        
        ax.set_ylabel('Número de Usuarios', fontsize=12)
        ax.set_title('Distribución de Usuarios por Nivel de Riesgo', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 10)
        
        return self._fig_to_svg(fig)
    
    def radar_chart_clusters(self, cluster_profiles: Dict) -> str:
        """
        Genera radar chart con perfil promedio de KPIs por cluster.
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        features = list(self.FEATURE_LABELS.keys())
        num_features = len(features)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el polígono
        
        for idx, (cluster_name, profile) in enumerate(cluster_profiles.items()):
            values = [profile.get(f, 0) for f in features]
            values += values[:1]  # Cerrar el polígono
            
            color = self.CLUSTER_COLORS[idx % len(self.CLUSTER_COLORS)]
            is_risk = profile.get('is_risk_cluster', False)
            
            label = f"{cluster_name} (n={profile.get('count', 0)})"
            if is_risk:
                label += " ⚠️ RIESGO"
                color = self.RISK_COLORS['ALTO_RIESGO']
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Configurar etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.FEATURE_LABELS[f] for f in features], size=9)
        ax.set_ylim(0, 1)
        ax.set_title('Perfil de KPIs por Cluster', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return self._fig_to_svg(fig)
    
    def severity_histogram(self, results_df: pd.DataFrame) -> str:
        """
        Genera histograma de distribución del índice de severidad.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        severity = results_df['severity_index']
        
        # Histograma con gradiente de colores
        n, bins, patches = ax.hist(severity, bins=20, edgecolor='white', linewidth=1)
        
        # Colorear barras según severidad
        for i, patch in enumerate(patches):
            if bins[i] < 30:
                patch.set_facecolor(self.RISK_COLORS['BAJO_RIESGO'])
            elif bins[i] < 60:
                patch.set_facecolor(self.RISK_COLORS['RIESGO_MODERADO'])
            else:
                patch.set_facecolor(self.RISK_COLORS['ALTO_RIESGO'])
        
        # Líneas de umbral
        ax.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Umbral Bajo')
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.7, label='Umbral Alto')
        
        ax.set_xlabel('Índice de Severidad (0-100)', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución del Índice de Severidad de Anomalía', fontsize=14, fontweight='bold')
        
        # Leyenda
        legend_elements = [
            Patch(facecolor=self.RISK_COLORS['BAJO_RIESGO'], label='Bajo (0-30)'),
            Patch(facecolor=self.RISK_COLORS['RIESGO_MODERADO'], label='Moderado (30-60)'),
            Patch(facecolor=self.RISK_COLORS['ALTO_RIESGO'], label='Alto (60-100)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return self._fig_to_svg(fig)
    
    def kmeans_clusters_scatter(self, results_df: pd.DataFrame, n_clusters: int) -> str:
        """
        Genera scatter plot coloreado por clusters K-Means.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for cluster_id in range(n_clusters):
            mask = results_df['kmeans_cluster'] == cluster_id
            if mask.sum() > 0:
                color = self.CLUSTER_COLORS[cluster_id % len(self.CLUSTER_COLORS)]
                ax.scatter(
                    results_df.loc[mask, 'pca_x'],
                    results_df.loc[mask, 'pca_y'],
                    c=color,
                    label=f'Cluster {cluster_id}',
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        ax.set_xlabel('Componente Principal 1', fontsize=12)
        ax.set_ylabel('Componente Principal 2', fontsize=12)
        ax.set_title('Clusters K-Means (Proyección PCA)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_svg(fig)
    
    def metrics_summary_chart(self, metrics: Dict) -> str:
        """
        Genera gráfico resumen de métricas de clustering.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Métricas de calidad
        ax1 = axes[0]
        metric_names = ['Silhouette\nScore', 'Calinski-Harabasz\n(normalizado)']
        silhouette = metrics.get('silhouette_score', 0)
        calinski = min(metrics.get('calinski_harabasz', 0) / 500, 1)  # Normalizar
        values = [silhouette, calinski]
        colors = ['#3B82F6', '#8B5CF6']
        
        bars = ax1.bar(metric_names, values, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylim(0, 1.2)
        ax1.set_title('Métricas de Calidad del Clustering', fontsize=12, fontweight='bold')
        ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Umbral aceptable')
        
        for bar, val in zip(bars, values):
            ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')
        
        # Gráfico 2: Distribución de riesgo (pie)
        ax2 = axes[1]
        risk_dist = metrics.get('risk_distribution', {})
        if risk_dist:
            labels = [k.replace('_', ' ') for k in risk_dist.keys()]
            sizes = list(risk_dist.values())
            colors = [self.RISK_COLORS.get(k, '#888888') for k in risk_dist.keys()]
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=[0.02]*len(sizes))
            ax2.set_title('Distribución de Riesgo', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_svg(fig)
