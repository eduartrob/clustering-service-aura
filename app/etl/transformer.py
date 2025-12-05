# app/etl/transformer.py
"""Fase T (Transform): Transformación, cálculo de KPIs y normalización."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import Optional


class DataTransformer:
    """Clase responsable de transformar y normalizar los datos (EDA)."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self._sentiment_analyzer = None
    
    @property
    def sentiment_analyzer(self):
        """Lazy loading del analizador de sentimiento."""
        if self._sentiment_analyzer is None:
            from app.nlp.sentiment_analyzer import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer
    
    def merge_datasets(self, data: dict) -> pd.DataFrame:
        """Unifica los datasets extraídos en un único DataFrame."""
        print("🔗 Unificando datasets...")
        
        df_social = data['social_metrics']
        df_messaging = data['messaging_metrics']
        
        # Merge Social + Messaging usando profile_id
        df_merged = pd.merge(
            df_social,
            df_messaging,
            left_on='profile_id_social',
            right_on='profile_id',
            how='left'
        )
        
        return df_merged
    
    def calculate_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula los KPIs a partir de los datos crudos."""
        print("📊 Calculando KPIs...")
        
        # KPI 1: Ratio de Reciprocidad Social
        df['reciprocity_ratio'] = df.apply(
            lambda row: row['following_count'] / row['followers_count'] 
            if pd.notna(row['followers_count']) and row['followers_count'] > 0 else 0.0,
            axis=1
        )
        
        # KPI 2: Días desde última conexión
        now = datetime.utcnow()
        df['days_since_last_seen'] = df['last_seen_at'].apply(
            lambda x: (now - x).total_seconds() / 86400 if pd.notna(x) else 365.0
        )
        
        # KPI 3: Ratio de mensajes nocturnos
        df['ratio_night_messages'] = df.apply(
            lambda row: row['night_messages'] / row['total_messages'] 
            if pd.notna(row['total_messages']) and row['total_messages'] > 0 else 0.0,
            axis=1
        )
        
        # KPI 4: Perfil incompleto (booleano)
        df['is_profile_incomplete'] = (
            df['profile_bio_missing'].fillna(True) & 
            df['complete_profile_missing'].fillna(True)
        )
        
        return df
    
    def calculate_sentiment_kpi(self, df: pd.DataFrame, text_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula el KPI 5 (Índice de Negatividad) usando NLP."""
        print("🧠 Calculando KPI 5 (Análisis de Sentimiento NLP)...")
        
        if text_data.empty:
            df['sentiment_negativity_index'] = 0.0
            return df
        
        # Agrupar textos por usuario
        user_texts = text_data.groupby('user_id')['content'].apply(list).reset_index()
        user_texts.columns = ['user_id', 'texts']
        
        # Calcular índice de negatividad para cada usuario
        user_texts['sentiment_negativity_index'] = user_texts['texts'].apply(
            self.sentiment_analyzer.calculate_negativity_index
        )
        
        # Merge con el DataFrame principal
        df = pd.merge(
            df,
            user_texts[['user_id', 'sentiment_negativity_index']],
            left_on='user_id_raiz',
            right_on='user_id',
            how='left'
        )
        
        # Rellenar NaN con 0 (usuarios sin textos)
        df['sentiment_negativity_index'] = df['sentiment_negativity_index'].fillna(0.0)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza las características numéricas usando MinMaxScaler."""
        print("⚖️ Normalizando características...")
        
        # Columnas a normalizar
        numerical_cols = [
            'reciprocity_ratio',
            'days_since_last_seen',
            'ratio_night_messages',
            'sentiment_negativity_index',
            'num_community_categories'
        ]
        
        # Filtrar columnas que existen en el DataFrame
        existing_cols = [col for col in numerical_cols if col in df.columns]
        
        # Manejar valores faltantes antes de normalizar
        for col in existing_cols:
            df[col] = df[col].fillna(0.0)
        
        # Aplicar normalización
        if existing_cols:
            df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
        
        # Renombrar columnas normalizadas
        rename_map = {
            'reciprocity_ratio': 'reciprocity_ratio_norm',
            'days_since_last_seen': 'days_since_last_seen_norm',
            'num_community_categories': 'num_community_categories_norm'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        return df
    
    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona y ordena las columnas finales para la carga."""
        final_columns = [
            'user_id_raiz',
            'reciprocity_ratio_norm',
            'days_since_last_seen_norm',
            'ratio_night_messages',
            'is_profile_incomplete',
            'sentiment_negativity_index',
            'num_community_categories_norm',
            'int_gaming_one_hot',
            'comm_voluntariado_one_hot'
        ]
        
        # Asegurar que las columnas one-hot existan
        if 'int_gaming_one_hot' not in df.columns:
            df['int_gaming_one_hot'] = False
        if 'comm_voluntariado_one_hot' not in df.columns:
            df['comm_voluntariado_one_hot'] = False
            
        # Asegurar que todas las columnas existan
        for col in final_columns:
            if col not in df.columns:
                df[col] = 0.0 if 'norm' in col or 'ratio' in col or 'index' in col else False
        
        return df[final_columns]
    
    def run_transformation(self, data: dict, skip_nlp: bool = False) -> pd.DataFrame:
        """Ejecuta el proceso completo de transformación."""
        print("🔄 Iniciando Fase T: Transformación de datos...")
        
        # 1. Unificar datasets
        df = self.merge_datasets(data)
        
        if df.empty:
            print("   ⚠️ No hay datos para transformar.")
            return pd.DataFrame()
        
        # 2. Calcular KPIs base
        df = self.calculate_kpis(df)
        
        # 3. Calcular KPI 5 (NLP) si hay datos de texto y no se omite
        if not skip_nlp and not data['text_content'].empty:
            df = self.calculate_sentiment_kpi(df, data['text_content'])
        else:
            df['sentiment_negativity_index'] = 0.0
            if skip_nlp:
                print("   ⏭️ Análisis NLP omitido (skip_nlp=True)")
        
        # 4. Normalizar características
        df = self.normalize_features(df)
        
        # 5. Seleccionar columnas finales
        df = self.select_final_columns(df)
        
        print(f"   ✅ Transformación completa: {len(df)} registros procesados")
        
        return df
