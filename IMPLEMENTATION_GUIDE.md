# 🚀 Guía de Implementación: Clustering Service AURA (FastAPI ETL)

> **Fecha:** 2025-12-05  
> **Directorio:** `data-mining/clustering-service-aura`  
> **Referencias:** [Documentación de Persistencia](../doc/data-persistence.md), [Análisis de KPIs](../doc/clustering-kpis-analysis.md)  
> **Tecnologías:** Python 3.11+, FastAPI, SQLAlchemy, PostgreSQL, Transformers (NLP)

---

Este documento presenta las instrucciones específicas y la estructura del código para implementar el servicio **AURA-Data-Miner** utilizando FastAPI en un entorno virtual de Python. Este servicio ejecutará el flujo completo de **ETL (Extract, Transform, Load)** para generar el **Vector de Características del Usuario** (`user_feature_vector`), listo para el Clustering.

---

## 📑 Tabla de Contenidos

1. [Configuración del Entorno Virtual](#-1-configuración-del-entorno-virtual-y-dependencias)
2. [Estructura del Proyecto](#-2-estructura-del-proyecto)
3. [Variables de Entorno](#-3-variables-de-entorno-env)
4. [Migración de Base de Datos](#-4-migración-y-creación-de-la-base-de-datos-analítica)
5. [Conexión a las Bases de Datos](#-5-conexión-a-las-bases-de-datos)
6. [Lógica ETL](#-6-lógica-etl-extracción-transformación-carga)
7. [Análisis de Sentimiento NLP](#-7-módulo-de-análisis-de-sentimiento-nlp)
8. [Implementación FastAPI](#-8-implementación-del-api-rest-con-fastapi)
9. [Ejecución y Pruebas](#-9-ejecución-y-pruebas)
10. [Flujo de Datos Completo](#-10-diagrama-del-flujo-de-datos)

---

## 🔧 1. Configuración del Entorno Virtual y Dependencias

### 1.1. Crear y Activar el Entorno Virtual

```bash
# Navegar al directorio del servicio
cd data-mining/clustering-service-aura

# Crear el entorno virtual
python3 -m venv .venv

# Activar el entorno virtual (Linux/macOS)
source .venv/bin/activate

# Activar en Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 1.2. Archivo `requirements.txt`

Crea el archivo `requirements.txt` con las siguientes dependencias:

```txt
# Framework Web
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0

# Base de Datos
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.0

# Procesamiento de Datos / EDA
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2

# NLP / Análisis de Sentimiento
transformers==4.35.2
torch>=2.2.0
sentencepiece==0.1.99

# Utilidades
pydantic==2.5.2
pydantic-settings==2.1.0
```

### 1.3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

> [!TIP]
> Para entornos con GPU (CUDA), instala PyTorch con soporte CUDA para acelerar la inferencia NLP:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🏗️ 2. Estructura del Proyecto

Organiza el código en la siguiente estructura modular:

```
clustering-service-aura/
├── .venv/                          # Entorno virtual (ignorar en git)
├── .env                            # Variables de entorno (ignorar en git)
├── .env.example                    # Plantilla de variables de entorno
├── requirements.txt                # Dependencias Python
├── alembic.ini                     # Configuración de Alembic (migraciones)
├── alembic/                        # Directorio de migraciones
│   ├── versions/                   # Archivos de migración
│   └── env.py                      # Configuración del entorno de Alembic
├── app/
│   ├── __init__.py
│   ├── main.py                     # Punto de entrada FastAPI
│   ├── config.py                   # Configuración centralizada (Settings)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py           # Conexiones SQLAlchemy (Source y Target)
│   │   └── models.py               # Modelos SQLAlchemy (UserFeatureVector)
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── extractor.py            # Fase E: Extracción de datos
│   │   ├── transformer.py          # Fase T: Transformación y EDA
│   │   └── loader.py               # Fase L: Carga a DB Analítica
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py   # Análisis de sentimiento con Transformers
│   └── api/
│       ├── __init__.py
│       └── routes.py               # Definición de endpoints
└── tests/                          # Tests unitarios (opcional)
    └── test_etl.py
```

---

## ⚙️ 3. Variables de Entorno (`.env`)

Crea el archivo `.env` en la raíz del proyecto:

```env
# ===== CONFIGURACIÓN DE BASE DE DATOS =====

# URL de la DB Analítica (Target) - Donde se guardará el vector de características
# Esta base de datos será creada por el servicio de minería
DATABASE_URL_ANALYTICS=postgresql://postgres:postgres@localhost:5432/aura_data_miner

# URLs de las DBs Fuente (Source) - Lectura desde los microservicios
DATABASE_URL_AUTH=postgresql://postgres:postgres@localhost:5432/aura_auth
DATABASE_URL_SOCIAL=postgresql://postgres:postgres@localhost:5432/aura_social
DATABASE_URL_MESSAGING=postgresql://postgres:postgres@localhost:5432/aura_messaging

# ===== CONFIGURACIÓN DEL SERVICIO =====
SERVICE_NAME=clustering-service-aura
SERVICE_PORT=8001
DEBUG=True

# ===== CONFIGURACIÓN NLP =====
# Modelo de análisis de sentimiento en español
NLP_MODEL_NAME=UMUTeam/roberta-spanish-sentiment-analysis
# Alternativa: dccuchile/bert-base-spanish-wwm-uncased

# ===== DOCKER (Producción) =====
# Cambiar 'localhost' por 'db' cuando se ejecute dentro de Docker Compose
# DATABASE_URL_ANALYTICS=postgresql://postgres:postgres@db:5432/aura_data_miner
```

Crea también `.env.example` como plantilla (sin credenciales reales).

---

## 💾 4. Migración y Creación de la Base de Datos Analítica

### 4.1. Inicializar Alembic

```bash
# Inicializar Alembic en el proyecto
alembic init alembic
```

### 4.2. Configurar `alembic/env.py`

Modifica el archivo `alembic/env.py` para usar la configuración del proyecto:

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.models import Base
from app.config import settings

# Configuración de Alembic
config = context.config

# Sobrescribir sqlalchemy.url con la variable de entorno
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_ANALYTICS)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Ejecutar migraciones en modo 'offline'."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Ejecutar migraciones en modo 'online'."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 4.3. Crear la Migración Inicial

```bash
# Generar migración automática basada en los modelos
alembic revision --autogenerate -m "create_user_feature_vector_table"

# Aplicar la migración (crear la tabla en la DB)
alembic upgrade head
```

> [!IMPORTANT]
> Antes de ejecutar la migración, asegúrate de que la base de datos `aura_data_miner` exista en PostgreSQL:
> ```sql
> CREATE DATABASE aura_data_miner;
> ```

---

## 🔌 5. Conexión a las Bases de Datos

### `app/config.py` - Configuración Centralizada

```python
# app/config.py
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
```

### `app/database/connection.py` - Motores de Conexión

```python
# app/database/connection.py
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
```

### `app/database/models.py` - Modelo de Datos

```python
# app/database/models.py
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
```

---

## 🔄 6. Lógica ETL (Extracción, Transformación, Carga)

### `app/etl/extractor.py` - Fase E: Extracción

```python
# app/etl/extractor.py
import pandas as pd
from sqlalchemy import text
from app.database.connection import auth_engine, social_engine, messaging_engine
from app.config import settings


class DataExtractor:
    """Clase responsable de extraer datos desde las DBs fuente."""
    
    def __init__(self):
        self.auth_engine = auth_engine
        self.social_engine = social_engine
        self.messaging_engine = messaging_engine
    
    def extract_user_base(self) -> pd.DataFrame:
        """Extrae la base de usuarios desde Auth y Social."""
        query = text("""
            SELECT 
                u.user_id AS user_id_raiz,
                up.id AS profile_id_social,
                up.followers_count,
                up.following_count,
                up.bio IS NULL AS profile_bio_missing
            FROM users u
            LEFT JOIN user_profiles up ON u.user_id = up.user_id
            WHERE u.id_role = 2  -- Solo usuarios normales, no admins
        """)
        
        # Ejecutar en la DB de Auth, haciendo JOIN lógico con Social
        with self.auth_engine.connect() as conn:
            df_auth = pd.read_sql(query, conn)
        
        return df_auth
    
    def extract_social_metrics(self) -> pd.DataFrame:
        """Extrae métricas sociales (perfiles, comunidades, posts)."""
        query = text("""
            SELECT 
                up.user_id AS user_id_raiz,
                up.id AS profile_id_social,
                up.followers_count,
                up.following_count,
                up.bio IS NULL AS profile_bio_missing,
                cp.user_id IS NULL AS complete_profile_missing,
                COALESCE(cm_stats.distinct_categories, 0) AS num_community_categories,
                COALESCE(cm_stats.is_in_voluntariado, FALSE) AS comm_voluntariado_one_hot
            FROM user_profiles up
            LEFT JOIN complete_profiles cp ON up.id = cp.user_id
            LEFT JOIN (
                SELECT 
                    cm.user_id,
                    COUNT(DISTINCT c.category) AS distinct_categories,
                    MAX(CASE WHEN c.category = 'Voluntariado' THEN 1 ELSE 0 END)::BOOLEAN AS is_in_voluntariado
                FROM community_members cm
                INNER JOIN communities c ON cm.community_id = c.id
                GROUP BY cm.user_id
            ) cm_stats ON up.id = cm_stats.user_id
            WHERE up.is_active = TRUE
        """)
        
        with self.social_engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def extract_messaging_metrics(self) -> pd.DataFrame:
        """Extrae métricas de mensajería (última conexión, mensajes nocturnos)."""
        query = text("""
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
            WHERE u.is_active = TRUE
        """)
        
        with self.messaging_engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def extract_text_content(self) -> pd.DataFrame:
        """Extrae contenido de texto para análisis de sentimiento NLP."""
        # Posts
        posts_query = text("""
            SELECT user_id, content, 'post' AS source
            FROM posts 
            WHERE content IS NOT NULL AND is_active = TRUE
        """)
        
        # Comentarios
        comments_query = text("""
            SELECT user_id, content, 'comment' AS source
            FROM comments 
            WHERE content IS NOT NULL AND is_active = TRUE
        """)
        
        with self.social_engine.connect() as conn:
            df_posts = pd.read_sql(posts_query, conn)
            df_comments = pd.read_sql(comments_query, conn)
        
        # Mensajes (desde messaging DB)
        messages_query = text("""
            SELECT sender_profile_id AS user_id, content, 'message' AS source
            FROM messages 
            WHERE content IS NOT NULL AND is_deleted = FALSE
        """)
        
        with self.messaging_engine.connect() as conn:
            df_messages = pd.read_sql(messages_query, conn)
        
        # Combinar todos los textos
        return pd.concat([df_posts, df_comments, df_messages], ignore_index=True)
    
    def run_extraction(self) -> dict:
        """Ejecuta el proceso completo de extracción."""
        print("📥 Iniciando Fase E: Extracción de datos...")
        
        data = {
            'social_metrics': self.extract_social_metrics(),
            'messaging_metrics': self.extract_messaging_metrics(),
            'text_content': self.extract_text_content()
        }
        
        print(f"   ✅ Social: {len(data['social_metrics'])} registros")
        print(f"   ✅ Messaging: {len(data['messaging_metrics'])} registros")
        print(f"   ✅ Textos: {len(data['text_content'])} registros")
        
        return data
```

### `app/etl/transformer.py` - Fase T: Transformación y EDA

```python
# app/etl/transformer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from app.nlp.sentiment_analyzer import SentimentAnalyzer


class DataTransformer:
    """Clase responsable de transformar y normalizar los datos (EDA)."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.sentiment_analyzer = SentimentAnalyzer()
    
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
            if row['followers_count'] > 0 else 0.0,
            axis=1
        )
        
        # KPI 2: Días desde última conexión
        now = datetime.utcnow()
        df['days_since_last_seen'] = df['last_seen_at'].apply(
            lambda x: (now - x).total_seconds() / 86400 if pd.notna(x) else 365.0  # Default: 1 año
        )
        
        # KPI 3: Ratio de mensajes nocturnos
        df['ratio_night_messages'] = df.apply(
            lambda row: row['night_messages'] / row['total_messages'] 
            if row['total_messages'] > 0 else 0.0,
            axis=1
        )
        
        # KPI 4: Perfil incompleto (booleano)
        df['is_profile_incomplete'] = (
            df['profile_bio_missing'] & df['complete_profile_missing']
        )
        
        # KPI 6: Densidad comunitaria (ya extraída como num_community_categories)
        
        return df
    
    def calculate_sentiment_kpi(self, df: pd.DataFrame, text_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula el KPI 5 (Índice de Negatividad) usando NLP."""
        print("🧠 Calculando KPI 5 (Análisis de Sentimiento NLP)...")
        
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
        
        # Manejar valores faltantes antes de normalizar
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Aplicar normalización
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Renombrar columnas normalizadas
        df.rename(columns={
            'reciprocity_ratio': 'reciprocity_ratio_norm',
            'days_since_last_seen': 'days_since_last_seen_norm',
            'num_community_categories': 'num_community_categories_norm'
        }, inplace=True)
        
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
        
        return df[final_columns]
    
    def run_transformation(self, data: dict) -> pd.DataFrame:
        """Ejecuta el proceso completo de transformación."""
        print("🔄 Iniciando Fase T: Transformación de datos...")
        
        # 1. Unificar datasets
        df = self.merge_datasets(data)
        
        # 2. Calcular KPIs base
        df = self.calculate_kpis(df)
        
        # 3. Calcular KPI 5 (NLP) si hay datos de texto
        if not data['text_content'].empty:
            df = self.calculate_sentiment_kpi(df, data['text_content'])
        else:
            df['sentiment_negativity_index'] = 0.0
        
        # 4. Normalizar características
        df = self.normalize_features(df)
        
        # 5. Seleccionar columnas finales
        df = self.select_final_columns(df)
        
        print(f"   ✅ Transformación completa: {len(df)} registros procesados")
        
        return df
```

### `app/etl/loader.py` - Fase L: Carga

```python
# app/etl/loader.py
import pandas as pd
from sqlalchemy import text
from datetime import datetime
from app.database.connection import analytics_engine, AnalyticsSessionLocal
from app.database.models import UserFeatureVector


class DataLoader:
    """Clase responsable de cargar los datos transformados a la DB Analítica."""
    
    def __init__(self):
        self.engine = analytics_engine
        self.table_name = UserFeatureVector.__tablename__
    
    def truncate_table(self):
        """Elimina todos los registros existentes antes de cargar nuevos."""
        print("🗑️ Truncando tabla existente...")
        with self.engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY;"))
    
    def load_dataframe(self, df: pd.DataFrame):
        """Carga el DataFrame a la base de datos."""
        print(f"💾 Cargando {len(df)} registros a la DB Analítica...")
        
        # Agregar fecha de extracción
        df['extraction_date'] = datetime.utcnow()
        
        # Usar pandas to_sql para carga masiva eficiente
        df.to_sql(
            name=self.table_name,
            con=self.engine,
            if_exists='append',
            index=False,
            method='multi',  # Inserciones en lote
            chunksize=1000   # Procesar en bloques de 1000
        )
    
    def run_load(self, df: pd.DataFrame, truncate_before: bool = True):
        """Ejecuta el proceso completo de carga."""
        print("📤 Iniciando Fase L: Carga de datos...")
        
        if df.empty:
            print("   ⚠️ No hay datos para cargar.")
            return 0
        
        if truncate_before:
            self.truncate_table()
        
        self.load_dataframe(df)
        
        print(f"   ✅ Carga completa: {len(df)} registros insertados")
        
        return len(df)
```

---

## 🧠 7. Módulo de Análisis de Sentimiento (NLP)

### `app/nlp/sentiment_analyzer.py`

```python
# app/nlp/sentiment_analyzer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import List
from app.config import settings


class SentimentAnalyzer:
    """Analizador de sentimiento usando modelos Transformer en español."""
    
    def __init__(self):
        self.model_name = settings.NLP_MODEL_NAME
        self.device = 0 if torch.cuda.is_available() else -1  # GPU si disponible
        self._pipeline = None
    
    @property
    def sentiment_pipeline(self):
        """Inicialización lazy del pipeline (solo cuando se necesita)."""
        if self._pipeline is None:
            print(f"🔄 Cargando modelo NLP: {self.model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            
            print("   ✅ Modelo NLP cargado correctamente")
        
        return self._pipeline
    
    def analyze_text(self, text: str) -> float:
        """
        Analiza el sentimiento de un texto individual.
        Retorna la probabilidad de sentimiento negativo (0.0 - 1.0).
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        try:
            # Truncar texto muy largo
            text = text[:512]
            
            result = self.sentiment_pipeline(text)[0]
            
            # Buscar el score de la etiqueta negativa
            neg_score = next(
                (item['score'] for item in result 
                 if item['label'].upper() in ['NEG', 'NEGATIVE', 'LABEL_0']),
                0.0
            )
            
            return neg_score
            
        except Exception as e:
            print(f"   ⚠️ Error analizando texto: {e}")
            return 0.0
    
    def calculate_negativity_index(self, texts: List[str]) -> float:
        """
        Calcula el índice de negatividad promedio para una lista de textos.
        Retorna un valor entre 0.0 y 1.0.
        """
        if not texts or len(texts) == 0:
            return 0.0
        
        scores = []
        for text in texts:
            score = self.analyze_text(text)
            if score > 0:  # Solo incluir análisis exitosos
                scores.append(score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Analiza múltiples textos en lotes para mayor eficiencia.
        """
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:512] if t else "" for t in batch]  # Truncar
            
            try:
                results = self.sentiment_pipeline(batch)
                
                for result in results:
                    neg_score = next(
                        (item['score'] for item in result 
                         if item['label'].upper() in ['NEG', 'NEGATIVE', 'LABEL_0']),
                        0.0
                    )
                    scores.append(neg_score)
                    
            except Exception as e:
                print(f"   ⚠️ Error en lote: {e}")
                scores.extend([0.0] * len(batch))
        
        return scores
```

---

## 🎯 8. Implementación del API REST con FastAPI

### `app/api/routes.py` - Definición de Endpoints

```python
# app/api/routes.py
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from app.etl.extractor import DataExtractor
from app.etl.transformer import DataTransformer
from app.etl.loader import DataLoader

router = APIRouter(prefix="/api/v1/data-miner", tags=["ETL Pipeline"])


class ETLResponse(BaseModel):
    """Modelo de respuesta para el endpoint ETL."""
    status: str
    message: str
    records_processed: int
    extraction_date: datetime
    next_step: str


class HealthResponse(BaseModel):
    """Modelo de respuesta para el endpoint de salud."""
    status: str
    service: str
    ready_for_etl: bool


@router.get("/status", response_model=HealthResponse)
def get_status():
    """Endpoint para verificar que el API está activo."""
    return HealthResponse(
        status="ok",
        service="AURA Data Miner",
        ready_for_etl=True
    )


@router.post("/execute-etl", response_model=ETLResponse)
def execute_etl_pipeline():
    """
    Ejecuta el flujo completo de ETL:
    1. Extracción (E): Datos desde las DB de Microservicios.
    2. Transformación (T): Normalización y Vectorización (EDA).
    3. Carga (L): A la DB Analítica (Target).
    """
    try:
        print("\n" + "="*60)
        print("🚀 INICIANDO PIPELINE ETL - AURA DATA MINER")
        print("="*60 + "\n")
        
        # Inicializar componentes
        extractor = DataExtractor()
        transformer = DataTransformer()
        loader = DataLoader()
        
        # FASE E: Extracción
        raw_data = extractor.run_extraction()
        
        if raw_data['social_metrics'].empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontraron registros de usuario válidos para el EDA."
            )
        
        # FASE T: Transformación
        processed_df = transformer.run_transformation(raw_data)
        
        # FASE L: Carga
        records_loaded = loader.run_load(processed_df)
        
        print("\n" + "="*60)
        print("✅ PIPELINE ETL COMPLETADO EXITOSAMENTE")
        print("="*60 + "\n")
        
        return ETLResponse(
            status="success",
            message="Flujo ETL de Vectorización completado con éxito.",
            records_processed=records_loaded,
            extraction_date=datetime.utcnow(),
            next_step="La tabla 'user_feature_vector' está lista para el algoritmo de Clustering."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO EN PIPELINE ETL: {e}\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en la ejecución del pipeline: {str(e)}"
        )


@router.post("/execute-etl-async")
async def execute_etl_async(background_tasks: BackgroundTasks):
    """
    Ejecuta el ETL de forma asíncrona en segundo plano.
    Útil para procesos largos sin bloquear el API.
    """
    def run_etl_background():
        try:
            extractor = DataExtractor()
            transformer = DataTransformer()
            loader = DataLoader()
            
            raw_data = extractor.run_extraction()
            processed_df = transformer.run_transformation(raw_data)
            loader.run_load(processed_df)
            
            print("✅ ETL en background completado")
        except Exception as e:
            print(f"❌ Error en ETL background: {e}")
    
    background_tasks.add_task(run_etl_background)
    
    return {
        "status": "accepted",
        "message": "El proceso ETL se está ejecutando en segundo plano.",
        "check_status_at": "/api/v1/data-miner/status"
    }
```

### `app/main.py` - Punto de Entrada FastAPI

```python
# app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.config import settings
from app.database.models import Base
from app.database.connection import analytics_engine
from app.api.routes import router as etl_router

# Cargar variables de entorno
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hook de ciclo de vida: ejecutar al iniciar y cerrar el servicio."""
    # Startup
    print(f"🚀 Iniciando {settings.SERVICE_NAME}...")
    
    # Crear tablas si no existen (alternativa a Alembic para desarrollo)
    if settings.DEBUG:
        Base.metadata.create_all(bind=analytics_engine)
        print("   ✅ Tablas de DB Analítica verificadas/creadas")
    
    print(f"   ✅ Servicio listo en puerto {settings.SERVICE_PORT}")
    
    yield
    
    # Shutdown
    print(f"👋 Cerrando {settings.SERVICE_NAME}...")


# Inicializar aplicación FastAPI
app = FastAPI(
    title="AURA Data Miner API",
    description="""
    API REST para la ejecución del flujo ETL (Extract, Transform, Load) 
    que genera el Vector de Características del Usuario para Clustering.
    
    ## Funcionalidades
    
    * **Extracción**: Obtiene datos desde las DBs de los microservicios (Auth, Social, Messaging)
    * **Transformación**: Calcula KPIs, aplica análisis NLP de sentimiento, y normaliza datos
    * **Carga**: Persiste el vector de características en la DB Analítica
    
    ## Endpoints Principales
    
    * `GET /api/v1/data-miner/status` - Verificar estado del servicio
    * `POST /api/v1/data-miner/execute-etl` - Ejecutar pipeline ETL completo
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(etl_router)


@app.get("/", tags=["Root"])
def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "status": "/api/v1/data-miner/status",
            "execute_etl": "/api/v1/data-miner/execute-etl"
        }
    }


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG
    )
```

---

## 🏃 9. Ejecución y Pruebas

### 9.1. Crear la Base de Datos Analítica

Antes de ejecutar el servicio, asegúrate de crear la base de datos:

```bash
# Conectar a PostgreSQL
psql -U postgres -h localhost -p 5432

# Crear la base de datos
CREATE DATABASE aura_data_miner;

# Salir
\q
```

### 9.2. Ejecutar Migraciones

```bash
# Aplicar migraciones de Alembic
alembic upgrade head
```

### 9.3. Iniciar el Servicio

```bash
# Desde el directorio clustering-service-aura
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 9.4. Probar los Endpoints

```bash
# Verificar estado del servicio
curl http://localhost:8001/api/v1/data-miner/status

# Ejecutar el pipeline ETL
curl -X POST http://localhost:8001/api/v1/data-miner/execute-etl
```

### 9.5. Acceder a la Documentación Interactiva

Abre en tu navegador:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## 📊 10. Diagrama del Flujo de Datos

```mermaid
flowchart TD
    subgraph Sources["🗄️ Bases de Datos Fuente"]
        AUTH[(aura_auth)]
        SOCIAL[(aura_social)]
        MSG[(aura_messaging)]
    end
    
    subgraph ETL["🔄 Pipeline ETL (FastAPI)"]
        E[📥 Extractor]
        T[🔄 Transformer]
        NLP[🧠 NLP Analyzer]
        L[📤 Loader]
    end
    
    subgraph Target["💾 Base de Datos Analítica"]
        ANALYTICS[(aura_data_miner)]
        UFV[user_feature_vector]
    end
    
    subgraph Next["🎯 Siguiente Fase"]
        CLUSTER[🔮 Algoritmo de Clustering]
    end
    
    AUTH --> E
    SOCIAL --> E
    MSG --> E
    
    E --> T
    T --> NLP
    NLP --> T
    T --> L
    
    L --> ANALYTICS
    ANALYTICS --> UFV
    
    UFV --> CLUSTER
```

---

## 📝 Notas Finales

> [!IMPORTANT]
> **Consideraciones de Seguridad:**
> - Las credenciales de las DBs fuente deben tener permisos de **solo lectura**.
> - Nunca hardcodear contraseñas en el código; usar siempre variables de entorno.
> - En producción, usar secrets managers (AWS Secrets Manager, HashiCorp Vault).

> [!TIP]
> **Optimizaciones Sugeridas:**
> - Implementar un scheduler (Celery + Redis) para ejecutar el ETL periódicamente.
> - Agregar logging estructurado con `structlog` o `loguru`.
> - Implementar health checks para monitoreo con Prometheus/Grafana.

---

*Documento generado para el proyecto AURA - Data Mining Service*
