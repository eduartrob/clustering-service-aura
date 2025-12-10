# 🚀 AURA Clustering Service - Data Miner ETL

> **Microservicio de Minería de Datos para Detección de Riesgos Psicoemocionales**  
> **Tecnologías:** Python 3.11+, FastAPI, SQLAlchemy, PostgreSQL, Transformers (NLP), WebSockets

API REST con FastAPI para la ejecución del flujo **ETL (Extract, Transform, Load)** que genera el **Vector de Características del Usuario** para Clustering de detección de riesgos en poblaciones juveniles.

---

## 📑 Tabla de Contenidos

1. [Quick Start](#-quick-start)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Variables de Entorno](#-variables-de-entorno)
4. [Endpoints API](#-endpoints-api)
5. [KPIs Calculados](#-kpis-calculados)
6. [Módulo de Clustering](#-módulo-de-clustering)
7. [Sistema en Tiempo Real (v2)](#-sistema-en-tiempo-real-v2)
8. [Configuración de Base de Datos](#-configuración-de-base-de-datos)
9. [Lógica ETL](#-lógica-etl)
10. [Análisis de Sentimiento NLP](#-análisis-de-sentimiento-nlp)

---

## 🚀 Quick Start

```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear la base de datos
sudo -u postgres psql -c "CREATE DATABASE aura_data_miner;"

# 4. Configurar permisos
sudo -u postgres psql -c "CREATE USER miner_user WITH PASSWORD 'miner_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aura_data_miner TO miner_user;"
sudo -u postgres psql -c "ALTER DATABASE aura_data_miner OWNER TO miner_user;"

# 5. Ejecutar migraciones
alembic upgrade head

# 6. Iniciar el servicio
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 📚 Documentación Interactiva

- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

---

## 🏗️ Estructura del Proyecto

```
clustering-service-aura/
├── .venv/                          # Entorno virtual
├── .env                            # Variables de entorno
├── requirements.txt                # Dependencias Python
├── alembic.ini                     # Configuración de Alembic
├── alembic/                        # Migraciones de base de datos
│   ├── versions/
│   └── env.py
├── app/
│   ├── __init__.py
│   ├── main.py                     # Punto de entrada FastAPI
│   ├── config.py                   # Configuración centralizada
│   ├── database/
│   │   ├── connection.py           # Conexiones SQLAlchemy
│   │   └── models.py               # Modelos (UserFeatureVector)
│   ├── etl/
│   │   ├── extractor.py            # Fase E: Extracción
│   │   ├── transformer.py          # Fase T: Transformación
│   │   └── loader.py               # Fase L: Carga
│   ├── clustering/
│   │   ├── ensemble.py             # Ensamble K-Means + DBSCAN + IsolationForest
│   │   └── visualizer.py           # Generador de gráficos SVG
│   ├── nlp/
│   │   └── sentiment_analyzer.py   # Análisis de sentimiento con Transformers
│   ├── realtime/                   # NUEVO: Módulo de tiempo real
│   │   ├── websocket_manager.py    # Gestor de conexiones WebSocket
│   │   ├── db_listener.py          # PostgreSQL LISTEN/NOTIFY (CDC)
│   │   └── streaming_pipeline.py   # ETL incremental
│   └── api/
│       ├── routes.py               # Endpoints ETL (v1)
│       ├── clustering_routes.py    # Endpoints Clustering (v1)
│       └── websocket_routes.py     # Endpoints WebSocket + JSON (v2)
└── sql/
    └── triggers/
        └── notify_triggers.sql     # Triggers para CDC
```

---

## ⚙️ Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Base de Datos Analítica (Target)
DATABASE_URL_ANALYTICS=postgresql://postgres:postgres@localhost:5432/aura_data_miner

# Bases de Datos Fuente (Source) - Solo lectura
DATABASE_URL_AUTH=postgresql://postgres:postgres@localhost:5432/aura_auth
DATABASE_URL_SOCIAL=postgresql://postgres:postgres@localhost:5432/aura_social
DATABASE_URL_MESSAGING=postgresql://postgres:postgres@localhost:5432/aura_messaging

# Configuración del Servicio
SERVICE_NAME=clustering-service-aura
SERVICE_PORT=8001
DEBUG=True

# Modelo NLP (Análisis de sentimiento en español)
NLP_MODEL_NAME=UMUTeam/roberta-spanish-sentiment-analysis
```

---

## 🔗 Endpoints API

### API v1 - Pipeline ETL y Clustering

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/data-miner/status` | Estado del servicio |
| POST | `/api/v1/data-miner/execute-etl` | Ejecutar pipeline ETL |
| POST | `/api/v1/data-miner/execute-etl-async` | ETL en background |
| GET | `/api/v1/data-miner/feature-vector/count` | Conteo de registros |
| POST | `/api/v1/clustering/execute` | Ejecutar Clustering |
| GET | `/api/v1/clustering/results` | Resultados del clustering |
| GET | `/api/v1/clustering/visualize/dashboard` | Dashboard SVG completo |
| GET | `/api/v1/clustering/visualize/scatter` | Scatter Plot PCA |
| GET | `/api/v1/clustering/visualize/distribution` | Distribución de riesgo |
| GET | `/api/v1/clustering/visualize/radar` | Radar Chart de Clusters |
| GET | `/api/v1/clustering/users/{risk_level}` | Usuarios por nivel de riesgo |

### API v2 - Tiempo Real (WebSocket + ChartJS)

| Tipo | Endpoint | Descripción |
|------|----------|-------------|
| WebSocket | `/api/v2/clustering/ws/live` | Actualizaciones en tiempo real |
| WebSocket | `/api/v2/clustering/ws/alerts` | Alertas críticas |
| GET | `/api/v2/clustering/data/distribution` | JSON para ChartJS (Bar/Pie) |
| GET | `/api/v2/clustering/data/scatter` | JSON para ChartJS (Scatter) |
| GET | `/api/v2/clustering/data/radar` | JSON para ChartJS (Radar) |
| GET | `/api/v2/clustering/data/severity-histogram` | Histograma de severidad |
| GET | `/api/v2/clustering/data/kpi-trends?hours=24` | Tendencias temporales |
| GET | `/api/v2/clustering/data/high-risk-users` | Usuarios en alto riesgo |
| GET | `/api/v2/clustering/status` | Estado del sistema real-time |

---

## 📊 KPIs Calculados

El sistema calcula 6 indicadores clave de riesgo psicoemocional:

| KPI | Nombre | Indicador de Riesgo |
|:---:|:-------|:--------------------|
| 1 | **Ratio de Reciprocidad Social** | Aislamiento (sigue a muchos sin ser seguido) |
| 2 | **Días desde Última Conexión** | Abandono/retiro de la plataforma |
| 3 | **Ratio de Mensajes Nocturnos** | Desorden circadiano (insomnio, ansiedad) |
| 4 | **Índice de Apatía del Perfil** | Perfil incompleto = anhedonia |
| 5 | **Índice de Negatividad (NLP)** | Depresión, crisis emocional |
| 6 | **Densidad Comunitaria** | Red de apoyo limitada |

---

## 🔮 Módulo de Clustering

### Sistema de Votación (Ensemble)

El sistema usa 3 algoritmos que votan independientemente:

1. **K-Means** → Identifica cluster con peor perfil de KPIs
2. **DBSCAN** → Detecta outliers estadísticos
3. **Isolation Forest** → Detecta anomalías comportamentales

**Regla de decisión:**
- ≥2 votos → 🔴 **ALTO RIESGO** (Intervención prioritaria)
- 1 voto → 🟡 **RIESGO MODERADO** (Monitoreo)
- 0 votos → 🟢 **BAJO RIESGO** (Normal)

### Índice de Severidad de Anomalía (ASI)

```
ASI = 0.5×(1-Score_IsolationForest) + 0.3×(Outlier_DBSCAN) + 0.2×(Distancia_Centroide)
```

Escala de 0-100 para priorizar intervención.

---

## 🚀 Sistema en Tiempo Real (v2)

### Arquitectura

```
[Microservicios AURA] → [PostgreSQL Trigger] → [pg_notify]
         ↓
[DatabaseListener] → [StreamingETLPipeline] → [WebSocketManager]
         ↓
[Clientes React/ChartJS]
```

### Activar Triggers SQL

Ejecutar en las bases de datos de los microservicios:

```bash
# aura_messaging
psql -U postgres -d aura_messaging -f sql/triggers/notify_triggers.sql

# aura_social
psql -U postgres -d aura_social -f sql/triggers/notify_triggers.sql
```

### Conectar desde React

```javascript
// WebSocket para actualizaciones en vivo
const ws = new WebSocket('ws://localhost:8001/api/v2/clustering/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'USER_RISK_UPDATE') {
    // Actualizar gráfica ChartJS
    updateChart(data.payload);
  }
  
  if (data.type === 'CRITICAL_ALERT') {
    // Mostrar notificación
    showAlert(data.payload);
  }
};

// Obtener datos para ChartJS
const response = await fetch('/api/v2/clustering/data/distribution');
const chartData = await response.json();
// chartData ya está en formato compatible con ChartJS
```

---

## 💾 Configuración de Base de Datos

### Crear Base de Datos Analítica

```sql
CREATE DATABASE aura_data_miner;
```

### Ejecutar Migraciones

```bash
# Generar migración automática
alembic revision --autogenerate -m "create_user_feature_vector_table"

# Aplicar migración
alembic upgrade head
```

### Modelo de Datos: `user_feature_vector`

| Campo | Tipo | Descripción |
|:------|:-----|:------------|
| `id` | SERIAL | Clave primaria |
| `user_id_raiz` | UUID | ID del usuario (Auth Service) |
| `extraction_date` | TIMESTAMP | Fecha del ETL |
| `reciprocity_ratio_norm` | FLOAT | KPI 1 normalizado |
| `days_since_last_seen_norm` | FLOAT | KPI 2 normalizado |
| `ratio_night_messages` | FLOAT | KPI 3 |
| `is_profile_incomplete` | BOOLEAN | KPI 4 |
| `sentiment_negativity_index` | FLOAT | KPI 5 (NLP) |
| `num_community_categories_norm` | FLOAT | KPI 6 normalizado |
| `cluster_label` | VARCHAR | Resultado del clustering |

---

## 🔄 Lógica ETL

### Fase E: Extracción (`app/etl/extractor.py`)

Extrae datos de 3 bases de datos:
- **aura_auth**: Usuarios base
- **aura_social**: Perfiles, posts, comentarios, comunidades
- **aura_messaging**: Mensajes, última conexión

### Fase T: Transformación (`app/etl/transformer.py`)

1. Unificación de datasets (merge por user_id)
2. Cálculo de los 6 KPIs
3. Análisis NLP de sentimiento
4. Normalización con MinMaxScaler

### Fase L: Carga (`app/etl/loader.py`)

Inserta los vectores de características en `user_feature_vector` usando pandas `to_sql` con inserción masiva.

---

## 🧠 Análisis de Sentimiento NLP

### Modelo Utilizado

**UMUTeam/roberta-spanish-sentiment-analysis** - RoBERTa fine-tuned para español.

### Proceso

1. Extrae textos (posts, comentarios, mensajes) por usuario
2. Tokeniza y procesa con el modelo Transformer
3. Calcula probabilidad de sentimiento negativo
4. Promedia por usuario → `sentiment_negativity_index`

### Optimización GPU

```bash
# Para acelerar inferencia con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Flujo de Datos Completo

```mermaid
flowchart TD
    subgraph Sources["🗄️ Bases de Datos Fuente"]
        AUTH[(aura_auth)]
        SOCIAL[(aura_social)]
        MSG[(aura_messaging)]
    end
    
    subgraph ETL["🔄 Pipeline ETL"]
        E[📥 Extractor]
        T[🔄 Transformer]
        NLP[🧠 NLP Analyzer]
        L[📤 Loader]
    end
    
    subgraph Clustering["🔮 Clustering Ensemble"]
        KMEANS[K-Means]
        DBSCAN[DBSCAN]
        ISO[Isolation Forest]
        VOTE[Votación]
    end
    
    subgraph Output["📊 Salidas"]
        SVG[Dashboard SVG]
        WS[WebSocket Live]
        JSON[API JSON v2]
    end
    
    AUTH --> E
    SOCIAL --> E
    MSG --> E
    
    E --> T
    T <--> NLP
    T --> L
    
    L --> KMEANS
    L --> DBSCAN
    L --> ISO
    
    KMEANS --> VOTE
    DBSCAN --> VOTE
    ISO --> VOTE
    
    VOTE --> SVG
    VOTE --> WS
    VOTE --> JSON
```

---

## 🔒 Consideraciones de Seguridad

> **Importante:**
> - Las credenciales de las DBs fuente deben tener permisos de **solo lectura**
> - Nunca hardcodear contraseñas en el código
> - En producción, usar secrets managers (AWS Secrets Manager, HashiCorp Vault)

---

## 📝 Próximas Mejoras

- [ ] Scheduler con Celery + Redis para ETL periódico
- [ ] Logging estructurado con `structlog`
- [ ] Health checks para Prometheus/Grafana
- [ ] Autenticación JWT para endpoints WebSocket

---

*Microservicio desarrollado para el proyecto AURA - Sistema de Detección de Riesgos Psicoemocionales en Poblaciones Juveniles*
