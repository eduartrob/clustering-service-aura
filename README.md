# AURA Clustering Service - Data Miner ETL

API REST con FastAPI para análisis de riesgo emocional de usuarios mediante Machine Learning.

## 🔐 Credenciales de Administrador

```
Email:    admin@aura.com
Usuario:  admin
Password: pezcadofrito.1
```

---

## 🚀 Quick Start (Docker)

```bash
# En EC2, después de clonar el repo
cd ~/orchestration-service-aura
git pull origin main

# Crear la base de datos (primera vez)
docker exec -it aura_postgres psql -U postgres -c "CREATE DATABASE aura_data_miner;"

# Build y levantar
docker compose build clustering-service
docker compose up -d clustering-service

# Verificar
curl http://localhost:8001/health
```

---

## 📚 Documentación Interactiva

| URL | Descripción |
|-----|-------------|
| http://localhost:8001/docs | **Swagger UI** (interactivo) |
| http://localhost:8001/redoc | ReDoc (documentación) |

---

## 🎨 Endpoints de Visualización para Admin Frontend

### Dashboard Completo
```
GET /api/v1/clustering/visualize/dashboard
```
Retorna HTML con dashboard completo incluyendo:
- Métricas generales (total usuarios, % alto riesgo, silhouette score)
- Distribución de riesgo (gráfico de barras)
- Proyección PCA (scatter plot)
- Índice de severidad (histograma)
- Perfil de clusters (radar chart)

### Gráficos Individuales (SVG embebido en HTML)

| Endpoint | Descripción | Uso |
|----------|-------------|-----|
| `GET /api/v1/clustering/visualize/scatter` | Scatter Plot PCA | Visualizar agrupamiento |
| `GET /api/v1/clustering/visualize/distribution` | Distribución de riesgo | Barras por nivel |
| `GET /api/v1/clustering/visualize/radar` | Radar Chart | Perfil KPIs por cluster |
| `GET /api/v1/clustering/visualize/severity` | Histograma severidad | Distribución índice |
| `GET /api/v1/clustering/visualize/kmeans` | Clusters K-Means | Visualización clusters |
| `GET /api/v1/clustering/visualize/metrics` | Métricas resumen | Calidad del clustering |

### Consulta de Usuarios

```
GET /api/v1/clustering/users/{risk_level}
```
Valores válidos: `ALTO_RIESGO`, `RIESGO_MODERADO`, `BAJO_RIESGO`

**Respuesta:**
```json
[
  {
    "user_id_raiz": "uuid-del-usuario",
    "risk_level": "ALTO_RIESGO",
    "severity_index": 0.75,
    "total_votes": 3
  }
]
```

---

## 🤖 Endpoint para Chat con IA

```
GET /api/v1/clustering/user-profile/{user_id}
```

**Respuesta:**
```json
{
  "user_id": "uuid",
  "risk_level": "ALTO_RIESGO",
  "severity_index": 0.68,
  "kpis": {
    "reciprocidad_social": 0.15,
    "dias_inactivo": 12,
    "mensajes_nocturnos": 0.45,
    "apatia_perfil": 0.8,
    "negatividad": 0.72,
    "participacion_comunitaria": 0.1
  },
  "has_data": true,
  "recommendation_context": "⚠️ Usuario identificado en ALTO RIESGO emocional. Responde con máxima empatía..."
}
```

---

## 🔄 Flujo ETL + Clustering

### 1. Ejecutar ETL (Extrae datos de todas las DBs)
```bash
curl -X POST "http://localhost:8001/api/v1/data-miner/execute-etl?skip_nlp=false"
```

### 2. Ejecutar Clustering (Clasifica usuarios)
```bash
curl -X POST "http://localhost:8001/api/v1/clustering/execute?n_clusters=4"
```

### 3. Ver Dashboard
```
http://localhost:8001/api/v1/clustering/visualize/dashboard
```

---

## 📊 KPIs Calculados

| KPI | Descripción | Señal de Riesgo |
|-----|-------------|-----------------|
| Ratio Reciprocidad | followers/following | Bajo = Aislamiento |
| Días Inactivo | Desde última conexión | Alto = Retirada |
| Mensajes Nocturnos | % mensajes 1-5am | Alto = Trastorno sueño |
| Apatía Perfil | Bio/perfil incompleto | Alto = Desinterés |
| Negatividad NLP | Sentimiento contenido | Alto = Estado negativo |
| Participación | Comunidades activas | Bajo = Poca red apoyo |

---

## 🗄️ Bases de Datos Conectadas

| DB | Propósito |
|----|-----------|
| `aura_auth` | Datos de usuarios |
| `aura_social` | Posts, perfiles, comunidades |
| `aura_messaging` | Mensajes, última conexión |
| `aura_data_miner` | **Vectores de características** (output) |

---

## 📅 Recomendación de Ejecución

| Proceso | Frecuencia | Descripción |
|---------|-----------|-------------|
| ETL Completo | Cada 6-12 horas | Actualiza vectores |
| Clustering | Después del ETL | Recalcula riesgos |
| Consulta en vivo | Por mensaje | `/user-profile/{id}` |
