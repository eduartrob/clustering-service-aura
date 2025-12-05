# AURA Clustering Service - API Documentation

## 🔐 Credenciales Admin
```
Email:    admin@aura.com
Usuario:  admin
Password: pezcadofrito.1
```

## 🌐 Puerto y Acceso
```
Puerto: 8001
URL Base: http://<IP-EC2>:8001
Swagger: http://<IP-EC2>:8001/docs
```

> **⚠️ Importante:** Abrir puerto 8001 en Security Groups de AWS EC2 (TCP Inbound)

---

## 📊 API Endpoints para Admin Frontend

### Base URL
```
http://<IP-EC2>:8001/api/v1
```

---

## 1️⃣ ETL - Ejecutar Pipeline (Requisito Previo)

### POST `/data-miner/execute-etl`
Extrae datos de todas las DBs y genera vectores de características.

```bash
curl -X POST "http://<IP>:8001/api/v1/data-miner/execute-etl?skip_nlp=false"
```

**Response:**
```json
{
  "status": "success",
  "message": "Flujo ETL de Vectorización completado con éxito.",
  "records_processed": 25,
  "extraction_date": "2025-12-05T18:00:00Z",
  "next_step": "La tabla 'user_feature_vector' está lista para el algoritmo de Clustering."
}
```

---

## 2️⃣ Clustering - Ejecutar Análisis

### POST `/clustering/execute`
Ejecuta K-Means, DBSCAN e Isolation Forest.

```bash
curl -X POST "http://<IP>:8001/api/v1/clustering/execute?n_clusters=4"
```

**Parámetros:**
- `n_clusters` (int, default=4): Número de clusters
- `contamination` (float, default=0.1): Proporción de anomalías

**Response:**
```json
{
  "status": "success",
  "execution_date": "2025-12-05T18:05:00Z",
  "total_users": 25,
  "risk_distribution": {
    "ALTO_RIESGO": 3,
    "RIESGO_MODERADO": 7,
    "BAJO_RIESGO": 15
  },
  "metrics": {
    "silhouette_score": 0.45,
    "calinski_harabasz": 120.5,
    "high_risk_percentage": 12.0
  }
}
```

---

## 3️⃣ Visualizaciones (HTML/SVG)

### GET `/clustering/visualize/dashboard`
Dashboard completo con todas las gráficas.
```
http://<IP>:8001/api/v1/clustering/visualize/dashboard
```
**Retorna:** HTML con CSS inline (puede embeberse en iframe)

### GET `/clustering/visualize/distribution`
Gráfico de barras: Distribución de niveles de riesgo.
```
http://<IP>:8001/api/v1/clustering/visualize/distribution
```

### GET `/clustering/visualize/scatter`
Scatter Plot PCA coloreado por nivel de riesgo.
```
http://<IP>:8001/api/v1/clustering/visualize/scatter
```

### GET `/clustering/visualize/radar`
Radar Chart con perfil de KPIs por cluster.
```
http://<IP>:8001/api/v1/clustering/visualize/radar
```

### GET `/clustering/visualize/severity`
Histograma de índice de severidad.
```
http://<IP>:8001/api/v1/clustering/visualize/severity
```

### GET `/clustering/visualize/kmeans`
Visualización de clusters K-Means.
```
http://<IP>:8001/api/v1/clustering/visualize/kmeans
```

---

## 4️⃣ Datos JSON para Frontend Personalizado

### GET `/clustering/results`
Métricas del último clustering.

```bash
curl "http://<IP>:8001/api/v1/clustering/results"
```

**Response:**
```json
{
  "execution_date": "2025-12-05T18:05:00Z",
  "metrics": {
    "silhouette_score": 0.45,
    "total_users": 25,
    "high_risk_percentage": 12.0
  },
  "risk_distribution": {
    "ALTO_RIESGO": 3,
    "RIESGO_MODERADO": 7,
    "BAJO_RIESGO": 15
  }
}
```

### GET `/clustering/users/{risk_level}`
Lista de usuarios por nivel de riesgo.

```bash
curl "http://<IP>:8001/api/v1/clustering/users/ALTO_RIESGO"
```

**Valores válidos:** `ALTO_RIESGO`, `RIESGO_MODERADO`, `BAJO_RIESGO`

**Response:**
```json
[
  {
    "user_id_raiz": "uuid-123",
    "risk_level": "ALTO_RIESGO",
    "severity_index": 0.85,
    "total_votes": 3
  }
]
```

### GET `/clustering/profiles`
Perfil promedio de KPIs por cluster.

```bash
curl "http://<IP>:8001/api/v1/clustering/profiles"
```

---

## 5️⃣ Endpoint para Chat IA

### GET `/clustering/user-profile/{user_id}`
Perfil de riesgo de un usuario específico.

```bash
curl "http://<IP>:8001/api/v1/clustering/user-profile/uuid-del-usuario"
```

**Response:**
```json
{
  "user_id": "uuid-123",
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
  "recommendation_context": "⚠️ Usuario en ALTO RIESGO emocional..."
}
```

---

## 🔧 Integración con Frontend

### Opción 1: Embeber Dashboard (iframe)
```html
<iframe 
  src="http://<IP>:8001/api/v1/clustering/visualize/dashboard" 
  width="100%" 
  height="800px"
  frameborder="0">
</iframe>
```

### Opción 2: Consumir API JSON
```javascript
// Ejemplo con fetch
const response = await fetch('http://<IP>:8001/api/v1/clustering/results');
const data = await response.json();

// Usar data.risk_distribution para crear gráficas con Chart.js, etc.
```

---

## 📅 Flujo Recomendado

1. **Ejecutar ETL** → `POST /data-miner/execute-etl`
2. **Ejecutar Clustering** → `POST /clustering/execute`
3. **Ver Dashboard** → `GET /clustering/visualize/dashboard`
4. **Consultar usuarios alto riesgo** → `GET /clustering/users/ALTO_RIESGO`

---

## 🔒 Puerto AWS Security Group

Agregar regla Inbound en EC2 Security Group:
- **Type:** Custom TCP
- **Port:** 8001
- **Source:** 0.0.0.0/0 (o IP específica)
