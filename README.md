# AURA Clustering Service - Data Miner ETL

API REST con FastAPI para la ejecución del flujo ETL que genera el Vector de Características del Usuario para Clustering.

## 🚀 Quick Start

```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear la base de datos
sudo -u postgres psql -c "CREATE DATABASE aura_data_miner;"

# 4. Ejecutar migraciones
sudo -u postgres psql -c "CREATE USER miner_user WITH PASSWORD 'miner_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aura_data_miner TO miner_user;"
sudo -u postgres psql -c "ALTER DATABASE aura_data_miner OWNER TO miner_user;"

alembic upgrade head

# 5. Iniciar el servicio
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📚 Documentación

- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

## 🔗 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/data-miner/status` | Estado del servicio |
| POST | `/api/v1/data-miner/execute-etl` | Ejecutar pipeline ETL |
| POST | `/api/v1/data-miner/execute-etl-async` | ETL en background |
| GET | `/api/v1/data-miner/feature-vector/count` | Conteo de registros |
| POST | `/api/v1/clustering/execute` | Ejecutar Clustering |
| GET | `/api/v1/clustering/visualize/dashboard` | **Dashboard de Visualización** |
| GET | `/api/v1/clustering/visualize/scatter` | Scatter Plot PCA |
| GET | `/api/v1/clustering/visualize/radar` | Radar Chart de Clusters |

## 📊 KPIs Calculados

1. Ratio de Reciprocidad Social
2. Días desde Última Conexión
3. Ratio de Mensajes Nocturnos
4. Índice de Apatía del Perfil
5. Índice de Negatividad (NLP)
6. Densidad de Participación Comunitaria
# Servicio-de-Clustering
