# app/main.py
"""Punto de entrada principal de la aplicación FastAPI."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.config import settings
from app.database.models import Base
from app.database.connection import analytics_engine
from app.api.routes import router as etl_router
from app.api.clustering_routes import router as clustering_router

# Cargar variables de entorno
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hook de ciclo de vida: ejecutar al iniciar y cerrar el servicio."""
    # === STARTUP ===
    print("\n" + "="*60)
    print(f"🚀 Iniciando {settings.SERVICE_NAME}...")
    print("="*60)
    
    # Crear tablas si no existen (modo desarrollo)
    if settings.DEBUG:
        try:
            Base.metadata.create_all(bind=analytics_engine)
            print("   ✅ Tablas de DB Analítica verificadas/creadas")
        except Exception as e:
            print(f"   ⚠️ Error creando tablas: {e}")
            print("   ℹ️ Asegúrate de que la base de datos 'aura_data_miner' exista")
    
    print(f"   ✅ Servicio listo en puerto {settings.SERVICE_PORT}")
    print(f"   📚 Documentación: http://localhost:{settings.SERVICE_PORT}/docs")
    print("="*60 + "\n")
    
    yield
    
    # === SHUTDOWN ===
    print(f"\n👋 Cerrando {settings.SERVICE_NAME}...")


# Inicializar aplicación FastAPI
app = FastAPI(
    title="AURA Data Miner API",
    description="""
## 🔬 API REST para Minería de Datos y Clustering de Usuarios AURA

Este servicio ejecuta el flujo completo de **ETL (Extract, Transform, Load)** 
para generar el **Vector de Características del Usuario** necesario para el Clustering.

### 📊 Funcionalidades

* **Extracción (E):** Obtiene datos desde las DBs de los microservicios (Auth, Social, Messaging)
* **Transformación (T):** Calcula KPIs, aplica análisis NLP de sentimiento, y normaliza datos
* **Carga (L):** Persiste el vector de características en la DB Analítica

### 🎯 KPIs Calculados

1. **Ratio de Reciprocidad Social** - Aislamiento social
2. **Días desde Última Conexión** - Retirada de la plataforma
3. **Ratio de Mensajes Nocturnos** - Desorden del ritmo circadiano
4. **Índice de Apatía del Perfil** - Incompletitud del perfil
5. **Índice de Negatividad (NLP)** - Tono emocional del contenido
6. **Densidad de Participación Comunitaria** - Amplitud de red de apoyo

### 🔗 Endpoints Principales

* `GET /api/v1/data-miner/status` - Verificar estado del servicio
* `POST /api/v1/data-miner/execute-etl` - Ejecutar pipeline ETL completo
* `POST /api/v1/data-miner/execute-etl-async` - Ejecutar ETL en background

### 🔮 Clustering y Visualización

* `POST /api/v1/clustering/execute` - Ejecutar ensamble de clustering
* `GET /api/v1/clustering/visualize/dashboard` - Dashboard completo con gráficos SVG
* `GET /api/v1/clustering/visualize/scatter` - Scatter plot PCA
* `GET /api/v1/clustering/visualize/distribution` - Distribución de riesgo
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(clustering_router)


@app.get("/", tags=["Root"])
def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "description": "AURA Data Miner - ETL Pipeline para Clustering",
        "documentation": "/docs",
        "endpoints": {
            "status": "/api/v1/data-miner/status",
            "execute_etl": "/api/v1/data-miner/execute-etl",
            "clustering_execute": "/api/v1/clustering/execute",
            "clustering_dashboard": "/api/v1/clustering/visualize/dashboard"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint para monitoreo."""
    return {"status": "healthy"}


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG
    )
