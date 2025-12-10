# app/main.py
"""Punto de entrada principal de la aplicación FastAPI con soporte en tiempo real."""

import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.config import settings
from app.database.models import Base
from app.database.connection import analytics_engine
from app.api.routes import router as etl_router
from app.api.clustering_routes import router as clustering_router
from app.api.websocket_routes import router as websocket_router

# Cargar variables de entorno
load_dotenv()

# Variables globales para componentes de tiempo real
_db_listener = None
_streaming_pipeline = None


async def _init_realtime_components():
    """Inicializa los componentes de tiempo real (opcional)."""
    global _db_listener, _streaming_pipeline
    
    try:
        from app.realtime.db_listener import DatabaseListener
        from app.realtime.streaming_pipeline import streaming_pipeline
        
        _db_listener = DatabaseListener()
        _streaming_pipeline = streaming_pipeline
        
        # Iniciar listener con el pipeline como callback
        await _db_listener.start(
            callback=_streaming_pipeline.process_notification
        )
        
        print("   ✅ Componentes de tiempo real inicializados")
        print("   🎧 PostgreSQL CDC Listener activo")
        print("   🔄 Streaming ETL Pipeline listo")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Tiempo real no disponible: {e}")
        print("   ℹ️ El servicio funcionará sin actualizaciones en tiempo real")
        return False


async def _shutdown_realtime_components():
    """Cierra los componentes de tiempo real."""
    global _db_listener
    
    if _db_listener is not None:
        await _db_listener.stop()
        print("   🔌 PostgreSQL Listeners cerrados")


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
    
    # Inicializar componentes de tiempo real
    realtime_enabled = await _init_realtime_components()
    
    print(f"   ✅ Servicio listo en puerto {settings.SERVICE_PORT}")
    print(f"   📚 Documentación: http://localhost:{settings.SERVICE_PORT}/docs")
    
    if realtime_enabled:
        print(f"   🔌 WebSocket: ws://localhost:{settings.SERVICE_PORT}/api/v2/clustering/ws/live")
    
    print("="*60 + "\n")
    
    yield
    
    # === SHUTDOWN ===
    print(f"\n👋 Cerrando {settings.SERVICE_NAME}...")
    await _shutdown_realtime_components()


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

### 🔗 API v1 - Endpoints Clásicos

* `GET /api/v1/data-miner/status` - Verificar estado del servicio
* `POST /api/v1/data-miner/execute-etl` - Ejecutar pipeline ETL completo
* `POST /api/v1/clustering/execute` - Ejecutar ensamble de clustering
* `GET /api/v1/clustering/visualize/dashboard` - Dashboard completo con gráficos SVG

### 🚀 API v2 - Tiempo Real (NUEVO)

* `WebSocket /api/v2/clustering/ws/live` - Actualizaciones en tiempo real
* `WebSocket /api/v2/clustering/ws/alerts` - Alertas críticas
* `GET /api/v2/clustering/data/distribution` - JSON para ChartJS
* `GET /api/v2/clustering/data/scatter` - Datos PCA para scatter plot
* `GET /api/v2/clustering/data/radar` - Perfil de KPIs por cluster
* `GET /api/v2/clustering/data/kpi-trends` - Tendencias temporales
    """,
    version="2.0.0",
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
app.include_router(websocket_router)  # API v2 + WebSocket


@app.get("/", tags=["Root"])
def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "2.0.0",
        "description": "AURA Data Miner - ETL Pipeline para Clustering con Tiempo Real",
        "documentation": "/docs",
        "endpoints": {
            "v1": {
                "status": "/api/v1/data-miner/status",
                "execute_etl": "/api/v1/data-miner/execute-etl",
                "clustering_execute": "/api/v1/clustering/execute",
                "clustering_dashboard": "/api/v1/clustering/visualize/dashboard"
            },
            "v2_realtime": {
                "websocket_live": "/api/v2/clustering/ws/live",
                "websocket_alerts": "/api/v2/clustering/ws/alerts",
                "data_distribution": "/api/v2/clustering/data/distribution",
                "data_scatter": "/api/v2/clustering/data/scatter",
                "data_radar": "/api/v2/clustering/data/radar",
                "data_trends": "/api/v2/clustering/data/kpi-trends",
                "status": "/api/v2/clustering/status"
            }
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint para monitoreo."""
    from app.realtime.websocket_manager import manager
    
    return {
        "status": "healthy",
        "realtime": {
            "enabled": _db_listener is not None,
            "websocket_connections": manager.get_connection_stats()
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

