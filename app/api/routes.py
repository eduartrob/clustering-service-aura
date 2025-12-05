# app/api/routes.py
"""Definición de endpoints del API REST."""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.etl.extractor import DataExtractor
from app.etl.transformer import DataTransformer
from app.etl.loader import DataLoader

router = APIRouter(prefix="/api/v1/data-miner", tags=["ETL Pipeline"])


# ===== Modelos de Respuesta =====

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
    nlp_available: bool


class AsyncResponse(BaseModel):
    """Modelo de respuesta para ejecución asíncrona."""
    status: str
    message: str
    check_status_at: str


# ===== Endpoints =====

@router.get("/status", response_model=HealthResponse)
def get_status():
    """
    Endpoint para verificar que el API está activo.
    
    Retorna el estado del servicio y disponibilidad del modelo NLP.
    """
    # Check NLP availability without loading the model
    nlp_available = True  # Will be determined on first use
    
    return HealthResponse(
        status="ok",
        service="AURA Data Miner - Clustering Service",
        ready_for_etl=True,
        nlp_available=nlp_available
    )


@router.post("/execute-etl", response_model=ETLResponse)
def execute_etl_pipeline(
    skip_nlp: bool = Query(
        default=False, 
        description="Omitir análisis NLP para ejecución más rápida"
    ),
    truncate_before: bool = Query(
        default=True,
        description="Truncar tabla antes de insertar nuevos datos"
    )
):
    """
    Ejecuta el flujo completo de ETL (Extract, Transform, Load).
    
    **Fases:**
    1. **Extracción (E):** Datos desde las DB de Microservicios (Auth, Social, Messaging)
    2. **Transformación (T):** Cálculo de KPIs, Normalización y Vectorización (EDA)
    3. **Carga (L):** Persistencia en la DB Analítica (Target)
    
    **Parámetros:**
    - `skip_nlp`: Si es `true`, omite el análisis de sentimiento NLP (más rápido)
    - `truncate_before`: Si es `true`, elimina registros existentes antes de cargar
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
        processed_df = transformer.run_transformation(raw_data, skip_nlp=skip_nlp)
        
        if processed_df.empty:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error en la transformación de datos."
            )
        
        # FASE L: Carga
        records_loaded = loader.run_load(processed_df, truncate_before=truncate_before)
        
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


@router.post("/execute-etl-async", response_model=AsyncResponse)
async def execute_etl_async(
    background_tasks: BackgroundTasks,
    skip_nlp: bool = Query(default=False),
    truncate_before: bool = Query(default=True)
):
    """
    Ejecuta el ETL de forma asíncrona en segundo plano.
    
    Útil para procesos largos sin bloquear el API.
    El estado del proceso puede verificarse en `/status`.
    """
    def run_etl_background():
        try:
            print("\n🔄 Iniciando ETL en background...")
            
            extractor = DataExtractor()
            transformer = DataTransformer()
            loader = DataLoader()
            
            raw_data = extractor.run_extraction()
            processed_df = transformer.run_transformation(raw_data, skip_nlp=skip_nlp)
            loader.run_load(processed_df, truncate_before=truncate_before)
            
            print("✅ ETL en background completado exitosamente")
            
        except Exception as e:
            print(f"❌ Error en ETL background: {e}")
    
    background_tasks.add_task(run_etl_background)
    
    return AsyncResponse(
        status="accepted",
        message="El proceso ETL se está ejecutando en segundo plano.",
        check_status_at="/api/v1/data-miner/status"
    )


@router.get("/feature-vector/count")
def get_feature_vector_count():
    """
    Retorna el número de registros en la tabla user_feature_vector.
    """
    from sqlalchemy import text
    from app.database.connection import analytics_engine
    
    try:
        with analytics_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM user_feature_vector"))
            count = result.scalar()
            
        return {
            "table": "user_feature_vector",
            "record_count": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando la base de datos: {str(e)}"
        )
