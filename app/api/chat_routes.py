"""
Rutas de API para el chat con IA de AURA.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

from ..ai.gemini_chat import (
    ChatRequest,
    ChatResponse,
    generate_ai_response,
    GEMINI_API_KEY
)
from ..database.connection import analytics_engine

router = APIRouter(prefix="/api/v1/chat", tags=["AI Chat"])


class MessageItem(BaseModel):
    """Item de mensaje en historial."""
    content: str
    is_user: bool


class ChatRequestModel(BaseModel):
    """Request para enviar mensaje al chat de IA."""
    user_id: str
    message: str
    conversation_history: Optional[List[MessageItem]] = None


class ChatResponseModel(BaseModel):
    """Respuesta del chat de IA."""
    success: bool
    response: str
    user_risk_level: Optional[str] = None
    sentiment_score: Optional[float] = None
    context_used: bool = False


# Variable global para el analizador de sentimiento (se carga una vez)
_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Obtiene el analizador de sentimiento (carga lazy)."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from transformers import pipeline
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="UMUTeam/roberta-spanish-sentiment-analysis"
            )
            print("✅ Analizador de sentimiento cargado")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el analizador de sentimiento: {e}")
            _sentiment_analyzer = None
    return _sentiment_analyzer


@router.get("/status")
def get_chat_status():
    """Verifica el estado del servicio de chat con IA."""
    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "sentiment_analyzer_available": get_sentiment_analyzer() is not None,
        "status": "ready" if GEMINI_API_KEY else "unconfigured"
    }


@router.post("/send", response_model=ChatResponseModel)
async def send_message(request: ChatRequestModel):
    """
    Envía un mensaje al chat de IA y obtiene una respuesta personalizada.
    
    El endpoint:
    1. Consulta el perfil de riesgo del usuario
    2. Analiza el sentimiento del mensaje
    3. Contextualiza el prompt para Gemini
    4. Retorna una respuesta empática y personalizada
    """
    
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío"
        )
    
    # Convertir historial a formato interno
    history = None
    if request.conversation_history:
        history = [
            {"content": msg.content, "is_user": msg.is_user}
            for msg in request.conversation_history
        ]
    
    # Generar respuesta
    result = await generate_ai_response(
        user_id=request.user_id,
        message=request.message,
        analytics_engine=analytics_engine,
        sentiment_analyzer=get_sentiment_analyzer(),
        conversation_history=history
    )
    
    return ChatResponseModel(
        success=True,
        response=result.response,
        user_risk_level=result.user_risk_level,
        sentiment_score=result.sentiment_score,
        context_used=result.context_used
    )


@router.post("/analyze-sentiment")
async def analyze_sentiment(message: str):
    """Analiza el sentimiento de un mensaje (endpoint de debug)."""
    analyzer = get_sentiment_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analizador de sentimiento no disponible"
        )
    
    try:
        result = analyzer(message)
        return {
            "message": message,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analizando sentimiento: {str(e)}"
        )
