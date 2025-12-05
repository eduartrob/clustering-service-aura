"""
Módulo de integración con Gemini AI para el chat de AURA.

Este módulo:
1. Recibe mensajes del usuario desde Flutter
2. Consulta el perfil de riesgo del usuario
3. Analiza el sentimiento del mensaje con Robertino
4. Construye un prompt contextualizado
5. Envía a Gemini y retorna la respuesta
"""

import os
import google.generativeai as genai
from typing import Optional
from pydantic import BaseModel
from sqlalchemy import text

# Configurar Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ChatRequest(BaseModel):
    """Request para el chat de IA."""
    user_id: str
    message: str
    conversation_history: Optional[list] = None


class ChatResponse(BaseModel):
    """Respuesta del chat de IA."""
    response: str
    user_risk_level: Optional[str] = None
    sentiment_score: Optional[float] = None
    context_used: bool = False


def get_user_risk_context(user_id: str, analytics_engine) -> dict:
    """Obtiene el contexto de riesgo del usuario desde la DB analítica."""
    try:
        query = text("""
            SELECT 
                cluster_label,
                ratio_reciprocidad_social,
                dias_desde_ultima_conexion,
                ratio_mensajes_nocturnos,
                indice_apatia_perfil,
                indice_negatividad_nlp,
                densidad_participacion_comunitaria
            FROM user_feature_vector
            WHERE user_id_raiz = :user_id
            ORDER BY etl_timestamp DESC
            LIMIT 1
        """)
        
        with analytics_engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id})
            row = result.fetchone()
        
        if row is None:
            return {
                "has_data": False,
                "risk_level": None,
                "context": "Usuario nuevo sin suficiente historial para análisis."
            }
        
        risk_level = row.cluster_label or "BAJO_RIESGO"
        
        # Construir contexto descriptivo
        context_parts = []
        
        if risk_level == "ALTO_RIESGO":
            context_parts.append("⚠️ Este usuario está identificado en ALTO RIESGO emocional.")
            context_parts.append("Responde con MÁXIMA EMPATÍA, comprensión y cuidado.")
            context_parts.append("Si detectas señales de crisis, sugiere amablemente buscar ayuda profesional.")
            context_parts.append("NO minimices sus sentimientos. Valida su experiencia.")
        elif risk_level == "RIESGO_MODERADO":
            context_parts.append("Este usuario muestra indicadores de riesgo MODERADO.")
            context_parts.append("Presta especial atención a señales de malestar en su mensaje.")
            context_parts.append("Ofrece apoyo pero sin ser alarmista.")
        else:
            context_parts.append("Usuario con perfil de bajo riesgo actual.")
            context_parts.append("Responde de manera amigable y positiva.")
        
        # Agregar indicadores específicos
        if row.dias_desde_ultima_conexion and row.dias_desde_ultima_conexion > 7:
            context_parts.append(f"Ha estado inactivo {int(row.dias_desde_ultima_conexion)} días - puede indicar aislamiento.")
        
        if row.indice_negatividad_nlp and row.indice_negatividad_nlp > 0.6:
            context_parts.append("Su contenido reciente muestra alta negatividad emocional.")
        
        if row.ratio_reciprocidad_social and row.ratio_reciprocidad_social < 0.2:
            context_parts.append("Tiene pocas conexiones sociales en la plataforma - puede sentirse solo/a.")
        
        if row.ratio_mensajes_nocturnos and row.ratio_mensajes_nocturnos > 0.4:
            context_parts.append("Patrones de actividad nocturna elevados - posibles problemas de sueño.")
        
        return {
            "has_data": True,
            "risk_level": risk_level,
            "context": " ".join(context_parts)
        }
        
    except Exception as e:
        print(f"Error obteniendo contexto de usuario: {e}")
        return {
            "has_data": False,
            "risk_level": None,
            "context": "No se pudo obtener el contexto del usuario."
        }


def analyze_message_sentiment(message: str, sentiment_analyzer) -> float:
    """Analiza el sentimiento del mensaje usando Robertino."""
    try:
        if sentiment_analyzer is None:
            return 0.5  # Neutral si no hay analizador
        
        result = sentiment_analyzer(message)
        if result and len(result) > 0:
            # Transformar a escala 0-1 donde 0 es muy negativo y 1 es muy positivo
            label = result[0].get('label', 'Neutral')
            score = result[0].get('score', 0.5)
            
            if 'NEG' in label.upper():
                return 1 - score  # Invertir para negativos
            elif 'POS' in label.upper():
                return score
            else:
                return 0.5
        return 0.5
    except Exception as e:
        print(f"Error analizando sentimiento: {e}")
        return 0.5


def build_system_prompt(user_context: dict, message_sentiment: float) -> str:
    """Construye el prompt del sistema para Gemini."""
    
    base_prompt = """Eres AURA, un coach de salud mental virtual amable, empático y profesional. 
Tu objetivo es apoyar emocionalmente a los usuarios, escucharlos activamente y ayudarles 
a sentirse comprendidos.

REGLAS IMPORTANTES:
1. NUNCA diagnostiques condiciones médicas o psicológicas
2. Si detectas señales de crisis (suicidio, autolesión), sugiere AMABLEMENTE buscar ayuda profesional
3. Mantén un tono cálido, cercano pero respetuoso
4. Valida las emociones del usuario antes de ofrecer consejos
5. Responde en español de manera natural y conversacional
6. Mantén respuestas concisas pero significativas (máximo 3-4 párrafos)
7. Usa emojis con moderación para transmitir calidez (1-2 máximo por respuesta)
"""

    # Agregar contexto del usuario si existe
    if user_context.get("has_data"):
        base_prompt += f"\n\nCONTEXTO DEL USUARIO:\n{user_context['context']}\n"
    
    # Agregar contexto del sentimiento del mensaje actual
    if message_sentiment < 0.3:
        base_prompt += "\n⚠️ El mensaje actual del usuario tiene un tono MUY NEGATIVO. Responde con especial cuidado y empatía."
    elif message_sentiment < 0.5:
        base_prompt += "\nEl mensaje actual tiene un tono algo negativo. Muestra comprensión."
    
    return base_prompt


async def generate_ai_response(
    user_id: str,
    message: str,
    analytics_engine,
    sentiment_analyzer=None,
    conversation_history: list = None
) -> ChatResponse:
    """
    Genera una respuesta de IA para el chat de AURA.
    
    Args:
        user_id: ID del usuario que envía el mensaje
        message: Mensaje del usuario
        analytics_engine: Engine de SQLAlchemy para la DB analítica
        sentiment_analyzer: Pipeline de HuggingFace para análisis de sentimiento
        conversation_history: Historial de conversación previo
    
    Returns:
        ChatResponse con la respuesta de Gemini
    """
    
    if not GEMINI_API_KEY:
        return ChatResponse(
            response="Lo siento, el servicio de IA no está configurado correctamente. Por favor, contacta al administrador. 😔",
            context_used=False
        )
    
    try:
        # 1. Obtener contexto de riesgo del usuario
        user_context = get_user_risk_context(user_id, analytics_engine)
        
        # 2. Analizar sentimiento del mensaje
        message_sentiment = analyze_message_sentiment(message, sentiment_analyzer)
        
        # 3. Construir prompt del sistema
        system_prompt = build_system_prompt(user_context, message_sentiment)
        
        # 4. Configurar modelo Gemini
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 500,
            }
        )
        
        # 5. Preparar historial de conversación
        chat_history = []
        if conversation_history:
            for msg in conversation_history[-10:]:  # Últimos 10 mensajes
                role = "user" if msg.get("is_user", True) else "model"
                chat_history.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
        
        # 6. Crear chat y enviar mensaje
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(message)
        
        return ChatResponse(
            response=response.text,
            user_risk_level=user_context.get("risk_level"),
            sentiment_score=message_sentiment,
            context_used=user_context.get("has_data", False)
        )
        
    except Exception as e:
        print(f"Error generando respuesta de IA: {e}")
        return ChatResponse(
            response="Disculpa, tuve un problema procesando tu mensaje. ¿Podrías intentar de nuevo? 🙏",
            context_used=False
        )
