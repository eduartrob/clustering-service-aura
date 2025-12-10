# app/realtime/websocket_manager.py
"""Gestión de conexiones WebSocket para actualizaciones en tiempo real."""

from fastapi import WebSocket
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio


class ConnectionManager:
    """
    Gestor de conexiones WebSocket.
    Maneja múltiples canales para diferentes tipos de suscripciones.
    """
    
    def __init__(self):
        # Diccionario de canales -> lista de conexiones
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Historial de mensajes recientes por canal (para nuevos clientes)
        self.message_history: Dict[str, List[dict]] = {}
        self.max_history = 50
        
    async def connect(self, websocket: WebSocket, channel: str = "clustering") -> None:
        """
        Acepta una conexión WebSocket y la registra en un canal.
        
        Args:
            websocket: Conexión WebSocket entrante
            channel: Canal de suscripción (ej: 'clustering', 'alerts')
        """
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = []
            self.message_history[channel] = []
            
        self.active_connections[channel].append(websocket)
        
        # Enviar estado inicial al nuevo cliente
        await self._send_initial_state(websocket, channel)
        
        print(f"✅ WebSocket conectado al canal '{channel}'. Total: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str = "clustering") -> None:
        """Desconecta un WebSocket de un canal."""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
                print(f"🔌 WebSocket desconectado del canal '{channel}'. Restantes: {len(self.active_connections[channel])}")
    
    async def broadcast(self, message: dict, channel: str = "clustering") -> None:
        """
        Envía un mensaje a todos los clientes conectados a un canal.
        
        Args:
            message: Diccionario con el mensaje a enviar
            channel: Canal destino
        """
        if channel not in self.active_connections:
            return
            
        # Agregar metadata de timestamp
        enriched_message = {
            **message,
            "server_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Guardar en historial
        self._add_to_history(enriched_message, channel)
        
        # Enviar a todos los clientes conectados
        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(enriched_message)
            except Exception as e:
                print(f"⚠️ Error enviando mensaje: {e}")
                disconnected.append(connection)
        
        # Limpiar conexiones muertas
        for conn in disconnected:
            self.disconnect(conn, channel)
    
    async def send_personal(self, message: dict, websocket: WebSocket) -> None:
        """Envía un mensaje a un cliente específico."""
        try:
            await websocket.send_json({
                **message,
                "server_timestamp": datetime.utcnow().isoformat() + "Z"
            })
        except Exception as e:
            print(f"⚠️ Error enviando mensaje personal: {e}")
    
    async def _send_initial_state(self, websocket: WebSocket, channel: str) -> None:
        """Envía el estado inicial y mensajes recientes a un nuevo cliente."""
        initial_message = {
            "type": "INITIAL_STATE",
            "channel": channel,
            "connected_clients": len(self.active_connections.get(channel, [])),
            "recent_messages": self.message_history.get(channel, [])[-10:]  # Últimos 10 mensajes
        }
        await self.send_personal(initial_message, websocket)
    
    def _add_to_history(self, message: dict, channel: str) -> None:
        """Añade un mensaje al historial del canal."""
        if channel not in self.message_history:
            self.message_history[channel] = []
            
        self.message_history[channel].append(message)
        
        # Limitar tamaño del historial
        if len(self.message_history[channel]) > self.max_history:
            self.message_history[channel] = self.message_history[channel][-self.max_history:]
    
    def get_connection_stats(self) -> Dict[str, int]:
        """Retorna estadísticas de conexiones por canal."""
        return {
            channel: len(connections) 
            for channel, connections in self.active_connections.items()
        }


# Instancia global del manager
manager = ConnectionManager()


# ===== Tipos de Mensajes WebSocket =====

class MessageType:
    """Constantes para tipos de mensajes WebSocket."""
    
    # Estado inicial al conectar
    INITIAL_STATE = "INITIAL_STATE"
    
    # Actualización de riesgo de un usuario específico
    USER_RISK_UPDATE = "USER_RISK_UPDATE"
    
    # Actualización de la distribución general
    DISTRIBUTION_UPDATE = "DISTRIBUTION_UPDATE"
    
    # Alerta crítica de usuario en alto riesgo
    CRITICAL_ALERT = "CRITICAL_ALERT"
    
    # Actualización de tendencias de KPIs
    KPI_TREND_UPDATE = "KPI_TREND_UPDATE"
    
    # Resultado de re-clustering completo
    CLUSTERING_COMPLETE = "CLUSTERING_COMPLETE"
    
    # Error del sistema
    ERROR = "ERROR"


def create_user_risk_message(
    user_id: str,
    risk_level: str,
    severity_index: float,
    previous_risk: Optional[str] = None,
    triggering_factors: Optional[List[dict]] = None
) -> dict:
    """
    Crea un mensaje de actualización de riesgo de usuario.
    
    Args:
        user_id: UUID del usuario
        risk_level: Nivel de riesgo actual (ALTO_RIESGO, RIESGO_MODERADO, BAJO_RIESGO)
        severity_index: Índice de severidad (0-100)
        previous_risk: Nivel de riesgo anterior (opcional)
        triggering_factors: Lista de factores que desencadenaron el cambio
    
    Returns:
        Diccionario con el mensaje formateado
    """
    return {
        "type": MessageType.USER_RISK_UPDATE,
        "payload": {
            "user_id": user_id,
            "risk_level": risk_level,
            "severity_index": round(severity_index, 2),
            "previous_risk": previous_risk,
            "risk_changed": previous_risk is not None and previous_risk != risk_level,
            "triggering_factors": triggering_factors or []
        }
    }


def create_distribution_message(distribution: Dict[str, int], total_users: int) -> dict:
    """
    Crea un mensaje de actualización de distribución de riesgo.
    
    Args:
        distribution: Diccionario con conteo por nivel de riesgo
        total_users: Total de usuarios analizados
    
    Returns:
        Diccionario con el mensaje formateado
    """
    return {
        "type": MessageType.DISTRIBUTION_UPDATE,
        "payload": {
            "distribution": distribution,
            "total_users": total_users,
            "percentages": {
                level: round(count / total_users * 100, 1) if total_users > 0 else 0
                for level, count in distribution.items()
            }
        }
    }


def create_critical_alert_message(
    user_id: str,
    severity_index: float,
    factors: List[str],
    suggested_action: str
) -> dict:
    """
    Crea un mensaje de alerta crítica para usuarios en alto riesgo.
    
    Args:
        user_id: UUID del usuario en riesgo
        severity_index: Índice de severidad
        factors: Lista de factores de riesgo detectados
        suggested_action: Acción recomendada
    
    Returns:
        Diccionario con la alerta formateada
    """
    return {
        "type": MessageType.CRITICAL_ALERT,
        "payload": {
            "priority": "URGENT" if severity_index > 80 else "HIGH",
            "user_id": user_id,
            "severity_index": round(severity_index, 2),
            "factors": factors,
            "suggested_action": suggested_action,
            "context": "Usuario requiere atención prioritaria del equipo de salud mental"
        }
    }
