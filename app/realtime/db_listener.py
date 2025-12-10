# app/realtime/db_listener.py
"""
Listener de PostgreSQL para Change Data Capture (CDC) usando LISTEN/NOTIFY.
Detecta cambios en las bases de datos de los microservicios AURA.
"""

import asyncio
import asyncpg
import json
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from app.config import settings


class DatabaseListener:
    """
    Listener asíncrono para notificaciones de PostgreSQL.
    Utiliza pg_notify para detectar cambios en las tablas fuente.
    """
    
    # Canales de notificación por tipo de evento
    CHANNELS = {
        "data_change": "aura_data_change",      # Cambios en datos de usuario
        "message_new": "aura_new_message",       # Nuevos mensajes
        "profile_update": "aura_profile_update"  # Actualizaciones de perfil
    }
    
    def __init__(self):
        self._connections: Dict[str, asyncpg.Connection] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._tasks: list = []
    
    async def start(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Inicia el listener en todas las bases de datos fuente.
        
        Args:
            callback: Función async que se llamará cuando llegue una notificación.
                     Recibe un diccionario con: {source, channel, payload}
        """
        self._running = True
        self._default_callback = callback
        
        print("🎧 Iniciando PostgreSQL Listeners...")
        
        # Conectar a cada base de datos fuente
        db_configs = [
            ("social", settings.DATABASE_URL_SOCIAL),
            ("messaging", settings.DATABASE_URL_MESSAGING),
        ]
        
        for name, url in db_configs:
            try:
                # Convertir SQLAlchemy URL a asyncpg format
                async_url = self._convert_url_to_asyncpg(url)
                conn = await asyncpg.connect(async_url)
                self._connections[name] = conn
                
                # Suscribirse al canal principal
                await conn.add_listener(
                    self.CHANNELS["data_change"],
                    lambda conn, pid, channel, payload, source=name: 
                        asyncio.create_task(self._handle_notification(source, channel, payload))
                )
                
                print(f"   ✅ Listener conectado: {name} -> {self.CHANNELS['data_change']}")
                
            except Exception as e:
                print(f"   ⚠️ Error conectando listener {name}: {e}")
        
        print("🎧 PostgreSQL Listeners iniciados\n")
    
    async def stop(self) -> None:
        """Detiene todos los listeners y cierra conexiones."""
        self._running = False
        
        for name, conn in self._connections.items():
            try:
                await conn.close()
                print(f"   🔌 Listener cerrado: {name}")
            except Exception as e:
                print(f"   ⚠️ Error cerrando listener {name}: {e}")
        
        self._connections.clear()
    
    async def _handle_notification(self, source: str, channel: str, payload: str) -> None:
        """
        Procesa una notificación entrante de PostgreSQL.
        
        Args:
            source: Nombre de la base de datos origen (social, messaging)
            channel: Canal de notificación
            payload: Datos JSON de la notificación
        """
        try:
            data = json.loads(payload) if payload else {}
            
            notification = {
                "source": source,
                "channel": channel,
                "payload": data,
                "received_at": datetime.utcnow().isoformat()
            }
            
            print(f"📨 Notificación recibida: {source}/{channel} - {data.get('table', 'unknown')}")
            
            # Llamar al callback registrado
            if self._default_callback:
                await self._default_callback(notification)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Error decodificando payload: {e}")
        except Exception as e:
            print(f"⚠️ Error procesando notificación: {e}")
    
    def _convert_url_to_asyncpg(self, sqlalchemy_url: str) -> str:
        """
        Convierte una URL de SQLAlchemy a formato asyncpg.
        postgresql://user:pass@host:port/db -> postgresql://user:pass@host:port/db
        """
        # asyncpg acepta el mismo formato, solo asegurar que no tenga parámetros extra
        if "?" in sqlalchemy_url:
            return sqlalchemy_url.split("?")[0]
        return sqlalchemy_url
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del listener."""
        return {
            "running": self._running,
            "connected_databases": list(self._connections.keys()),
            "channels": list(self.CHANNELS.values())
        }


# ===== Scripts SQL para Triggers =====

TRIGGER_SQL_TEMPLATE = """
-- Función genérica para notificar cambios
CREATE OR REPLACE FUNCTION notify_{table}_change() 
RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    -- Construir payload con información del cambio
    payload = json_build_object(
        'table', '{table}',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.{user_id_column}, OLD.{user_id_column}),
        'timestamp', NOW()
    );
    
    -- Enviar notificación
    PERFORM pg_notify('{channel}', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Crear trigger
DROP TRIGGER IF EXISTS {table}_change_trigger ON {table};
CREATE TRIGGER {table}_change_trigger
    AFTER INSERT OR UPDATE OR DELETE ON {table}
    FOR EACH ROW
    EXECUTE FUNCTION notify_{table}_change();
"""


def generate_trigger_sql(table: str, user_id_column: str, channel: str = "aura_data_change") -> str:
    """
    Genera el SQL para crear un trigger de notificación.
    
    Args:
        table: Nombre de la tabla
        user_id_column: Columna que contiene el user_id
        channel: Canal de notificación
    
    Returns:
        Script SQL para crear el trigger
    """
    return TRIGGER_SQL_TEMPLATE.format(
        table=table,
        user_id_column=user_id_column,
        channel=channel
    )


# Triggers necesarios para cada microservicio
REQUIRED_TRIGGERS = {
    "aura_messaging": [
        {"table": "messages", "user_id_column": "sender_profile_id"},
        {"table": "users", "user_id_column": "profile_id"},
    ],
    "aura_social": [
        {"table": "posts", "user_id_column": "user_id"},
        {"table": "comments", "user_id_column": "user_id"},
        {"table": "user_profiles", "user_id_column": "user_id"},
    ]
}


def generate_all_trigger_scripts() -> Dict[str, str]:
    """
    Genera todos los scripts SQL de triggers necesarios.
    
    Returns:
        Diccionario con nombre de DB -> script SQL completo
    """
    scripts = {}
    
    for db_name, triggers in REQUIRED_TRIGGERS.items():
        script_parts = [f"-- Triggers para {db_name}\n-- Generado automáticamente\n"]
        
        for trigger_config in triggers:
            script_parts.append(generate_trigger_sql(**trigger_config))
            script_parts.append("\n")
        
        scripts[db_name] = "\n".join(script_parts)
    
    return scripts
