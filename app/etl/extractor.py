# app/etl/extractor.py
"""Fase E (Extract): Extracción de datos desde las bases de datos fuente."""

import pandas as pd
from sqlalchemy import text
from app.database.connection import auth_engine, social_engine, messaging_engine
from app.config import settings


class DataExtractor:
    """Clase responsable de extraer datos desde las DBs fuente."""
    
    def __init__(self):
        self.auth_engine = auth_engine
        self.social_engine = social_engine
        self.messaging_engine = messaging_engine
    
    def extract_social_metrics(self) -> pd.DataFrame:
        """Extrae métricas sociales (perfiles, comunidades, posts)."""
        query = text("""
            SELECT 
                up.user_id AS user_id_raiz,
                up.id AS profile_id_social,
                up.followers_count,
                up.following_count,
                up.bio IS NULL AS profile_bio_missing,
                cp.user_id IS NULL AS complete_profile_missing,
                COALESCE(cm_stats.distinct_categories, 0) AS num_community_categories,
                COALESCE(cm_stats.is_in_voluntariado, FALSE) AS comm_voluntariado_one_hot
            FROM user_profiles up
            LEFT JOIN complete_profiles cp ON up.id = cp.user_id
            LEFT JOIN (
                SELECT 
                    cm.user_id,
                    COUNT(DISTINCT c.category) AS distinct_categories,
                    MAX(CASE WHEN c.category = 'Voluntariado' THEN 1 ELSE 0 END)::BOOLEAN AS is_in_voluntariado
                FROM community_members cm
                INNER JOIN communities c ON cm.community_id = c.id
                GROUP BY cm.user_id
            ) cm_stats ON up.id = cm_stats.user_id
            WHERE up.is_active = TRUE
        """)
        
        with self.social_engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def extract_messaging_metrics(self) -> pd.DataFrame:
        """Extrae métricas de mensajería (última conexión, mensajes nocturnos)."""
        query = text("""
            SELECT 
                u.profile_id,
                u.last_seen_at,
                COALESCE(msg_stats.total_messages, 0) AS total_messages,
                COALESCE(msg_stats.night_messages, 0) AS night_messages
            FROM users u
            LEFT JOIN (
                SELECT 
                    sender_profile_id,
                    COUNT(*) AS total_messages,
                    SUM(CASE 
                        WHEN EXTRACT(HOUR FROM created_at) BETWEEN 1 AND 5 THEN 1 
                        ELSE 0 
                    END) AS night_messages
                FROM messages
                WHERE is_deleted = FALSE
                GROUP BY sender_profile_id
            ) msg_stats ON u.id = msg_stats.sender_profile_id
            WHERE u.is_active = TRUE
        """)
        
        with self.messaging_engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def extract_text_content(self) -> pd.DataFrame:
        """Extrae contenido de texto para análisis de sentimiento NLP."""
        # Posts
        posts_query = text("""
            SELECT user_id, content, 'post' AS source
            FROM posts 
            WHERE content IS NOT NULL AND is_active = TRUE
        """)
        
        # Comentarios
        comments_query = text("""
            SELECT user_id, content, 'comment' AS source
            FROM comments 
            WHERE content IS NOT NULL AND is_active = TRUE
        """)
        
        with self.social_engine.connect() as conn:
            df_posts = pd.read_sql(posts_query, conn)
            df_comments = pd.read_sql(comments_query, conn)
        
        # Mensajes (desde messaging DB)
        messages_query = text("""
            SELECT sender_profile_id AS user_id, content, 'message' AS source
            FROM messages 
            WHERE content IS NOT NULL AND is_deleted = FALSE
        """)
        
        with self.messaging_engine.connect() as conn:
            df_messages = pd.read_sql(messages_query, conn)
        
        # Combinar todos los textos
        return pd.concat([df_posts, df_comments, df_messages], ignore_index=True)
    
    def run_extraction(self) -> dict:
        """Ejecuta el proceso completo de extracción."""
        print("📥 Iniciando Fase E: Extracción de datos...")
        
        data = {
            'social_metrics': self.extract_social_metrics(),
            'messaging_metrics': self.extract_messaging_metrics(),
            'text_content': self.extract_text_content()
        }
        
        print(f"   ✅ Social: {len(data['social_metrics'])} registros")
        print(f"   ✅ Messaging: {len(data['messaging_metrics'])} registros")
        print(f"   ✅ Textos: {len(data['text_content'])} registros")
        
        return data
