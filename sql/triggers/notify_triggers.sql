-- ============================================================
-- AURA Data Miner - PostgreSQL Triggers para CDC (Change Data Capture)
-- ============================================================
-- 
-- Estos triggers notifican cambios en las tablas de los microservicios
-- al servicio de clustering para procesamiento en tiempo real.
--
-- INSTRUCCIONES DE INSTALACIÓN:
-- 1. Ejecutar este script en cada base de datos correspondiente
-- 2. Asegúrate de tener permisos de superusuario o TRIGGER en las tablas
--
-- ============================================================

-- ============================================================
-- PARA BASE DE DATOS: aura_messaging
-- ============================================================

-- Función genérica de notificación para mensajes
CREATE OR REPLACE FUNCTION notify_messages_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    -- Construir payload con información del cambio
    payload = json_build_object(
        'table', 'messages',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.sender_profile_id, OLD.sender_profile_id),
        'message_id', COALESCE(NEW.id, OLD.id),
        'timestamp', NOW()
    );
    
    -- Enviar notificación al canal
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para la tabla messages
DROP TRIGGER IF EXISTS messages_change_trigger ON messages;
CREATE TRIGGER messages_change_trigger
    AFTER INSERT OR UPDATE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION notify_messages_change();

-- Función para cambios en usuarios de mensajería
CREATE OR REPLACE FUNCTION notify_messaging_users_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    payload = json_build_object(
        'table', 'users',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.profile_id, OLD.profile_id),
        'field_changed', CASE 
            WHEN TG_OP = 'UPDATE' AND OLD.last_seen_at IS DISTINCT FROM NEW.last_seen_at THEN 'last_seen_at'
            WHEN TG_OP = 'UPDATE' AND OLD.is_online IS DISTINCT FROM NEW.is_online THEN 'is_online'
            ELSE 'other'
        END,
        'timestamp', NOW()
    );
    
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para la tabla users (mensajería)
DROP TRIGGER IF EXISTS messaging_users_change_trigger ON users;
CREATE TRIGGER messaging_users_change_trigger
    AFTER INSERT OR UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION notify_messaging_users_change();


-- ============================================================
-- PARA BASE DE DATOS: aura_social
-- ============================================================

-- Función para cambios en posts
CREATE OR REPLACE FUNCTION notify_posts_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    payload = json_build_object(
        'table', 'posts',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.user_id, OLD.user_id),
        'post_id', COALESCE(NEW.id, OLD.id),
        'timestamp', NOW()
    );
    
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para posts
DROP TRIGGER IF EXISTS posts_change_trigger ON posts;
CREATE TRIGGER posts_change_trigger
    AFTER INSERT OR UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION notify_posts_change();

-- Función para cambios en comentarios
CREATE OR REPLACE FUNCTION notify_comments_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    payload = json_build_object(
        'table', 'comments',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.user_id, OLD.user_id),
        'comment_id', COALESCE(NEW.id, OLD.id),
        'timestamp', NOW()
    );
    
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para comments
DROP TRIGGER IF EXISTS comments_change_trigger ON comments;
CREATE TRIGGER comments_change_trigger
    AFTER INSERT OR UPDATE ON comments
    FOR EACH ROW
    EXECUTE FUNCTION notify_comments_change();

-- Función para cambios en perfiles
CREATE OR REPLACE FUNCTION notify_user_profiles_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    payload = json_build_object(
        'table', 'user_profiles',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.user_id, OLD.user_id),
        'profile_id', COALESCE(NEW.id, OLD.id),
        'timestamp', NOW()
    );
    
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para user_profiles
DROP TRIGGER IF EXISTS user_profiles_change_trigger ON user_profiles;
CREATE TRIGGER user_profiles_change_trigger
    AFTER INSERT OR UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION notify_user_profiles_change();

-- Función para cambios en membresías de comunidad
CREATE OR REPLACE FUNCTION notify_community_members_change() RETURNS trigger AS $$
DECLARE
    payload JSON;
BEGIN
    payload = json_build_object(
        'table', 'community_members',
        'operation', TG_OP,
        'user_id', COALESCE(NEW.user_id, OLD.user_id),
        'community_id', COALESCE(NEW.community_id, OLD.community_id),
        'timestamp', NOW()
    );
    
    PERFORM pg_notify('aura_data_change', payload::text);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para community_members
DROP TRIGGER IF EXISTS community_members_change_trigger ON community_members;
CREATE TRIGGER community_members_change_trigger
    AFTER INSERT OR UPDATE OR DELETE ON community_members
    FOR EACH ROW
    EXECUTE FUNCTION notify_community_members_change();


-- ============================================================
-- VERIFICACIÓN
-- ============================================================
-- 
-- Para verificar que los triggers están instalados:
--
-- SELECT trigger_name, event_manipulation, event_object_table
-- FROM information_schema.triggers
-- WHERE trigger_schema = 'public' AND trigger_name LIKE '%_change_trigger';
--
-- Para probar una notificación manualmente:
-- LISTEN aura_data_change;
-- (en otra sesión) UPDATE messages SET content = content WHERE id = '...'
--
-- ============================================================
