"""create user_feature_vector table

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Crear tabla user_feature_vector
    op.create_table(
        'user_feature_vector',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id_raiz', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('extraction_date', sa.DateTime(), nullable=True),
        sa.Column('reciprocity_ratio_norm', sa.Float(), nullable=True),
        sa.Column('days_since_last_seen_norm', sa.Float(), nullable=True),
        sa.Column('ratio_night_messages', sa.Float(), nullable=True),
        sa.Column('is_profile_incomplete', sa.Boolean(), nullable=True),
        sa.Column('sentiment_negativity_index', sa.Float(), nullable=True),
        sa.Column('num_community_categories_norm', sa.Float(), nullable=True),
        sa.Column('int_gaming_one_hot', sa.Boolean(), nullable=True),
        sa.Column('comm_voluntariado_one_hot', sa.Boolean(), nullable=True),
        sa.Column('cluster_label', sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Crear índices
    op.create_index('ix_user_feature_vector_user_id_raiz', 'user_feature_vector', ['user_id_raiz'], unique=True)
    op.create_index('ix_user_feature_vector_extraction_date', 'user_feature_vector', ['extraction_date'], unique=False)
    op.create_index('idx_extraction_cluster', 'user_feature_vector', ['extraction_date', 'cluster_label'], unique=False)


def downgrade() -> None:
    # Eliminar índices
    op.drop_index('idx_extraction_cluster', table_name='user_feature_vector')
    op.drop_index('ix_user_feature_vector_extraction_date', table_name='user_feature_vector')
    op.drop_index('ix_user_feature_vector_user_id_raiz', table_name='user_feature_vector')
    
    # Eliminar tabla
    op.drop_table('user_feature_vector')
