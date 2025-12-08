"""Add multi-modal pill recognition fields

Revision ID: multi_modal_fields
Revises: 0ff7c5c33b2a
Create Date: 2024-12-05 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'multi_modal_fields'
down_revision = '0ff7c5c33b2a'
branch_labels = None
depends_on = None


def upgrade():
    """Add multi-modal pill recognition fields to medications table."""
    
    # Add new columns for pill features
    op.add_column('medications', sa.Column('shape', sa.String(50), nullable=True))
    op.add_column('medications', sa.Column('color_primary', sa.String(50), nullable=True))
    op.add_column('medications', sa.Column('color_secondary', sa.String(50), nullable=True))
    op.add_column('medications', sa.Column('imprint_code', sa.String(100), nullable=True))
    op.add_column('medications', sa.Column('diameter_mm', sa.Float(), nullable=True))
    op.add_column('medications', sa.Column('length_mm', sa.Float(), nullable=True))
    op.add_column('medications', sa.Column('thickness_mm', sa.Float(), nullable=True))
    op.add_column('medications', sa.Column('has_score_line', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('medications', sa.Column('is_coated', sa.Boolean(), server_default='false', nullable=False))
    
    # Create index on imprint_code for fast lookups
    op.create_index('idx_medications_imprint_code', 'medications', ['imprint_code'])
    
    # Create composite index for shape + color search
    op.create_index('idx_medications_shape_color', 'medications', ['shape', 'color_primary'])


def downgrade():
    """Remove multi-modal pill recognition fields."""
    
    # Drop indexes
    op.drop_index('idx_medications_imprint_code', table_name='medications')
    op.drop_index('idx_medications_shape_color', table_name='medications')
    
    # Drop columns
    op.drop_column('medications', 'is_coated')
    op.drop_column('medications', 'has_score_line')
    op.drop_column('medications', 'thickness_mm')
    op.drop_column('medications', 'length_mm')
    op.drop_column('medications', 'diameter_mm')
    op.drop_column('medications', 'imprint_code')
    op.drop_column('medications', 'color_secondary')
    op.drop_column('medications', 'color_primary')
    op.drop_column('medications', 'shape')
