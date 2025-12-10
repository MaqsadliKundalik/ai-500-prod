"""add_ai_enhancements_tables

Revision ID: 05c10e97e228
Revises: multi_modal_fields
Create Date: 2025-12-09 10:50:57.521214

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '05c10e97e228'
down_revision: Union[str, None] = 'multi_modal_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Pharmacy Inventory Table (for price comparison)
    op.create_table(
        'pharmacy_inventory',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('pharmacy_id', sa.UUID(), nullable=False),
        sa.Column('medication_id', sa.UUID(), nullable=False),
        sa.Column('price', sa.Numeric(10, 2), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, server_default='UZS'),
        sa.Column('in_stock', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('stock_quantity', sa.Integer(), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['pharmacy_id'], ['pharmacies.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['medication_id'], ['medications.id'], ondelete='CASCADE')
    )
    op.create_index('idx_pharmacy_inventory_pharmacy', 'pharmacy_inventory', ['pharmacy_id'])
    op.create_index('idx_pharmacy_inventory_medication', 'pharmacy_inventory', ['medication_id'])
    op.create_index('idx_pharmacy_inventory_price', 'pharmacy_inventory', ['price'])
    
    # Medication Recalls Table (FDA/WHO alerts)
    op.create_table(
        'medication_recalls',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('medication_id', sa.UUID(), nullable=True),
        sa.Column('recall_number', sa.String(50), nullable=True),
        sa.Column('source', sa.String(20), nullable=False),  # FDA, WHO, UZ_MOH
        sa.Column('product_description', sa.Text(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('classification', sa.String(20), nullable=True),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('recall_date', sa.Date(), nullable=True),
        sa.Column('batch_number', sa.String(100), nullable=True),
        sa.Column('company', sa.String(200), nullable=True),
        sa.Column('distribution_pattern', sa.Text(), nullable=True),
        sa.Column('action_required', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['medication_id'], ['medications.id'], ondelete='SET NULL')
    )
    op.create_index('idx_recalls_medication', 'medication_recalls', ['medication_id'])
    op.create_index('idx_recalls_severity', 'medication_recalls', ['severity'])
    op.create_index('idx_recalls_status', 'medication_recalls', ['status'])
    op.create_index('idx_recalls_batch', 'medication_recalls', ['batch_number'])
    
    # User Notifications Table
    op.create_table(
        'user_notifications',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),  # recall_alert, price_drop, interaction_warning
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, server_default='info'),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('action_url', sa.String(500), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('idx_notifications_user', 'user_notifications', ['user_id'])
    op.create_index('idx_notifications_read', 'user_notifications', ['is_read'])
    op.create_index('idx_notifications_created', 'user_notifications', ['created_at'])
    
    # Pharmacy Reviews Table
    op.create_table(
        'pharmacy_reviews',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('pharmacy_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),  # 1-5
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('service_rating', sa.Integer(), nullable=True),
        sa.Column('price_rating', sa.Integer(), nullable=True),
        sa.Column('availability_rating', sa.Integer(), nullable=True),
        sa.Column('cleanliness_rating', sa.Integer(), nullable=True),
        sa.Column('helpful_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['pharmacy_id'], ['pharmacies.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.CheckConstraint('rating >= 1 AND rating <= 5', name='check_rating_range')
    )
    op.create_index('idx_reviews_pharmacy', 'pharmacy_reviews', ['pharmacy_id'])
    op.create_index('idx_reviews_user', 'pharmacy_reviews', ['user_id'])
    op.create_index('idx_reviews_rating', 'pharmacy_reviews', ['rating'])


def downgrade() -> None:
    op.drop_table('pharmacy_reviews')
    op.drop_table('user_notifications')
    op.drop_table('medication_recalls')
    op.drop_table('pharmacy_inventory')
