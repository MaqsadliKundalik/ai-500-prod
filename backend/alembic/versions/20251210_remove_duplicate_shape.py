"""Remove duplicate shape column from medications

Revision ID: remove_shape_202512
Revises: 20251204_0406_0ff7c5c33b2a
Create Date: 2025-12-10 09:20:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'remove_shape_202512'
down_revision = '20251204_0406_0ff7c5c33b2a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the duplicate 'shape' column from medications table
    op.drop_column('medications', 'shape')


def downgrade() -> None:
    # Re-add the shape column if needed
    op.add_column('medications', sa.Column('shape', sa.String(length=50), nullable=True))
