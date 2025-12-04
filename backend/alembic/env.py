"""
Alembic Environment Configuration
=================================
"""

from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy import create_engine
from alembic import context

import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.db.session import Base

# Import all models so Alembic can detect them
from app.models.user import User, FamilyMember, UserBadge, PointsHistory
from app.models.medication import Medication, MedicationPrice, UserMedication
from app.models.interaction import DrugInteraction, FoodInteraction, Contraindication
from app.models.scan import Scan, BatchRecall, AdherenceLog
from app.models.pharmacy import Pharmacy, PharmacyInventory, PharmacyReport

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate
target_metadata = Base.metadata


def get_url():
    """Get database URL from settings."""
    return settings.sync_database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_engine(get_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
