#!/bin/bash
# Database Seed Script for Docker
# ===============================
# Seeds the database with initial data for AI enhancements

echo "🌱 Starting database seeding..."

# Wait for database to be ready
echo "⏳ Waiting for database..."
python -c "
import time
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

async def wait_for_db():
    engine = create_async_engine(settings.DATABASE_URL)
    max_retries = 30
    for i in range(max_retries):
        try:
            async with engine.connect() as conn:
                await conn.execute('SELECT 1')
            print('✅ Database is ready!')
            return
        except Exception as e:
            if i < max_retries - 1:
                print(f'⏳ Waiting for database... ({i+1}/{max_retries})')
                time.sleep(2)
            else:
                print(f'❌ Database connection failed: {e}')
                raise

asyncio.run(wait_for_db())
"

# Run migrations
echo "📦 Running database migrations..."
alembic upgrade head

# Seed data
echo "🌱 Seeding initial data..."
python app/scripts/seed_ai_enhancements.py

echo "✅ Database seeding completed!"
