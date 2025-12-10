#!/bin/bash
# Sentinel-RX Startup Script for Render.com

set -e

echo "🚀 Starting Sentinel-RX..."

# Wait for database to be ready
echo "⏳ Waiting for database..."
until python -c "from app.db.session import async_session_maker; import asyncio; from sqlalchemy import text; asyncio.run((lambda: async_session_maker().__aenter__())()).execute(text('SELECT 1'))" 2>/dev/null; do
    echo "Database is unavailable - sleeping"
    sleep 2
done

echo "✅ Database is ready!"

# Run migrations
echo "📦 Running database migrations..."
cd /app && alembic upgrade head

echo "✅ Migrations completed!"

# Seed database if empty
echo "🌱 Checking if database needs seeding..."
MEDICATION_COUNT=$(python -c "
import asyncio
from app.db.session import async_session_maker
from sqlalchemy import text

async def check():
    try:
        async with async_session_maker() as db:
            result = await db.execute(text('SELECT COUNT(*) FROM medications'))
            count = result.scalar()
            print(count)
    except:
        print(0)

asyncio.run(check())
" 2>/dev/null || echo "0")

if [ "$MEDICATION_COUNT" = "0" ]; then
    echo "🌱 Database is empty, seeding with initial data..."
    python app/scripts/seed_data.py
    echo "✅ Database seeding completed!"
else
    echo "✅ Database already contains ${MEDICATION_COUNT} medications, skipping seed."
fi

# Start application
echo "🎯 Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}
