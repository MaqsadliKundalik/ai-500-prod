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

# Start application
echo "🎯 Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}
