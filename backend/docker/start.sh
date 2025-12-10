#!/bin/bash
# Render Startup Script
# =====================
# This script runs automatically when deploying to Render

set -e  # Exit on error

echo "🚀 Starting AI-500 Backend on Render..."
echo ""

# Wait for database to be ready
echo "⏳ Waiting for database connection..."
python -c "
import time
import psycopg2
from urllib.parse import urlparse
import os

db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('❌ DATABASE_URL not set')
    exit(1)

# Parse DATABASE_URL
result = urlparse(db_url)
username = result.username
password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port or 5432

# Wait for database
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        conn = psycopg2.connect(
            dbname=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        conn.close()
        print('✅ Database is ready!')
        break
    except Exception as e:
        retry_count += 1
        print(f'⏳ Database not ready yet (attempt {retry_count}/{max_retries})...')
        time.sleep(2)

if retry_count >= max_retries:
    print('❌ Database connection timeout')
    exit(1)
"

# Download AI models if URLs are provided
if [ -n "$PILL_RECOGNITION_MODEL_URL" ] && [ ! -f "/app/models/pill_recognition.pt" ]; then
    echo "📥 Downloading pill recognition model..."
    curl -L "$PILL_RECOGNITION_MODEL_URL" -o /app/models/pill_recognition.pt
    echo "✅ Pill recognition model downloaded"
fi

if [ -n "$DDI_MODEL_URL" ] && [ ! -f "/app/models/biobert_ddi_model.pt" ]; then
    echo "📥 Downloading drug interaction model..."
    curl -L "$DDI_MODEL_URL" -o /app/models/biobert_ddi_model.pt
    echo "✅ Drug interaction model downloaded"
fi

# Run database migrations if enabled
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "📦 Running database migrations..."
    alembic upgrade head
    echo "✅ Migrations completed"
else
    echo "⏭️  Skipping migrations (RUN_MIGRATIONS=false)"
fi

# Seed database if enabled and empty
if [ "$SEED_DATABASE" = "true" ]; then
    echo "🌱 Checking if database needs seeding..."
    python -c "
import asyncio
from sqlalchemy import text
from app.db.session import AsyncSessionLocal

async def check_and_seed():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text('SELECT COUNT(*) FROM users'))
        count = result.scalar()
        
        if count == 0:
            print('📥 Database is empty, seeding data...')
            
            # Seed base data
            print('🌱 Seeding base data...')
            import subprocess
            subprocess.run(['python', 'app/scripts/seed_data.py'], check=True)
            
            # Seed AI enhancements data
            print('🤖 Seeding AI enhancements data...')
            subprocess.run(['python', 'app/scripts/seed_ai_enhancements.py'], check=True)
            
            print('✅ Database seeding completed')
        else:
            print(f'✅ Database already has {count} users, skipping seeding')

asyncio.run(check_and_seed())
    "
else
    echo "⏭️  Skipping database seeding (SEED_DATABASE=false)"
fi

echo ""
echo "🎉 Starting FastAPI server..."
echo "📡 Health check: /health"
echo "📚 API docs: /docs"
echo ""

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers --forwarded-allow-ips='*'
