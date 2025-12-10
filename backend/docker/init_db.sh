#!/bin/bash
# Docker Database Initialization Script
# =====================================
# Run this script to initialize and seed the database in Docker

set -e  # Exit on error

echo "🐳 Starting Docker Database Initialization..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start database and redis only
echo "🚀 Starting database and Redis..."
docker-compose up -d db redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run migrations
echo "📦 Running database migrations..."
docker-compose run --rm backend alembic upgrade head

# Seed base data
echo "🌱 Seeding base data (users, medications, pharmacies)..."
docker-compose run --rm backend python app/scripts/seed_data.py

# Seed AI enhancements data
echo "🤖 Seeding AI enhancements data..."
docker-compose run --rm backend python app/scripts/seed_ai_enhancements.py

echo ""
echo "✅ Database initialization completed!"
echo ""
echo "📊 Database Statistics:"
docker-compose exec -T db psql -U postgres -d ai500 -c "
SELECT 
    'users' as table_name, COUNT(*) as count FROM users
UNION ALL
SELECT 'medications', COUNT(*) FROM medications
UNION ALL
SELECT 'pharmacies', COUNT(*) FROM pharmacies
UNION ALL
SELECT 'pharmacy_inventory', COUNT(*) FROM pharmacy_inventory
UNION ALL
SELECT 'medication_recalls', COUNT(*) FROM medication_recalls
UNION ALL
SELECT 'pharmacy_reviews', COUNT(*) FROM pharmacy_reviews
UNION ALL
SELECT 'user_notifications', COUNT(*) FROM user_notifications;
"

echo ""
echo "🎉 All done! You can now start the backend:"
echo "   docker-compose up backend"
echo ""
echo "📝 Database access:"
echo "   Host: localhost"
echo "   Port: 5433"
echo "   User: postgres"
echo "   Password: admin"
echo "   Database: ai500"
echo ""
echo "🌐 pgAdmin: http://localhost:5050"
echo "   Email: admin@admin.com"
echo "   Password: admin"
