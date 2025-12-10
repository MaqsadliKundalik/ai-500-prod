# Docker Database Initialization Script (Windows)
# ===============================================
# Run this script to initialize and seed the database in Docker

Write-Host "🐳 Starting Docker Database Initialization..." -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down

# Start database and redis only
Write-Host "🚀 Starting database and Redis..." -ForegroundColor Green
docker-compose up -d db redis

# Wait for database to be ready
Write-Host "⏳ Waiting for database to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Run migrations
Write-Host "📦 Running database migrations..." -ForegroundColor Cyan
docker-compose run --rm backend alembic upgrade head

# Seed base data
Write-Host "🌱 Seeding base data (users, medications, pharmacies)..." -ForegroundColor Green
docker-compose run --rm backend python app/scripts/seed_data.py

# Seed AI enhancements data
Write-Host "🤖 Seeding AI enhancements data..." -ForegroundColor Magenta
docker-compose run --rm backend python app/scripts/seed_ai_enhancements.py

Write-Host ""
Write-Host "✅ Database initialization completed!" -ForegroundColor Green
Write-Host ""

# Get database statistics
Write-Host "📊 Database Statistics:" -ForegroundColor Cyan
docker-compose exec -T db psql -U postgres -d ai500 -c @"
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
"@

Write-Host ""
Write-Host "🎉 All done! You can now start the backend:" -ForegroundColor Green
Write-Host "   docker-compose up backend" -ForegroundColor Yellow
Write-Host ""
Write-Host "📝 Database access:" -ForegroundColor Cyan
Write-Host "   Host: localhost"
Write-Host "   Port: 5433"
Write-Host "   User: postgres"
Write-Host "   Password: admin"
Write-Host "   Database: ai500"
Write-Host ""
Write-Host "🌐 pgAdmin: http://localhost:5050" -ForegroundColor Cyan
Write-Host "   Email: admin@admin.com"
Write-Host "   Password: admin"
