# Docker Quick Start Guide
# ========================

## Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker

## Quick Start (Development)

### 1. Initialize Database (First Time Only)

**Windows:**
```powershell
.\docker\init_db.ps1
```

**Linux/Mac:**
```bash
chmod +x docker/init_db.sh
./docker/init_db.sh
```

This will:
- Start PostgreSQL and Redis
- Run database migrations
- Seed initial data (users, medications, pharmacies)
- Seed AI enhancements data (inventory, recalls, reviews, notifications)

### 2. Start All Services

```bash
docker-compose up
```

Or in detached mode:
```bash
docker-compose up -d
```

### 3. Access Services

- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **pgAdmin**: http://localhost:5050
  - Email: admin@admin.com
  - Password: admin
- **Database**:
  - Host: localhost
  - Port: 5433
  - User: postgres
  - Password: admin
  - Database: ai500

## Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f db
```

### Restart Services
```bash
docker-compose restart
```

### Stop Services
```bash
docker-compose down
```

### Rebuild Containers
```bash
docker-compose build
docker-compose up -d
```

### Run Database Migrations
```bash
docker-compose run --rm backend alembic upgrade head
```

### Seed Data Manually
```bash
# Base data
docker-compose run --rm backend python app/scripts/seed_data.py

# AI enhancements data
docker-compose run --rm backend python app/scripts/seed_ai_enhancements.py
```

### Access Database Shell
```bash
docker-compose exec db psql -U postgres -d ai500
```

### Access Backend Shell
```bash
docker-compose exec backend bash
```

## Database Statistics Query

```bash
docker-compose exec db psql -U postgres -d ai500 -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

## Troubleshooting

### Port Already in Use
If ports 5433, 6380, or 8001 are already in use, edit `docker-compose.yml`:
```yaml
ports:
  - "5434:5432"  # Change 5433 to 5434
```

### Database Connection Failed
```bash
# Check if database is running
docker-compose ps

# View database logs
docker-compose logs db

# Restart database
docker-compose restart db
```

### Reset Everything
```bash
# Stop and remove all containers, volumes
docker-compose down -v

# Reinitialize
./docker/init_db.sh  # or init_db.ps1 on Windows
docker-compose up
```

## Production Deployment

For production, use:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

Make sure to:
1. Set strong passwords in `.env`
2. Use external database (not Docker)
3. Configure proper SSL/TLS
4. Set up backups
5. Use Redis password
6. Configure monitoring (Sentry)
