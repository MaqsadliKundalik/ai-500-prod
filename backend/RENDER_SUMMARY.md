# AI-500 Backend - Render Deployment Summary
# ==========================================

## ✅ Deployment Ready Files Created

### 1. **render.yaml** - Blueprint Configuration
- PostgreSQL database service (free tier)
- Redis cache service (free tier)
- Backend web service with auto-deployment
- All environment variables configured
- Automatic migrations and seeding enabled

### 2. **Dockerfile** - Multi-stage Production Build
- Optimized for Render deployment
- Multi-stage build (reduces image size)
- Non-root user for security
- Health checks enabled
- Automatic startup script

### 3. **docker/start.sh** - Render Startup Script
- Database readiness check (30 retries)
- AI model download from S3 (if URLs provided)
- Automatic database migrations
- Smart database seeding (only if empty)
- FastAPI server with 2 workers

### 4. **RENDER_DEPLOYMENT.md** - Complete Guide
- Prerequisites and setup steps
- Environment variable reference
- Service plans and pricing
- Database management commands
- Monitoring and troubleshooting
- Custom domain setup
- CI/CD configuration

### 5. **RENDER_CHECKLIST.md** - Quick Start Guide
- Pre-deployment checklist
- Step-by-step deployment instructions
- Post-deployment verification
- Testing commands
- Troubleshooting common issues

### 6. **DOCKER.md** - Docker Development Guide
- Local development with Docker
- Database initialization scripts
- Common Docker commands
- Troubleshooting tips

## 🚀 How to Deploy to Render

### Option 1: Automatic with Blueprint (Recommended)
```bash
# 1. Push to GitHub
git add .
git commit -m "feat: Add Render deployment configuration"
git push origin main

# 2. Go to Render Dashboard
https://dashboard.render.com

# 3. New + → Blueprint
# 4. Select your repository
# 5. Click "Apply"
# ✅ Done! Render will automatically:
#    - Create database and Redis
#    - Build Docker image
#    - Run migrations
#    - Seed database
#    - Start server
```

### Option 2: Manual Setup
See RENDER_DEPLOYMENT.md for detailed manual setup instructions.

## 📋 Required Manual Steps

After Blueprint deployment, set these environment variables in Render dashboard:

### Security Keys (Generate these)
```bash
# Generate with OpenSSL:
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET_KEY

# Or PowerShell (Windows):
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

### Model URLs (Upload to S3 first)
```bash
PILL_RECOGNITION_MODEL_URL=https://your-bucket.s3.amazonaws.com/models/pill_recognition.pt
DDI_MODEL_URL=https://your-bucket.s3.amazonaws.com/models/biobert_ddi_model.pt
```

### CORS Origins (Update with frontend URL)
```bash
CORS_ORIGINS=https://your-frontend.onrender.com,https://yourdomain.com
```

### FDA API Key (Optional)
```bash
FDA_API_KEY=<get-from-https://open.fda.gov/apis/authentication/>
```

## 🔧 What Happens on Deployment

### Build Phase (2-3 minutes)
1. ✅ Clone repository from GitHub
2. ✅ Build Docker image (multi-stage)
3. ✅ Install dependencies from requirements.txt
4. ✅ Copy application code

### Startup Phase (2-5 minutes)
1. ✅ Wait for PostgreSQL database to be ready
2. ✅ Download AI models from S3 (if URLs provided)
3. ✅ Run Alembic migrations (alembic upgrade head)
4. ✅ Seed database with initial data (if empty)
   - Users (3 samples)
   - Medications (20 samples)
   - Pharmacies (10 samples)
   - Pharmacy inventory (100-150 records)
   - Medication recalls (3 samples)
   - Pharmacy reviews (50-150 records)
   - User notifications (20-80 records)
5. ✅ Start FastAPI server on port 8000

### First Deployment Time: ~5-10 minutes

## 🌐 Your Deployed Services

After successful deployment:

- **Backend API**: `https://ai500-backend.onrender.com`
- **API Documentation**: `https://ai500-backend.onrender.com/docs`
- **Health Check**: `https://ai500-backend.onrender.com/health`
- **Database**: `postgresql://...` (Internal Render URL)
- **Redis**: `redis://...` (Internal Render URL)

## 📊 Service Plans & Pricing

### Free Tier (For Testing)
- ✅ Web: 512 MB RAM, sleeps after 15 min inactivity
- ✅ Database: 1 GB storage, expires in 90 days
- ✅ Redis: 25 MB, expires in 90 days
- ❌ No persistent disk (use S3 for models)
- **Cost: $0/month**

### Starter Tier (For Production)
- ✅ Web: $7/month - Always on, 512 MB RAM
- ✅ Database: $7/month - Persistent 1 GB
- ✅ Redis: $10/month - Persistent 256 MB
- ✅ Disk: $1/GB/month (optional, for models)
- **Total: ~$25/month**

### Recommendation
- **Development/Testing**: Free tier
- **Production**: Starter tier minimum
- **High Traffic**: Standard tier ($75/month)

## ⚠️ Important Notes

### 1. Large AI Models (255 MB biobert_ddi_model.pt)
**Problem**: Too large for GitHub (100 MB limit)
**Solution**: Upload to S3/Spaces, set model URLs in environment variables

### 2. Free Tier Sleep
**Problem**: Service sleeps after 15 minutes of inactivity
**Solution**: 
- Use free cron job service (https://cron-job.org) to ping every 10 min
- OR upgrade to Starter tier ($7/month)

### 3. Database Seeding
**Automatic**: Script checks if database is empty before seeding
**Manual**: SSH into service and run:
```bash
python app/scripts/seed_data.py
python app/scripts/seed_ai_enhancements.py
```

### 4. AI Models Not Loading
**Check**: Model URLs are accessible with curl
**Fix**: Verify S3 bucket permissions (public read)
**Alternative**: Deploy without models, add later

## 🧪 Testing Deployment

```bash
# 1. Health check
curl https://ai500-backend.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "database": {"status": "connected", "latency_ms": 45},
  "redis": {"status": "connected"},
  "ai_models": {"status": "2/3 models available"},
  "version": "1.0.0",
  "environment": "production"
}

# 2. API docs
# Open in browser: https://ai500-backend.onrender.com/docs

# 3. Test endpoint
curl -X POST https://ai500-backend.onrender.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","full_name":"Test User"}'
```

## 📈 Monitoring

### Logs
View in real-time:
- Dashboard → ai500-backend → Logs

### Metrics
- Dashboard → ai500-backend → Metrics
- CPU, Memory, Network usage
- Request count and latency

### Alerts (Optional)
- Set up Sentry: https://sentry.io
- Add SENTRY_DSN to environment variables

## 🔄 CI/CD - Auto Deployment

With `render.yaml` and `autoDeploy: true`:
- Push to `main` branch → Automatic deployment
- No manual intervention needed
- Zero-downtime deployments

**Disable auto-deploy**:
```yaml
# In render.yaml
services:
  - type: web
    autoDeploy: false  # Change to false
```

## 🆘 Troubleshooting

### Build Failed
```bash
# View build logs
Dashboard → ai500-backend → Events → Latest Build

# Common fixes:
# 1. Out of memory → Upgrade to Starter tier
# 2. Dependency error → Check requirements.txt
# 3. Docker error → Check Dockerfile syntax
```

### Migration Failed
```bash
# SSH into service
Dashboard → ai500-backend → Shell

# Run manually
alembic upgrade head

# Check database
psql $DATABASE_URL
\dt  # List tables
```

### Service Unhealthy
```bash
# Check health endpoint response
curl https://ai500-backend.onrender.com/health

# If database status = "error":
# → Check DATABASE_URL in environment variables
# → Restart database service
# → View database logs

# If redis status = "not_configured":
# → This is OK, Redis is optional
# → Application will work without it
```

## 📚 Documentation Files

1. **RENDER_DEPLOYMENT.md** - Complete deployment guide
2. **RENDER_CHECKLIST.md** - Quick deployment checklist
3. **DOCKER.md** - Local Docker development
4. **API_REFERENCE.md** - API endpoints documentation
5. **README.md** - Project overview

## ✅ Deployment Checklist

Before deploying:
- [x] render.yaml created with all services
- [x] Dockerfile optimized for production
- [x] start.sh script with auto-migration/seeding
- [x] Health check endpoint updated
- [x] Environment variables documented
- [ ] AI models uploaded to S3
- [ ] SECRET_KEY and JWT_SECRET_KEY generated
- [ ] FDA_API_KEY obtained (optional)
- [ ] Code pushed to GitHub
- [ ] Render account created

After deploying:
- [ ] Set required environment variables in Render
- [ ] Verify health check returns "healthy"
- [ ] Test API endpoints with curl
- [ ] Check database has seeded data
- [ ] Update frontend CORS_ORIGINS
- [ ] Set up monitoring (Sentry)
- [ ] Configure custom domain (optional)
- [ ] Enable database backups

## 🎉 Next Steps

1. **Deploy to Render** using Blueprint
2. **Test thoroughly** - All API endpoints
3. **Update frontend** - Point to new backend URL
4. **Monitor** - Set up Sentry and logs
5. **Scale** - Upgrade plans as needed

---

**Ready to deploy!** Follow RENDER_CHECKLIST.md for step-by-step instructions.
