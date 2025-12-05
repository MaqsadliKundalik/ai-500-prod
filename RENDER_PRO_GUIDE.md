# Sentinel-RX Pro Deployment Guide for Render.com
===============================================

## 🚀 Pro Tier Features

### Resources
- **Memory**: 2GB+ RAM (vs 512MB free)
- **CPU**: Dedicated cores
- **Workers**: 4 workers (vs 1)
- **Uptime**: 99.95% SLA
- **No cold starts**: Always-on
- **Custom domains**: Multiple domains supported
- **SSL**: Automatic with custom domains

### AI Features Enabled
✅ PyTorch pill recognition models
✅ Drug interaction detection
✅ Price anomaly detection  
✅ Voice assistant (Whisper)
✅ Image processing (OpenCV)
✅ Real-time notifications

---

## 📋 Pre-Deployment Checklist

### 1. API Keys Tayyorlash
- [ ] OpenAI API Key (required for AI features)
- [ ] Google Maps API Key (pharmacy locations)
- [ ] Sentry DSN (error monitoring)
- [ ] SMS Provider API (optional)
- [ ] Exchange Rate API (currency conversion)

### 2. Secret Key Generate Qilish
```bash
# PowerShell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | % {[char]$_})

# yoki online: https://randomkeygen.com/
```

### 3. Domain Sozlash (Optional)
- Domain provider'da CNAME record qo'shish
- Render'da custom domain qo'shish

---

## 🎯 Deployment Steps

### Step 1: Render.com Pro Setup

1. **Dashboard** → https://dashboard.render.com
2. **Upgrade to Pro**: Billing → Upgrade to Pro Plan

### Step 2: PostgreSQL Database (Pro)

1. **New** → **PostgreSQL**
2. **Settings**:
   - Name: `sentinel-rx-postgres-pro`
   - Database: `sentinel_rx`
   - Region: `Oregon (US West)`
   - Plan: **Standard** ($7/month - 5GB storage, 2 vCPU)
   
3. **Advanced**:
   - PostgreSQL Version: 15
   - Enable Point-in-Time Recovery: ✅
   - Connection pooling: ✅

4. **Create Database** → Copy **Internal Database URL**

### Step 3: Redis Cache (Pro)

1. **New** → **Redis**
2. **Settings**:
   - Name: `sentinel-rx-redis-pro`
   - Region: `Oregon (US West)`
   - Plan: **Standard** ($10/month - 250MB, persistent)
   - Eviction Policy: `allkeys-lru`
   
3. **Create Redis** → Copy **Internal Redis URL**

### Step 4: Web Service (Pro)

1. **New** → **Web Service**
2. **Connect Repository**: `MaqsadliKundalik/ai-500-prod`
3. **Basic Settings**:
   - Name: `sentinel-rx-api-pro`
   - Region: `Oregon`
   - Branch: `main`
   - Runtime: `Docker`
   - Instance Type: **Pro** ($25/month - 2GB RAM, 1 vCPU)

4. **Advanced Settings**:
   - Health Check Path: `/health`
   - Auto-Deploy: ✅ Yes

### Step 5: Environment Variables

Dashboard → Service → Environment → Add Environment Variables:

```bash
# Database & Redis (auto-injected)
DATABASE_URL=[Paste from PostgreSQL]
REDIS_URL=[Paste from Redis]

# Security (GENERATE NEW!)
SECRET_KEY=[Your generated 64-char key]
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Application
APP_NAME=Sentinel-RX
APP_VERSION=0.1.0
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4

# CORS (update with your domain)
CORS_ORIGINS=["https://api.sentinel-rx.com","https://sentinel-rx.com","https://app.sentinel-rx.com"]

# AI Services (REQUIRED)
OPENAI_API_KEY=[Your OpenAI API Key]
PILL_RECOGNITION_MODEL_PATH=./models/pill_recognition.pt
DRUG_INTERACTION_MODEL_PATH=./models/drug_interaction.pkl

# External APIs (Optional but recommended)
GOOGLE_MAPS_API_KEY=[Your Google Maps Key]
EXCHANGE_RATE_API_KEY=[Your Exchange Rate Key]

# Monitoring (Recommended)
SENTRY_DSN=[Your Sentry DSN]
ENABLE_METRICS=true

# File Upload
MAX_UPLOAD_SIZE=10485760
UPLOAD_DIR=./uploads

# Rate Limiting (Pro allows higher)
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

### Step 6: Deploy!

Click **"Create Web Service"**
- Build time: ~10-15 minutes (full AI dependencies)
- Deploy time: ~2-3 minutes

---

## 🔧 Post-Deployment Setup

### 1. Run Database Migrations

Service → **Shell** tab:
```bash
cd /app
alembic upgrade head
```

### 2. Create Admin User (Optional)

```bash
python -c "
from app.services.user_service import UserService
from app.db.session import SessionLocal
db = SessionLocal()
# Create admin user
"
```

### 3. Upload AI Models (If needed)

If you have pre-trained models:
```bash
# Using Render Disk or S3
# Upload to /app/models/ directory
```

### 4. Test Health Check

```bash
curl https://sentinel-rx-api-pro.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": {"status": "connected", "latency_ms": 10.5},
  "redis": {"status": "connected"},
  "ai_models": {
    "status": "3/3 models available",
    "details": {
      "pill_recognition": true,
      "drug_interaction": true,
      "price_anomaly": true
    }
  }
}
```

---

## 🌐 Custom Domain Setup

### 1. Render'da Domain Qo'shish

Service → **Settings** → **Custom Domain**
- Add: `api.sentinel-rx.com`
- Add: `sentinel-rx.com`

### 2. DNS Configuration

Domain provider'da (GoDaddy, Namecheap, etc):

**For subdomain (api.sentinel-rx.com):**
```
Type: CNAME
Name: api
Value: sentinel-rx-api-pro.onrender.com
```

**For root domain (sentinel-rx.com):**
```
Type: A
Name: @
Value: [Render IP from dashboard]
```

### 3. SSL Certificate

Render avtomatik Let's Encrypt SSL beradi (2-5 daqiqa)

---

## 📊 Monitoring & Logs

### Real-time Logs
Dashboard → Service → **Logs** tab

### Metrics
Dashboard → Service → **Metrics** tab
- CPU usage
- Memory usage
- Request rate
- Response times

### Sentry Integration

`SENTRY_DSN` env variable sozlang:
```bash
SENTRY_DSN=https://[key]@o[org].ingest.sentry.io/[project]
```

---

## 🔄 CI/CD Auto-Deploy

Har safar `git push origin main` qilganingizda:
1. Render avtomatik detect qiladi
2. Docker image rebuild bo'ladi
3. Health check pass bo'lsa deploy bo'ladi
4. Zero-downtime deployment (Pro feature)

---

## 💾 Backup Strategy

### Database Backups

PostgreSQL Pro plan:
- Automatic daily backups (30 days retention)
- Point-in-Time Recovery (PITR)
- Manual backup: Dashboard → Database → **Manual Backup**

### Manual Backup

```bash
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql
```

---

## 🔐 Security Best Practices

1. **Environment Variables**: Hech qachon Git'ga commit qilmang
2. **API Keys**: Render dashboard'da encrypted saqlanadi
3. **Database**: Private networking (internal URL)
4. **Redis**: Password protected
5. **Rate Limiting**: Enable to prevent abuse
6. **CORS**: Faqat trusted domains

---

## 📈 Scaling Options

### Vertical Scaling (More Resources)
Service → Settings → Instance Type:
- **Pro**: 2GB RAM, 1 vCPU ($25/month)
- **Pro Plus**: 4GB RAM, 2 vCPU ($85/month)
- **Pro Max**: 8GB RAM, 4 vCPU ($225/month)

### Horizontal Scaling (More Instances)
- Add load balancer
- Multiple service instances
- Shared database & Redis

---

## 🐛 Troubleshooting

### High Memory Usage
- Reduce workers: `WORKERS=2`
- Optimize AI models
- Enable caching

### Slow Response Times
- Check database query performance
- Add Redis caching
- Optimize endpoints

### Deployment Failed
- Check build logs
- Verify all dependencies in requirements.prod.txt
- Test locally with Docker first

---

## 📞 Support

**Render Support:**
- Pro Plan: Priority email support
- Dashboard → Help → Contact Support
- Response time: < 24 hours

**Emergency:**
- Status: https://status.render.com
- Twitter: @render

---

## 💰 Cost Breakdown (Monthly)

| Service | Plan | Cost |
|---------|------|------|
| Web Service | Pro (2GB) | $25 |
| PostgreSQL | Standard (5GB) | $7 |
| Redis | Standard (250MB) | $10 |
| **Total** | | **$42/month** |

**Additional costs:**
- Custom domains: Free
- SSL certificates: Free
- Bandwidth: Free (100GB included)
- Extra bandwidth: $0.10/GB

---

## ✅ Final Checklist

- [ ] Pro plan activated
- [ ] Database created & migrated
- [ ] Redis configured
- [ ] Web service deployed & healthy
- [ ] All environment variables set
- [ ] AI models loaded
- [ ] Custom domain configured (optional)
- [ ] SSL certificate active
- [ ] Monitoring setup (Sentry)
- [ ] Backups enabled
- [ ] Documentation updated

---

**🎉 Congratulations! Your production API is live!**

**API URL**: `https://sentinel-rx-api-pro.onrender.com`
**Docs**: `https://sentinel-rx-api-pro.onrender.com/api/docs`
**Health**: `https://sentinel-rx-api-pro.onrender.com/health`
