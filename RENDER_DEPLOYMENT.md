# Render.com Deployment Guide - Sentinel-RX
===========================================

## 📋 Tayyorgarlik

### 1. GitHub Repository
Loyiha allaqachon GitHub'da: `https://github.com/MaqsadliKundalik/ai-500-prod`

### 2. Render.com Account
1. [Render.com](https://render.com) saytiga kiring
2. GitHub bilan bog'lang

## 🚀 Deploy Qilish Qadamlari

### Usul 1: Blueprint orqali (Tavsiya etiladi)

1. **Render Dashboard'ga kiring**
   - [https://dashboard.render.com](https://dashboard.render.com)

2. **Blueprint yarating**
   - "New" → "Blueprint Instance" ni tanlang
   - Repository: `MaqsadliKundalik/ai-500-prod` ni tanlang
   - Branch: `main` ni tanlang
   - Blueprint file: `render.yaml` ni aniqlaydi

3. **Environment Variables sozlang**
   - `OPENAI_API_KEY` - OpenAI API kalitingiz
   - `SECRET_KEY` - Avtomatik generate bo'ladi
   - `CORS_ORIGINS` - Frontend domeningiz
   - Qolgan o'zgaruvchilar avtomatik sozlanadi

4. **Deploy qiling**
   - "Apply" tugmasini bosing
   - 5-10 daqiqa kutib turing

### Usul 2: Manual Deploy

#### A. PostgreSQL Database
1. Dashboard → "New" → "PostgreSQL"
2. Name: `sentinel-rx-postgres`
3. Database: `sentinel_rx`
4. Region: Oregon (Free)
5. Plan: Free
6. "Create Database" ni bosing

#### B. Redis Cache
1. Dashboard → "New" → "Redis"
2. Name: `sentinel-rx-redis`
3. Region: Oregon (Free)
4. Plan: Free
5. "Create Redis" ni bosing

#### C. Web Service (Backend)
1. Dashboard → "New" → "Web Service"
2. Connect repository: `MaqsadliKundalik/ai-500-prod`
3. Sozlamalar:
   - **Name**: `sentinel-rx-api`
   - **Region**: Oregon
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Environment**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free

4. **Environment Variables qo'shing**:
   ```
   DATABASE_URL=[PostgreSQL Internal URL]
   REDIS_URL=[Redis Internal URL]
   SECRET_KEY=[Generate with: openssl rand -hex 32]
   ENVIRONMENT=production
   DEBUG=false
   CORS_ORIGINS=["https://your-frontend.com"]
   OPENAI_API_KEY=[Your OpenAI Key]
   LOG_LEVEL=INFO
   WORKERS=2
   ```

5. **Advanced Settings**:
   - Health Check Path: `/health`
   - Auto-Deploy: Yes

6. "Create Web Service" ni bosing

## 🔧 Migration va Setup

### 1. Database Migration
Deploy bo'lgandan so'ng, Render Shell orqali:

```bash
# Render Dashboard → Service → Shell
cd backend
alembic upgrade head
```

### 2. Seed Data (Ixtiyoriy)
```bash
python app/scripts/seed_data.py
```

## 📝 Health Check Endpoint Qo'shish

Backend `main.py` fayliga qo'shing:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

## 🌐 URL'lar

Deploy bo'lgandan keyin:
- **API URL**: `https://sentinel-rx-api.onrender.com`
- **Database**: Internal connection (dashboard'da)
- **Redis**: Internal connection (dashboard'da)

## 🔐 Environment Variables

Render Dashboard'da sozlang:

### Majburiy:
- `DATABASE_URL` - Render tomonidan avtomatik beriladi
- `REDIS_URL` - Render tomonidan avtomatik beriladi
- `SECRET_KEY` - Generate qiling
- `OPENAI_API_KEY` - OpenAI kalitingiz

### Ixtiyoriy:
- `GOOGLE_MAPS_API_KEY`
- `SMS_API_KEY`
- `SENTRY_DSN`
- `CORS_ORIGINS`

## 📊 Monitoring

### Logs ko'rish:
```
Render Dashboard → Service → Logs
```

### Metrics:
```
Render Dashboard → Service → Metrics
```

## 🐛 Troubleshooting

### Build muvaffaqiyatsiz bo'lsa:
1. Dockerfile syntax tekshiring
2. requirements.txt dependencies tekshiring
3. Render logs'ni ko'ring

### Database ulanmasa:
1. DATABASE_URL to'g'ri ekanligini tekshiring
2. PostgreSQL service ishlab turganini tekshiring
3. Migration run qilganingizni tekshiring

### Redis ulanmasa:
1. REDIS_URL to'g'ri ekanligini tekshiring
2. Redis service ishlab turganini tekshiring

## 💰 Cost Optimization

### Free Tier Limits:
- Web Service: 750 saat/oy
- PostgreSQL: 1GB storage
- Redis: 25MB memory
- 15 daqiqadan keyin inactive bo'lsa uyquga ketadi

### Production uchun:
- Upgrade to Starter plan ($7/month per service)
- Custom domain qo'shish
- SSL avtomatik faollashadi

## 🔄 Continuous Deployment

GitHub'ga push qilganingizda avtomatik deploy bo'ladi:

```bash
git add .
git commit -m "Update"
git push origin main
```

Render avtomatik detect qiladi va rebuild qiladi.

## 📱 Frontend Integration

Frontend uchun API URL:
```javascript
const API_URL = "https://sentinel-rx-api.onrender.com/api/v1"
```

## 🔗 Foydali Linklar

- [Render Docs](https://render.com/docs)
- [Render Status](https://status.render.com)
- [Render Community](https://community.render.com)
- [Blueprint Spec](https://render.com/docs/blueprint-spec)

## ⚠️ Muhim Eslatmalar

1. **Free tier'da ilk request sekin bo'lishi mumkin** (cold start)
2. **Environment variables'ni himoya qiling** (SECRET_KEY, API keys)
3. **Database backup'larini oling** (manual yoki automated)
4. **CORS sozlamalarini to'g'ri qiling**
5. **Health check endpoint qo'shing**

## 📞 Support

Muammo bo'lsa:
- Render Support: support@render.com
- Community Forum: community.render.com
- Documentation: render.com/docs
