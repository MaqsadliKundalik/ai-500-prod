# 🚀 Render.com'ga Deploy Qilish - Tezkor Yo'riqnoma

## 1️⃣ Tayyorgarlik (5 daqiqa)

### GitHub Repository
✅ Loyiha allaqachon GitHub'da: `https://github.com/MaqsadliKundalik/ai-500-prod`

### Render.com Account
1. [render.com](https://render.com) saytiga kiring
2. "Get Started for Free" tugmasini bosing
3. GitHub bilan bog'lang
4. Repository access bering

---

## 2️⃣ Deploy Qilish (10 daqiqa)

### Usul 1: Blueprint (Eng Oson) ⭐

1. **Render Dashboard'ga kiring**
   - [https://dashboard.render.com](https://dashboard.render.com)

2. **New Blueprint Instance yarating**
   - Dashboard → "New" → "Blueprint Instance"
   - "Connect Account" → GitHub'ni tanlang
   - Repository: `MaqsadliKundalik/ai-500-prod`
   - Branch: `main`
   - Blueprint: `render.yaml` (avtomatik topiladi)

3. **Service Name'larni tasdiqlang**
   - `sentinel-rx-postgres` - Database
   - `sentinel-rx-redis` - Cache
   - `sentinel-rx-api` - Backend API

4. **Environment Variables qo'shing**
   ```
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```
   
5. **"Apply" tugmasini bosing**
   - ⏳ Deploy jarayoni: ~5-10 daqiqa
   - ✅ Status "Live" bo'lganda tayyor!

---

### Usul 2: Manual (Qo'lda)

#### A. PostgreSQL Database yarating
1. Dashboard → "New" → "PostgreSQL"
2. Settings:
   - **Name**: `sentinel-rx-postgres`
   - **Database**: `sentinel_rx`
   - **Region**: Oregon (Free)
   - **Plan**: Free
3. "Create Database" → ⏳ 2-3 daqiqa kutish

#### B. Redis yarating
1. Dashboard → "New" → "Redis"
2. Settings:
   - **Name**: `sentinel-rx-redis`
   - **Region**: Oregon (Free)
   - **Plan**: Free
3. "Create Redis" → ⏳ 1-2 daqiqa kutish

#### C. Web Service yarating
1. Dashboard → "New" → "Web Service"
2. "Connect Repository" → `MaqsadliKundalik/ai-500-prod`
3. Settings:
   - **Name**: `sentinel-rx-api`
   - **Region**: Oregon
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Environment**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free

4. **Environment Variables**:
   ```
   DATABASE_URL = [Copy from PostgreSQL service Internal URL]
   REDIS_URL = [Copy from Redis service Internal URL]
   SECRET_KEY = [Generate: openssl rand -hex 32]
   ENVIRONMENT = production
   DEBUG = false
   CORS_ORIGINS = ["*"]
   OPENAI_API_KEY = [Your OpenAI API Key]
   LOG_LEVEL = INFO
   WORKERS = 2
   PORT = 8000
   ```

5. **Advanced**:
   - Health Check Path: `/health`
   - Auto-Deploy: ✅ Yes

6. "Create Web Service" → ⏳ 5-10 daqiqa kutish

---

## 3️⃣ Tekshirish

### API URL
Deploy bo'lgandan keyin:
```
https://sentinel-rx-api.onrender.com
```

### Health Check
```bash
curl https://sentinel-rx-api.onrender.com/health
```

### API Docs (Development)
```
https://sentinel-rx-api.onrender.com/api/docs
```

---

## 4️⃣ Database Migration

Deploy bo'lgan so'ng, Shell'da migration bajaring:

1. Render Dashboard → `sentinel-rx-api` → "Shell" tab
2. Quyidagi buyruqni kiriting:
   ```bash
   alembic upgrade head
   ```

---

## 5️⃣ Git Push bilan Auto-Deploy

Har safar GitHub'ga push qilganingizda avtomatik deploy bo'ladi:

```bash
git add .
git commit -m "Update: feature name"
git push origin main
```

Render avtomatik yangilanadi! 🔄

---

## 🎯 Keyingi Qadamlar

### Frontend Integration
API URL'ni frontendga qo'shing:
```javascript
const API_URL = "https://sentinel-rx-api.onrender.com/api/v1"
```

### Custom Domain (Ixtiyoriy)
1. Dashboard → Service → "Settings"
2. "Custom Domain" → Add your domain
3. DNS'ga CNAME record qo'shing

### Upgrade to Paid Plan (Production uchun)
- ✅ Always-on (no cold starts)
- ✅ More resources
- ✅ Better performance
- 💰 $7/month per service

---

## ⚠️ Muhim Eslatmalar

### Free Tier Limitations:
- ⏰ 15 daqiqa inactive bo'lsa uyquga ketadi
- 🐌 Birinchi request sekin (cold start: ~30 soniya)
- 💾 Database: 1GB storage
- 💾 Redis: 25MB memory
- ⏱️ 750 soat/oy runtime

### Cold Start'dan qochish:
Har 10 daqiqada ping yuborish:
```bash
# Cron job (UptimeRobot yoki cron-job.org)
*/10 * * * * curl https://sentinel-rx-api.onrender.com/health
```

---

## 🐛 Troubleshooting

### Build Failed?
1. Render Dashboard → Logs'ni tekshiring
2. Dockerfile syntax to'g'rimi?
3. requirements.txt to'liqmi?

### Database Connection Error?
1. DATABASE_URL to'g'ri berilganmi?
2. PostgreSQL service running'mi?
3. Migration run qilganmisiz?

### 502 Bad Gateway?
1. Service starting'mi? (Logs'ni ko'ring)
2. Health check working'mi?
3. Port 8000'da listening'mi?

---

## 📞 Yordam

### Render Support:
- 📧 Email: support@render.com
- 💬 Community: [community.render.com](https://community.render.com)
- 📚 Docs: [render.com/docs](https://render.com/docs)

### Loyiha Issues:
- 🐛 GitHub Issues: [github.com/MaqsadliKundalik/ai-500-prod/issues](https://github.com/MaqsadliKundalik/ai-500-prod/issues)

---

## ✅ Checklist

- [ ] Render account yaratdim
- [ ] GitHub repository bog'ladim
- [ ] Blueprint deploy qildim / Manual deploy qildim
- [ ] Database migration run qildim
- [ ] Health check ishlayapti
- [ ] API endpoints test qildim
- [ ] Frontend bilan integrate qildim
- [ ] Environment variables to'g'ri sozladim
- [ ] OPENAI_API_KEY qo'shdim
- [ ] Auto-deploy sozladim

---

**Omad yor bo'lsin! 🚀**
