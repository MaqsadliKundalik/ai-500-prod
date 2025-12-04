# Sentinel-RX Deployment Guide
================================

## 🚀 Production Deployment

### Prerequisites
- VPS/Cloud server (DigitalOcean, AWS, Linode, etc.)
- Domain name with DNS configured
- Docker & Docker Compose installed
- SSL certificate (Let's Encrypt recommended)

### Option 1: Manual VPS Deployment

#### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Clone repository
git clone https://github.com/MaxmudovMaqsudbek/PharmaCheck.git
cd PharmaCheck/backend
```

#### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env

# Required changes:
# - DB_PASSWORD: Strong password
# - JWT_SECRET_KEY: Generate with: openssl rand -hex 32
# - REDIS_PASSWORD: Strong password
# - CORS_ORIGINS: Your domain
# - OPENAI_API_KEY: Your OpenAI key
```

#### 3. SSL Certificate (Let's Encrypt)
```bash
# Install certbot
sudo apt install certbot -y

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates
sudo mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/
```

#### 4. Update Configuration
```bash
# Edit nginx config
nano nginx/nginx.conf

# Replace 'yourdomain.com' with your actual domain

# Edit docker-compose.prod.yml
nano docker-compose.prod.yml

# Update CORS_ORIGINS with your domain
```

#### 5. Deploy
```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Run database migrations
docker exec sentinel-rx-api alembic upgrade head

# Check health
curl http://localhost/health
```

#### 6. Firewall Configuration
```bash
# Allow HTTP, HTTPS, SSH
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Option 2: Render.com Deployment (Easiest)

#### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin master
```

#### 2. Create Render Services

**Database:**
- Go to Render.com → New → PostgreSQL
- Name: sentinel-rx-db
- Plan: Free or Starter
- Copy Internal Database URL

**Redis:**
- New → Redis
- Name: sentinel-rx-redis
- Plan: Free
- Copy Internal Redis URL

**Web Service:**
- New → Web Service
- Connect GitHub repository
- Name: sentinel-rx-api
- Environment: Docker
- Branch: master
- Plan: Starter ($7/month minimum for always-on)

**Environment Variables:**
```
ENVIRONMENT=production
DEBUG=False
DATABASE_URL=<internal_postgres_url>
REDIS_URL=<internal_redis_url>
JWT_SECRET_KEY=<generate_with_openssl>
OPENAI_API_KEY=<your_key>
CORS_ORIGINS=["https://sentinel-rx-api.onrender.com"]
```

### Option 3: Railway.app Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add services
railway add postgres redis

# Deploy
railway up

# Set environment variables in Railway dashboard
```

### Option 4: DigitalOcean App Platform

1. Push to GitHub
2. Go to DigitalOcean → Apps → Create App
3. Select repository
4. Configure:
   - Name: sentinel-rx
   - Type: Docker Hub or Dockerfile
   - Environment: Production
5. Add PostgreSQL database
6. Add Redis cache
7. Configure environment variables
8. Deploy

---

## 🔧 Post-Deployment

### 1. Test API
```bash
# Health check
curl https://yourdomain.com/health

# API docs
open https://yourdomain.com/api/docs

# Test login
curl -X POST https://yourdomain.com/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=password123"
```

### 2. Monitor Logs
```bash
# Application logs
docker logs -f sentinel-rx-api

# Nginx logs
docker logs -f sentinel-rx-nginx

# Database logs
docker logs -f sentinel-rx-db
```

### 3. Backup Database
```bash
# Create backup
docker exec sentinel-rx-db pg_dump -U postgres ai500 > backup.sql

# Restore backup
cat backup.sql | docker exec -i sentinel-rx-db psql -U postgres ai500
```

### 4. Update Application
```bash
# Pull latest changes
git pull origin master

# Rebuild
docker-compose -f docker-compose.prod.yml up -d --build

# Run migrations
docker exec sentinel-rx-api alembic upgrade head
```

---

## 📊 Monitoring

### Sentry Integration (Error Tracking)
```python
# Already configured in main.py
# Just add SENTRY_DSN to .env
SENTRY_DSN=https://xxxxx@sentry.io/xxxxx
```

### Prometheus Metrics
```bash
# Metrics endpoint
curl https://yourdomain.com/metrics
```

### Health Check Monitoring
Set up UptimeRobot or similar to ping:
```
https://yourdomain.com/health
```

---

## 🔒 Security Checklist

- [x] Strong passwords for DB, Redis
- [x] JWT secret key properly generated
- [x] HTTPS enabled
- [x] CORS configured correctly
- [x] Rate limiting enabled
- [x] File upload size limits
- [x] Firewall configured
- [x] Regular backups
- [ ] Setup automated SSL renewal
- [ ] Enable 2FA for server access

---

## 📱 API Base URLs

**Production:**
```
https://yourdomain.com/api/v1
```

**Documentation:**
```
https://yourdomain.com/api/docs
```

**Health:**
```
https://yourdomain.com/health
```

---

## 🤝 Frontend Integration

Share these with your frontend developers:

```javascript
// API Configuration
const API_BASE_URL = 'https://yourdomain.com/api/v1';

// Example: Login
const response = await fetch(`${API_BASE_URL}/auth/login`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    username: 'user@example.com',
    password: 'password123'
  })
});

const { access_token } = await response.json();

// Example: Scan medication
const formData = new FormData();
formData.append('image', imageFile);

const scanResponse = await fetch(`${API_BASE_URL}/scans/image`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`
  },
  body: formData
});

const result = await scanResponse.json();
```

---

## 🆘 Troubleshooting

**Container won't start:**
```bash
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up --build
```

**Database connection error:**
```bash
# Check database is running
docker ps | grep db

# Check connection string
docker exec sentinel-rx-api env | grep DATABASE_URL
```

**SSL certificate issues:**
```bash
# Renew certificate
sudo certbot renew

# Copy to nginx
sudo cp /etc/letsencrypt/live/yourdomain.com/* nginx/ssl/

# Restart nginx
docker-compose -f docker-compose.prod.yml restart nginx
```

---

## 📞 Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Check health: `curl http://localhost/health`
3. GitHub Issues: https://github.com/MaxmudovMaqsudbek/PharmaCheck/issues
