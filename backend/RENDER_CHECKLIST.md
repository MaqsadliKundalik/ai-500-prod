# Quick Render Deployment Checklist
# ==================================

## Before Deployment

### 1. Prepare AI Models (Required)
- [ ] Upload `pill_recognition.pt` to S3/Spaces
- [ ] Upload `biobert_ddi_model.pt` to S3/Spaces (255 MB)
- [ ] Get public URLs for both models
- [ ] Test URLs with `curl -I <url>` to verify accessibility

### 2. Get FDA API Key (Optional but Recommended)
- [ ] Visit https://open.fda.gov/apis/authentication/
- [ ] Register and get API key
- [ ] Test with: `curl "https://api.fda.gov/drug/enforcement.json?api_key=YOUR_KEY&limit=1"`

### 3. Generate Security Keys
```bash
# SECRET_KEY (copy output)
openssl rand -hex 32

# JWT_SECRET_KEY (copy output)
openssl rand -hex 32
```

## Deployment Steps

### Step 1: Push Code to GitHub
```bash
cd backend
git add .
git commit -m "feat: Add Render deployment configuration"
git push origin main
```

### Step 2: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub account
3. Authorize Render to access your repositories

### Step 3: Deploy with Blueprint
1. Dashboard → "New +" → "Blueprint"
2. Select repository: `ai-500-prod`
3. Branch: `main`
4. Render detects `render.yaml`
5. Click "Apply"

### Step 4: Set Environment Variables
**In ai500-backend service settings:**

**Required (Set Immediately):**
```
SECRET_KEY=<paste-generated-key>
JWT_SECRET_KEY=<paste-generated-key>
CORS_ORIGINS=https://your-frontend.onrender.com
```

**AI Models (Set Your S3 URLs):**
```
PILL_RECOGNITION_MODEL_URL=https://your-bucket.s3.amazonaws.com/models/pill_recognition.pt
DDI_MODEL_URL=https://your-bucket.s3.amazonaws.com/models/biobert_ddi_model.pt
```

**Optional (Recommended):**
```
FDA_API_KEY=<your-fda-key>
SENTRY_DSN=<your-sentry-dsn>
```

### Step 5: Wait for Deployment
⏳ First deployment: **5-10 minutes**
- Building Docker image
- Starting database
- Downloading models
- Running migrations
- Seeding data

### Step 6: Verify Deployment
```bash
# Check health
curl https://ai500-backend.onrender.com/health

# Expected:
# {"status":"healthy","database":"connected","redis":"connected"}

# Check API docs
# https://ai500-backend.onrender.com/docs
```

## Post-Deployment

### Test API Endpoints
```bash
# 1. Register user
curl -X POST https://ai500-backend.onrender.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","full_name":"Test User"}'

# 2. Login
curl -X POST https://ai500-backend.onrender.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@example.com","password":"Test123!"}'

# 3. Test Uzbek NLU
curl -X POST https://ai500-backend.onrender.com/api/v1/ai/nlu/understand \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"text":"Aspirin dori bor mi?"}'
```

### Configure Monitoring (Optional)
1. **Sentry**:
   - Create project at https://sentry.io
   - Copy DSN
   - Add to environment variables

2. **Uptime Monitoring**:
   - Free tier sleeps after 15 min
   - Use https://cron-job.org to ping every 10 min
   - URL: `https://ai500-backend.onrender.com/health`

### Set Up Custom Domain (Optional)
1. Service → Settings → Custom Domain
2. Add: `api.yourdomain.com`
3. Add CNAME record in DNS:
   ```
   Type: CNAME
   Name: api
   Value: ai500-backend.onrender.com
   ```

## Troubleshooting

### Deployment Failed
```bash
# View logs
Dashboard → ai500-backend → Logs

# Common issues:
# 1. Out of memory → Upgrade to Starter ($7/month)
# 2. Model download failed → Check S3 URLs
# 3. Database timeout → Wait and retry
```

### Service Not Responding
```bash
# Check service status
Dashboard → ai500-backend → Events

# Restart service
Dashboard → ai500-backend → Manual Deploy → Deploy latest commit
```

### Database Empty
```bash
# SSH into service
Dashboard → ai500-backend → Shell

# Run seeding manually
python app/scripts/seed_data.py
python app/scripts/seed_ai_enhancements.py
```

## Upgrading Plans

### Free Tier Limitations
- ❌ Sleeps after 15 minutes of inactivity
- ❌ 512 MB RAM (may be insufficient for AI models)
- ❌ No persistent disk (must use S3 for models)
- ✅ Good for: Testing, demos, low-traffic apps

### Starter Tier ($25/month)
- ✅ Always on (no sleep)
- ✅ 512 MB RAM per service
- ✅ Persistent disk (1 GB included)
- ✅ Automatic SSL
- ✅ Good for: Small production apps

### Standard Tier ($75/month)
- ✅ 2 GB RAM per service
- ✅ Multiple workers (horizontal scaling)
- ✅ 10 GB disk
- ✅ Priority support
- ✅ Good for: Medium traffic, production

**Recommendation**: Start with Free tier, upgrade to Starter when ready for production.

## Important URLs

After deployment, save these:

- **Backend**: https://ai500-backend.onrender.com
- **API Docs**: https://ai500-backend.onrender.com/docs
- **Health**: https://ai500-backend.onrender.com/health
- **Database**: `postgresql://...` (from Render dashboard)
- **Redis**: `redis://...` (from Render dashboard)

## Security Checklist

- [x] Strong SECRET_KEY generated
- [x] Strong JWT_SECRET_KEY generated
- [x] CORS restricted to frontend domain only
- [ ] FDA_API_KEY not exposed in logs
- [ ] Database password auto-generated by Render
- [ ] HTTPS enabled (automatic with Render)
- [ ] Non-root user in Docker (already configured)

## Cost Tracking

### Free Tier (For Testing)
- Web Service: $0
- PostgreSQL: $0
- Redis: $0
- **Total: $0/month**

### Starter Tier (For Production)
- Web Service: $7
- PostgreSQL: $7
- Redis: $10
- Disk: $1 (1 GB)
- **Total: $25/month**

### Data Transfer
- Free: 100 GB/month
- Overage: $0.10/GB

## Next Actions

After successful deployment:

1. **Test thoroughly** - Try all API endpoints
2. **Update frontend** - Point to new backend URL
3. **Monitor errors** - Set up Sentry
4. **Configure backups** - Enable in Render dashboard
5. **Document for team** - Share URLs and credentials
6. **Plan scaling** - Monitor usage, upgrade when needed

## Support

- **Render Docs**: https://render.com/docs
- **Render Status**: https://status.render.com
- **Support**: support@render.com
- **This Project**: Check RENDER_DEPLOYMENT.md for detailed guide

---

✅ **Ready to deploy!** Follow steps 1-6 above.
