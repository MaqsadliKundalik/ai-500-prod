# 🚀 Quick Start for API Integration

## Backend is Ready! ✅

### API Base URL (Local Development)
```
http://localhost:8001/api/v1
```

### API Base URL (Production - After Deployment)
```
https://yourdomain.com/api/v1
```

---

## 📋 Quick Test

### 1. Test Health
```bash
curl http://localhost:8001/health
```

Expected Response:
```json
{
  "status": "healthy",
  "ai_models": {
    "status": "3/3 models available"
  }
}
```

### 2. Test Login
```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=password123"
```

### 3. Test Medication Search
```bash
# Get token first
TOKEN="your_token_here"

curl -X GET "http://localhost:8001/api/v1/medications/search?q=aspirin" \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Test Scan Endpoint
```bash
curl -X POST http://localhost:8001/api/v1/scans/image \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@/path/to/pill.jpg"
```

---

## 🌐 Interactive API Documentation

Visit: **http://localhost:8001/api/docs**

This provides:
- ✅ All endpoints with descriptions
- ✅ Try it out functionality
- ✅ Request/response examples
- ✅ Authentication testing

---

## 🔑 Test Credentials

```
Email: test@example.com
Password: password123

OR

Email: demo@sentinel-rx.uz
Password: demo123
```

---

## 📱 Frontend Integration Examples

### React/Next.js Example

```typescript
// api/client.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1';

export class SentinelAPI {
  private token: string | null = null;

  async login(email: string, password: string) {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ username: email, password }),
    });
    
    const data = await response.json();
    this.token = data.access_token;
    localStorage.setItem('token', this.token);
    return data;
  }

  async scanMedication(imageFile: File) {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/scans/image`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
      },
      body: formData,
    });

    return response.json();
  }

  async searchMedications(query: string) {
    const response = await fetch(
      `${API_BASE_URL}/medications/search?q=${encodeURIComponent(query)}`,
      {
        headers: {
          'Authorization': `Bearer ${this.token}`,
        },
      }
    );

    return response.json();
  }

  async findNearbyPharmacies(lat: number, lon: number) {
    const response = await fetch(
      `${API_BASE_URL}/pharmacies/nearby?latitude=${lat}&longitude=${lon}&radius_km=5`,
      {
        headers: {
          'Authorization': `Bearer ${this.token}`,
        },
      }
    );

    return response.json();
  }
}

export const api = new SentinelAPI();
```

### React Native Example

```typescript
// services/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor for auth token
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const authAPI = {
  login: (email: string, password: string) =>
    apiClient.post('/auth/login', 
      new URLSearchParams({ username: email, password }),
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' }}
    ),
  
  register: (data: RegisterData) =>
    apiClient.post('/auth/register', data),
};

export const scanAPI = {
  scanImage: (imageFile: File) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    return apiClient.post('/scans/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  getHistory: (skip = 0, limit = 20) =>
    apiClient.get(`/scans/history?skip=${skip}&limit=${limit}`),
};

export const medicationAPI = {
  search: (query: string) =>
    apiClient.get(`/medications/search?q=${query}`),
  
  getDetails: (id: string) =>
    apiClient.get(`/medications/${id}`),
  
  getPrices: (id: string, lat?: number, lon?: number) =>
    apiClient.get(`/medications/${id}/prices`, {
      params: { latitude: lat, longitude: lon },
    }),
};

export const pharmacyAPI = {
  findNearby: (lat: number, lon: number, radius = 5) =>
    apiClient.get('/pharmacies/nearby', {
      params: { latitude: lat, longitude: lon, radius_km: radius },
    }),
  
  getDetails: (id: string) =>
    apiClient.get(`/pharmacies/${id}`),
};
```

---

## 📊 Available Endpoints

### Core Features
- ✅ **Authentication** - Register, Login, Refresh Token
- ✅ **User Management** - Profile, Family Members
- ✅ **Medications** - Search, Details, Prices
- ✅ **Pharmacies** - Find Nearby, Details, Inventory
- ✅ **Scans** - Image Recognition, QR/Barcode, History
- ✅ **Interactions** - Drug-Drug Interaction Checks
- ✅ **Voice** - STT, Query Processing
- ✅ **Dashboard** - Family Overview
- ✅ **Gamification** - Points, Badges, Leaderboard

### AI Features (Working)
- ✅ **Pill Recognition** - CNN model trained
- ✅ **Drug Interaction Detection** - Random Forest (87.5% accuracy)
- ✅ **Price Anomaly Detection** - Isolation Forest

---

## 🔧 Environment Setup

### Option 1: Use Existing Docker Setup
```bash
# Already running!
docker ps
# You should see: backend-backend-1, backend-db-1, backend-redis-1
```

### Option 2: Local Development
```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup database
alembic upgrade head

# Run server
uvicorn app.main:app --reload --port 8001
```

---

## 🚀 Deployment Options

### 1. Render.com (Recommended - Free Tier Available)
- Follow: [DEPLOYMENT.md](./DEPLOYMENT.md#option-2-rendercom-deployment-easiest)
- Deploy time: ~10 minutes
- Cost: Free (with limitations) or $7/month

### 2. DigitalOcean / AWS / Linode
- Follow: [DEPLOYMENT.md](./DEPLOYMENT.md#option-1-manual-vps-deployment)
- Deploy time: ~30 minutes
- Cost: $5-10/month

### 3. Railway.app
- Follow: [DEPLOYMENT.md](./DEPLOYMENT.md#option-3-railwayapp-deployment)
- Deploy time: ~5 minutes
- Cost: Free with usage limits

---

## 📖 Full Documentation

- **API Reference**: [API_REFERENCE.md](./API_REFERENCE.md)
- **Deployment Guide**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Interactive Docs**: http://localhost:8001/api/docs

---

## 🆘 Common Issues

**CORS Error:**
```bash
# Update backend/.env
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

**Database Connection Error:**
```bash
# Check if containers running
docker ps

# Restart if needed
docker-compose -f backend/docker-compose.yml restart
```

**Model Loading Error:**
```bash
# Check models exist
ls backend/models/

# Should see:
# - drug_interaction.pkl
# - price_anomaly.pkl
# - pill_recognition.pt
```

---

## 📞 Support

- **GitHub**: https://github.com/MaxmudovMaqsudbek/PharmaCheck
- **Issues**: Create issue with `[Frontend]` tag
- **Health Check**: http://localhost:8001/health

---

## ✅ Ready for Integration!

Backend is **100% ready** for frontend integration:
- ✅ All endpoints working
- ✅ AI models trained and loaded
- ✅ Database seeded with test data
- ✅ Authentication working
- ✅ API documentation available
- ✅ Deployment guides ready

**Start building your frontend! 🎨**
