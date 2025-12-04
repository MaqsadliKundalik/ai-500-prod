# 📋 API Endpoints Reference

Base URL: `https://yourdomain.com/api/v1`

## 🔐 Authentication

### Register
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe",
  "phone_number": "+998901234567"
}
```

### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=password123
```

Response:
```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

---

## 👤 Users

### Get Current User
```http
GET /users/me
Authorization: Bearer <access_token>
```

### Update Profile
```http
PUT /users/me
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "full_name": "John Updated",
  "phone_number": "+998901234567",
  "date_of_birth": "1990-01-01",
  "language": "uz"
}
```

### Add Family Member
```http
POST /users/me/family-members
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "full_name": "Jane Doe",
  "relationship": "spouse",
  "date_of_birth": "1992-05-15"
}
```

---

## 💊 Medications

### Search Medications
```http
GET /medications/search?q=aspirin&limit=20
Authorization: Bearer <access_token>
```

### Get Medication Details
```http
GET /medications/{medication_id}
Authorization: Bearer <access_token>
```

### Get Medication Prices
```http
GET /medications/{medication_id}/prices?latitude=41.2995&longitude=69.2401
Authorization: Bearer <access_token>
```

---

## 🏥 Pharmacies

### Find Nearby Pharmacies
```http
GET /pharmacies/nearby?latitude=41.2995&longitude=69.2401&radius_km=5&limit=10
Authorization: Bearer <access_token>
```

### Get Pharmacy Details
```http
GET /pharmacies/{pharmacy_id}
Authorization: Bearer <access_token>
```

### Get Pharmacy Inventory
```http
GET /pharmacies/{pharmacy_id}/inventory
Authorization: Bearer <access_token>
```

---

## 🔬 Scans (AI Features)

### Scan Medication Image
```http
POST /scans/image
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

image: <file>
```

Response:
```json
{
  "scan_id": "scan_1234567890",
  "scan_type": "image",
  "scanned_at": "2025-12-04T12:00:00",
  "recognized": true,
  "medication": {
    "id": "uuid",
    "name": "Aspirin",
    "brand_name": "Bayer",
    "dosage_form": "tablet",
    "strength": "500mg"
  },
  "confidence": 0.95,
  "interactions": {
    "has_interactions": true,
    "total_count": 2,
    "severe_count": 1,
    "interactions": [...]
  },
  "price_analysis": {
    "average_price": 15000,
    "min_price": 12000,
    "max_price": 20000,
    "anomalies_found": 1
  },
  "nearby_pharmacies": [...],
  "batch_recall": {
    "is_recalled": false
  },
  "personalized_insights": [...],
  "points_earned": 5
}
```

### Scan QR/Barcode
```http
POST /scans/qr
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "qr_data": "MED123456",
  "latitude": 41.2995,
  "longitude": 69.2401
}
```

### Get Scan History
```http
GET /scans/history?skip=0&limit=20
Authorization: Bearer <access_token>
```

---

## ⚠️ Drug Interactions

### Check Interactions
```http
POST /interactions/check
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "medication_ids": [
    "uuid1",
    "uuid2"
  ]
}
```

Response:
```json
{
  "total_interactions": 2,
  "severe_interactions": 1,
  "interactions": [
    {
      "medication_name": "Aspirin",
      "interacting_with": "Warfarin",
      "severity": "severe",
      "description": "Increased risk of bleeding",
      "recommendation": "Avoid combination. Consult doctor."
    }
  ]
}
```

### Get User Medication Interactions
```http
GET /interactions/my-medications
Authorization: Bearer <access_token>
```

---

## 🎤 Voice Assistant

### Transcribe Voice
```http
POST /voice/transcribe
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

audio: <file>
language: uz
```

Response:
```json
{
  "transcription": "Aspirinning narxi qancha?",
  "language": "uz",
  "confidence": 0.92
}
```

### Query with Voice
```http
POST /voice/query
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

audio: <file>
language: uz
latitude: 41.2995
longitude: 69.2401
```

Response:
```json
{
  "query": "Aspirinning narxi qancha?",
  "intent": "price_query",
  "response": "Aspirin narxi 12,000 - 20,000 so'm oralig'ida",
  "audio_url": "/path/to/response.mp3",
  "data": {
    "medication": {...},
    "prices": [...]
  }
}
```

---

## 📊 Dashboard

### Get Family Dashboard
```http
GET /dashboard/family
Authorization: Bearer <access_token>
```

Response:
```json
{
  "family_members": 4,
  "total_medications": 12,
  "active_medications": 8,
  "recent_scans": [...],
  "adherence_rate": 0.85,
  "interaction_alerts": 2,
  "upcoming_refills": [...]
}
```

---

## 🎮 Gamification

### Get User Points
```http
GET /gamification/points
Authorization: Bearer <access_token>
```

### Get Badges
```http
GET /gamification/badges
Authorization: Bearer <access_token>
```

### Get Leaderboard
```http
GET /gamification/leaderboard?limit=10
Authorization: Bearer <access_token>
```

---

## 🏥 Health Check

### API Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "database": {
    "status": "connected",
    "latency_ms": 11.2
  },
  "redis": {
    "status": "connected"
  },
  "ai_models": {
    "status": "2/3 models available",
    "details": {
      "pill_recognition": false,
      "drug_interaction": true,
      "price_anomaly": true
    }
  },
  "version": "0.1.0",
  "environment": "production"
}
```

---

## 📝 Error Responses

All endpoints return errors in this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {},
  "path": "/api/v1/endpoint"
}
```

Common HTTP Status Codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

---

## 🔑 Authentication Headers

All protected endpoints require:
```http
Authorization: Bearer <access_token>
```

---

## 📊 Rate Limits

- General API: 60 requests/minute
- Scan endpoints: 10 requests/minute
- Voice endpoints: 20 requests/minute

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1638360000
```

---

## 🌐 CORS

Allowed origins configured in `.env`:
```
CORS_ORIGINS=["https://yourdomain.com"]
```

---

For interactive documentation, visit:
**https://yourdomain.com/api/docs**
