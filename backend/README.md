# 🏥 Sentinel-RX Backend

**AI-powered Medication Safety Platform**

Bu loyiha 11 ta AI model bilan ishlaydigan dori xavfsizligi platformasining backend qismi.

## 🚀 Quick Start

### 1. Virtual environment yaratish

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Dependencies o'rnatish

```bash
pip install -r requirements.txt
```

### 3. Environment sozlash

```bash
copy .env.example .env
# .env faylini tahrirlang va o'z qiymatlaringizni kiriting
```

### 4. Database yaratish

```bash
# PostgreSQL da database yarating
createdb sentinel_rx

# Migratsiyalarni bajarish
alembic upgrade head
```

### 5. Serverni ishga tushirish

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/api/docs

## 📁 Project Structure

```
backend/
├── app/
│   ├── api/                 # API endpoints
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── auth.py         # Authentication
│   │           ├── users.py        # User management
│   │           ├── medications.py  # Medication database
│   │           ├── scans.py        # Pill scanning
│   │           ├── interactions.py # Drug interactions
│   │           ├── pharmacies.py   # Pharmacy finder
│   │           ├── voice.py        # Voice assistant
│   │           ├── dashboard.py    # Family dashboard
│   │           └── gamification.py # Points & rewards
│   ├── core/                # Core functionality
│   │   ├── config.py        # Settings
│   │   ├── security.py      # JWT, password hashing
│   │   └── dependencies.py  # FastAPI dependencies
│   ├── db/                  # Database
│   │   ├── session.py       # SQLAlchemy setup
│   │   └── base.py          # Base model
│   ├── models/              # SQLAlchemy models
│   │   ├── user.py
│   │   ├── medication.py
│   │   ├── interaction.py
│   │   ├── scan.py
│   │   └── pharmacy.py
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   │   └── ai/              # AI models
│   └── utils/               # Utilities
├── alembic/                 # Database migrations
├── tests/                   # Unit tests
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## 🧠 11 AI Models

| # | Model | Description |
|---|-------|-------------|
| 1 | Visual Pill Recognition | PyTorch CNN for pill identification |
| 2 | Drug Interaction AI | DrugBank + OpenFDA data |
| 3 | Personalized Health | Rule-based + ML insights |
| 4 | Price Anomaly Detection | Isolation Forest |
| 5 | Pharmacy Finder | OpenStreetMap + OSRM |
| 6 | Batch Recall Prediction | Random Forest |
| 7 | Voice Assistant | Faster-Whisper + gTTS |
| 8 | Family Dashboard | Real-time monitoring |
| 9 | Gamification | Points, badges, rewards |
| 10 | Medical Tourism | Translation + currency |
| 11 | AI Orchestrator | Unified intelligence layer |

## 🔗 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token

### Scans
- `POST /api/v1/scans/image` - Scan medication image
- `POST /api/v1/scans/qr` - Scan QR/barcode
- `GET /api/v1/scans/history` - Scan history

### Medications
- `GET /api/v1/medications/search` - Search medications
- `GET /api/v1/medications/{id}` - Get medication details
- `POST /api/v1/medications/my/list` - Add to my medications

### Interactions
- `POST /api/v1/interactions/check` - Check drug interactions
- `GET /api/v1/interactions/{id}` - Get interactions for medication

### Pharmacies
- `GET /api/v1/pharmacies/nearby` - Find nearby pharmacies
- `GET /api/v1/pharmacies/{id}/availability` - Check availability

### Voice
- `POST /api/v1/voice/query` - Voice query (audio)
- `POST /api/v1/voice/text-query` - Text query
- `POST /api/v1/voice/tts` - Text to speech

### Dashboard
- `GET /api/v1/dashboard/summary` - Family dashboard
- `POST /api/v1/dashboard/adherence/log` - Log medication taken

### Gamification
- `GET /api/v1/gamification/points` - Get user points
- `GET /api/v1/gamification/leaderboard` - Leaderboard
- `POST /api/v1/gamification/rewards/{id}/redeem` - Redeem reward

## 🧪 Testing

```bash
pytest
pytest --cov=app  # With coverage
```

## 🚀 Deployment

### Render.com (FREE tier)

1. GitHub repo'ga push qiling
2. Render.com da "New Web Service" yarating
3. Environment variables qo'shing
4. Auto-deploy yoqiladi

### Docker

```bash
docker build -t sentinel-rx-backend .
docker run -p 8000:8000 sentinel-rx-backend
```

## 📝 License

MIT License

## 👥 Team

Sentinel-RX Team - AI-500 Hackathon
