# 🚀 Production Data Seeding - Quick Start

## Render Shell'da Ishlatish

### 1. Render Dashboard'ga Kiring
- https://dashboard.render.com
- Loyihangizni toping: **ai-500-prod**

### 2. Shell Oching
- **Shell** tabiga o'ting
- Kutib turing (30-60 soniya yuklanadi)

### 3. Scriptni Ishga Tushuring

```bash
# Check current data
python -c "
import asyncio
from app.db.session import async_session_maker
from app.models.medication import Medication
from sqlalchemy import select, func

async def check():
    async with async_session_maker() as db:
        result = await db.execute(select(func.count(Medication.id)))
        print(f'Medications: {result.scalar()}')

asyncio.run(check())
"

# Run seeding script
python app/scripts/seed_production_data.py
```

### 4. Kutib Turing
- Script 30-60 soniya ishlaydi
- 5000+ ma'lumot yoziladi

### 5. Natijani Ko'ring
```
✅ PRODUCTION DATA SEEDING COMPLETED SUCCESSFULLY!

📊 Summary:
   • Medications: 103
   • Pharmacies: 55
   • Inventory records: ~3575
   • Drug interactions: ~487
   • Test users: 4
```

## Test Qilish

### Medications API
```bash
curl https://ai-500-prod.onrender.com/api/v1/medications/search?query=aspirin
```

### Pharmacies API
```bash
curl https://ai-500-prod.onrender.com/api/v1/pharmacies
```

### Price Comparison (AI narx taqqoslash)
```bash
# First get a medication ID
MEDICATION_ID=$(curl -s "https://ai-500-prod.onrender.com/api/v1/medications/search?query=aspirin" | jq -r '.items[0].id')

# Then compare prices
curl -X POST https://ai-500-prod.onrender.com/api/v1/pharmacies/compare-prices \
  -H "Content-Type: application/json" \
  -d "{
    \"medication_id\": \"$MEDICATION_ID\",
    \"latitude\": 41.2995,
    \"longitude\": 69.2401,
    \"radius_km\": 10
  }"
```

### Drug Interactions
```bash
# Get medication IDs
ASPIRIN_ID=$(curl -s "https://ai-500-prod.onrender.com/api/v1/medications/search?query=aspirin" | jq -r '.items[0].id')
IBUPROFEN_ID=$(curl -s "https://ai-500-prod.onrender.com/api/v1/medications/search?query=ibuprofen" | jq -r '.items[0].id')

# Check interactions
curl "https://ai-500-prod.onrender.com/api/v1/interactions/check?medication_ids=$ASPIRIN_ID&medication_ids=$IBUPROFEN_ID"
```

## Muammolar

### "Database connection failed"
```bash
# Check environment variables
env | grep DATABASE_URL
```

### "Module not found"
```bash
# Install dependencies
pip install -r requirements.txt
```

### Script juda sekin
- Bu normal (5000+ record)
- 30-60 soniya kutish kerak

## Ma'lumotlar Haqida

### 🏥 Dorixonalar
- 55 ta dorixona
- 7 ta shahar: Toshkent, Samarqand, Buxoro, Andijon, Namangan, Farg'ona, Nukus
- Real koordinatalar (Google Maps)
- Rating: 4.1★ - 4.6★

### 💊 Dorilar
- 103 ta real dori
- To'liq ma'lumot (dozalar, yon ta'sirlar, ogohlantirishlar)
- Retsept talab qiladigan va OTC dorilar

### 💰 Narxlar
- 3500+ narx ma'lumoti
- Har dorixonada 50-80 xil dori
- 5% anomaliya (AI training uchun)
- Real narxlar (UZS)

### ⚠️ Interaksiyalar
- 487 ta tasdiqlangan interaksiya
- Darajalar: mild, moderate, severe
- Warfarin, Aspirin, Ibuprofen, SSRI va boshqalar

## Test Userlar

```
Email: demo@pharmacheck.uz
Password: password123
Role: USER

Email: admin@pharmacheck.uz
Password: password123
Role: ADMIN
```

## Keyingi Qadamlar

1. ✅ Data seeded → AI models ready
2. 🧪 Test qiling → postman/curl bilan
3. 📱 Frontend ulang → React/Flutter
4. 🚀 Production deploy → Mobile app release

---

**Qo'shimcha ma'lumot:** `backend/PRODUCTION_SEEDING.md`
