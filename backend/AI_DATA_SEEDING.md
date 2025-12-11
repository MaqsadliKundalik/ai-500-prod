# AI Models Data Seeding Guide

## Overview
AI modellar ishlashi uchun qo'shimcha ma'lumotlar kerak:

## 1. Price Anomaly Detection (Narx Anomaliyasi)

### Kerakli ma'lumotlar:
- `pharmacy_inventory` - Dorixonalardagi dori narxlari
- Har bir dori uchun turli dorixonalarda turli narxlar

### Seed qilish:
```bash
# Production (Render)
# Render Shell'da:
cd /app
python app/scripts/seed_pharmacy_inventory.py

# Local
cd backend
python app/scripts/seed_pharmacy_inventory.py
```

### Nima qo'shiladi:
- ~100-500 ta inventory records
- Har bir dorixonada 70-100% dorilar
- Narx variantlari:
  - Normal narxlar (base price ± 10%)
  - Anomaliyalar (5%):
    - Underpriced: 30-50% arzon
    - Overpriced: 50-100% qimmat
- Stock mavjudligi (85% in stock)

### Foydalanish:
```python
# Price anomaly detection API
GET /api/v1/ai-enhancements/pharmacies/compare-prices/{medication_id}
```

## 2. Drug Interaction Detection (Dori O'zaro Ta'siri)

### Status: ✅ Allaqachon seed qilingan

### Mavjud ma'lumotlar:
- Aspirin + Ibuprofen interaction (NSAIDs - both increase bleeding risk)
- Aspirin + Warfarin contraindication (severe bleeding risk)

### Foydalanish:
```python
# Check interactions
POST /api/v1/interactions/check
{
  "medication_ids": ["uuid1", "uuid2"]
}

# GET version
GET /api/v1/interactions/check?medication_ids=uuid1&medication_ids=uuid2
```

## 3. Pill Recognition (Tabletka Tanish)

### Status: ⚠️ Training dataset kerak

### Kerakli ma'lumotlar:
- Pill images (RGB, 224x224px minimum)
- Metadata: shape, color, imprint
- Labels: medication ID

### Dataset yaratish:
1. Real pill images to'plash
2. Yoki NIH Pill Image dataset yuklab olish
3. Training script ishlatish:

```bash
# Local training
cd backend
python app/scripts/train_pill_recognition_v2.py

# Creates: models/pill_recognition.pt
```

### Hozirgi holat:
- Model structure tayyor
- Dataset kerak (rasmlar)
- Seed_data.py'da pill metadata bor (shape, color, imprint)

## 4. Current Database State

### Seeded tables:
- ✅ `medications` - 10+ dorilar
- ✅ `pharmacies` - 5 ta dorixonalar
- ✅ `drug_interactions` - 2+ o'zaro ta'sirlar
- ✅ `food_interactions` - Aspirin + Alcohol
- ⏳ `pharmacy_inventory` - Seed qilish kerak (yuqoridagi script)

### Missing for full AI functionality:
- ⏳ Pharmacy inventory (price data)
- ❌ Pill images dataset
- ❌ User medication history (for personalized recommendations)
- ❌ Historical price data (for trend analysis)

## Quick Start

### 1. Seed basic data (if not done):
```bash
python app/scripts/seed_data.py
```

### 2. Seed pharmacy inventory:
```bash
python app/scripts/seed_pharmacy_inventory.py
```

### 3. Test API:
```bash
# Health check
curl https://ai-500-prod.onrender.com/health

# Search medications
curl https://ai-500-prod.onrender.com/api/v1/medications/search?q=aspirin

# Check pharmacies
curl https://ai-500-prod.onrender.com/api/v1/pharmacies/

# Check interactions
curl "https://ai-500-prod.onrender.com/api/v1/interactions/check?medication_ids=UUID1&medication_ids=UUID2"
```

## Render Production Seeding

### Via Render Shell:
1. Go to Render Dashboard → Your Service
2. Click "Shell" tab
3. Run:
```bash
cd /app
python app/scripts/seed_pharmacy_inventory.py
```

### Via One-off Job:
1. Go to Render Dashboard → Your Service
2. Click "Manual Deploy" → "Create Job"
3. Command: `python app/scripts/seed_pharmacy_inventory.py`
4. Click "Run"

## Notes

- Seed scriptlar idempotent emas - har safar yangi data qo'shadi
- Production'da ehtiyotkorlik bilan ishlatish
- Agar xato bo'lsa, database backup olish tavsiya etiladi
