# DailyMed Integration

## Overview

**DailyMed** - National Library of Medicine (NLM) ning rasmiy dori ma'lumotlar bazasi. FDA-tasdiqlangan barcha dorilar haqida to'liq ma'lumot beradi.

## Nima Uchun DailyMed?

### 1️⃣ **Rasmiy FDA Ma'lumotlari**
- ✅ FDA tomonidan tasdiqlangan dori yorliqlari (SPL documents)
- ✅ NDC kodlari (National Drug Code)
- ✅ Dori rasmlari va qadoq fotolari
- ✅ Faol moddalar ro'yxati
- ✅ Imprint kodlar (doriga yozilgan belgilar)

### 2️⃣ **Xavfsizlik Ma'lumotlari**
- ⚠️ FDA xavfsizlik ogohlantishlari
- ⚠️ Qaytarib olingan dorilar (recalls)
- ⚠️ Yon ta'sirlar
- ⚠️ Ziddiyatlar (contraindications)

### 3️⃣ **Bepul va Ochiq**
- 💰 To'liq bepul API
- 🌐 Rasmiy davlat ma'lumotlar bazasi
- 📊 Kundalik yangilanadi
- 🔓 API key talab qilmaydi

## Bizning Loyihamizda Qanday Foydalanish Mumkin?

### ✅ 1. Pill Recognition Database ni To'ldirish

```python
from app.services.external.dailymed_service import DailyMedService

service = DailyMedService()

# NDC kod bo'yicha qidirish
medication = await service.search_by_ndc("0002-3227-30")

# Database ga saqlash
if medication:
    db_med = Medication(
        name=medication["title"],
        ndc_code=medication["ndc_codes"][0],
        imprint_code=medication.get("imprint_code"),
        shape=medication.get("shape"),
        color_primary=medication.get("color"),
        image_url=medication["images"][0] if medication["images"] else None,
        manufacturer=medication["manufacturer"],
        # ... other fields
    )
    db.add(db_med)
```

### ✅ 2. Real-Time Verification

Multi-modal pill recognition paytida DailyMed'dan tekshirish:

```python
async def verify_with_dailymed(ndc_code: str, imprint: str) -> bool:
    """Verify pill against official FDA database."""
    service = DailyMedService()
    
    # Get official data
    official_data = await service.search_by_ndc(ndc_code)
    
    if not official_data:
        return False
    
    # Get pill characteristics
    pill_info = await service.get_pill_imprint_info(official_data["setid"])
    
    # Verify imprint matches
    if pill_info and pill_info["imprint"]:
        return pill_info["imprint"].upper() == imprint.upper()
    
    return False
```

### ✅ 3. Safety Warnings

```python
# Get official FDA warnings
medication = await service.get_medication_by_setid(setid)

warnings = []
if medication.get("boxed_warning"):
    warnings.append({
        "type": "BLACK_BOX",
        "severity": "CRITICAL",
        "text": medication["boxed_warning"]
    })

if medication.get("warnings"):
    warnings.append({
        "type": "WARNING",
        "severity": "HIGH",
        "text": medication["warnings"]
    })
```

### ✅ 4. Drug Images

```python
# Get official packaging images
image_url = await service.get_drug_image(setid)

# Display to user for visual comparison
```

## API Endpoints

### Base URL
```
https://dailymed.nlm.nih.gov/dailymed/services/v2/
```

### Key Endpoints

#### 1. Search by NDC
```http
GET /spls.json?ndc={ndc_code}
```

Example:
```bash
curl "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?ndc=0002-3227-30"
```

#### 2. Get Medication Details
```http
GET /spls/{setid}.json
```

Example:
```bash
curl "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/07a59edb-318c-388a-e063-6294a90a986f.json"
```

#### 3. Get NDC Codes
```http
GET /spls/{setid}/ndcs.json
```

#### 4. Get Packaging Info
```http
GET /spls/{setid}/packaging.json
```

#### 5. Get Images
```http
GET /spls/{setid}/media.json
```

#### 6. Search by Name
```http
GET /spls.json?drug_name={name}
```

Example:
```bash
curl "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=aspirin"
```

## Response Example

```json
{
  "metadata": {
    "page_size": 1,
    "total_elements": 1
  },
  "data": [
    {
      "setid": "07a59edb-318c-388a-e063-6294a90a986f",
      "title": "ASPIRIN",
      "published_date": "2024-11-15",
      "spl_version": "5",
      "brand_name": "Bayer Aspirin",
      "generic_drug": [
        {
          "name": "ASPIRIN",
          "unii": "R16CO5Y76E"
        }
      ],
      "dosage_form": ["TABLET"],
      "route": ["ORAL"],
      "strength": "81 mg",
      "manufacturer_name": "Bayer HealthCare LLC",
      "marketing_category": "OTC monograph final",
      "application_number": null,
      "dea_schedule_code": null
    }
  ]
}
```

## Integration Steps

### Step 1: Install Dependencies
```bash
pip install httpx
```

### Step 2: Create Service
```python
from app.services.external.dailymed_service import DailyMedService

service = DailyMedService()
```

### Step 3: Use in Medication Service

```python
# In app/services/medication_service.py

from app.services.external.dailymed_service import enrich_medication_with_dailymed

async def create_medication(medication_data: dict):
    # Enrich with DailyMed data
    enriched = await enrich_medication_with_dailymed(
        medication_data,
        ndc_code=medication_data.get("ndc_code")
    )
    
    # Save to database
    medication = Medication(**enriched)
    db.add(medication)
    await db.commit()
    
    return medication
```

### Step 4: Add to Scan Endpoint

```python
# In app/api/v1/endpoints/scans.py

@router.post("/", response_model=ScanResponse)
async def create_scan(
    file: UploadFile,
    db: AsyncSession = Depends(get_db)
):
    # ... existing recognition code ...
    
    # Verify with DailyMed
    if recognized_medication.ndc_code:
        dailymed_service = DailyMedService()
        official_data = await dailymed_service.search_by_ndc(
            recognized_medication.ndc_code
        )
        
        if official_data:
            # Add official verification badge
            response["official_verification"] = {
                "verified": True,
                "source": "FDA DailyMed",
                "dailymed_url": official_data["dailymed_url"],
                "label_pdf": official_data["label_pdf"]
            }
    
    return response
```

## Database Migration

Add DailyMed fields to medications table:

```python
# alembic/versions/add_dailymed_fields.py

def upgrade():
    op.add_column('medications', sa.Column('dailymed_setid', sa.String(100), nullable=True))
    op.add_column('medications', sa.Column('dailymed_url', sa.String(500), nullable=True))
    op.add_column('medications', sa.Column('official_label_pdf', sa.String(500), nullable=True))
    op.add_column('medications', sa.Column('last_verified_dailymed', sa.DateTime(), nullable=True))
    
    op.create_index('idx_medications_dailymed_setid', 'medications', ['dailymed_setid'])
```

## Caching Strategy

DailyMed responses are cached for 24 hours to reduce API calls:

```python
# In-memory cache
self._cache = {}
self._cache_ttl = timedelta(hours=24)

# Check cache before API call
if cache_key in self._cache:
    cached_time, data = self._cache[cache_key]
    if datetime.utcnow() - cached_time < self._cache_ttl:
        return data
```

For production, use Redis:

```python
# config.py
DAILYMED_CACHE_TTL = 86400  # 24 hours

# In service
async def search_by_ndc(self, ndc_code: str):
    cache_key = f"dailymed:ndc:{ndc_code}"
    
    # Try Redis first
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Fetch from API
    data = await self._fetch_from_api(ndc_code)
    
    # Cache in Redis
    await redis.setex(cache_key, DAILYMED_CACHE_TTL, json.dumps(data))
    
    return data
```

## Benefits for Our Project

### 🎯 1. Enhanced Pill Recognition
- Official imprint codes from FDA
- High-quality pill images for training
- Accurate shape/color data

### 🛡️ 2. Safety Verification
- Real-time verification against FDA database
- Official warnings and contraindications
- Recall information

### 📊 3. Database Enrichment
- Auto-populate medication database
- Keep data updated from official source
- Reduce manual data entry

### 🏥 4. Trust & Credibility
- "Verified by FDA DailyMed" badge
- Link to official FDA documentation
- Professional healthcare compliance

## Rate Limits

DailyMed does not have explicit rate limits, but:
- ✅ Be respectful - cache aggressively
- ✅ Don't hammer API - use reasonable delays
- ✅ Consider bulk downloads for initial database population

## Alternatives

If DailyMed is insufficient, consider:

1. **OpenFDA API** - More comprehensive FDA data
   - https://open.fda.gov/apis/
   
2. **RxNorm API** - Clinical drug terminology
   - https://rxnav.nlm.nih.gov/RxNormAPIs.html
   
3. **NIH RxImage** - Pill images database
   - https://rximage.nlm.nih.gov/

## Example: Complete Integration

```python
from app.services.external.dailymed_service import DailyMedService
from app.services.ai.models.pill_recognition_multi_modal import MultiModalPillRecognizer

async def recognize_and_verify_pill(image_data: bytes):
    """Complete pill recognition with DailyMed verification."""
    
    # Step 1: Multi-modal recognition
    recognizer = MultiModalPillRecognizer(...)
    result = await recognizer.recognize_pill(image, features)
    
    # Step 2: DailyMed verification
    dailymed = DailyMedService()
    
    if result.medication_id:
        # Get medication from our DB
        medication = await db.get(Medication, result.medication_id)
        
        if medication.ndc_code:
            # Verify with DailyMed
            official = await dailymed.search_by_ndc(medication.ndc_code)
            
            if official:
                # Compare imprint codes
                pill_info = await dailymed.get_pill_imprint_info(official["setid"])
                
                if pill_info and pill_info["imprint"]:
                    if pill_info["imprint"].upper() == result.imprint_text.upper():
                        result.warnings.insert(0, "✅ FDA VERIFIED: Imprint matches official database")
                    else:
                        result.critical_warning = "🚨 IMPRINT MISMATCH WITH FDA DATABASE!"
    
    return result
```

## Next Steps

1. ✅ Integrate DailyMed service (DONE)
2. ⏳ Add to scan endpoint
3. ⏳ Create database migration for DailyMed fields
4. ⏳ Add caching layer (Redis)
5. ⏳ Build admin tool to populate database from DailyMed
6. ⏳ Add "FDA Verified" badge to UI

## Resources

- **Official Docs**: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
- **API Base URL**: https://dailymed.nlm.nih.gov/dailymed/services/v2/
- **Support**: https://support.nlm.nih.gov/support/create-case/
- **Terms**: Public domain (US Government)
