"""
Seed Database with Initial Data
================================
Populate medications, pharmacies, and drug interactions
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import async_session_maker
from app.models.medication import Medication, MedicationPrice
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.models.interaction import DrugInteraction, FoodInteraction, Contraindication
from app.models.user import User
from app.core.security import get_password_hash


async def seed_medications(db: AsyncSession):
    """Seed common medications"""
    print("🔄 Seeding medications...")
    
    medications = [
        {
            "name": "Aspirin",
            "brand_name": "Bayer Aspirin",
            "generic_name": "Acetylsalicylic acid",
            "dosage_form": "tablet",
            "strength": "500mg",
            "manufacturer": "Bayer",
            "description": "Pain reliever and fever reducer",
            "prescription_required": False,
            "active_ingredients": [{"name": "Acetylsalicylic acid", "amount": "500mg"}],
            "indications": ["Pain relief", "Fever reduction", "Anti-inflammatory"],
            "side_effects": ["Stomach upset", "Heartburn", "Nausea"],
            "contraindications": ["Aspirin allergy", "Active bleeding"],
            "pregnancy_category": "C",
            "pill_shape": "round",
            "pill_color": "white",
            "pill_imprint": "BAYER"
        },
        {
            "name": "Paracetamol",
            "brand_name": "Panadol",
            "generic_name": "Acetaminophen",
            "dosage_form": "tablet",
            "strength": "500mg",
            "manufacturer": "GSK",
            "description": "Pain reliever and fever reducer",
            "prescription_required": False,
            "active_ingredients": [{"name": "Acetaminophen", "amount": "500mg"}],
            "indications": ["Pain relief", "Fever reduction"],
            "side_effects": ["Rare: skin rash", "Liver damage with overdose"],
            "pregnancy_category": "B"
        },
        {
            "name": "Ibuprofen",
            "brand_name": "Nurofen",
            "generic_name": "Ibuprofen",
            "dosage_form": "tablet",
            "strength": "400mg",
            "manufacturer": "Reckitt Benckiser",
            "description": "Nonsteroidal anti-inflammatory drug (NSAID)",
            "prescription_required": False,
            "active_ingredients": [{"name": "Ibuprofen", "amount": "400mg"}],
            "indications": ["Pain relief", "Fever reduction", "Anti-inflammatory"],
            "side_effects": ["Stomach upset", "Dizziness", "Heartburn"],
            "contraindications": ["NSAID allergy", "Active GI bleeding"],
            "pregnancy_category": "C"
        },
        {
            "name": "Amoxicillin",
            "brand_name": "Amoxil",
            "generic_name": "Amoxicillin",
            "dosage_form": "capsule",
            "strength": "500mg",
            "manufacturer": "GSK",
            "description": "Antibiotic for bacterial infections",
            "prescription_required": True,
            "active_ingredients": [{"name": "Amoxicillin", "amount": "500mg"}],
            "indications": ["Bacterial infections", "Respiratory infections"],
            "side_effects": ["Nausea", "Diarrhea", "Skin rash"],
            "pregnancy_category": "B"
        },
        {
            "name": "Metformin",
            "brand_name": "Glucophage",
            "generic_name": "Metformin HCl",
            "dosage_form": "tablet",
            "strength": "850mg",
            "manufacturer": "Bristol-Myers Squibb",
            "description": "Type 2 diabetes medication",
            "prescription_required": True,
            "active_ingredients": [{"name": "Metformin Hydrochloride", "amount": "850mg"}],
            "indications": ["Type 2 diabetes", "Blood sugar control"],
            "side_effects": ["Nausea", "Diarrhea", "Stomach upset"],
            "contraindications": ["Renal impairment", "Lactic acidosis risk"],
            "pregnancy_category": "B"
        }
    ]
    
    for med_data in medications:
        med = Medication(**med_data)
        db.add(med)
    
    await db.commit()
    print(f"✅ Added {len(medications)} medications")


async def seed_pharmacies(db: AsyncSession):
    """Seed pharmacies in Tashkent"""
    print("🔄 Seeding pharmacies...")
    
    pharmacies = [
        {
            "name": "MEDPLUS Pharmacy",
            "address": "Amir Temur Avenue 107, Tashkent",
            "phone": "+998712345678",
            "latitude": 41.311151,
            "longitude": 69.279737,
            "working_hours": {"mon-fri": "09:00-21:00", "sat-sun": "10:00-20:00"},
            "is_24_hours": False,
            "accepts_insurance": True,
            "has_delivery": True,
            "rating": 4.5,
            "website": "https://medplus.uz"
        },
        {
            "name": "SOGLOM APTEKA",
            "address": "Chilanzar 1, Tashkent",
            "phone": "+998712345679",
            "latitude": 41.275518,
            "longitude": 69.203451,
            "working_hours": {"everyday": "00:00-24:00"},
            "is_24_hours": True,
            "accepts_insurance": True,
            "has_delivery": True,
            "rating": 4.7,
            "website": "https://soglikapt.uz"
        },
        {
            "name": "FARMATSIYA PLUS",
            "address": "Yunusabad 10, Tashkent",
            "phone": "+998712345680",
            "latitude": 41.363029,
            "longitude": 69.289042,
            "working_hours": {"mon-sun": "08:00-22:00"},
            "is_24_hours": False,
            "accepts_insurance": True,
            "has_delivery": False,
            "rating": 4.3,
        },
        {
            "name": "DORIXONA 24/7",
            "address": "Mirzo Ulugbek Street 50, Tashkent",
            "phone": "+998712345681",
            "latitude": 41.338353,
            "longitude": 69.334421,
            "working_hours": {"everyday": "00:00-24:00"},
            "is_24_hours": True,
            "accepts_insurance": False,
            "has_delivery": True,
            "rating": 4.6,
        },
        {
            "name": "GREEN PHARMACY",
            "address": "Sergeli District, Tashkent",
            "phone": "+998712345682",
            "latitude": 41.219722,
            "longitude": 69.222222,
            "working_hours": {"mon-sat": "09:00-19:00"},
            "is_24_hours": False,
            "accepts_insurance": True,
            "has_delivery": False,
            "rating": 4.4,
        }
    ]
    
    for pharm_data in pharmacies:
        pharm = Pharmacy(**pharm_data)
        db.add(pharm)
    
    await db.commit()
    print(f"✅ Added {len(pharmacies)} pharmacies")


async def seed_interactions(db: AsyncSession):
    """Seed drug interactions"""
    print("🔄 Seeding drug interactions...")
    
    # Get medications first
    from sqlalchemy import select
    result = await db.execute(select(Medication))
    medications = {m.name: m for m in result.scalars().all()}
    
    if "Aspirin" not in medications or "Ibuprofen" not in medications:
        print("⚠️  Skipping interactions - medications not found")
        return
    
    interactions = [
        {
            "medication_id": medications["Aspirin"].id,
            "interacting_medication_id": medications["Ibuprofen"].id,
            "severity": "moderate",
            "description": "Taking aspirin and ibuprofen together may reduce the effectiveness of aspirin",
            "mechanism": "Competition for COX enzyme binding sites",
            "management": "Take ibuprofen at least 2 hours after aspirin"
        }
    ]
    
    for int_data in interactions:
        interaction = DrugInteraction(**int_data)
        db.add(interaction)
    
    await db.commit()
    print(f"✅ Added {len(interactions)} drug interactions")


async def seed_food_interactions(db: AsyncSession):
    """Seed food-drug interactions"""
    print("🔄 Seeding food interactions...")
    
    # Simplified - skip for now
    print("⚠️  Skipping food interactions")
    return


async def seed_test_users(db: AsyncSession):
    """Create test users"""
    print("🔄 Seeding test users...")
    
    users = [
        {
            "email": "test@example.com",
            "full_name": "Test User",
            "hashed_password": get_password_hash("password123"),
            "phone": "+998901234567",
            "language": "uz"
        },
        {
            "email": "demo@sentinel-rx.uz",
            "full_name": "Demo User",
            "hashed_password": get_password_hash("demo123"),
            "phone": "+998901234568",
            "language": "ru"
        }
    ]
    
    for user_data in users:
        user = User(**user_data)
        db.add(user)
    
    await db.commit()
    print(f"✅ Added {len(users)} test users")


async def main():
    """Run all seeders"""
    print("\n🌱 Starting database seeding...\n")
    
    async with async_session_maker() as db:
        try:
            await seed_medications(db)
            await seed_pharmacies(db)
            await seed_interactions(db)
            await seed_food_interactions(db)
            await seed_test_users(db)
            
            print("\n✅ Database seeding completed successfully!\n")
            print("📋 Test credentials:")
            print("   Email: test@example.com")
            print("   Password: password123")
            print("\n   Email: demo@sentinel-rx.uz")
            print("   Password: demo123\n")
            
        except Exception as e:
            print(f"\n❌ Error during seeding: {e}")
            await db.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(main())
