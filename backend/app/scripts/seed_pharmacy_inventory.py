"""
Seed Pharmacy Inventory and Price Data
=======================================
For price anomaly detection AI model
"""

import asyncio
import sys
from pathlib import Path
import random
from datetime import datetime
from uuid import uuid4

sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.db.session import async_session_maker
from app.models.pharmacy import Pharmacy, PharmacyInventory
from app.models.medication import Medication


PRICE_MULTIPLIERS = {
    # Different pharmacy chains have different pricing strategies
    "MEDPLUS": 1.1,      # Slightly higher
    "SOGLOM": 0.95,      # Competitive
    "FARMATSIYA": 1.0,   # Average
    "DORIXONA": 1.05,    # Slightly higher
    "GREEN": 0.92        # Budget-friendly
}

BASE_PRICES = {
    # Base prices in UZS
    "Aspirin": 5000,
    "Paracetamol": 3000,
    "Ibuprofen": 7000,
    "Amoxicillin": 25000,
    "Lisinopril": 45000,
    "Metformin": 15000,
    "Atorvastatin": 55000,
    "Omeprazole": 12000,
}


async def seed_inventory(db: AsyncSession):
    """Seed pharmacy inventory with realistic price variations."""
    print("📦 Seeding pharmacy inventory...")
    
    # Get all pharmacies and medications
    pharm_result = await db.execute(select(Pharmacy))
    pharmacies = list(pharm_result.scalars().all())
    
    med_result = await db.execute(select(Medication))
    medications = list(med_result.scalars().all())
    
    if not pharmacies:
        print("❌ No pharmacies found. Run seed_data.py first!")
        return
    
    if not medications:
        print("❌ No medications found. Run seed_data.py first!")
        return
    
    print(f"Found {len(pharmacies)} pharmacies and {len(medications)} medications")
    
    inventory_count = 0
    
    for pharmacy in pharmacies:
        # Get pharmacy name prefix for pricing strategy
        name_prefix = pharmacy.name.split()[0].upper()
        base_multiplier = 1.0
        
        for key, multiplier in PRICE_MULTIPLIERS.items():
            if key in name_prefix:
                base_multiplier = multiplier
                break
        
        # Each pharmacy stocks 70-100% of medications
        num_to_stock = random.randint(int(len(medications) * 0.7), len(medications))
        stocked_meds = random.sample(medications, num_to_stock)
        
        for medication in stocked_meds:
            # Get base price or generate random if not in dict
            base_price = BASE_PRICES.get(medication.name, random.randint(5000, 80000))
            
            # Apply pharmacy multiplier
            pharmacy_price = base_price * base_multiplier
            
            # Add random variation (±10%)
            price_variation = random.uniform(0.9, 1.1)
            final_price = int(pharmacy_price * price_variation)
            
            # Add some price anomalies (5% chance)
            if random.random() < 0.05:
                anomaly_type = random.choice(['underpriced', 'overpriced'])
                if anomaly_type == 'underpriced':
                    final_price = int(final_price * random.uniform(0.5, 0.7))  # 30-50% lower
                else:
                    final_price = int(final_price * random.uniform(1.5, 2.0))  # 50-100% higher
            
            # Stock availability
            in_stock = random.random() > 0.15  # 85% in stock
            stock_quantity = random.randint(10, 200) if in_stock else 0
            
            inventory = PharmacyInventory(
                pharmacy_id=pharmacy.id,
                medication_id=medication.id,
                price=final_price,
                currency="UZS",
                in_stock=in_stock,
                stock_quantity=stock_quantity,
                last_updated=datetime.utcnow()
            )
            
            db.add(inventory)
            inventory_count += 1
    
    await db.commit()
    print(f"✅ Added {inventory_count} pharmacy inventory records")
    
    # Show sample prices
    print("\n📊 Sample prices (UZS):")
    result = await db.execute(
        text("""
            SELECT m.name, p.name, pi.price, pi.in_stock
            FROM pharmacy_inventory pi
            JOIN medications m ON pi.medication_id = m.id
            JOIN pharmacies p ON pi.pharmacy_id = p.id
            ORDER BY m.name, pi.price
            LIMIT 10
        """)
    )
    
    for row in result:
        stock_status = "✅" if row[3] else "❌"
        print(f"  {row[0]:15} @ {row[1]:20} = {row[2]:6,} UZS {stock_status}")


async def main():
    """Run inventory seeding."""
    print("\n🌱 Starting pharmacy inventory seeding...\n")
    
    async with async_session_maker() as db:
        try:
            await seed_inventory(db)
            print("\n✅ Pharmacy inventory seeding completed!\n")
        except Exception as e:
            print(f"\n❌ Error during seeding: {e}")
            await db.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(main())
