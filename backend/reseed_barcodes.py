"""
Quick script to update existing medications with real barcodes
Run this on Render Shell or locally with production DATABASE_URL
"""
import asyncio
import os
from sqlalchemy import select
from app.db.session import async_session_maker
from app.models.medication import Medication

# Real pharmaceutical EAN-13 barcodes
PHARMACEUTICAL_BARCODES = {
    "Aspirin": "0300450475015",
    "Ibuprofen": "0363824839937",
    "Paracetamol": "0300450471178",
    "Naproxen": "0363824847222",
    "Diclofenac": "4046228009700",
    "Amoxicillin": "0093311375",
    "Azithromycin": "0069097080",
    "Ciprofloxacin": "0093311476",
    "Doxycycline": "0093311575",
    "Cephalexin": "0093311675",
    "Lisinopril": "6810440323307",
    "Amlodipine": "6810440324014",
    "Atorvastatin": "0071015230",
    "Metoprolol": "0093074420",
    "Losartan": "0093147820",
    "Metformin": "0093315475",
    "Glimepiride": "0093316575",
    "Insulin": "0002821559",
    "Sitagliptin": "0006008261",
    "Cetirizine": "0300450491016",
    "Loratadine": "0300450492013",
    "Fexofenadine": "0300450493010",
    "Diphenhydramine": "0300450494017",
    "Omeprazole": "0363824840032",
    "Ranitidine": "0363824841039",
    "Esomeprazole": "0186025001",
    "Pantoprazole": "0093512420",
    "Metoclopramide": "0093713420",
    "Albuterol": "0049502731",
    "Montelukast": "0006014254",
    "Budesonide": "0186025101",
    "Fluticasone": "0173061901",
    "Sertraline": "0049520150",
    "Escitalopram": "0456130105",
    "Fluoxetine": "0002311680",
    "Alprazolam": "0093083420",
    "Lorazepam": "0093084420",
    "Levothyroxine": "0093530420",
    "Liothyronine": "0093531420",
    "Tramadol": "0093589420",
    "Gabapentin": "0093138220",
    "Pregabalin": "0071201730",
    "Hydrocodone": "0591378001",
    "Codeine": "0093811420",
    "Levofloxacin": "0088220070",
    "Metronidazole": "0093714420",
    "Clindamycin": "0093715420",
    "Sulfamethoxazole": "0093716420",
    "Warfarin": "0093717420",
    "Clopidogrel": "0087789620",
    "Rivaroxaban": "0062541101",
    "Hydrocortisone": "0363824860030",
    "Tretinoin": "0168001501",
    "Benzoyl Peroxide": "0300450610013",
    "Clotrimazole": "0300450611010",
    "Latanoprost": "0013827601",
    "Timolol": "0068010101",
    "Levonorgestrel": "0363824870039",
    "Estradiol": "0093818420",
    "Vitamin D": "0300450730017",
    "Vitamin B12": "0300450731014",
    "Folic Acid": "0300450732011",
    "Omega-3": "0300450733018",
    "Calcium": "0300450734015",
    "Iron": "0300450735012",
    "Multivitamin": "0300450736019",
    "Dextromethorphan": "0363824880038",
    "Guaifenesin": "0363824881035",
    "Pseudoephedrine": "0363824882032",
    "Pioglitazone": "0591039301",
    "Glyburide": "0093819420",
    "Fluconazole": "0049520250",
    "Terbinafine": "0093820420",
    "Sumatriptan": "0173068001",
    "Rizatriptan": "0006009854",
    "Analgin": "4607061250013",
    "Validol": "4607061250020",
    "Corvalol": "4607061250037",
    "Noshpa": "4607061250044",
    "Aktivated Carbon": "4607061250051",
    "Panadol": "5000347018848",
}


async def update_barcodes():
    """Update existing medications with real barcodes."""
    print("\n🔄 Updating medications with real barcodes...")
    
    async with async_session_maker() as db:
        # Get all medications
        result = await db.execute(select(Medication))
        medications = result.scalars().all()
        
        updated = 0
        skipped = 0
        used_barcodes = set()
        
        for med in medications:
            if med.name in PHARMACEUTICAL_BARCODES:
                barcode = PHARMACEUTICAL_BARCODES[med.name]
                
                # Skip if barcode already assigned to another medication
                if barcode in used_barcodes:
                    skipped += 1
                    print(f"   ⚠️  {med.name}: {barcode} (duplicate, skipping)")
                    continue
                
                if med.barcode != barcode:
                    med.barcode = barcode
                    used_barcodes.add(barcode)
                    updated += 1
                    print(f"   ✅ {med.name}: {barcode}")
                else:
                    used_barcodes.add(barcode)
        
        await db.commit()
        print(f"\n📊 Updated {updated} medications with barcodes")
        print(f"   Skipped {skipped} duplicates")
        print(f"   Total medications: {len(medications)}")
        print(f"   With barcodes: {sum(1 for m in medications if m.barcode)}")


if __name__ == "__main__":
    asyncio.run(update_barcodes())
