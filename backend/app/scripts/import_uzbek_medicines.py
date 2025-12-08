"""
Import Uzbek Medicines to Database
====================================
Import scraped medicines from uzpharm-control.uz into database
"""

import asyncio
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import sys
import logging

sys.path.insert(0, '.')

from app.models.medication import Medication
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def import_uzbek_medicines(json_file: str = "datasets/uzbek_medicines.json"):
    """Import medicines from JSON file to database."""
    
    # Read JSON
    data_path = Path(json_file)
    if not data_path.exists():
        logger.error(f"File not found: {json_file}")
        return
    
    medicines_data = json.loads(data_path.read_text(encoding='utf-8'))
    logger.info(f"Loaded {len(medicines_data)} medicines from {json_file}")
    
    # Setup database
    engine = create_async_engine(
        settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
        echo=False
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    imported = 0
    skipped = 0
    errors = 0
    
    async with async_session() as session:
        for i, med_data in enumerate(medicines_data, 1):
            try:
                # Extract data
                package_name = med_data.get('package_name', '')
                inn = med_data.get('inn', '')
                
                if not package_name or not inn:
                    logger.warning(f"Skipping {i}: Missing name or INN")
                    skipped += 1
                    continue
                
                # Parse prices
                prices = med_data.get('prices', {})
                retail_price_str = prices.get('retail_price_uzs', '0')
                # Remove "UZS" and whitespace, convert to float
                try:
                    retail_price = float(retail_price_str.replace('UZS', '').replace(',', '').strip())
                except:
                    retail_price = 0.0
                
                # Create medication with proper field mapping
                medication = Medication(
                    name=package_name[:255],
                    brand_name=med_data.get('medicine_name_ru', '')[:255],
                    generic_name=inn[:255],
                    manufacturer=med_data.get('manufacturer', '')[:255],
                    description=f"ATX: {med_data.get('atx_code', '')}. {med_data.get('pharmacotherapeutic_group', '')[:500]}",
                    barcode=med_data.get('registration_number', '')[:50],
                    # Price in proper field
                    price=retail_price,
                    prescription_required=med_data.get('prescription_required', False),
                    # Visual characteristics (placeholder for now)
                    pill_shape="unknown",
                    pill_color="unknown",
                    pill_imprint=med_data.get('atx_code', '')[:100],
                    # Additional metadata
                    indications=[med_data.get('pharmacotherapeutic_group', '')],
                    active_ingredients=[{"name": inn, "atx_code": med_data.get('atx_code', '')}],
                )
                
                session.add(medication)
                imported += 1
                
                if i % 10 == 0:
                    await session.commit()
                    logger.info(f"Progress: {i}/{len(medicines_data)} - Imported: {imported}")
                
            except Exception as e:
                logger.error(f"Error importing medicine {i} ({med_data.get('package_name', 'Unknown')}): {e}")
                errors += 1
                continue
        
        # Final commit
        await session.commit()
    
    await engine.dispose()
    
    logger.info("\n" + "=" * 70)
    logger.info("Import Complete!")
    logger.info("=" * 70)
    logger.info(f"✅ Imported: {imported}")
    logger.info(f"⚠️  Skipped: {skipped}")
    logger.info(f"❌ Errors: {errors}")
    logger.info(f"📊 Total: {len(medicines_data)}")


async def main():
    print("=" * 70)
    print("Importing Uzbek Medicines to Database")
    print("=" * 70)
    print("Source: datasets/uzbek_medicines.json")
    print("From: https://www.uzpharm-control.uz")
    print("=" * 70)
    
    await import_uzbek_medicines()
    
    print("\n✅ Database updated with official Uzbekistan medicines!")


if __name__ == "__main__":
    asyncio.run(main())
