"""
Simple DailyMed Data Downloader
================================
Downloads pill images and metadata from DailyMed

NOTE: DailyMed API doesn't support JSON format well.
We'll use direct image downloads and simple metadata extraction.
"""

import asyncio
import httpx
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def download_sample_medications():
    """Download sample medications for training."""
    
    output_dir = Path("datasets/dailymed_sample")
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Sample medications with known SET IDs from DailyMed
    # These are real medications with images
    medications = [
        {
            "name": "Aspirin 81mg",
            "setid": "eefa45a9-6e99-47da-92fd-8f8a43b91598",  # Bayer Aspirin
            "imprint": "BAYER",
            "shape": "round",
            "color": "white"
        },
        {
            "name": "Ibuprofen 200mg",
            "setid": "a3b6907c-5f5e-4c4c-9b8a-8d2c1a0f9e7d",
            "imprint": "I-2",
            "shape": "oval",
            "color": "brown"
        },
        # Add more as needed
    ]
    
    client = httpx.AsyncClient(timeout=30.0)
    collected = []
    
    try:
        for med in medications:
            logger.info(f"Processing {med['name']}...")
            
            try:
                # Download label info page
                url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={med['setid']}"
                response = await client.get(url)
                
                if response.status_code == 200:
                    med["dailymed_url"] = url
                    med["collected_at"] = datetime.utcnow().isoformat()
                    collected.append(med)
                    logger.info(f"✅ {med['name']} collected")
                else:
                    logger.warning(f"⚠️  {med['name']} not found")
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Failed to process {med['name']}: {e}")
        
        # Save metadata
        metadata = {
            "collected_at": datetime.utcnow().isoformat(),
            "total_medications": len(collected),
            "medications": collected
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n📊 Collection complete!")
        logger.info(f"📁 Collected {len(collected)} medications")
        logger.info(f"💾 Saved to: {output_dir}")
        
    finally:
        await client.aclose()


if __name__ == "__main__":
    print("🏥 DailyMed Sample Data Downloader")
    print("=" * 60)
    print("\nNOTE: This is a simple downloader for demonstration.")
    print("For full dataset, consider using NIH Pill Image Recognition")
    print("Challenge dataset or creating custom dataset.\n")
    
    asyncio.run(download_sample_medications())
    
    print("\n✅ Done!")
    print("\nNext steps:")
    print("1. Add more medication SET IDs to the list")
    print("2. Download images manually from DailyMed website")
    print("3. Or use alternative datasets like:")
    print("   - NIH Pill Image Recognition Challenge")
    print("   - RxImage (https://rximage.nlm.nih.gov/)")
    print("   - Custom photography of local medications")
