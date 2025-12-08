"""
Uzpharm-Control.uz Scraper
===========================
Scrape official Uzbekistan medicine registry for real data

Data includes:
- Medicine names (UZ/RU)
- INN (International Nonproprietary Names)
- Registration numbers
- Manufacturers
- Prices (wholesale/retail)
- ATX codes
- Instructions (PDF links)
"""

import httpx
from bs4 import BeautifulSoup
import json
from pathlib import Path
import asyncio
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UzpharmScraper:
    """Scraper for uzpharm-control.uz official medicine registry."""
    
    BASE_URL = "https://www.uzpharm-control.uz"
    REGISTRY_URL = f"{BASE_URL}/uz/registries/api-mpip"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False,  # Disable SSL verification for scraping
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    async def get_medicine_details(self, medicine_id: int) -> Dict:
        """Get detailed info for a specific medicine."""
        url = f"{self.REGISTRY_URL}/view/{medicine_id}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse table data
            data = {}
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) == 2:
                        key = cols[0].get_text(strip=True)
                        value = cols[1].get_text(strip=True)
                        data[key] = value
            
            return {
                "id": medicine_id,
                "url": url,
                "package_name": data.get("Qadoqlash nomi", ""),
                "registration_number": data.get("Ro'yxatdan o'tish guvohnomasi raqami", ""),
                "registration_date": data.get("Ro'yxatdan o'tish guvohnomasining boshlanish sanasi", ""),
                "manufacturer": data.get("Ishlab chiqaruvchining nomi", ""),
                "inn": data.get("МНН", ""),
                "atx_code": data.get("ATX kodi", ""),
                "prescription_required": data.get("Beriladigan dori vositasi", "") == "С рецептом врача",
                "medicine_name_ru": data.get("Dorivor mahsulotning nomi", ""),
                "pharmacotherapeutic_group": data.get("Farmakoterapevtik guruh nomi", ""),
                "prices": {
                    "limit_price_usd": data.get("Narxni cheklash", ""),
                    "wholesale_price_usd": data.get("Ulgurji narx", ""),
                    "retail_price_usd": data.get("Chakana savdo narxi", ""),
                    "limit_price_uzs": data.get("Cheklangan narx", ""),
                    "wholesale_price_uzs": data.get("Ulgurji narx", "").split()[0] if len(data.get("Ulgurji narx", "").split()) > 0 else "",
                    "retail_price_uzs": data.get("Chakana savdo narxi", "").split()[0] if len(data.get("Chakana savdo narxi", "").split()) > 0 else "",
                },
                "currency": data.get("Valyuta nomi", ""),
                "price_date": data.get("O'rnatish sanasi", ""),
                "updated_date": data.get("Yangilangan sana", ""),
                "raw_data": data
            }
            
        except Exception as e:
            logger.error(f"Error fetching medicine {medicine_id}: {e}")
            return {"id": medicine_id, "error": str(e)}
    
    async def scrape_page(self, page: int = 1, per_page: int = 50) -> List[int]:
        """Scrape medicine IDs from a page (note: site uses AJAX, may need different approach)."""
        # The site uses JavaScript/AJAX to load data, so we need to:
        # 1. Try direct access
        # 2. Or use Selenium/Playwright
        # 3. Or find API endpoint
        
        logger.info(f"Attempting to scrape page {page}...")
        
        # For now, let's try sequential IDs
        # Based on the example URL: /view/1, /view/2, etc.
        medicine_ids = list(range(1, 101))  # Try first 100
        
        return medicine_ids
    
    async def scrape_all(self, max_medicines: int = 100, output_file: str = "uzbek_medicines.json"):
        """Scrape all medicines and save to file."""
        logger.info(f"Starting scrape of {max_medicines} medicines...")
        
        medicines = []
        medicine_ids = list(range(1, max_medicines + 1))
        
        for i, med_id in enumerate(medicine_ids, 1):
            logger.info(f"Scraping medicine {i}/{len(medicine_ids)}: ID {med_id}")
            
            medicine = await self.get_medicine_details(med_id)
            
            if "error" not in medicine:
                medicines.append(medicine)
                logger.info(f"  ✅ {medicine.get('package_name', 'Unknown')}")
            else:
                logger.warning(f"  ⚠️  Failed: {medicine['error']}")
            
            # Rate limiting
            await asyncio.sleep(1)
            
            # Save checkpoint every 10 medicines
            if i % 10 == 0:
                self._save_checkpoint(medicines, output_file)
        
        # Final save
        output_path = Path(output_file)
        output_path.write_text(json.dumps(medicines, ensure_ascii=False, indent=2), encoding='utf-8')
        
        logger.info(f"\n✅ Scraping complete!")
        logger.info(f"📁 Saved {len(medicines)} medicines to {output_file}")
        
        await self.client.aclose()
        
        return medicines
    
    def _save_checkpoint(self, medicines: List[Dict], output_file: str):
        """Save checkpoint to file."""
        checkpoint_file = f"{output_file}.checkpoint"
        Path(checkpoint_file).write_text(
            json.dumps(medicines, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.info(f"  💾 Checkpoint saved: {len(medicines)} medicines")


async def main():
    print("=== Scraping Uzpharm-Control.uz Medicine Registry ===")
    print("=" * 70)
    print("Source: https://www.uzpharm-control.uz/uz/registries/api-mpip")
    print("=" * 70)
    
    scraper = UzpharmScraper()
    
    # Scrape first 100 medicines
    medicines = await scraper.scrape_all(
        max_medicines=100,
        output_file="datasets/uzbek_medicines.json"
    )
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total medicines: {len(medicines)}")
    
    # Count by features
    prescription_required = sum(1 for m in medicines if m.get('prescription_required'))
    has_price = sum(1 for m in medicines if m.get('prices', {}).get('retail_price_uzs'))
    
    print(f"  Prescription required: {prescription_required}")
    print(f"  Has retail price: {has_price}")
    
    # Sample medicine
    if medicines:
        print(f"\nSample medicine:")
        sample = medicines[0]
        print(f"  Name: {sample.get('package_name')}")
        print(f"  INN: {sample.get('inn')}")
        print(f"  Manufacturer: {sample.get('manufacturer')}")
        print(f"  ATX: {sample.get('atx_code')}")
        if sample.get('prices', {}).get('retail_price_uzs'):
            print(f"  Price: {sample['prices']['retail_price_uzs']} UZS")


if __name__ == "__main__":
    asyncio.run(main())
