"""
Pharmacy Price Scraper for Uzbekistan
======================================
Scrapes prices from multiple online pharmacies to build price anomaly detection dataset
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
from datetime import datetime
import re


class PharmacyPriceScraper:
    """Scraper for multiple Uzbekistan online pharmacies."""
    
    def __init__(self):
        self.prices_data = []
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.aclose()
    
    async def search_remedy_uz(self, medicine_name: str) -> Optional[Dict]:
        """Search medicine on remedy.uz"""
        try:
            # Try searching by name
            search_url = f"https://remedy.uz/search?q={medicine_name}"
            response = await self.session.get(search_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Parse price from search results
                # Note: This is a template - actual selectors need to be updated based on site structure
                price_elements = soup.select('.product-price, .price, [class*="price"]')
                
                for price_elem in price_elements:
                    price_text = price_elem.get_text(strip=True)
                    price_match = re.search(r'([\d\s]+)', price_text.replace(',', ''))
                    if price_match:
                        price = int(price_match.group(1).replace(' ', ''))
                        return {
                            'pharmacy': 'remedy.uz',
                            'price': price,
                            'currency': 'UZS',
                            'url': search_url,
                            'date': datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"Error searching remedy.uz for {medicine_name}: {e}")
        return None
    
    async def search_medicstore_uz(self, medicine_name: str) -> Optional[Dict]:
        """Search medicine on medicstore.uz"""
        try:
            # Placeholder for medicstore.uz scraping
            # Similar structure to remedy.uz
            pass
        except Exception as e:
            print(f"Error searching medicstore.uz for {medicine_name}: {e}")
        return None
    
    async def search_apteka_uz(self, medicine_name: str) -> Optional[Dict]:
        """Search medicine on apteka.uz"""
        try:
            # Placeholder for apteka.uz scraping
            pass
        except Exception as e:
            print(f"Error searching apteka.uz for {medicine_name}: {e}")
        return None
    
    async def get_all_prices(self, medicine_name: str, inn: str) -> List[Dict]:
        """Get prices from all pharmacies for a given medicine."""
        prices = []
        
        # Try searching by both package name and INN
        search_terms = [medicine_name, inn]
        
        for term in search_terms:
            if term:
                # Search all pharmacies in parallel
                tasks = [
                    self.search_remedy_uz(term),
                    self.search_medicstore_uz(term),
                    self.search_apteka_uz(term),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result:
                        prices.append(result)
                
                if prices:  # If we found prices, no need to try other search terms
                    break
        
        return prices
    
    async def scrape_all_medicines(self, medicines_file: str = "datasets/uzbek_medicines.json"):
        """Scrape prices for all medicines in our database."""
        
        # Load medicines
        with open(medicines_file, 'r', encoding='utf-8') as f:
            medicines = json.load(f)
        
        print(f"Scraping prices for {len(medicines)} medicines...")
        
        for i, medicine in enumerate(medicines, 1):
            print(f"[{i}/{len(medicines)}] Processing: {medicine['package_name'][:50]}...")
            
            # Get official price as baseline
            official_price = None
            if medicine.get('prices', {}).get('retail_price_uzs'):
                price_str = medicine['prices']['retail_price_uzs'].replace(' UZS', '').replace(',', '')
                try:
                    official_price = float(price_str)
                except:
                    pass
            
            # Search online pharmacies
            pharmacy_prices = await self.get_all_prices(
                medicine['package_name'],
                medicine['inn']
            )
            
            # Combine all price data
            price_record = {
                'medicine_id': medicine['id'],
                'medicine_name': medicine['package_name'],
                'inn': medicine['inn'],
                'atx_code': medicine['atx_code'],
                'official_retail_price': official_price,
                'pharmacy_prices': pharmacy_prices,
                'scrape_date': datetime.now().isoformat()
            }
            
            self.prices_data.append(price_record)
            
            # Rate limiting
            await asyncio.sleep(2)
            
            # Save checkpoint every 10 medicines
            if i % 10 == 0:
                self._save_checkpoint()
        
        # Final save
        self._save_checkpoint()
        print(f"\nCompleted! Scraped prices for {len(self.prices_data)} medicines")
    
    def _save_checkpoint(self):
        """Save current progress."""
        output_file = "datasets/pharmacy_prices_raw.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.prices_data, f, ensure_ascii=False, indent=2)
        print(f"Saved checkpoint: {len(self.prices_data)} records")


async def main():
    """Main scraping function."""
    async with PharmacyPriceScraper() as scraper:
        await scraper.scrape_all_medicines()


if __name__ == "__main__":
    asyncio.run(main())
