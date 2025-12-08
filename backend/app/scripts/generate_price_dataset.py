"""
Generate Synthetic Price Dataset for Anomaly Detection
========================================================
Creates realistic price variations across pharmacies and regions
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np


UZBEKISTAN_REGIONS = [
    "Tashkent", "Samarkand", "Bukhara", "Andijan", "Fergana",
    "Namangan", "Kashkadarya", "Surkhandarya", "Jizzakh", "Khorezm",
    "Navoi", "Sirdaryo", "Karakalpakstan"
]

PHARMACY_CHAINS = [
    "Remedy", "Medicstore", "Apteka Plus", "Dorixona 24", "MedLife",
    "HealthCare", "Farmland", "Vita Pharm", "MedCity", "Dorillar"
]

PRICE_FACTORS = {
    "region_multiplier": {
        "Tashkent": 1.0,      # Base price
        "Samarkand": 0.95,
        "Bukhara": 0.93,
        "Andijan": 0.90,
        "Fergana": 0.92,
        "Namangan": 0.91,
        "Kashkadarya": 0.88,
        "Surkhandarya": 0.85,
        "Jizzakh": 0.87,
        "Khorezm": 0.86,
        "Navoi": 0.89,
        "Sirdaryo": 0.88,
        "Karakalpakstan": 0.83
    },
    "pharmacy_markup": {
        "Remedy": 1.15,       # Premium pharmacy
        "Medicstore": 1.12,
        "Apteka Plus": 1.08,
        "Dorixona 24": 1.10,
        "MedLife": 1.05,
        "HealthCare": 1.07,
        "Farmland": 1.03,
        "Vita Pharm": 1.06,
        "MedCity": 1.09,
        "Dorillar": 1.04
    }
}


def generate_price_variations(base_price: float, region: str, pharmacy: str, 
                              add_anomalies: bool = False) -> float:
    """Generate realistic price with regional and pharmacy variations."""
    
    if base_price <= 0:
        return 0
    
    # Apply regional multiplier
    regional_price = base_price * PRICE_FACTORS["region_multiplier"][region]
    
    # Apply pharmacy markup
    pharmacy_price = regional_price * PRICE_FACTORS["pharmacy_markup"][pharmacy]
    
    # Add random variation (±5%)
    random_variation = random.uniform(0.95, 1.05)
    final_price = pharmacy_price * random_variation
    
    # Generate anomalies (5% chance)
    if add_anomalies and random.random() < 0.05:
        anomaly_type = random.choice(['underpriced', 'overpriced', 'extreme'])
        
        if anomaly_type == 'underpriced':
            final_price *= random.uniform(0.5, 0.7)  # 30-50% lower
        elif anomaly_type == 'overpriced':
            final_price *= random.uniform(1.4, 1.8)  # 40-80% higher
        else:  # extreme
            final_price *= random.uniform(2.0, 3.0)  # 2-3x price (fraud alert)
    
    return round(final_price, 2)


def generate_time_series_prices(base_price: float, days: int = 90) -> List[Dict]:
    """Generate price history over time with trends."""
    prices = []
    current_date = datetime.now() - timedelta(days=days)
    current_price = base_price
    
    for day in range(days):
        # Add seasonal trend (±0.2% per day)
        trend = random.uniform(-0.002, 0.002)
        current_price *= (1 + trend)
        
        prices.append({
            'date': (current_date + timedelta(days=day)).strftime('%Y-%m-%d'),
            'price': round(current_price, 2)
        })
    
    return prices


def generate_full_dataset(medicines_file: str = "datasets/uzbek_medicines.json"):
    """Generate comprehensive price dataset."""
    
    # Load medicines
    with open(medicines_file, 'r', encoding='utf-8') as f:
        medicines = json.load(f)
    
    print(f"Generating price dataset for {len(medicines)} medicines...")
    
    dataset = []
    anomaly_count = 0
    
    for medicine in medicines:
        # Get base price
        base_price = 0
        if medicine.get('prices', {}).get('retail_price_uzs'):
            price_str = medicine['prices']['retail_price_uzs'].replace(' UZS', '').replace(',', '')
            try:
                base_price = float(price_str)
            except:
                continue
        
        if base_price <= 0:
            continue
        
        # Generate prices for each region and pharmacy combination
        for region in UZBEKISTAN_REGIONS:
            for pharmacy in PHARMACY_CHAINS:
                # Generate current price
                price = generate_price_variations(base_price, region, pharmacy, add_anomalies=True)
                
                # Check if it's an anomaly (>30% deviation from base)
                is_anomaly = abs(price - base_price) / base_price > 0.3
                if is_anomaly:
                    anomaly_count += 1
                
                record = {
                    'medicine_id': medicine['id'],
                    'medicine_name': medicine['package_name'],
                    'inn': medicine['inn'],
                    'atx_code': medicine['atx_code'],
                    'manufacturer': medicine['manufacturer'],
                    'base_price': base_price,
                    'region': region,
                    'pharmacy': pharmacy,
                    'price': price,
                    'currency': 'UZS',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'is_anomaly': is_anomaly,
                    'deviation_percent': round((price - base_price) / base_price * 100, 2)
                }
                
                dataset.append(record)
    
    # Save dataset
    output_file = "datasets/pharmacy_prices_synthetic.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset generated successfully!")
    print(f"Total records: {len(dataset)}")
    print(f"Anomalies: {anomaly_count} ({anomaly_count/len(dataset)*100:.2f}%)")
    print(f"Regions: {len(UZBEKISTAN_REGIONS)}")
    print(f"Pharmacies: {len(PHARMACY_CHAINS)}")
    print(f"Saved to: {output_file}")
    
    # Generate summary statistics
    generate_summary(dataset)


def generate_summary(dataset: List[Dict]):
    """Generate dataset summary statistics."""
    
    prices = [d['price'] for d in dataset]
    base_prices = [d['base_price'] for d in dataset]
    deviations = [d['deviation_percent'] for d in dataset]
    
    summary = {
        'total_records': len(dataset),
        'unique_medicines': len(set(d['medicine_id'] for d in dataset)),
        'unique_regions': len(set(d['region'] for d in dataset)),
        'unique_pharmacies': len(set(d['pharmacy'] for d in dataset)),
        'price_statistics': {
            'min': round(min(prices), 2),
            'max': round(max(prices), 2),
            'mean': round(np.mean(prices), 2),
            'median': round(np.median(prices), 2),
            'std': round(np.std(prices), 2)
        },
        'deviation_statistics': {
            'min': round(min(deviations), 2),
            'max': round(max(deviations), 2),
            'mean': round(np.mean(deviations), 2),
            'median': round(np.median(deviations), 2)
        },
        'anomaly_count': sum(1 for d in dataset if d['is_anomaly']),
        'anomaly_percentage': round(sum(1 for d in dataset if d['is_anomaly']) / len(dataset) * 100, 2)
    }
    
    # Regional price comparison
    regional_avg = {}
    for region in UZBEKISTAN_REGIONS:
        region_prices = [d['price'] for d in dataset if d['region'] == region]
        if region_prices:
            regional_avg[region] = round(np.mean(region_prices), 2)
    
    summary['regional_averages'] = regional_avg
    
    # Pharmacy price comparison
    pharmacy_avg = {}
    for pharmacy in PHARMACY_CHAINS:
        pharmacy_prices = [d['price'] for d in dataset if d['pharmacy'] == pharmacy]
        if pharmacy_prices:
            pharmacy_avg[pharmacy] = round(np.mean(pharmacy_prices), 2)
    
    summary['pharmacy_averages'] = pharmacy_avg
    
    # Save summary
    with open('datasets/price_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n--- Dataset Summary ---")
    print(f"Total Records: {summary['total_records']}")
    print(f"Price Range: {summary['price_statistics']['min']} - {summary['price_statistics']['max']} UZS")
    print(f"Average Price: {summary['price_statistics']['mean']} UZS")
    print(f"Anomalies: {summary['anomaly_count']} ({summary['anomaly_percentage']}%)")


if __name__ == "__main__":
    generate_full_dataset()
