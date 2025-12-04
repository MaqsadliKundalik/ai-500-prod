"""
Training Script for Price Anomaly Detector
==========================================
Train Isolation Forest for anomaly detection
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.services.ai.models.price_anomaly import PriceAnomalyDetector, build_price_dataset


def generate_synthetic_price_data(n_samples=2000):
    """
    Generate synthetic medication pricing data.
    
    Real data sources:
    - GoodRx API
    - Pharmacy price databases
    - Historical transaction data
    """
    print("📦 Generating synthetic price data...")
    
    # Base prices for different medication types
    base_prices = {
        'generic': np.random.normal(5000, 1000, n_samples // 4),
        'brand': np.random.normal(15000, 3000, n_samples // 4),
        'specialty': np.random.normal(50000, 10000, n_samples // 4),
        'otc': np.random.normal(3000, 500, n_samples // 4)
    }
    
    prices = []
    
    for medication_type, base_price_array in base_prices.items():
        for base_price in base_price_array:
            # Normal price variation factors
            pharmacy_factor = np.random.uniform(0.9, 1.1)  # ±10%
            location_factor = np.random.uniform(0.95, 1.05)  # ±5%
            demand_factor = np.random.uniform(0.98, 1.02)  # ±2%
            
            price = base_price * pharmacy_factor * location_factor * demand_factor
            
            # Add some anomalies (overpriced)
            if np.random.random() < 0.1:  # 10% anomalies
                price *= np.random.uniform(1.5, 3.0)  # 50-200% markup
            
            prices.append({
                'price': max(0, price),
                'avg_market_price': base_price,
                'pharmacy_rating': np.random.uniform(3.0, 5.0),
                'is_chain': np.random.choice([True, False]),
                'distance_from_center_km': np.random.uniform(0, 20),
                'day_of_week': np.random.randint(0, 7),
                'medication_demand_score': np.random.uniform(0.3, 0.9),
                'stock_level': np.random.randint(10, 100)
            })
    
    return prices


def evaluate_detector(detector, test_data):
    """Evaluate anomaly detector performance."""
    print("\n📊 Evaluating detector...")
    
    true_anomalies = 0
    detected_anomalies = 0
    false_positives = 0
    false_negatives = 0
    
    for record in test_data:
        # Ground truth: price > 1.5x average is anomaly
        is_true_anomaly = record['price'] > (record['avg_market_price'] * 1.5)
        
        # Detector prediction
        result = detector.detect_anomaly(record)
        is_detected = result['is_anomaly']
        
        if is_true_anomaly:
            true_anomalies += 1
            if is_detected:
                detected_anomalies += 1
            else:
                false_negatives += 1
        else:
            if is_detected:
                false_positives += 1
    
    # Calculate metrics
    precision = detected_anomalies / (detected_anomalies + false_positives) if (detected_anomalies + false_positives) > 0 else 0
    recall = detected_anomalies / true_anomalies if true_anomalies > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Detection Metrics:")
    print(f"   Total samples: {len(test_data)}")
    print(f"   True anomalies: {true_anomalies}")
    print(f"   Detected: {detected_anomalies}")
    print(f"   False positives: {false_positives}")
    print(f"   False negatives: {false_negatives}")
    print(f"\n   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")


def main():
    """Main training function."""
    print("🚀 Starting Price Anomaly Detector Training")
    print("=" * 60)
    
    # Generate dataset
    price_data = generate_synthetic_price_data(n_samples=2000)
    print(f"✅ Generated {len(price_data)} price records")
    
    # Calculate anomaly rate in dataset
    anomaly_count = sum(1 for p in price_data if p['price'] > p['avg_market_price'] * 1.5)
    print(f"✅ Anomalies in dataset: {anomaly_count} ({anomaly_count/len(price_data)*100:.1f}%)")
    
    # Split data
    train_size = int(0.8 * len(price_data))
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]
    
    print(f"\n📊 Train set: {len(train_data)} samples")
    print(f"📊 Test set: {len(test_data)} samples")
    
    # Build feature matrix
    detector = PriceAnomalyDetector(contamination=0.1)
    X_train = build_price_dataset(train_data)
    
    print(f"\n🏗️  Feature matrix shape: {X_train.shape}")
    print(f"   Features: {', '.join(detector.feature_names)}")
    
    # Train model
    print("\n🏋️  Training Isolation Forest...")
    detector.train(X_train)
    
    # Evaluate
    evaluate_detector(detector, test_data)
    
    # Example predictions
    print("\n🔍 Sample Predictions:")
    for i in range(min(5, len(test_data))):
        record = test_data[i]
        result = detector.detect_anomaly(record)
        
        print(f"\n   Sample {i+1}:")
        print(f"   Price: {record['price']:.0f} UZS")
        print(f"   Market Avg: {record['avg_market_price']:.0f} UZS")
        print(f"   Anomaly: {result['is_anomaly']}")
        print(f"   Severity: {result['severity']}")
        print(f"   {result['explanation'][:60]}...")
    
    # Save model
    model_path = Path("models/price_anomaly.pkl")
    model_path.parent.mkdir(exist_ok=True)
    detector.save_model(str(model_path))
    
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    main()
