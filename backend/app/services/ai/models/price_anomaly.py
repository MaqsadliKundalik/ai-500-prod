"""
Price Anomaly Detection
========================
Isolation Forest model for detecting overpriced medications
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import pickle
from pathlib import Path


class PriceAnomalyDetector:
    """
    Anomaly detection model for identifying overpriced medications.
    
    Uses Isolation Forest algorithm to detect price outliers based on:
    - Historical prices
    - Geographic location
    - Pharmacy type
    - Medication characteristics
    """
    
    def __init__(self, model_path: Optional[str] = None, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,  # Expected proportion of outliers
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'price',
            'avg_market_price',
            'pharmacy_rating',
            'is_chain',
            'distance_from_center',
            'day_of_week',
            'medication_demand',
            'stock_level'
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"✅ Loaded price anomaly detector from {model_path}")
        else:
            print("⚠️  Using untrained model")
    
    def extract_features(self, price_data: Dict) -> np.ndarray:
        """
        Extract features for anomaly detection.
        
        Args:
            price_data: Dictionary with pricing information
            
        Returns:
            Feature array
        """
        features = [
            price_data.get('price', 0.0),
            price_data.get('avg_market_price', 0.0),
            price_data.get('pharmacy_rating', 0.0),
            1.0 if price_data.get('is_chain', False) else 0.0,
            price_data.get('distance_from_center_km', 0.0),
            price_data.get('day_of_week', 0),
            price_data.get('medication_demand_score', 0.5),
            price_data.get('stock_level', 50)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def detect_anomaly(self, price_data: Dict) -> Dict:
        """
        Detect if medication price is anomalous.
        
        Returns:
            Dictionary with anomaly status and score
        """
        features = self.extract_features(price_data)
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Predict
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(features_scaled)[0]
            anomaly_score = self.model.score_samples(features_scaled)[0]
            is_anomaly = prediction == -1
        else:
            # Fallback for untrained model
            price = price_data.get('price', 0)
            avg_price = price_data.get('avg_market_price', price)
            price_ratio = price / avg_price if avg_price > 0 else 1.0
            
            is_anomaly = price_ratio > 1.5  # More than 50% above average
            anomaly_score = -(price_ratio - 1.0)  # Negative score for anomalies
        
        # Calculate percentage above market average
        price = price_data.get('price', 0)
        avg_price = price_data.get('avg_market_price', price)
        percent_above_avg = ((price - avg_price) / avg_price * 100) if avg_price > 0 else 0
        
        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "severity": self._calculate_severity(percent_above_avg),
            "percent_above_average": round(percent_above_avg, 2),
            "explanation": self._generate_explanation(price_data, is_anomaly, percent_above_avg),
            "recommendation": self._generate_recommendation(is_anomaly, percent_above_avg)
        }
    
    def _calculate_severity(self, percent_above_avg: float) -> str:
        """Calculate anomaly severity."""
        if percent_above_avg < 10:
            return "normal"
        elif percent_above_avg < 25:
            return "slightly_elevated"
        elif percent_above_avg < 50:
            return "elevated"
        elif percent_above_avg < 100:
            return "high"
        else:
            return "extreme"
    
    def _generate_explanation(self, price_data: Dict, is_anomaly: bool, percent_above: float) -> str:
        """Generate human-readable explanation."""
        if not is_anomaly:
            return "Price is within normal market range."
        
        factors = []
        
        if percent_above > 50:
            factors.append(f"Price is {abs(percent_above):.0f}% above market average")
        
        if not price_data.get('is_chain', False):
            factors.append("Independent pharmacy (typically higher prices)")
        
        if price_data.get('distance_from_center_km', 0) > 10:
            factors.append("Remote location may increase costs")
        
        if price_data.get('stock_level', 50) < 20:
            factors.append("Low stock level may inflate price")
        
        return ". ".join(factors) if factors else "Price appears overpriced."
    
    def _generate_recommendation(self, is_anomaly: bool, percent_above: float) -> str:
        """Generate actionable recommendation."""
        if not is_anomaly:
            return "Fair price. Safe to purchase."
        
        if percent_above < 25:
            return "Slightly overpriced. Consider checking nearby pharmacies."
        elif percent_above < 50:
            return "Overpriced. Recommend shopping at alternative pharmacies."
        else:
            return "Significantly overpriced. Strong recommendation to find alternative source."
    
    def train(self, X_train: np.ndarray):
        """
        Train anomaly detection model.
        
        Args:
            X_train: Training feature matrix (no labels needed for Isolation Forest)
        """
        # Fit scaler
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Train model
        self.model.fit(X_scaled)
        
        print(f"✅ Anomaly detector trained on {len(X_train)} samples")
    
    def save_model(self, path: str):
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', self.feature_names)


def build_price_dataset(price_history: List[Dict]) -> np.ndarray:
    """
    Build training dataset from historical price data.
    
    Args:
        price_history: List of historical price records
        
    Returns:
        Feature matrix for training
    """
    detector = PriceAnomalyDetector()
    X_list = []
    
    for record in price_history:
        features = detector.extract_features(record)
        X_list.append(features[0])
    
    return np.array(X_list)


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_prices = [
        {
            'price': 15000,
            'avg_market_price': 12000,
            'pharmacy_rating': 4.5,
            'is_chain': True,
            'distance_from_center_km': 3,
            'day_of_week': 2,
            'medication_demand_score': 0.7,
            'stock_level': 80
        },
        {
            'price': 25000,  # Anomaly
            'avg_market_price': 12000,
            'pharmacy_rating': 3.5,
            'is_chain': False,
            'distance_from_center_km': 15,
            'day_of_week': 5,
            'medication_demand_score': 0.9,
            'stock_level': 10
        }
    ]
    
    detector = PriceAnomalyDetector()
    
    for price_data in sample_prices:
        result = detector.detect_anomaly(price_data)
        print(f"\nPrice: {price_data['price']} UZS")
        print(f"Is Anomaly: {result['is_anomaly']}")
        print(f"Severity: {result['severity']}")
        print(f"Explanation: {result['explanation']}")
