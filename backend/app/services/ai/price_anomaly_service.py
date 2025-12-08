"""
Price Anomaly Detection Service
================================
Detects anomalous medicine prices using trained models
"""

import numpy as np
import joblib
import torch
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PriceAnomalyService:
    """Service for detecting price anomalies."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.isolation_forest = None
        self.autoencoder = None
        self.price_scaler = None
        self.encoders = None
        self.autoencoder_config = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models."""
        try:
            # Load Isolation Forest
            if_path = self.models_dir / "isolation_forest_price_anomaly.pkl"
            if if_path.exists():
                self.isolation_forest = joblib.load(if_path)
                logger.info("Loaded Isolation Forest model")
            
            # Load Autoencoder
            ae_path = self.models_dir / "autoencoder_price_anomaly.pt"
            if ae_path.exists():
                from app.scripts.train_price_anomaly_detection import PriceAutoencoder
                self.autoencoder = PriceAutoencoder(input_dim=11)
                self.autoencoder.load_state_dict(torch.load(ae_path, map_location='cpu'))
                self.autoencoder.eval()
                logger.info("Loaded Autoencoder model")
            
            # Load scaler and encoders
            scaler_path = self.models_dir / "price_scaler.pkl"
            if scaler_path.exists():
                self.price_scaler = joblib.load(scaler_path)
            
            encoders_path = self.models_dir / "price_encoders.pkl"
            if encoders_path.exists():
                self.encoders = joblib.load(encoders_path)
            
            config_path = self.models_dir / "autoencoder_config.pkl"
            if config_path.exists():
                self.autoencoder_config = joblib.load(config_path)
            
        except Exception as e:
            logger.error(f"Error loading price anomaly models: {e}")
    
    def _prepare_features(self, 
                         region: str,
                         pharmacy: str,
                         inn: str,
                         atx_code: str,
                         base_price: float,
                         current_price: float) -> np.ndarray:
        """Prepare features for model input."""
        
        if not self.encoders:
            raise ValueError("Encoders not loaded")
        
        # Encode categorical variables
        try:
            region_encoded = self.encoders['region'].transform([region])[0]
        except:
            region_encoded = 0  # Unknown region
        
        try:
            pharmacy_encoded = self.encoders['pharmacy'].transform([pharmacy])[0]
        except:
            pharmacy_encoded = 0  # Unknown pharmacy
        
        try:
            inn_encoded = self.encoders['inn'].transform([inn])[0]
        except:
            inn_encoded = 0  # Unknown INN
        
        try:
            atx_encoded = self.encoders['atx_code'].transform([atx_code])[0]
        except:
            atx_encoded = 0  # Unknown ATX
        
        # Calculate derived features
        price_per_base = current_price / base_price if base_price > 0 else 0
        log_price = np.log1p(current_price)
        log_base_price = np.log1p(base_price)
        price_deviation = current_price - base_price
        price_deviation_log = np.log1p(np.abs(price_deviation))
        
        # Create feature vector
        features = np.array([[
            region_encoded,
            pharmacy_encoded,
            inn_encoded,
            atx_encoded,
            base_price,
            current_price,
            log_price,
            log_base_price,
            price_per_base,
            price_deviation,
            price_deviation_log
        ]])
        
        return features
    
    def detect_anomaly(self,
                      medicine_name: str,
                      region: str,
                      pharmacy: str,
                      current_price: float,
                      inn: str = "",
                      atx_code: str = "",
                      base_price: Optional[float] = None) -> Dict:
        """
        Detect if a price is anomalous.
        
        Args:
            medicine_name: Name of the medicine
            region: Region (e.g., "Tashkent")
            pharmacy: Pharmacy name
            current_price: Current price to check
            inn: International Nonproprietary Name
            atx_code: Anatomical Therapeutic Chemical code
            base_price: Official/baseline price (optional)
        
        Returns:
            Dictionary with anomaly detection results
        """
        
        # If no base price provided, use current price as baseline
        if base_price is None:
            base_price = current_price
        
        # Prepare features
        features = self._prepare_features(
            region=region,
            pharmacy=pharmacy,
            inn=inn,
            atx_code=atx_code,
            base_price=base_price,
            current_price=current_price
        )
        
        results = {
            'medicine_name': medicine_name,
            'region': region,
            'pharmacy': pharmacy,
            'current_price': current_price,
            'base_price': base_price,
            'deviation_percent': ((current_price - base_price) / base_price * 100) if base_price > 0 else 0,
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'confidence': 0.0,
            'methods': {}
        }
        
        # Isolation Forest detection
        if self.isolation_forest and self.price_scaler:
            features_scaled = self.price_scaler.transform(features)
            prediction = self.isolation_forest.predict(features_scaled)[0]
            anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
            
            results['methods']['isolation_forest'] = {
                'is_anomaly': prediction == -1,
                'anomaly_score': float(anomaly_score)
            }
        
        # Autoencoder detection
        if self.autoencoder and self.autoencoder_config:
            scaler = self.autoencoder_config['scaler']
            threshold = self.autoencoder_config['threshold']
            
            features_scaled = scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled)
            
            with torch.no_grad():
                reconstructed = self.autoencoder(features_tensor)
                reconstruction_error = torch.mean((reconstructed - features_tensor) ** 2, dim=1).item()
            
            is_anomaly_ae = reconstruction_error > threshold
            
            results['methods']['autoencoder'] = {
                'is_anomaly': is_anomaly_ae,
                'reconstruction_error': float(reconstruction_error),
                'threshold': float(threshold)
            }
        
        # Combine predictions (ensemble)
        anomaly_votes = sum([
            results['methods'].get('isolation_forest', {}).get('is_anomaly', False),
            results['methods'].get('autoencoder', {}).get('is_anomaly', False)
        ])
        
        results['is_anomaly'] = anomaly_votes >= 1  # At least one model agrees
        results['confidence'] = anomaly_votes / 2.0  # Confidence based on agreement
        
        # Severity level
        deviation = abs(results['deviation_percent'])
        if deviation > 50:
            results['severity'] = 'critical'
        elif deviation > 30:
            results['severity'] = 'high'
        elif deviation > 15:
            results['severity'] = 'medium'
        else:
            results['severity'] = 'low'
        
        return results
    
    def compare_regional_prices(self,
                               medicine_name: str,
                               inn: str,
                               atx_code: str,
                               base_price: float,
                               regions: List[str] = None) -> Dict:
        """
        Compare prices across different regions.
        
        Returns expected price range for each region.
        """
        
        if regions is None:
            regions = [
                "Tashkent", "Samarkand", "Bukhara", "Andijan", "Fergana",
                "Namangan", "Kashkadarya", "Surkhandarya"
            ]
        
        regional_analysis = {}
        
        for region in regions:
            # Estimate expected price for this region
            # Using simple multipliers based on regional economics
            regional_multipliers = {
                "Tashkent": 1.0,
                "Samarkand": 0.95,
                "Bukhara": 0.93,
                "Andijan": 0.90,
                "Fergana": 0.92,
                "Namangan": 0.91,
                "Kashkadarya": 0.88,
                "Surkhandarya": 0.85
            }
            
            multiplier = regional_multipliers.get(region, 0.90)
            expected_price = base_price * multiplier
            
            regional_analysis[region] = {
                'expected_price': round(expected_price, 2),
                'price_range': {
                    'min': round(expected_price * 0.90, 2),
                    'max': round(expected_price * 1.15, 2)
                }
            }
        
        return {
            'medicine_name': medicine_name,
            'base_price': base_price,
            'regional_prices': regional_analysis
        }


# Global instance
_price_anomaly_service = None


def get_price_anomaly_service() -> PriceAnomalyService:
    """Get or create PriceAnomalyService instance."""
    global _price_anomaly_service
    if _price_anomaly_service is None:
        _price_anomaly_service = PriceAnomalyService()
    return _price_anomaly_service
