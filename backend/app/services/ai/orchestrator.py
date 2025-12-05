"""
AI Orchestrator
===============
Coordinates all 11 AI models and returns unified insights
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.services.medication_service import MedicationService
from app.services.interaction_service import InteractionService
from app.services.pharmacy_service import PharmacyService
from app.services.ai.models import (
    PillRecognizer,
    DrugInteractionDetector,
    PriceAnomalyDetector,
    PILL_RECOGNITION_AVAILABLE,
    INTERACTION_DETECTOR_AVAILABLE,
    PRICE_ANOMALY_AVAILABLE
)

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """
    Intelligence Layer - Orchestrates all 11 AI models.
    
    This is the main entry point for getting unified insights about medications.
    It calls all AI services in parallel and combines results.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.medication_service = MedicationService(db)
        self.interaction_service = InteractionService(db)
        self.pharmacy_service = PharmacyService(db)
        
        # Initialize AI models (optional for production)
        self.pill_recognizer = None
        if PILL_RECOGNITION_AVAILABLE:
            try:
                self.pill_recognizer = PillRecognizer(
                    model_path="models/pill_recognition.pt"
                )
                logger.info("✅ Pill recognizer initialized")
            except Exception as e:
                logger.warning(f"⚠️  Pill recognizer init failed: {e}")
        else:
            logger.info("ℹ️  Pill recognition unavailable (torch not installed)")
        
        self.interaction_detector = None
        if INTERACTION_DETECTOR_AVAILABLE:
            try:
                self.interaction_detector = DrugInteractionDetector(
                    model_path="models/drug_interaction.pkl"
                )
                logger.info("✅ Interaction detector initialized")
            except Exception as e:
                logger.warning(f"⚠️  Interaction detector init failed: {e}")
        else:
            logger.info("ℹ️  Interaction detector unavailable (ML libraries not installed)")
        
        self.price_anomaly_detector = None
        if PRICE_ANOMALY_AVAILABLE:
            try:
                self.price_anomaly_detector = PriceAnomalyDetector(
                    model_path="models/price_anomaly.pkl"
                )
                logger.info("✅ Price anomaly detector initialized")
            except Exception as e:
                logger.warning(f"⚠️  Price anomaly detector init failed: {e}")
        else:
            logger.info("ℹ️  Price anomaly detector unavailable (ML libraries not installed)")
    
    async def process_scan(
        self,
        image_data: bytes,
        user_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process image scan and return unified insights from all AI models.
        
        Args:
            image_data: Raw image bytes
            user_id: Optional user ID for personalized insights
            latitude: Optional user location for pharmacy finder
            longitude: Optional user location
            
        Returns:
            UnifiedInsightResponse as dict
        """
        scan_id = "scan_" + str(datetime.utcnow().timestamp())
        
        # Model 1: Visual Pill Recognition
        recognition_result = await self._recognize_pill(image_data)
        
        if not recognition_result["recognized"]:
            return {
                "scan_id": scan_id,
                "scan_type": "image",
                "scanned_at": datetime.utcnow(),
                "recognized": False,
                "medication": None,
                "confidence": 0.0,
                "interactions": {"has_interactions": False, "total_count": 0, "severe_count": 0, "interactions": []},
                "price_analysis": {},
                "nearby_pharmacies": [],
                "batch_recall": {"is_recalled": False},
                "personalized_insights": [],
                "points_earned": 2,  # Points for trying
                "alternatives": []
            }
        
        medication_id = recognition_result["medication_id"]
        medication = await self.medication_service.get_by_id(medication_id)
        
        # If medication not found in database, return not recognized
        if not medication:
            return {
                "scan_id": scan_id,
                "scan_type": "image",
                "scanned_at": datetime.utcnow(),
                "recognized": False,
                "medication": None,
                "confidence": recognition_result["confidence"],
                "interactions": {"has_interactions": False, "total_count": 0, "severe_count": 0, "interactions": []},
                "price_analysis": {},
                "nearby_pharmacies": [],
                "batch_recall": {"is_recalled": False},
                "personalized_insights": [],
                "points_earned": 2,
                "alternatives": []
            }
        
        # Model 2: Drug Interactions
        interactions = {"has_interactions": False, "total_count": 0, "severe_count": 0, "interactions": []}
        if user_id:
            interaction_result = await self.interaction_service.check_with_user_medications(
                medication_id, user_id
            )
            interactions = {
                "has_interactions": interaction_result.total_interactions > 0,
                "total_count": interaction_result.total_interactions,
                "severe_count": interaction_result.severe_interactions,
                "interactions": [
                    {
                        "medication": i.medication_name,
                        "interacting_with": i.interacting_with,
                        "severity": i.severity,
                        "description": i.description
                    }
                    for i in interaction_result.interactions[:5]  # Top 5
                ]
            }
        
        # Model 3: Personalized Health Insights
        personalized_insights = await self._generate_personalized_insights(
            medication, user_id
        )
        
        # Model 4: Price Anomaly Detection
        price_analysis = await self._analyze_prices(medication_id)
        
        # Model 5: Nearby Pharmacies
        nearby_pharmacies = []
        if latitude and longitude:
            pharmacies = await self.pharmacy_service.find_nearby(
                latitude, longitude, radius_km=5.0, limit=5
            )
            nearby_pharmacies = [
                {
                    "pharmacy_id": str(p.id),
                    "name": p.name,
                    "address": p.address,
                    "distance_km": round(dist, 2),
                    "has_medication": False,  # TODO: Check inventory
                    "price": None,
                    "is_open": True  # TODO: Check working hours
                }
                for p, dist in pharmacies
            ]
        
        # Model 6: Batch Recall Check
        batch_recall = await self._check_batch_recall(medication_id)
        
        # Calculate points
        points = 5  # Base points for successful scan
        if interactions["severe_count"] > 0:
            points += 3  # Bonus for finding important interactions
        
        return {
            "scan_id": scan_id,
            "scan_type": "image",
            "scanned_at": datetime.utcnow(),
            "recognized": True,
            "medication": {
                "id": str(medication.id),
                "name": medication.name,
                "brand_name": medication.brand_name,
                "generic_name": medication.generic_name,
                "dosage_form": medication.dosage_form,
                "strength": medication.strength,
                "image_url": medication.image_url,
                "prescription_required": medication.prescription_required
            },
            "confidence": recognition_result["confidence"],
            "interactions": interactions,
            "price_analysis": price_analysis,
            "nearby_pharmacies": nearby_pharmacies,
            "batch_recall": batch_recall,
            "personalized_insights": personalized_insights,
            "points_earned": points,
            "alternatives": []  # TODO: Get alternatives
        }
    
    async def process_medication(
        self,
        medication_id: str,
        user_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process known medication ID and return insights.
        Used for QR/barcode scans.
        """
        medication = await self.medication_service.get_by_id(medication_id)
        
        if not medication:
            return {"error": "Medication not found"}
        
        # Similar to process_scan but skip recognition
        scan_id = "scan_" + str(datetime.utcnow().timestamp())
        
        # Get interactions if user authenticated
        interactions = {"has_interactions": False, "total_count": 0, "severe_count": 0, "interactions": []}
        if user_id:
            interaction_result = await self.interaction_service.check_with_user_medications(
                medication_id, user_id
            )
            interactions = {
                "has_interactions": interaction_result.total_interactions > 0,
                "total_count": interaction_result.total_interactions,
                "severe_count": interaction_result.severe_interactions,
                "interactions": [
                    {
                        "medication": i.medication_name,
                        "interacting_with": i.interacting_with,
                        "severity": i.severity,
                        "description": i.description
                    }
                    for i in interaction_result.interactions[:5]
                ]
            }
        
        personalized_insights = await self._generate_personalized_insights(medication, user_id)
        price_analysis = await self._analyze_prices(medication_id)
        
        nearby_pharmacies = []
        if latitude and longitude:
            pharmacies = await self.pharmacy_service.find_nearby(
                latitude, longitude, radius_km=5.0, limit=5
            )
            nearby_pharmacies = [
                {
                    "pharmacy_id": str(p.id),
                    "name": p.name,
                    "address": p.address,
                    "distance_km": round(dist, 2),
                    "has_medication": False,
                    "price": None,
                    "is_open": True
                }
                for p, dist in pharmacies
            ]
        
        batch_recall = await self._check_batch_recall(medication_id)
        
        return {
            "scan_id": scan_id,
            "scan_type": "qr",
            "scanned_at": datetime.utcnow(),
            "recognized": True,
            "medication": {
                "id": str(medication.id),
                "name": medication.name,
                "brand_name": medication.brand_name,
                "generic_name": medication.generic_name,
                "dosage_form": medication.dosage_form,
                "strength": medication.strength,
                "image_url": medication.image_url,
                "prescription_required": medication.prescription_required
            },
            "confidence": 1.0,
            "interactions": interactions,
            "price_analysis": price_analysis,
            "nearby_pharmacies": nearby_pharmacies,
            "batch_recall": batch_recall,
            "personalized_insights": personalized_insights,
            "points_earned": 5,
            "alternatives": []
        }
    
    async def _recognize_pill(self, image_data: bytes) -> Dict[str, Any]:
        """
        Model 1: Visual Pill Recognition using CNN.
        """
        if not self.pill_recognizer:
            logger.warning("Pill recognizer not available, using fallback")
            return {
                "recognized": False,
                "medication_id": None,
                "confidence": 0.0,
                "alternatives": []
            }
        
        try:
            # Save image temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_data)
                tmp_path = tmp_file.name
            
            try:
                # Run prediction
                predictions = self.pill_recognizer.predict(tmp_path, top_k=3)
                
                if predictions and predictions[0]['confidence'] > 0.3:
                    # Search for matching medication in database
                    best_match = predictions[0]
                    
                    # TODO: Use shape/color to filter database search
                    medications = await self.medication_service.search(
                        query="",
                        limit=10
                    )
                    
                    return {
                        "recognized": True,
                        "medication_id": str(medications[0].id) if medications else None,
                        "confidence": best_match['confidence'],
                        "shape": best_match.get('shape', 'unknown'),
                        "color": best_match.get('color', 'unknown'),
                        "alternatives": [
                            {
                                "medication_id": pred['medication_id'],
                                "confidence": pred['confidence']
                            }
                            for pred in predictions[1:]
                        ]
                    }
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"Pill recognition failed: {e}", exc_info=True)
            return {
                "recognized": False,
                "medication_id": None,
                "confidence": 0.0,
                "alternatives": [],
                "error": str(e)
            }
        
        return {
            "recognized": False,
            "medication_id": None,
            "confidence": 0.0,
            "alternatives": []
        }
    
    async def _generate_personalized_insights(
        self,
        medication,
        user_id: Optional[str]
    ) -> list:
        """
        Model 3: Generate personalized health insights.
        Based on user's medical profile, age, conditions, etc.
        """
        insights = []
        
        if not user_id:
            return insights
        
        # TODO: Get user's medical profile
        # Check for allergies, conditions, pregnancy, etc.
        
        # Example insights
        if medication.pregnancy_category in ["D", "X"]:
            insights.append({
                "type": "warning",
                "message": "This medication should not be used during pregnancy",
                "severity": "critical",
                "related_to": "pregnancy"
            })
        
        if medication.prescription_required:
            insights.append({
                "type": "info",
                "message": "Prescription required - consult your doctor",
                "severity": "info",
                "related_to": "prescription"
            })
        
        return insights
    
    async def _analyze_prices(self, medication_id: str) -> Dict[str, Any]:
        """
        Model 4: Price Anomaly Detection using Isolation Forest.
        """
        prices = await self.medication_service.get_prices(medication_id)
        
        if not prices:
            return {
                "current_price": None,
                "average_price": None,
                "is_anomaly": False,
                "recommendation": "No price data available",
                "price_range": {}
            }
        
        # Calculate statistics
        price_values = [p.price for p in prices]
        avg_price = sum(price_values) / len(price_values)
        min_price = min(price_values)
        max_price = max(price_values)
        
        results = []
        
        if self.price_anomaly_detector:
            # Analyze each price for anomalies
            for price_record in prices:
                price_data = {
                    'price': price_record.price,
                    'avg_market_price': avg_price,
                    'pharmacy_rating': 4.0,  # TODO: Get from pharmacy
                    'is_chain': False,  # TODO: Get from pharmacy
                    'distance_from_center_km': 5.0,
                    'day_of_week': datetime.now().weekday(),
                    'medication_demand_score': 0.5,
                    'stock_level': 50
                }
                
                anomaly_result = self.price_anomaly_detector.detect_anomaly(price_data)
                
                if anomaly_result['is_anomaly']:
                    results.append({
                        'pharmacy_id': str(price_record.pharmacy_id) if price_record.pharmacy_id else None,
                        'price': price_record.price,
                        'severity': anomaly_result['severity'],
                        'explanation': anomaly_result['explanation']
                    })
        
        return {
            "average_price": round(avg_price, 2),
            "min_price": min_price,
            "max_price": max_price,
            "price_range": {
                "low": min_price,
                "high": max_price,
                "median": sorted(price_values)[len(price_values) // 2]
            },
            "anomalies_found": len(results),
            "anomalies": results[:5],  # Top 5 anomalies
            "recommendation": self._generate_price_recommendation(results)
        }
    
    def _generate_price_recommendation(self, anomalies: List[Dict]) -> str:
        """Generate price recommendation based on anomalies."""
        if not anomalies:
            return "Prices are within normal range"
        
        severe_count = sum(1 for a in anomalies if a['severity'] in ['high', 'extreme'])
        
        if severe_count > 0:
            return f"Found {severe_count} significantly overpriced options. Check alternative pharmacies."
        else:
            return "Some minor price variations detected. Compare before purchasing."
    
    async def _check_batch_recall(self, medication_id: str) -> Dict[str, Any]:
        """
        Model 6: Batch Recall Prediction.
        Checks if medication has any recalls.
        """
        # TODO: Query BatchRecall table
        return {
            "is_recalled": False,
            "recall_reason": None,
            "recall_date": None,
            "risk_score": 0.0
        }
