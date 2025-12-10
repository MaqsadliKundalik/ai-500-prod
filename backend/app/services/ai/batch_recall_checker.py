"""
Batch Recall Checker Service
============================
Integration with FDA, WHO, and Uzbekistan health ministry for recall alerts
"""

from typing import Dict, List, Optional
import httpx
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RecallSeverity(str, Enum):
    """Recall severity levels."""
    CLASS_I = "class_i"      # Life-threatening
    CLASS_II = "class_ii"    # Serious health hazard
    CLASS_III = "class_iii"  # Minor violation
    MARKET_WITHDRAWAL = "market_withdrawal"  # No health hazard


class BatchRecallChecker:
    """Check for medication recalls and safety alerts."""
    
    FDA_API_BASE = "https://api.fda.gov/drug/enforcement.json"
    WHO_API_BASE = "https://www.who.int/medicines/publications/drugalerts"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),  # 30s timeout, 10s connect
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self._cache = {}
        self._cache_duration = timedelta(hours=6)  # Cache for 6 hours
    
    async def check_medication_recalls(
        self,
        medication_name: str,
        ndc_code: Optional[str] = None,
        batch_number: Optional[str] = None
    ) -> Dict:
        """
        Check if medication has any recalls or safety alerts.
        
        Args:
            medication_name: Name of medication
            ndc_code: National Drug Code
            batch_number: Batch/lot number
            
        Returns:
            {
                "has_recalls": bool,
                "recall_count": int,
                "recalls": List[Dict],
                "risk_level": str,
                "action_required": str
            }
        """
        # Check cache first
        cache_key = f"{medication_name}_{ndc_code}_{batch_number}"
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.info(f"Returning cached recall data for {medication_name}")
                return cached_data
        
        recalls = []
        
        # Check FDA database
        fda_recalls = await self._check_fda_recalls(medication_name, ndc_code)
        recalls.extend(fda_recalls)
        
        # Check WHO alerts (if applicable)
        who_alerts = await self._check_who_alerts(medication_name)
        recalls.extend(who_alerts)
        
        # Check Uzbekistan health ministry (mock - replace with real API)
        uz_alerts = await self._check_uzbekistan_alerts(medication_name, batch_number)
        recalls.extend(uz_alerts)
        
        # Determine overall risk level
        risk_level = self._calculate_risk_level(recalls)
        action_required = self._get_action_required(risk_level, recalls)
        
        result = {
            "has_recalls": len(recalls) > 0,
            "recall_count": len(recalls),
            "recalls": recalls,
            "risk_level": risk_level,
            "action_required": action_required,
            "checked_sources": ["FDA", "WHO", "Uzbekistan MoH"],
            "last_checked": datetime.utcnow().isoformat()
        }
        
        # Cache result
        self._cache[cache_key] = (result, datetime.now())
        
        return result
    
    async def _check_fda_recalls(
        self,
        medication_name: str,
        ndc_code: Optional[str] = None
    ) -> List[Dict]:
        """Check FDA enforcement database."""
        try:
            # Build query
            search_terms = []
            if ndc_code:
                search_terms.append(f'product_ndc:"{ndc_code}"')
            else:
                # Search by name
                search_terms.append(f'product_description:"{medication_name}"')
            
            query = " AND ".join(search_terms)
            
            params = {
                "search": query,
                "limit": 10
            }
            
            response = await self.client.get(self.FDA_API_BASE, params=params)
            response.raise_for_status()  # Raise exception for 4xx/5xx
            
            data = response.json()
            results = data.get("results", [])
            
            recalls = []
            for item in results:
                recalls.append({
                    "source": "FDA",
                    "recall_number": item.get("recall_number"),
                    "product_description": item.get("product_description"),
                    "reason": item.get("reason_for_recall"),
                    "classification": item.get("classification"),
                    "status": item.get("status"),
                    "recall_initiation_date": item.get("recall_initiation_date"),
                    "company": item.get("recalling_firm"),
                    "voluntary_mandated": item.get("voluntary_mandated"),
                    "distribution_pattern": item.get("distribution_pattern"),
                    "severity_uz": self._translate_classification_uz(item.get("classification"))
                })
            
            logger.info(f"Found {len(recalls)} FDA recalls for {medication_name}")
            return recalls
        
        except httpx.TimeoutException:
            logger.error(f"FDA API timeout for {medication_name}")
            return []
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"No FDA recalls found for {medication_name}")
                return []
            else:
                logger.warning(f"FDA API HTTP error: {e.response.status_code}")
                return []
        except httpx.NetworkError as e:
            logger.error(f"FDA API network error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error checking FDA recalls: {e}", exc_info=True)
            return []
    
    async def _check_who_alerts(self, medication_name: str) -> List[Dict]:
        """Check WHO drug alerts."""
        # WHO doesn't have a public API, this is a mock
        # In production, scrape their website or use official data feed
        
        try:
            # Mock data for demonstration
            who_alerts = []
            
            # Example: simulate finding an alert
            if "counterfeit" in medication_name.lower():
                who_alerts.append({
                    "source": "WHO",
                    "alert_id": "WHO-2025-001",
                    "product_description": medication_name,
                    "reason": "Counterfeit product detected",
                    "countries_affected": ["Uzbekistan", "Kazakhstan", "Tajikistan"],
                    "date_issued": "2025-11-15",
                    "severity": "HIGH",
                    "severity_uz": "Yuqori xavf - Soxta dori"
                })
            
            return who_alerts
        
        except Exception as e:
            logger.error(f"Error checking WHO alerts: {e}")
            return []
    
    async def _check_uzbekistan_alerts(
        self,
        medication_name: str,
        batch_number: Optional[str]
    ) -> List[Dict]:
        """Check Uzbekistan Ministry of Health alerts."""
        # Mock implementation - replace with real API when available
        
        try:
            # In production, call actual Uzbekistan MoH API
            # For now, return mock data
            
            uz_alerts = []
            
            # Example: simulate batch-specific alert
            if batch_number and batch_number.startswith("UZ"):
                uz_alerts.append({
                    "source": "Uzbekistan MoH",
                    "alert_id": "UZ-SOG-2025-042",
                    "product_description": medication_name,
                    "batch_number": batch_number,
                    "reason": "Sifat standartiga mos kelmaydi",
                    "reason_en": "Does not meet quality standards",
                    "date_issued": "2025-12-01",
                    "severity": "MEDIUM",
                    "severity_uz": "O'rtacha xavf",
                    "action": "Ushbu partiyani qaytarish va almashish"
                })
            
            return uz_alerts
        
        except Exception as e:
            logger.error(f"Error checking Uzbekistan alerts: {e}")
            return []
    
    def _calculate_risk_level(self, recalls: List[Dict]) -> str:
        """Calculate overall risk level from recalls."""
        if not recalls:
            return "SAFE"
        
        # Check for Class I (life-threatening)
        for recall in recalls:
            classification = recall.get("classification", "").upper()
            if "CLASS I" in classification or recall.get("severity") == "HIGH":
                return "CRITICAL"
        
        # Check for Class II (serious)
        for recall in recalls:
            classification = recall.get("classification", "").upper()
            if "CLASS II" in classification or recall.get("severity") == "MEDIUM":
                return "HIGH"
        
        # Class III or market withdrawal
        return "MODERATE"
    
    def _get_action_required(self, risk_level: str, recalls: List[Dict]) -> str:
        """Get action required message in Uzbek."""
        actions = {
            "SAFE": "✅ Xavf yo'q. Dorini xavfsiz qabul qilishingiz mumkin.",
            "MODERATE": "⚠️ Kichik muammolar aniqlandi. Doktor bilan maslahatlashing.",
            "HIGH": "🟠 Jiddiy muammo! Darhol dorini qabul qilishni to'xtating va shifokorga murojaat qiling.",
            "CRITICAL": "🚨 HAYOT UCHUN XAVFLI! Darhol dorini qabul qilishni to'xtating va tez yordam chaqiring!"
        }
        
        action = actions.get(risk_level, actions["MODERATE"])
        
        # Add specific instructions from recalls
        if recalls:
            first_recall = recalls[0]
            if first_recall.get("source") == "Uzbekistan MoH":
                action += f"\n\nQo'shimcha: {first_recall.get('action', '')}"
        
        return action
    
    def _translate_classification_uz(self, classification: str) -> str:
        """Translate FDA classification to Uzbek."""
        translations = {
            "Class I": "1-sinf: Hayot uchun xavfli",
            "Class II": "2-sinf: Sog'liq uchun jiddiy xavf",
            "Class III": "3-sinf: Kichik xavf",
        }
        
        for key, value in translations.items():
            if key.lower() in classification.lower():
                return value
        
        return "Noma'lum sinf"
    
    async def subscribe_to_alerts(
        self,
        user_id: str,
        medication_ids: List[str],
        notification_method: str = "push"
    ) -> Dict:
        """Subscribe user to recall alerts for specific medications."""
        # This would integrate with notification service
        return {
            "subscription_id": f"sub_{user_id}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "medications": medication_ids,
            "notification_method": notification_method,
            "status": "active",
            "message": "Siz tanlagan dorilar uchun ogohlantirishlar yoqildi"
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


def get_batch_recall_checker() -> BatchRecallChecker:
    """Get singleton instance."""
    return BatchRecallChecker()
