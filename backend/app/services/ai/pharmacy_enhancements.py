"""
Enhanced Pharmacy Service
=========================
Price comparison, availability tracking, route optimization
"""

from typing import Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import logging
from datetime import datetime, time

from app.models.pharmacy import Pharmacy
from app.models.medication import Medication

logger = logging.getLogger(__name__)


class PharmacyEnhancements:
    """Enhanced pharmacy features."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compare_prices(
        self,
        medication_id: str,
        latitude: float,
        longitude: float,
        max_distance_km: float = 5.0
    ) -> List[Dict]:
        """
        Compare medication prices across nearby pharmacies.
        
        Returns list sorted by price (cheapest first) with:
        - pharmacy info
        - price
        - distance
        - availability
        - savings compared to average
        """
        # Find nearby pharmacies
        nearby = await self._find_nearby_pharmacies(latitude, longitude, max_distance_km)
        
        if not nearby:
            return []
        
        # Get prices from each pharmacy
        price_comparisons = []
        total_price = 0
        count = 0
        
        for pharmacy in nearby:
            # Mock price data - replace with real database query
            price_info = await self._get_medication_price(medication_id, pharmacy.id)
            
            if price_info:
                price_comparisons.append({
                    "pharmacy": {
                        "id": str(pharmacy.id),
                        "name": pharmacy.name,
                        "address": pharmacy.address,
                        "phone": pharmacy.phone_number,
                        "rating": pharmacy.rating or 4.0,
                        "is_24h": pharmacy.is_24h
                    },
                    "price": price_info["price"],
                    "currency": "UZS",
                    "distance_km": pharmacy.distance_km,
                    "in_stock": price_info["in_stock"],
                    "stock_quantity": price_info.get("stock_quantity", 0),
                    "last_updated": price_info.get("last_updated"),
                })
                
                total_price += price_info["price"]
                count += 1
        
        if count == 0:
            return []
        
        # Calculate average price and savings
        avg_price = total_price / count
        
        for item in price_comparisons:
            item["savings_vs_average"] = avg_price - item["price"]
            item["savings_percentage"] = ((avg_price - item["price"]) / avg_price * 100) if avg_price > 0 else 0
        
        # Sort by price (cheapest first)
        price_comparisons.sort(key=lambda x: x["price"])
        
        # Add rankings
        for idx, item in enumerate(price_comparisons, 1):
            item["price_rank"] = idx
            
            # Add badges
            badges = []
            if idx == 1:
                badges.append("🏆 Eng arzon")
            if item["pharmacy"]["is_24h"]:
                badges.append("🕐 24 soat")
            if item["pharmacy"]["rating"] >= 4.5:
                badges.append("⭐ Yuqori reyting")
            if item["in_stock"] and item["stock_quantity"] > 10:
                badges.append("✅ Ko'p miqdorda")
            
            item["badges"] = badges
        
        return price_comparisons
    
    async def check_availability(
        self,
        medication_id: str,
        pharmacy_ids: Optional[List[str]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_distance_km: float = 10.0
    ) -> Dict:
        """
        Check medication availability across pharmacies.
        
        Returns:
            {
                "medication_id": str,
                "total_pharmacies_checked": int,
                "in_stock_count": int,
                "out_of_stock_count": int,
                "pharmacies": [...]
            }
        """
        pharmacies = []
        
        if pharmacy_ids:
            # Check specific pharmacies
            for pharmacy_id in pharmacy_ids:
                result = await self.db.execute(
                    select(Pharmacy).where(Pharmacy.id == pharmacy_id)
                )
                pharmacy = result.scalar_one_or_none()
                if pharmacy:
                    pharmacies.append(pharmacy)
        else:
            # Check nearby pharmacies
            pharmacies = await self._find_nearby_pharmacies(latitude, longitude, max_distance_km)
        
        availability_list = []
        in_stock = 0
        out_of_stock = 0
        
        for pharmacy in pharmacies:
            stock_info = await self._get_medication_price(medication_id, pharmacy.id)
            
            is_available = stock_info and stock_info.get("in_stock", False)
            
            availability_list.append({
                "pharmacy_id": str(pharmacy.id),
                "pharmacy_name": pharmacy.name,
                "in_stock": is_available,
                "quantity": stock_info.get("stock_quantity", 0) if stock_info else 0,
                "price": stock_info.get("price") if stock_info else None,
                "last_checked": datetime.utcnow().isoformat()
            })
            
            if is_available:
                in_stock += 1
            else:
                out_of_stock += 1
        
        return {
            "medication_id": medication_id,
            "total_pharmacies_checked": len(pharmacies),
            "in_stock_count": in_stock,
            "out_of_stock_count": out_of_stock,
            "availability_rate": (in_stock / len(pharmacies) * 100) if pharmacies else 0,
            "pharmacies": availability_list
        }
    
    def calculate_route_optimization(
        self,
        user_location: Tuple[float, float],
        pharmacy_locations: List[Dict],
        medication_list: List[str]
    ) -> Dict:
        """
        Optimize route to visit multiple pharmacies for best prices.
        
        Simple greedy algorithm: nearest pharmacy with best price combination.
        """
        # This is a simplified version
        # In production, use proper routing API (Google Maps, OSM)
        
        current_location = user_location
        route = []
        total_distance = 0
        total_cost = 0
        remaining_medications = set(medication_list)
        
        while remaining_medications and pharmacy_locations:
            # Find nearest pharmacy with needed medications
            best_pharmacy = None
            best_score = float('inf')
            
            for pharmacy in pharmacy_locations:
                # Calculate distance
                distance = self._haversine_distance(
                    current_location[0], current_location[1],
                    pharmacy["latitude"], pharmacy["longitude"]
                )
                
                # Calculate available medications at this pharmacy
                available = [m for m in remaining_medications if m in pharmacy.get("available_medications", [])]
                
                if available:
                    # Score: distance / number of available meds (lower is better)
                    score = distance / len(available)
                    
                    if score < best_score:
                        best_score = score
                        best_pharmacy = pharmacy
                        best_pharmacy["medications_to_buy"] = available
            
            if best_pharmacy:
                route.append(best_pharmacy)
                total_distance += self._haversine_distance(
                    current_location[0], current_location[1],
                    best_pharmacy["latitude"], best_pharmacy["longitude"]
                )
                
                # Update remaining medications
                for med in best_pharmacy["medications_to_buy"]:
                    remaining_medications.discard(med)
                
                # Update current location
                current_location = (best_pharmacy["latitude"], best_pharmacy["longitude"])
                
                # Remove pharmacy from list
                pharmacy_locations.remove(best_pharmacy)
            else:
                break
        
        return {
            "route": route,
            "total_distance_km": round(total_distance, 2),
            "total_pharmacies": len(route),
            "estimated_time_minutes": int(total_distance * 2),  # Assume 30 km/h average
            "medications_found": len(medication_list) - len(remaining_medications),
            "medications_not_found": list(remaining_medications)
        }
    
    async def get_pharmacy_ratings(
        self,
        pharmacy_id: str,
        include_reviews: bool = True
    ) -> Dict:
        """Get pharmacy ratings and reviews."""
        # Mock data - replace with real reviews database
        return {
            "pharmacy_id": pharmacy_id,
            "overall_rating": 4.3,
            "total_reviews": 127,
            "rating_breakdown": {
                "5_star": 65,
                "4_star": 42,
                "3_star": 15,
                "2_star": 3,
                "1_star": 2
            },
            "aspects": {
                "service": 4.5,
                "prices": 4.0,
                "availability": 4.2,
                "cleanliness": 4.6
            },
            "recent_reviews": [
                {
                    "rating": 5,
                    "comment": "Xizmat a'lo! Tez va sifatli",
                    "date": "2025-12-08",
                    "helpful_count": 12
                },
                {
                    "rating": 4,
                    "comment": "Yaxshi dorixona, biroz qimmat",
                    "date": "2025-12-07",
                    "helpful_count": 5
                }
            ] if include_reviews else []
        }
    
    async def _find_nearby_pharmacies(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float
    ) -> List[Pharmacy]:
        """Find pharmacies within radius."""
        # Simplified - use PostGIS in production
        result = await self.db.execute(
            select(Pharmacy).where(Pharmacy.is_active == True)
        )
        pharmacies = result.scalars().all()
        
        # Calculate distances and filter
        nearby = []
        for pharmacy in pharmacies:
            if pharmacy.latitude and pharmacy.longitude:
                distance = self._haversine_distance(
                    latitude, longitude,
                    pharmacy.latitude, pharmacy.longitude
                )
                
                if distance <= max_distance_km:
                    pharmacy.distance_km = round(distance, 2)
                    nearby.append(pharmacy)
        
        # Sort by distance
        nearby.sort(key=lambda p: p.distance_km)
        return nearby
    
    async def _get_medication_price(self, medication_id: str, pharmacy_id: str) -> Optional[Dict]:
        """Get medication price and availability at specific pharmacy."""
        # Mock data - replace with real pharmacy_inventory table query
        import random
        
        # Simulate some pharmacies have it, some don't
        if random.random() > 0.3:  # 70% in stock
            return {
                "price": random.randint(5000, 50000),  # UZS
                "in_stock": True,
                "stock_quantity": random.randint(5, 100),
                "last_updated": datetime.utcnow().isoformat()
            }
        return None
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c


def get_pharmacy_enhancements(db: AsyncSession) -> PharmacyEnhancements:
    """Get pharmacy enhancements instance."""
    return PharmacyEnhancements(db)
