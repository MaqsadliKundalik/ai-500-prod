"""
Pharmacy Service
================
Business logic for pharmacy finding, availability checking
"""

from typing import Optional, List, Tuple
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import math

from app.models.pharmacy import Pharmacy, PharmacyInventory, PharmacyReport


class PharmacyService:
    """Service for pharmacy-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        Returns distance in kilometers.
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    async def get_by_id(self, pharmacy_id: str) -> Optional[Pharmacy]:
        """Get pharmacy by ID."""
        result = await self.db.execute(
            select(Pharmacy)
            .where(Pharmacy.id == pharmacy_id)
            .where(Pharmacy.is_active == True)
        )
        
        return result.scalar_one_or_none()
    
    async def find_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        limit: int = 20
    ) -> List[Tuple[Pharmacy, float]]:
        """
        Find nearby pharmacies within radius.
        Returns list of (Pharmacy, distance_km) tuples.
        """
        # Get all active pharmacies
        result = await self.db.execute(
            select(Pharmacy)
            .where(Pharmacy.is_active == True)
            .where(Pharmacy.temporarily_closed == False)
        )
        
        pharmacies = result.scalars().all()
        
        # Calculate distances and filter by radius
        nearby = []
        for pharmacy in pharmacies:
            distance = self._calculate_distance(
                latitude, longitude,
                pharmacy.latitude, pharmacy.longitude
            )
            
            if distance <= radius_km:
                nearby.append((pharmacy, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        
        return nearby[:limit]
    
    async def check_availability(
        self,
        pharmacy_id: str,
        medication_id: str
    ) -> Optional[PharmacyInventory]:
        """Check if medication is available at pharmacy."""
        result = await self.db.execute(
            select(PharmacyInventory)
            .where(PharmacyInventory.pharmacy_id == pharmacy_id)
            .where(PharmacyInventory.medication_id == medication_id)
            .where(PharmacyInventory.is_available == True)
        )
        
        return result.scalar_one_or_none()
    
    async def find_with_medication(
        self,
        medication_id: str,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0
    ) -> List[Tuple[Pharmacy, float, Optional[float]]]:
        """
        Find pharmacies that have a specific medication in stock.
        Returns list of (Pharmacy, distance_km, price) tuples.
        """
        # Get pharmacies with this medication
        result = await self.db.execute(
            select(PharmacyInventory, Pharmacy)
            .join(Pharmacy, PharmacyInventory.pharmacy_id == Pharmacy.id)
            .where(PharmacyInventory.medication_id == medication_id)
            .where(PharmacyInventory.is_available == True)
            .where(Pharmacy.is_active == True)
        )
        
        inventory_pharmacy_pairs = result.all()
        
        # Calculate distances
        nearby = []
        for inventory, pharmacy in inventory_pharmacy_pairs:
            distance = self._calculate_distance(
                latitude, longitude,
                pharmacy.latitude, pharmacy.longitude
            )
            
            if distance <= radius_km:
                nearby.append((pharmacy, distance, inventory.price))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        
        return nearby
    
    async def get_directions(
        self,
        pharmacy_id: str,
        from_lat: float,
        from_lon: float
    ) -> dict:
        """
        Get directions to pharmacy.
        TODO: Integrate with OSRM or Google Maps API
        """
        pharmacy = await self.get_by_id(pharmacy_id)
        if not pharmacy:
            return None
        
        distance = self._calculate_distance(
            from_lat, from_lon,
            pharmacy.latitude, pharmacy.longitude
        )
        
        # Rough estimation: 5 minutes per km
        duration_minutes = int(distance * 5)
        
        return {
            "pharmacy_id": str(pharmacy.id),
            "pharmacy_name": pharmacy.name,
            "pharmacy_address": pharmacy.address,
            "distance_km": round(distance, 2),
            "duration_minutes": duration_minutes,
            "route_coordinates": [
                [from_lat, from_lon],
                [pharmacy.latitude, pharmacy.longitude]
            ],
            "instructions": [
                f"Head towards {pharmacy.name}",
                f"Arrive at {pharmacy.address}"
            ]
        }
    
    async def create_report(
        self,
        pharmacy_id: str,
        user_id: str,
        report_type: str,
        description: Optional[str] = None
    ) -> PharmacyReport:
        """Create a report about a pharmacy."""
        report = PharmacyReport(
            pharmacy_id=pharmacy_id,
            user_id=user_id,
            report_type=report_type,
            description=description
        )
        
        self.db.add(report)
        await self.db.flush()
        await self.db.refresh(report)
        
        return report
