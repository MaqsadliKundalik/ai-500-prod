"""
Scan Service
============
Business logic for medication scans, history, QR/barcode processing
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.scan import Scan
from app.models.medication import Medication


class ScanService:
    """Service for scan-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_scan(
        self,
        user_id: str,
        scan_result: dict
    ) -> Scan:
        """
        Save a scan to database.
        
        Args:
            user_id: User who performed the scan
            scan_result: UnifiedInsightResponse as dict
        """
        # Extract medication ID safely
        medication_id = None
        if scan_result.get("medication") and isinstance(scan_result["medication"], dict):
            medication_id = scan_result["medication"].get("id")
        
        # Convert datetime objects to ISO format strings for JSON storage
        insights_json = scan_result.copy()
        if "scanned_at" in insights_json and hasattr(insights_json["scanned_at"], "isoformat"):
            insights_json["scanned_at"] = insights_json["scanned_at"].isoformat()
        
        scan = Scan(
            user_id=user_id,
            medication_id=medication_id,
            scan_type=scan_result.get("scan_type", "image"),
            recognized=scan_result.get("recognized", False),
            confidence_score=scan_result.get("confidence", 0.0),
            insights=insights_json,  # Store full result as JSONB
            interactions_count=scan_result.get("interactions", {}).get("total_count", 0),
            severe_interactions=scan_result.get("interactions", {}).get("severe_count", 0),
            is_price_anomaly=scan_result.get("price_analysis", {}).get("is_anomaly", False),
            points_earned=scan_result.get("points_earned", 0)
        )
        
        self.db.add(scan)
        await self.db.flush()
        await self.db.refresh(scan)
        
        return scan
    
    async def get_scan(self, scan_id: str, user_id: str) -> Optional[Scan]:
        """Get a specific scan by ID."""
        result = await self.db.execute(
            select(Scan)
            .where(Scan.id == scan_id)
            .where(Scan.user_id == user_id)
        )
        
        return result.scalar_one_or_none()
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Scan]:
        """Get user's scan history."""
        result = await self.db.execute(
            select(Scan)
            .where(Scan.user_id == user_id)
            .order_by(Scan.scanned_at.desc())
            .limit(limit)
        )
        
        return result.scalars().all()
    
    async def delete_scan(self, scan_id: str, user_id: str) -> bool:
        """Delete a scan from history."""
        result = await self.db.execute(
            select(Scan)
            .where(Scan.id == scan_id)
            .where(Scan.user_id == user_id)
        )
        
        scan = result.scalar_one_or_none()
        if not scan:
            return False
        
        await self.db.delete(scan)
        await self.db.flush()
        
        return True
    
    async def lookup_by_code(
        self,
        code: str,
        code_type: str
    ) -> Optional[Medication]:
        """
        Lookup medication by QR/barcode.
        
        Args:
            code: The scanned code
            code_type: Type of code (qr, ean13, etc.)
        """
        # For now, assume code is barcode
        # TODO: Implement different code type handling
        result = await self.db.execute(
            select(Medication)
            .where(Medication.barcode == code)
            .where(Medication.is_active == True)
        )
        
        return result.scalar_one_or_none()
    
    async def get_scan_statistics(self, user_id: str) -> dict:
        """Get user's scan statistics."""
        from sqlalchemy import func
        
        result = await self.db.execute(
            select(
                func.count(Scan.id).label('total_scans'),
                func.count(Scan.id).filter(Scan.recognized == True).label('recognized'),
                func.count(Scan.id).filter(Scan.severe_interactions > 0).label('with_warnings'),
                func.sum(Scan.points_earned).label('total_points')
            )
            .where(Scan.user_id == user_id)
        )
        
        stats = result.first()
        
        return {
            "total_scans": stats.total_scans or 0,
            "recognized": stats.recognized or 0,
            "with_warnings": stats.with_warnings or 0,
            "total_points_earned": stats.total_points or 0
        }
