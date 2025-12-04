"""
Dashboard Service
=================
Business logic for family dashboard, adherence tracking
"""

from typing import Optional, List
from datetime import datetime, date, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.scan import AdherenceLog
from app.models.user import FamilyMember
from app.models.medication import UserMedication


class DashboardService:
    """Service for family dashboard and adherence tracking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_summary(self, user_id: str) -> dict:
        """
        Get dashboard summary for user.
        TODO: Implement full logic with family members
        """
        # Get user's adherence today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        today_end = datetime.utcnow().replace(hour=23, minute=59, second=59)
        
        adherence_today = await self._get_adherence_stats(
            user_id, today_start, today_end
        )
        
        # Get week adherence
        week_start = datetime.utcnow() - timedelta(days=7)
        adherence_week = await self._get_adherence_stats(
            user_id, week_start, datetime.utcnow()
        )
        
        return {
            "user_adherence_today": adherence_today,
            "user_adherence_week": adherence_week,
            "current_streak": 0,  # TODO: Calculate streak
            "family_members": [],
            "upcoming_reminders": [],
            "unread_alerts_count": 0,
            "recent_alerts": [],
            "points_today": 0,
            "total_points": 0,
            "level": 1
        }
    
    async def _get_adherence_stats(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        family_member_id: Optional[str] = None
    ) -> dict:
        """Calculate adherence statistics for a period."""
        query = select(
            func.count(AdherenceLog.id).label('total'),
            func.count(AdherenceLog.id).filter(AdherenceLog.taken == True).label('taken'),
            func.count(AdherenceLog.id).filter(AdherenceLog.taken == False).label('missed')
        ).where(
            AdherenceLog.user_id == user_id,
            AdherenceLog.scheduled_time >= start_date,
            AdherenceLog.scheduled_time <= end_date
        )
        
        if family_member_id:
            query = query.where(AdherenceLog.family_member_id == family_member_id)
        
        result = await self.db.execute(query)
        stats = result.first()
        
        total = stats.total or 0
        taken = stats.taken or 0
        missed = stats.missed or 0
        
        adherence_rate = (taken / total * 100) if total > 0 else 0.0
        
        return {
            "total_scheduled": total,
            "taken": taken,
            "missed": missed,
            "adherence_rate": round(adherence_rate, 2)
        }
    
    async def get_member_dashboard(
        self,
        user_id: str,
        member_id: str
    ) -> Optional[dict]:
        """Get detailed dashboard for a family member."""
        # Verify family member belongs to user
        result = await self.db.execute(
            select(FamilyMember)
            .where(FamilyMember.id == member_id)
            .where(FamilyMember.user_id == user_id)
        )
        
        member = result.scalar_one_or_none()
        if not member:
            return None
        
        # Get adherence stats
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        adherence_today = await self._get_adherence_stats(
            user_id, today_start, datetime.utcnow(), member_id
        )
        
        return {
            "id": str(member.id),
            "name": member.name,
            "relationship": member.relationship,
            "age": member.age,
            "avatar_url": member.avatar_url,
            "active_medications": 0,
            "medications": [],
            "adherence_today": adherence_today,
            "adherence_week": {},
            "adherence_month": {},
            "upcoming_reminders": [],
            "alerts": []
        }
    
    async def get_adherence_history(
        self,
        user_id: str,
        member_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[AdherenceLog]:
        """Get adherence history."""
        query = select(AdherenceLog).where(
            AdherenceLog.user_id == user_id
        )
        
        if member_id:
            query = query.where(AdherenceLog.family_member_id == member_id)
        
        if start_date:
            query = query.where(AdherenceLog.scheduled_time >= start_date)
        
        if end_date:
            query = query.where(AdherenceLog.scheduled_time <= end_date)
        
        query = query.order_by(AdherenceLog.scheduled_time.desc())
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def log_adherence(
        self,
        user_id: str,
        medication_id: str,
        taken: bool,
        member_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> AdherenceLog:
        """Log medication adherence."""
        log = AdherenceLog(
            user_id=user_id,
            user_medication_id=medication_id,
            family_member_id=member_id,
            taken=taken,
            scheduled_time=datetime.utcnow(),
            actual_time=datetime.utcnow() if taken else None,
            notes=notes,
            points_earned=5 if taken else 0
        )
        
        self.db.add(log)
        await self.db.flush()
        await self.db.refresh(log)
        
        # Award points if taken
        if taken:
            from app.services.user_service import UserService
            user_service = UserService(self.db)
            await user_service.add_points(
                user_id, 5, "adherence", "Medication taken on time"
            )
        
        return log
    
    async def get_reminders(self, user_id: str) -> List[dict]:
        """Get medication reminders."""
        # TODO: Implement reminder system
        return []
    
    async def create_reminder(
        self,
        user_id: str,
        medication_id: str,
        reminder_time: str,
        repeat: str,
        member_id: Optional[str] = None
    ) -> dict:
        """Create a medication reminder."""
        # TODO: Implement reminder creation
        return {}
    
    async def delete_reminder(self, user_id: str, reminder_id: str) -> bool:
        """Delete a reminder."""
        # TODO: Implement reminder deletion
        return True
    
    async def get_alerts(
        self,
        user_id: str,
        unread_only: bool = False
    ) -> List[dict]:
        """Get user alerts."""
        # TODO: Implement alert system
        return []
    
    async def mark_alert_read(self, user_id: str, alert_id: str) -> bool:
        """Mark an alert as read."""
        # TODO: Implement alert marking
        return True
