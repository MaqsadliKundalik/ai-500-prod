"""
Dashboard Schemas
=================
Request/Response models for family dashboard endpoints
"""

from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel


class AdherenceStats(BaseModel):
    """Adherence statistics."""
    
    total_scheduled: int = 0
    taken: int = 0
    missed: int = 0
    adherence_rate: float = 0.0  # 0-100%


class MedicationReminder(BaseModel):
    """Medication reminder."""
    
    id: str
    medication_id: str
    medication_name: str
    dosage: Optional[str] = None
    reminder_time: str  # "08:00"
    repeat: str  # "daily", "weekly"
    is_active: bool = True
    family_member_id: Optional[str] = None
    family_member_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class UpcomingReminder(BaseModel):
    """Upcoming medication reminder."""
    
    medication_name: str
    dosage: Optional[str] = None
    scheduled_time: datetime
    family_member_name: Optional[str] = None


class AlertResponse(BaseModel):
    """Alert/notification."""
    
    id: str
    type: str  # "missed_dose", "interaction_warning", "refill_reminder", "recall_alert"
    title: str
    message: str
    severity: str = "info"  # "info", "warning", "critical"
    is_read: bool = False
    created_at: datetime
    related_medication: Optional[str] = None
    related_family_member: Optional[str] = None
    
    class Config:
        from_attributes = True


class FamilyMemberStatus(BaseModel):
    """Family member status summary."""
    
    id: str
    name: str
    relationship: str
    avatar_url: Optional[str] = None
    adherence_today: AdherenceStats
    next_medication: Optional[UpcomingReminder] = None
    has_alerts: bool = False


class DashboardSummary(BaseModel):
    """Main dashboard summary."""
    
    # User stats
    user_adherence_today: AdherenceStats
    user_adherence_week: AdherenceStats
    current_streak: int = 0
    
    # Family
    family_members: List[FamilyMemberStatus] = []
    
    # Upcoming
    upcoming_reminders: List[UpcomingReminder] = []
    
    # Alerts
    unread_alerts_count: int = 0
    recent_alerts: List[AlertResponse] = []
    
    # Points
    points_today: int = 0
    total_points: int = 0
    level: int = 1


class FamilyMemberDashboard(BaseModel):
    """Detailed dashboard for a family member."""
    
    id: str
    name: str
    relationship: str
    age: Optional[int] = None
    avatar_url: Optional[str] = None
    
    # Medications
    active_medications: int = 0
    medications: List[dict] = []
    
    # Adherence
    adherence_today: AdherenceStats
    adherence_week: AdherenceStats
    adherence_month: AdherenceStats
    
    # Reminders
    upcoming_reminders: List[UpcomingReminder] = []
    
    # Alerts
    alerts: List[AlertResponse] = []


class AdherenceRecord(BaseModel):
    """Adherence log record."""
    
    id: str
    medication_name: str
    taken: bool
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    notes: Optional[str] = None
    skip_reason: Optional[str] = None
    family_member_name: Optional[str] = None
    
    class Config:
        from_attributes = True
