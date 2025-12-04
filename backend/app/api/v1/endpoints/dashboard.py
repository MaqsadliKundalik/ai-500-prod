"""
Family Dashboard Endpoints
==========================
Family medication monitoring, adherence tracking
"""

from typing import List, Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user
from app.schemas.dashboard import (
    DashboardSummary,
    FamilyMemberDashboard,
    AdherenceRecord,
    MedicationReminder,
    AlertResponse
)
from app.services.dashboard_service import DashboardService

router = APIRouter()


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    👨‍👩‍👧 Get family dashboard summary.
    
    Returns:
    - User's medication adherence
    - Family members' status
    - Upcoming reminders
    - Recent alerts
    """
    dashboard_service = DashboardService(db)
    summary = await dashboard_service.get_summary(current_user.id)
    return summary


@router.get("/family/{member_id}", response_model=FamilyMemberDashboard)
async def get_family_member_dashboard(
    member_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed dashboard for a family member.
    """
    dashboard_service = DashboardService(db)
    dashboard = await dashboard_service.get_member_dashboard(
        user_id=current_user.id,
        member_id=member_id
    )
    
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Family member not found"
        )
    
    return dashboard


@router.get("/adherence", response_model=List[AdherenceRecord])
async def get_adherence_history(
    member_id: Optional[str] = None,
    start_date: date = Query(None),
    end_date: date = Query(None),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get medication adherence history.
    
    - **member_id**: Optional family member ID (default: current user)
    - **start_date**: Start of date range
    - **end_date**: End of date range
    """
    dashboard_service = DashboardService(db)
    history = await dashboard_service.get_adherence_history(
        user_id=current_user.id,
        member_id=member_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return history


@router.post("/adherence/log")
async def log_medication_taken(
    medication_id: str,
    member_id: Optional[str] = None,
    taken: bool = True,
    notes: Optional[str] = None,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Log that a medication was taken (or missed).
    
    - **medication_id**: User medication ID
    - **member_id**: Optional family member ID
    - **taken**: Whether medication was taken
    - **notes**: Optional notes
    """
    dashboard_service = DashboardService(db)
    await dashboard_service.log_adherence(
        user_id=current_user.id,
        member_id=member_id,
        medication_id=medication_id,
        taken=taken,
        notes=notes
    )
    
    return {"message": "Adherence logged successfully"}


# Reminders
@router.get("/reminders", response_model=List[MedicationReminder])
async def get_reminders(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all medication reminders.
    """
    dashboard_service = DashboardService(db)
    reminders = await dashboard_service.get_reminders(current_user.id)
    return reminders


@router.post("/reminders")
async def create_reminder(
    medication_id: str,
    reminder_time: str = Query(..., description="Time in HH:MM format"),
    repeat: str = Query("daily", description="daily, weekly, specific_days"),
    member_id: Optional[str] = None,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a medication reminder.
    """
    dashboard_service = DashboardService(db)
    reminder = await dashboard_service.create_reminder(
        user_id=current_user.id,
        medication_id=medication_id,
        reminder_time=reminder_time,
        repeat=repeat,
        member_id=member_id
    )
    
    return reminder


@router.delete("/reminders/{reminder_id}")
async def delete_reminder(
    reminder_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a reminder.
    """
    dashboard_service = DashboardService(db)
    await dashboard_service.delete_reminder(current_user.id, reminder_id)
    return {"message": "Reminder deleted"}


# Alerts
@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    unread_only: bool = False,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get alerts and notifications.
    
    Alert types:
    - missed_dose: Family member missed a medication
    - interaction_warning: New interaction detected
    - refill_reminder: Medication running low
    - recall_alert: Medication batch recalled
    """
    dashboard_service = DashboardService(db)
    alerts = await dashboard_service.get_alerts(
        current_user.id,
        unread_only=unread_only
    )
    return alerts


@router.put("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Mark an alert as read.
    """
    dashboard_service = DashboardService(db)
    await dashboard_service.mark_alert_read(current_user.id, alert_id)
    return {"message": "Alert marked as read"}
