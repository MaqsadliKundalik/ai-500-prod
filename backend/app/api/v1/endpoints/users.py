"""
User Management Endpoints
=========================
User profile, settings, family members
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.dependencies import get_db, get_current_active_user
from app.schemas.user import (
    UserResponse,
    UserUpdate,
    UserProfileResponse,
    FamilyMemberCreate,
    FamilyMemberResponse
)
from app.services.user_service import UserService
from app.models.user import User
from app.models.medication import Medication
from app.models.scan import Scan

router = APIRouter()


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user = Depends(get_current_active_user)
):
    """
    Get current authenticated user's profile.
    """
    # Convert UUID to string for response
    return UserProfileResponse(
        id=str(current_user.id),
        email=current_user.email,
        phone=current_user.phone,
        full_name=current_user.full_name,
        avatar_url=current_user.avatar_url,
        date_of_birth=current_user.date_of_birth,
        language=current_user.language,
        notifications_enabled=current_user.notifications_enabled,
        reminder_time=current_user.reminder_time,
        theme=current_user.theme,
        medical_conditions=current_user.medical_conditions or [],
        allergies=current_user.allergies or [],
        pregnancy_status=current_user.pregnancy_status,
        total_points=current_user.total_points,
        level=current_user.level,
        current_streak=current_user.current_streak,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user's profile.
    """
    user_service = UserService(db)
    updated_user = await user_service.update(current_user.id, user_data)
    return updated_user


@router.delete("/me")
async def delete_current_user(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete current user's account (soft delete).
    """
    user_service = UserService(db)
    await user_service.soft_delete(current_user.id)
    return {"message": "Account deleted successfully"}


# Family Members Management
@router.get("/me/family", response_model=List[FamilyMemberResponse])
async def get_family_members(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all family members linked to current user.
    """
    user_service = UserService(db)
    family = await user_service.get_family_members(current_user.id)
    return family


@router.post("/me/family", response_model=FamilyMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_family_member(
    member_data: FamilyMemberCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a family member (for medication monitoring).
    
    - **name**: Family member's name
    - **relationship**: e.g., "mother", "father", "child"
    - **age**: Age for dosage recommendations
    """
    user_service = UserService(db)
    member = await user_service.add_family_member(current_user.id, member_data)
    return member


@router.delete("/me/family/{member_id}")
async def remove_family_member(
    member_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a family member.
    """
    user_service = UserService(db)
    await user_service.remove_family_member(current_user.id, member_id)
    return {"message": "Family member removed"}


# Settings
@router.get("/me/settings")
async def get_user_settings(
    current_user = Depends(get_current_active_user)
):
    """
    Get user settings (notifications, language, etc.).
    """
    return {
        "language": current_user.language,
        "notifications_enabled": current_user.notifications_enabled,
        "reminder_time": current_user.reminder_time,
        "theme": current_user.theme
    }


@router.put("/me/settings")
async def update_user_settings(
    settings: dict,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user settings.
    """
    user_service = UserService(db)
    await user_service.update_settings(current_user.id, settings)
    return {"message": "Settings updated"}


@router.get("/stats")
async def get_system_stats(
    db: AsyncSession = Depends(get_db)
):
    """
    📊 Get system statistics (public endpoint).
    
    Returns:
    - Total users
    - Total medications
    - Total scans
    """
    # Count users
    user_count_query = select(func.count(User.id)).where(User.deleted_at.is_(None))
    user_result = await db.execute(user_count_query)
    total_users = user_result.scalar() or 0
    
    # Count medications
    med_count_query = select(func.count(Medication.id)).where(Medication.deleted_at.is_(None))
    med_result = await db.execute(med_count_query)
    total_medications = med_result.scalar() or 0
    
    # Count scans
    scan_count_query = select(func.count(Scan.id))
    scan_result = await db.execute(scan_count_query)
    total_scans = scan_result.scalar() or 0
    
    return {
        "total_users": total_users,
        "total_medications": total_medications,
        "total_scans": total_scans,
        "status": "operational"
    }
