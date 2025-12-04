"""
Gamification Endpoints
======================
Points, badges, leaderboards, rewards
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user
from app.schemas.gamification import (
    UserPointsResponse,
    BadgeResponse,
    LeaderboardEntry,
    RewardResponse,
    AchievementResponse
)
from app.services.gamification_service import GamificationService

router = APIRouter()


@router.get("/points", response_model=UserPointsResponse)
async def get_user_points(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    🎮 Get current user's points and level.
    
    Returns:
    - Total points
    - Current level
    - Points to next level
    - Weekly points
    """
    gamification_service = GamificationService(db)
    points = await gamification_service.get_user_points(current_user.id)
    return points


@router.get("/points/history")
async def get_points_history(
    limit: int = Query(50, ge=1, le=100),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get points earning history.
    """
    gamification_service = GamificationService(db)
    history = await gamification_service.get_points_history(current_user.id, limit)
    return history


@router.get("/badges", response_model=List[BadgeResponse])
async def get_user_badges(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    🏅 Get user's earned badges.
    """
    gamification_service = GamificationService(db)
    badges = await gamification_service.get_user_badges(current_user.id)
    return badges


@router.get("/badges/available", response_model=List[BadgeResponse])
async def get_available_badges(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all available badges and progress toward earning them.
    """
    gamification_service = GamificationService(db)
    badges = await gamification_service.get_available_badges(current_user.id)
    return badges


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    period: str = Query("weekly", description="weekly, monthly, all_time"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    🏆 Get the points leaderboard.
    
    - **period**: Time period (weekly, monthly, all_time)
    - **limit**: Number of entries
    """
    gamification_service = GamificationService(db)
    leaderboard = await gamification_service.get_leaderboard(period, limit)
    return leaderboard


@router.get("/leaderboard/my-rank")
async def get_my_rank(
    period: str = Query("weekly", description="weekly, monthly, all_time"),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's rank on the leaderboard.
    """
    gamification_service = GamificationService(db)
    rank = await gamification_service.get_user_rank(current_user.id, period)
    return rank


@router.get("/achievements", response_model=List[AchievementResponse])
async def get_achievements(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's achievements and progress.
    """
    gamification_service = GamificationService(db)
    achievements = await gamification_service.get_achievements(current_user.id)
    return achievements


# Rewards
@router.get("/rewards", response_model=List[RewardResponse])
async def get_available_rewards(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    🎁 Get available rewards to redeem.
    
    Rewards include:
    - Pharmacy discount codes
    - Premium features
    - Partner offers
    """
    gamification_service = GamificationService(db)
    rewards = await gamification_service.get_available_rewards(current_user.id)
    return rewards


@router.post("/rewards/{reward_id}/redeem")
async def redeem_reward(
    reward_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Redeem a reward using points.
    """
    gamification_service = GamificationService(db)
    
    result = await gamification_service.redeem_reward(
        current_user.id,
        reward_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result


@router.get("/streaks")
async def get_streaks(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's current streaks.
    
    Streaks:
    - Daily scan streak
    - Medication adherence streak
    - Weekly activity streak
    """
    gamification_service = GamificationService(db)
    streaks = await gamification_service.get_streaks(current_user.id)
    return streaks
