"""
Gamification Schemas
====================
Request/Response models for gamification endpoints
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class UserPointsResponse(BaseModel):
    """User points summary."""
    
    total_points: int = 0
    weekly_points: int = 0
    monthly_points: int = 0
    level: int = 1
    level_name: str = "Beginner"
    points_to_next_level: int = 0
    next_level_threshold: int = 100
    
    class Config:
        from_attributes = True


class PointsHistoryItem(BaseModel):
    """Points history entry."""
    
    id: str
    points: int
    action: str  # "scan", "adherence", "streak_bonus", "badge_earned"
    description: Optional[str] = None
    earned_at: datetime
    
    class Config:
        from_attributes = True


class BadgeResponse(BaseModel):
    """Badge information."""
    
    id: str
    name: str
    description: str
    icon_url: Optional[str] = None
    category: str  # "scanning", "adherence", "streak", "social"
    is_earned: bool = False
    earned_at: Optional[datetime] = None
    progress: Optional[float] = None  # 0-100% for unearned badges
    requirement: Optional[str] = None  # "Scan 10 medications"
    
    class Config:
        from_attributes = True


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    
    rank: int
    user_id: str
    user_name: str
    avatar_url: Optional[str] = None
    points: int
    level: int
    badge_count: int = 0
    is_current_user: bool = False


class AchievementResponse(BaseModel):
    """Achievement/milestone."""
    
    id: str
    name: str
    description: str
    icon_url: Optional[str] = None
    is_completed: bool = False
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0-100%
    current_value: int = 0
    target_value: int = 0
    reward_points: int = 0
    
    class Config:
        from_attributes = True


class RewardResponse(BaseModel):
    """Available reward."""
    
    id: str
    name: str
    description: str
    image_url: Optional[str] = None
    points_required: int
    category: str  # "discount", "premium", "partner"
    is_available: bool = True
    quantity_available: Optional[int] = None
    partner_name: Optional[str] = None  # Pharmacy name for discounts
    expiry_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class RedeemedReward(BaseModel):
    """Redeemed reward."""
    
    id: str
    reward_id: str
    reward_name: str
    code: Optional[str] = None  # Discount code
    redeemed_at: datetime
    expires_at: Optional[datetime] = None
    is_used: bool = False
    
    class Config:
        from_attributes = True


class StreaksResponse(BaseModel):
    """User streaks."""
    
    daily_scan_streak: int = 0
    adherence_streak: int = 0
    weekly_activity_streak: int = 0
    longest_scan_streak: int = 0
    longest_adherence_streak: int = 0
