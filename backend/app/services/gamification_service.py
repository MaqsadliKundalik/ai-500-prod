"""
Gamification Service
====================
Business logic for points, badges, leaderboards, rewards
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserBadge, PointsHistory


class GamificationService:
    """Service for gamification features."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # Badge definitions (can be moved to database later)
    BADGES = {
        "first_scan": {
            "name": "First Scan",
            "description": "Complete your first medication scan",
            "icon_url": "/badges/first_scan.png"
        },
        "scan_master": {
            "name": "Scan Master",
            "description": "Scan 10 medications",
            "icon_url": "/badges/scan_master.png"
        },
        "adherence_hero": {
            "name": "Adherence Hero",
            "description": "7-day medication adherence streak",
            "icon_url": "/badges/adherence_hero.png"
        },
        "family_guardian": {
            "name": "Family Guardian",
            "description": "Add 3 family members",
            "icon_url": "/badges/family_guardian.png"
        },
        "safety_advocate": {
            "name": "Safety Advocate",
            "description": "Check 20 drug interactions",
            "icon_url": "/badges/safety_advocate.png"
        }
    }
    
    async def get_user_points(self, user_id: str) -> dict:
        """Get user's points summary."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        # Calculate points to next level
        current_level_threshold = (user.level - 1) * 100
        next_level_threshold = user.level * 100
        points_to_next = next_level_threshold - user.total_points
        
        # Get weekly points
        week_ago = datetime.utcnow() - timedelta(days=7)
        result = await self.db.execute(
            select(func.sum(PointsHistory.points))
            .where(PointsHistory.user_id == user_id)
            .where(PointsHistory.created_at >= week_ago)
        )
        weekly_points = result.scalar() or 0
        
        # Get monthly points
        month_ago = datetime.utcnow() - timedelta(days=30)
        result = await self.db.execute(
            select(func.sum(PointsHistory.points))
            .where(PointsHistory.user_id == user_id)
            .where(PointsHistory.created_at >= month_ago)
        )
        monthly_points = result.scalar() or 0
        
        return {
            "total_points": user.total_points,
            "weekly_points": int(weekly_points),
            "monthly_points": int(monthly_points),
            "level": user.level,
            "level_name": self._get_level_name(user.level),
            "points_to_next_level": max(0, points_to_next),
            "next_level_threshold": next_level_threshold
        }
    
    def _get_level_name(self, level: int) -> str:
        """Get level name based on level number."""
        if level == 1:
            return "Beginner"
        elif level <= 5:
            return "Novice"
        elif level <= 10:
            return "Intermediate"
        elif level <= 20:
            return "Advanced"
        elif level <= 50:
            return "Expert"
        else:
            return "Master"
    
    async def get_points_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[PointsHistory]:
        """Get user's points earning history."""
        result = await self.db.execute(
            select(PointsHistory)
            .where(PointsHistory.user_id == user_id)
            .order_by(desc(PointsHistory.created_at))
            .limit(limit)
        )
        
        return result.scalars().all()
    
    async def get_user_badges(self, user_id: str) -> List[dict]:
        """Get badges earned by user."""
        result = await self.db.execute(
            select(UserBadge)
            .where(UserBadge.user_id == user_id)
            .order_by(desc(UserBadge.earned_at))
        )
        
        user_badges = result.scalars().all()
        
        return [
            {
                "id": badge.badge_id,
                "name": self.BADGES.get(badge.badge_id, {}).get("name", badge.badge_id),
                "description": self.BADGES.get(badge.badge_id, {}).get("description", ""),
                "icon_url": self.BADGES.get(badge.badge_id, {}).get("icon_url", ""),
                "category": "achievement",
                "is_earned": True,
                "earned_at": badge.earned_at
            }
            for badge in user_badges
        ]
    
    async def get_available_badges(self, user_id: str) -> List[dict]:
        """Get all badges with progress."""
        # Get earned badges
        earned = await self.get_user_badges(user_id)
        earned_ids = {badge["id"] for badge in earned}
        
        # All badges with progress
        all_badges = []
        for badge_id, badge_info in self.BADGES.items():
            is_earned = badge_id in earned_ids
            
            all_badges.append({
                "id": badge_id,
                "name": badge_info["name"],
                "description": badge_info["description"],
                "icon_url": badge_info.get("icon_url", ""),
                "category": "achievement",
                "is_earned": is_earned,
                "earned_at": None if not is_earned else next(
                    (b["earned_at"] for b in earned if b["id"] == badge_id), None
                ),
                "progress": 100.0 if is_earned else 0.0,  # TODO: Calculate actual progress
                "requirement": badge_info.get("description", "")
            })
        
        return all_badges
    
    async def get_leaderboard(
        self,
        period: str = "weekly",
        limit: int = 20
    ) -> List[dict]:
        """Get leaderboard."""
        if period == "weekly":
            since = datetime.utcnow() - timedelta(days=7)
            # Get weekly points
            query = select(
                User.id,
                User.full_name,
                User.avatar_url,
                User.level,
                func.sum(PointsHistory.points).label('points')
            ).join(
                PointsHistory, User.id == PointsHistory.user_id
            ).where(
                PointsHistory.created_at >= since
            ).group_by(
                User.id
            ).order_by(
                desc('points')
            ).limit(limit)
        elif period == "monthly":
            since = datetime.utcnow() - timedelta(days=30)
            query = select(
                User.id,
                User.full_name,
                User.avatar_url,
                User.level,
                func.sum(PointsHistory.points).label('points')
            ).join(
                PointsHistory, User.id == PointsHistory.user_id
            ).where(
                PointsHistory.created_at >= since
            ).group_by(
                User.id
            ).order_by(
                desc('points')
            ).limit(limit)
        else:  # all_time
            query = select(
                User.id,
                User.full_name,
                User.avatar_url,
                User.level,
                User.total_points.label('points')
            ).order_by(
                desc(User.total_points)
            ).limit(limit)
        
        result = await self.db.execute(query)
        entries = result.all()
        
        leaderboard = []
        for rank, entry in enumerate(entries, 1):
            leaderboard.append({
                "rank": rank,
                "user_id": str(entry.id),
                "user_name": entry.full_name,
                "avatar_url": entry.avatar_url,
                "points": int(entry.points),
                "level": entry.level,
                "badge_count": 0,  # TODO: Count badges
                "is_current_user": False
            })
        
        return leaderboard
    
    async def get_user_rank(self, user_id: str, period: str = "weekly") -> dict:
        """Get user's rank on leaderboard."""
        leaderboard = await self.get_leaderboard(period, limit=1000)
        
        for entry in leaderboard:
            if entry["user_id"] == user_id:
                return entry
        
        return {
            "rank": 0,
            "user_id": user_id,
            "points": 0,
            "message": "Not ranked yet"
        }
    
    async def get_achievements(self, user_id: str) -> List[dict]:
        """Get user's achievements."""
        # TODO: Implement achievements system
        return []
    
    async def get_available_rewards(self, user_id: str) -> List[dict]:
        """Get available rewards to redeem."""
        # TODO: Implement rewards system
        return [
            {
                "id": "discount_10",
                "name": "10% Pharmacy Discount",
                "description": "Get 10% off at partner pharmacies",
                "image_url": "/rewards/discount.png",
                "points_required": 100,
                "category": "discount",
                "is_available": True,
                "partner_name": "Oson Apteka"
            },
            {
                "id": "premium_month",
                "name": "Premium Features - 1 Month",
                "description": "Unlock premium features for 30 days",
                "image_url": "/rewards/premium.png",
                "points_required": 500,
                "category": "premium",
                "is_available": True
            }
        ]
    
    async def redeem_reward(self, user_id: str, reward_id: str) -> dict:
        """Redeem a reward."""
        # Get user points
        user_points = await self.get_user_points(user_id)
        
        # Get reward info
        rewards = await self.get_available_rewards(user_id)
        reward = next((r for r in rewards if r["id"] == reward_id), None)
        
        if not reward:
            return {"success": False, "message": "Reward not found"}
        
        if user_points["total_points"] < reward["points_required"]:
            return {"success": False, "message": "Insufficient points"}
        
        # Deduct points
        from app.services.user_service import UserService
        user_service = UserService(self.db)
        await user_service.add_points(
            user_id,
            -reward["points_required"],
            "reward_redemption",
            f"Redeemed: {reward['name']}"
        )
        
        # TODO: Generate reward code, send to user
        
        return {
            "success": True,
            "message": "Reward redeemed successfully",
            "code": "DISCOUNT2025",  # TODO: Generate actual code
            "reward": reward
        }
    
    async def get_streaks(self, user_id: str) -> dict:
        """Get user's current streaks."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return {
            "daily_scan_streak": user.current_streak,
            "adherence_streak": 0,  # TODO: Calculate from adherence logs
            "weekly_activity_streak": 0,
            "longest_scan_streak": user.longest_streak,
            "longest_adherence_streak": 0
        }
