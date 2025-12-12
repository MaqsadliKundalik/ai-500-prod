"""
User Service
============
Business logic for user management, authentication, family members
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User, FamilyMember, UserBadge, PointsHistory
from app.schemas.user import UserCreate, UserUpdate, FamilyMemberCreate
from app.schemas.auth import UserResponse
from app.core.security import get_password_hash, verify_password


class UserService:
    """Service for user-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
            .where(User.is_deleted == False)
            .options(selectinload(User.family_members))
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(
            select(User)
            .where(User.email == email)
            .where(User.is_deleted == False)
        )
        return result.scalar_one_or_none()
    
    async def get_by_phone(self, phone: str) -> Optional[User]:
        """Get user by phone number."""
        result = await self.db.execute(
            select(User)
            .where(User.phone == phone)
            .where(User.is_deleted == False)
        )
        return result.scalar_one_or_none()
    
    async def create(self, user_data: UserCreate) -> User:
        """Create a new user."""
        from app.models.user import UserRole, Language
        
        user = User(
            email=user_data.email,
            phone=user_data.phone,
            hashed_password=get_password_hash(user_data.password),
            full_name=user_data.full_name,
            language=Language(user_data.language),  # Convert string to enum
            # Gamification defaults
            total_points=0,
            level=1,
            current_streak=0,
            longest_streak=0,
            # Status defaults
            is_active=True,
            is_verified=False,
            role=UserRole.USER
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def update(self, user_id: str, user_data: UserUpdate) -> Optional[User]:
        """Update user profile."""
        from app.models.user import Language
        
        user = await self.get_by_id(user_id)
        if not user:
            return None
        
        update_data = user_data.model_dump(exclude_unset=True)
        
        # Convert language string to enum if present
        if 'language' in update_data and update_data['language']:
            update_data['language'] = Language(update_data['language'])
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = await self.get_by_email(email)
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        await self.db.flush()
        
        return user
    
    async def soft_delete(self, user_id: str) -> bool:
        """Soft delete a user account."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        user.is_deleted = True
        user.deleted_at = datetime.utcnow()
        user.is_active = False
        await self.db.flush()
        
        return True
    
    # Family Members
    async def get_family_members(self, user_id: str) -> List[FamilyMember]:
        """Get all family members for a user."""
        result = await self.db.execute(
            select(FamilyMember)
            .where(FamilyMember.user_id == user_id)
            .order_by(FamilyMember.created_at)
        )
        return result.scalars().all()
    
    async def add_family_member(
        self,
        user_id: str,
        member_data: FamilyMemberCreate
    ) -> FamilyMember:
        """Add a family member."""
        member = FamilyMember(
            user_id=user_id,
            name=member_data.name,
            relationship=member_data.relationship,
            age=member_data.age,
            medical_conditions=member_data.medical_conditions,
            allergies=member_data.allergies,
            pregnancy_status=member_data.pregnancy_status
        )
        
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        
        return member
    
    async def remove_family_member(self, user_id: str, member_id: str) -> bool:
        """Remove a family member."""
        result = await self.db.execute(
            select(FamilyMember)
            .where(FamilyMember.id == member_id)
            .where(FamilyMember.user_id == user_id)
        )
        member = result.scalar_one_or_none()
        
        if not member:
            return False
        
        await self.db.delete(member)
        await self.db.flush()
        
        return True
    
    # Settings
    async def update_settings(self, user_id: str, settings: dict) -> bool:
        """Update user settings."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        
        # Update allowed settings
        allowed_fields = [
            'language', 'notifications_enabled', 
            'reminder_time', 'theme'
        ]
        
        for field, value in settings.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        await self.db.flush()
        
        return True
    
    # Gamification
    async def add_points(
        self,
        user_id: str,
        points: int,
        action: str,
        description: Optional[str] = None
    ) -> int:
        """Add points to user and return new total."""
        user = await self.get_by_id(user_id)
        if not user:
            return 0
        
        # Add points
        user.total_points += points
        
        # Update level (simple formula: level = points // 100 + 1)
        user.level = user.total_points // 100 + 1
        
        # Log points
        points_log = PointsHistory(
            user_id=user_id,
            points=points,
            action=action,
            description=description
        )
        self.db.add(points_log)
        
        await self.db.flush()
        await self.db.refresh(user)
        
        return user.total_points
    
    async def award_badge(self, user_id: str, badge_id: str) -> bool:
        """Award a badge to user."""
        # Check if already earned
        result = await self.db.execute(
            select(UserBadge)
            .where(UserBadge.user_id == user_id)
            .where(UserBadge.badge_id == badge_id)
        )
        
        if result.scalar_one_or_none():
            return False  # Already has badge
        
        badge = UserBadge(
            user_id=user_id,
            badge_id=badge_id
        )
        self.db.add(badge)
        await self.db.flush()
        
        return True
