"""
User Model
==========
User accounts, authentication, family members
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum

from app.db.base import BaseModel, SoftDeleteMixin


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class Language(str, enum.Enum):
    UZ = "uz"
    RU = "ru"
    EN = "en"


class User(BaseModel, SoftDeleteMixin):
    """
    User account model.
    """
    __tablename__ = "users"
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone = Column(String(20), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    full_name = Column(String(255), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    
    # Settings
    language = Column(SQLEnum(Language), default=Language.UZ, nullable=False)
    notifications_enabled = Column(Boolean, default=True)
    reminder_time = Column(String(5), default="09:00")  # HH:MM format
    theme = Column(String(20), default="light")
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    
    # Medical profile (for personalized insights)
    medical_conditions = Column(JSONB, default=list)  # ["diabetes", "hypertension"]
    allergies = Column(JSONB, default=list)  # ["penicillin", "aspirin"]
    pregnancy_status = Column(String(20), nullable=True)  # "pregnant", "breastfeeding", None
    age_group = Column(String(20), nullable=True)  # "child", "adult", "elderly"
    
    # Gamification
    total_points = Column(Integer, default=0)
    level = Column(Integer, default=1)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    
    # Timestamps
    last_login_at = Column(DateTime, nullable=True)
    verified_at = Column(DateTime, nullable=True)
    
    # Relationships
    family_members = relationship("FamilyMember", back_populates="owner", cascade="all, delete-orphan")
    medications = relationship("UserMedication", back_populates="user", cascade="all, delete-orphan")
    scans = relationship("Scan", back_populates="user", cascade="all, delete-orphan")
    badges = relationship("UserBadge", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class FamilyMember(BaseModel):
    """
    Family member for medication monitoring.
    """
    __tablename__ = "family_members"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(255), nullable=False)
    relation_type = Column(String(50), nullable=False)  # mother, father, child, spouse
    age = Column(Integer, nullable=True)
    
    # Medical profile
    medical_conditions = Column(JSONB, default=list)
    allergies = Column(JSONB, default=list)
    pregnancy_status = Column(String(20), nullable=True)
    
    # Avatar
    avatar_url = Column(String(500), nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="family_members")
    medications = relationship("UserMedication", back_populates="family_member", cascade="all, delete-orphan")


class UserBadge(BaseModel):
    """
    Badges earned by users.
    """
    __tablename__ = "user_badges"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    badge_id = Column(String(50), nullable=False)  # Reference to badge type
    
    earned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="badges")


class PointsHistory(BaseModel):
    """
    Points earning history for gamification.
    """
    __tablename__ = "points_history"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    points = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)  # scan, adherence, streak, etc.
    description = Column(String(255), nullable=True)
