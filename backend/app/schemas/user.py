"""
User Schemas
============
Request/Response models for user endpoints
"""

from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator


class UserCreate(BaseModel):
    """Schema for user registration."""
    
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=255)
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9]{9,15}$")
    language: str = Field(default="uz", pattern=r"^(uz|ru|en)$")


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9]{9,15}$")
    avatar_url: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    language: Optional[str] = Field(None, pattern=r"^(uz|ru|en)$")
    medical_conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    pregnancy_status: Optional[str] = None


class UserResponse(BaseModel):
    """Basic user response schema."""
    
    id: str
    email: str
    phone: Optional[str] = None
    full_name: str
    avatar_url: Optional[str] = None
    language: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    @field_validator("id", mode="before")
    @classmethod
    def convert_uuid_to_string(cls, v):
        """Convert UUID to string."""
        return str(v)
    
    model_config = ConfigDict(from_attributes=True, json_encoders={UUID: str})


class UserProfileResponse(BaseModel):
    """Detailed user profile response."""
    
    id: str
    email: str
    phone: Optional[str] = None
    full_name: str
    avatar_url: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    language: str
    notifications_enabled: bool
    reminder_time: str
    theme: str
    medical_conditions: List[str] = []
    allergies: List[str] = []
    pregnancy_status: Optional[str] = None
    total_points: int
    level: int
    current_streak: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True, json_encoders={UUID: str})


class FamilyMemberCreate(BaseModel):
    """Schema for adding a family member."""
    
    name: str = Field(..., min_length=2, max_length=255)
    relationship: str = Field(..., min_length=2, max_length=50)
    age: Optional[int] = Field(None, ge=0, le=150)
    medical_conditions: List[str] = []
    allergies: List[str] = []
    pregnancy_status: Optional[str] = None


class FamilyMemberResponse(BaseModel):
    """Family member response schema."""
    
    id: str
    name: str
    relationship: str
    age: Optional[int] = None
    avatar_url: Optional[str] = None
    medical_conditions: List[str] = []
    allergies: List[str] = []
    pregnancy_status: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class FamilyMemberUpdate(BaseModel):
    """Schema for updating a family member."""
    
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    relationship: Optional[str] = Field(None, min_length=2, max_length=50)
    age: Optional[int] = Field(None, ge=0, le=150)
    medical_conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    pregnancy_status: Optional[str] = None
