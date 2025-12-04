"""
Authentication Schemas
======================
Request/Response models for auth endpoints
"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class UserCreate(BaseModel):
    """Schema for user registration."""
    
    email: EmailStr
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9]{9,15}$")
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=255)
    language: str = Field(default="uz", pattern=r"^(uz|ru|en)$")
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    """Schema for user response."""
    
    id: str
    email: str
    phone: Optional[str] = None
    full_name: str
    language: str
    is_active: bool
    is_verified: bool
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for token response."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""
    
    refresh_token: str


class PasswordReset(BaseModel):
    """Schema for password reset request."""
    
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
