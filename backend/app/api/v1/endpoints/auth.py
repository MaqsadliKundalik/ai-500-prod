"""
Authentication Endpoints
========================
Login, register, token refresh, password reset
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_password,
    get_password_hash
)
from app.core.config import settings
from app.schemas.auth import (
    Token,
    TokenRefresh,
    UserCreate,
    UserResponse,
    PasswordReset,
    PasswordResetConfirm
)
from app.services.user_service import UserService

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    - **email**: Valid email address
    - **phone**: Phone number (optional, for SMS verification)
    - **password**: Minimum 8 characters
    - **full_name**: User's full name
    - **language**: Preferred language (uz, ru, en)
    """
    user_service = UserService(db)
    
    # Check if email already exists
    existing_user = await user_service.get_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = await user_service.create(user_data)
    
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password to obtain JWT tokens.
    
    Returns access_token and refresh_token.
    """
    user_service = UserService(db)
    
    # Authenticate user
    user = await user_service.authenticate(
        email=form_data.username,
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    user_id = verify_token(token_data.refresh_token, token_type="refresh")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify user still exists and is active
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token = create_access_token(data={"sub": user_id})
    refresh_token = create_refresh_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/password-reset")
async def request_password_reset(
    data: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """
    Request a password reset email/SMS.
    """
    user_service = UserService(db)
    user = await user_service.get_by_email(data.email)
    
    # Always return success to prevent email enumeration
    if user:
        # TODO: Send password reset email/SMS
        pass
    
    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm password reset with token and new password.
    """
    # TODO: Implement token verification and password update
    return {"message": "Password has been reset successfully"}


@router.post("/logout")
async def logout():
    """
    Logout user (client should discard tokens).
    
    Note: With JWT, logout is handled client-side.
    For extra security, implement token blacklisting with Redis.
    """
    return {"message": "Successfully logged out"}
