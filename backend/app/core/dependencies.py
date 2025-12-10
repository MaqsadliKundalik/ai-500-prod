"""
Sentinel-RX Dependency Injection
================================
FastAPI dependencies for authentication, database sessions, etc.
"""

from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import oauth2_scheme, verify_token
from app.db.session import async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user_id(
    token: Optional[str] = Depends(oauth2_scheme)
) -> str:
    """
    Dependency to get current authenticated user ID from token.
    
    Args:
        token: JWT access token from Authorization header
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    from app.core.config import get_settings
    settings = get_settings()
    
    # Bypass auth if disabled (for testing) or no token provided
    if settings.disable_auth or not token:
        return "00000000-0000-0000-0000-000000000000"  # Dummy user ID
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    user_id = verify_token(token, token_type="access")
    
    if user_id is None:
        raise credentials_exception
    
    return user_id


async def get_current_user(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Dependency to get current authenticated user object.
    
    Args:
        user_id: User ID from token
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If user not found
    """
    from app.core.config import get_settings
    from app.services.user_service import UserService
    from app.models.user import User
    from uuid import UUID
    
    settings = get_settings()
    
    # Bypass auth - return dummy user object
    if settings.disable_auth:
        dummy_user = User(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            email="demo@example.com",
            phone="+998901234567",
            full_name="Demo User",
            is_active=True,
            is_verified=True
        )
        return dummy_user
    
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


async def get_current_active_user(
    current_user = Depends(get_current_user)
):
    """
    Dependency to get current active (non-disabled) user.
    
    Args:
        current_user: Current user object
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is disabled
    """
    from app.core.config import get_settings
    settings = get_settings()
    
    # Bypass auth - always return user as active
    if settings.disable_auth:
        return current_user
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return current_user


def get_optional_user_id(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[str]:
    """
    Dependency to optionally get user ID (for public endpoints).
    
    Args:
        token: Optional JWT token
        
    Returns:
        User ID if token valid, None otherwise
    """
    if token is None:
        return None
    
    return verify_token(token, token_type="access")
