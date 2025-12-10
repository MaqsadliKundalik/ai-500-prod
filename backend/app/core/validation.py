"""
Input Validation Utilities
==========================
Sanitize and validate user inputs to prevent injection attacks
"""

import re
from typing import Optional
from fastapi import HTTPException, status
import html
import logging

logger = logging.getLogger(__name__)

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\bUNION\b.*\bSELECT\b)",
    r"(\bSELECT\b.*\bFROM\b)",
    r"(\bINSERT\b.*\bINTO\b)",
    r"(\bUPDATE\b.*\bSET\b)",
    r"(\bDELETE\b.*\bFROM\b)",
    r"(\bDROP\b.*\bTABLE\b)",
    r"(--|\#|\/\*|\*\/)",  # SQL comments
    r"(\bOR\b.*=.*)",
    r"(';|\")",  # Common SQL injection endings
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",  # Event handlers
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
]


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input by removing dangerous characters.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        HTTPException: If input is invalid
    """
    if not value or not isinstance(value, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input must be a non-empty string"
        )
    
    # Trim whitespace
    value = value.strip()
    
    # Check if empty after trimming
    if len(value) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input cannot be empty or only whitespace"
        )
    
    # Check length
    if len(value) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long. Maximum {max_length} characters allowed."
        )
    
    # Check for SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            logger.warning(f"SQL injection attempt detected: {value[:50]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input. Potential SQL injection detected."
            )
    
    # Check for XSS
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            logger.warning(f"XSS attempt detected: {value[:50]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input. Potential XSS attack detected."
            )
    
    # HTML escape
    value = html.escape(value)
    
    return value


def validate_medication_name(name: str) -> str:
    """
    Validate medication name input.
    
    Args:
        name: Medication name
        
    Returns:
        Sanitized medication name
    """
    # Allow letters, numbers, spaces, hyphens, and common characters
    pattern = r"^[a-zA-Z0-9\s\-\.\(\)]+$"
    
    name = sanitize_string(name, max_length=200)
    
    if not re.match(pattern, name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid medication name. Use only letters, numbers, spaces, and basic punctuation."
        )
    
    return name


def validate_search_query(query: str, min_length: int = 2) -> str:
    """
    Validate search query input.
    
    Args:
        query: Search query
        min_length: Minimum query length
        
    Returns:
        Sanitized query
    """
    query = sanitize_string(query, max_length=500)
    
    if len(query) < min_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Search query too short. Minimum {min_length} characters required."
        )
    
    return query


def validate_coordinates(latitude: float, longitude: float) -> tuple:
    """
    Validate geographic coordinates.
    
    Args:
        latitude: Latitude (-90 to 90)
        longitude: Longitude (-180 to 180)
        
    Returns:
        Tuple of (latitude, longitude)
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordinates must be numeric values"
        )
    
    if not (-90 <= latitude <= 90):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Latitude must be between -90 and 90"
        )
    
    if not (-180 <= longitude <= 180):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Longitude must be between -180 and 180"
        )
    
    return latitude, longitude


def validate_pagination(skip: int, limit: int, max_limit: int = 100) -> tuple:
    """
    Validate pagination parameters.
    
    Args:
        skip: Number of items to skip
        limit: Number of items to return
        max_limit: Maximum allowed limit
        
    Returns:
        Tuple of (skip, limit)
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter cannot be negative"
        )
    
    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be at least 1"
        )
    
    if limit > max_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Limit too large. Maximum {max_limit} items allowed."
        )
    
    return skip, limit


def validate_email(email: str) -> str:
    """
    Validate email address format.
    
    Args:
        email: Email address
        
    Returns:
        Sanitized email
    """
    email = sanitize_string(email, max_length=254)
    
    # Basic email regex
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if not re.match(pattern, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address format"
        )
    
    return email.lower()


def validate_password(password: str) -> str:
    """
    Validate password strength.
    
    Args:
        password: Password string
        
    Returns:
        Password if valid
    """
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    if len(password) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too long. Maximum 128 characters."
        )
    
    # Check for at least one letter and one number
    if not re.search(r"[a-zA-Z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one letter"
        )
    
    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one number"
        )
    
    return password


def validate_phone_number(phone: Optional[str]) -> Optional[str]:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number
        
    Returns:
        Sanitized phone number or None
    """
    if not phone:
        return None
    
    # Remove common formatting characters
    phone = re.sub(r"[\s\-\(\)]", "", phone)
    
    # Check if valid phone number (digits and +)
    if not re.match(r"^\+?\d{10,15}$", phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format. Use international format with country code."
        )
    
    return phone
