"""
Custom Exceptions
=================
Application-specific exception classes
"""

from fastapi import HTTPException, status


class SentinelRXException(Exception):
    """Base exception for Sentinel-RX application."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ResourceNotFoundException(SentinelRXException):
    """Resource not found exception."""
    
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f" with ID: {resource_id}"
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)


class UnauthorizedException(SentinelRXException):
    """Unauthorized access exception."""
    
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(message, status_code=status.HTTP_401_UNAUTHORIZED)


class ForbiddenException(SentinelRXException):
    """Forbidden access exception."""
    
    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=status.HTTP_403_FORBIDDEN)


class ValidationException(SentinelRXException):
    """Validation error exception."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class DatabaseException(SentinelRXException):
    """Database operation exception."""
    
    def __init__(self, message: str = "Database error occurred"):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ExternalAPIException(SentinelRXException):
    """External API call exception."""
    
    def __init__(self, service: str, message: str = None):
        msg = f"External service '{service}' error"
        if message:
            msg += f": {message}"
        super().__init__(msg, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class FileUploadException(SentinelRXException):
    """File upload exception."""
    
    def __init__(self, message: str = "File upload failed"):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class RateLimitException(SentinelRXException):
    """Rate limit exceeded exception."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=status.HTTP_429_TOO_MANY_REQUESTS)
