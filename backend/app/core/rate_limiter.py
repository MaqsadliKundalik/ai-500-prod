"""
Rate Limiting Middleware
========================
Protect API endpoints from abuse and DDoS attacks
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per minute", "2000 per hour"],
    storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
    strategy="fixed-window"
)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Handle rate limit exceeded errors.
    
    Returns a 429 Too Many Requests response.
    """
    logger.warning(
        f"Rate limit exceeded for {get_remote_address(request)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "ip": get_remote_address(request)
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "message": "Too many requests. Please slow down.",
            "detail": str(exc),
            "retry_after": "60 seconds"
        },
        headers={
            "Retry-After": "60"
        }
    )


# Rate limit decorators for different endpoint types
def limit_scan_endpoints():
    """Rate limit for scan endpoints (resource intensive)"""
    return limiter.limit("30 per minute")


def limit_search_endpoints():
    """Rate limit for search endpoints"""
    return limiter.limit("100 per minute")


def limit_auth_endpoints():
    """Rate limit for authentication endpoints (prevent brute force)"""
    return limiter.limit("10 per minute")


def limit_upload_endpoints():
    """Rate limit for file upload endpoints"""
    return limiter.limit("20 per minute")


def limit_ai_endpoints():
    """Rate limit for AI processing endpoints"""
    return limiter.limit("50 per minute")
