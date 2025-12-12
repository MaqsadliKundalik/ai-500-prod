"""
Error Handlers
==============
Global exception handlers for the application
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import ValidationError

from app.core.exceptions import SentinelRXException

logger = logging.getLogger(__name__)


async def sentinel_rx_exception_handler(request: Request, exc: SentinelRXException):
    """Handle custom Sentinel-RX exceptions."""
    logger.error(
        f"SentinelRX Exception: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": request.url.path
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    # Convert errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        error_dict = {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "input": str(error.get("input")) if error.get("input") is not None else None,
        }
        # Convert ctx if present (may contain non-serializable objects)
        if "ctx" in error:
            error_dict["ctx"] = {
                k: str(v) for k, v in error["ctx"].items()
            }
        errors.append(error_dict)
    
    logger.warning(
        f"Validation error: {errors}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": errors,
            "path": request.url.path
        }
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle SQLAlchemy database errors."""
    logger.error(
        f"Database error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True
    )
    
    # Check for specific error types
    if isinstance(exc, IntegrityError):
        error_msg = str(exc.orig) if hasattr(exc, 'orig') else str(exc)
        
        # Parse user-friendly messages for common constraints
        if 'ix_users_email' in error_msg or 'users_email_key' in error_msg:
            message = "Email address already registered. Please use a different email or login."
            field = "email"
        elif 'ix_users_phone' in error_msg or 'users_phone_key' in error_msg:
            message = "Phone number already registered. Please use a different phone number or login."
            field = "phone"
        elif 'ix_medications_barcode' in error_msg:
            message = "Barcode already exists in the system."
            field = "barcode"
        else:
            message = "Data integrity constraint violated. This record may already exist."
            field = None
        
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "IntegrityError",
                "message": message,
                "field": field,
                "path": request.url.path
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "DatabaseError",
            "message": "An error occurred while processing your request",
            "path": request.url.path
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "path": request.url.path
        }
    )
