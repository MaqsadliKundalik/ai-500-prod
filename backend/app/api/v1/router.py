"""
Sentinel-RX API v1 Router
=========================
Main router that includes all v1 endpoint routers
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    users,
    medications,
    scans,
    interactions,
    pharmacies,
    voice,
    dashboard,
    gamification,
    test_model,
)

api_router = APIRouter()

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# User management endpoints
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

# Medication endpoints
api_router.include_router(
    medications.router,
    prefix="/medications",
    tags=["Medications"]
)

# Scan endpoints (pill recognition, QR scanning)
api_router.include_router(
    scans.router,
    prefix="/scans",
    tags=["Scans"]
)

# Drug interaction endpoints
api_router.include_router(
    interactions.router,
    prefix="/interactions",
    tags=["Drug Interactions"]
)

# Pharmacy finder endpoints
api_router.include_router(
    pharmacies.router,
    prefix="/pharmacies",
    tags=["Pharmacies"]
)

# Voice assistant endpoints
api_router.include_router(
    voice.router,
    prefix="/voice",
    tags=["Voice Assistant"]
)

# Family dashboard endpoints
api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["Family Dashboard"]
)

# Gamification endpoints
api_router.include_router(
    gamification.router,
    prefix="/gamification",
    tags=["Gamification"]
)

# Test/Development endpoints
api_router.include_router(
    test_model.router,
    prefix="/dev",
    tags=["Development"]
)
