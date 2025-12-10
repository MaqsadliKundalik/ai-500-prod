"""
Sentinel-RX Main Application
============================
FastAPI application factory and configuration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.logging_config import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.error_handlers import (
    sentinel_rx_exception_handler,
    validation_exception_handler,
    sqlalchemy_exception_handler,
    generic_exception_handler
)
from app.core.exceptions import SentinelRXException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    setup_logging(
        level=settings.log_level,
        log_file="logs/sentinel_rx.log",
        json_format=(settings.environment == "production")
    )
    
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📍 Environment: {settings.environment}")
    print(f"🔗 Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'configured'}")
    
    # Initialize services, connections, ML models here
    # await init_database()
    # await init_redis()
    # await load_ml_models()
    
    yield
    
    # Shutdown
    print(f"👋 Shutting down {settings.app_name}")
    # Close connections, cleanup resources
    # await close_database()
    # await close_redis()


def create_application() -> FastAPI:
    """
    Application factory - creates and configures FastAPI app.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
        🏥 **Sentinel-RX API** - AI-powered Medication Safety Platform
        
        ## Features
        
        * 📸 **Visual Pill Recognition** - Scan medications with camera
        * 💊 **Drug Interaction Detection** - Check for dangerous combinations
        * 💰 **Price Anomaly Detection** - Find overpriced medications
        * 🗺️ **Pharmacy Finder** - Locate nearest legitimate pharmacies
        * 🎤 **Voice Assistant** - Uzbek/Russian/English support
        * 👨‍👩‍👧 **Family Dashboard** - Monitor family medication adherence
        * 🎮 **Gamification** - Rewards for medication compliance
        * ✈️ **Medical Tourism** - Multi-currency, translation support
        
        ## Authentication
        
        All protected endpoints require JWT Bearer token.
        Use `/api/v1/auth/login` to obtain tokens.
        """,
        version=settings.app_version,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    
    # Add exception handlers
    app.add_exception_handler(SentinelRXException, sentinel_rx_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Include API router
    app.include_router(api_router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint - API health check."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "healthy",
            "environment": settings.environment,
            "docs": "/api/docs" if settings.debug else "disabled"
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Detailed health check endpoint for monitoring and Render."""
        from app.db.session import async_session_maker
        from sqlalchemy import text
        from pathlib import Path
        import os
        
        health_data = {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment
        }
        
        # Check database connection
        db_status = "disconnected"
        db_latency = None
        try:
            import time
            start = time.time()
            async with async_session_maker() as session:
                await session.execute(text("SELECT 1"))
            db_latency = round((time.time() - start) * 1000, 2)  # ms
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
            health_data["status"] = "degraded"
        
        health_data["database"] = {
            "status": db_status,
            "latency_ms": db_latency
        }
        
        # Check Redis connection (optional, don't fail if not available)
        redis_status = "not_configured"
        try:
            import redis as redis_client
            redis_host = os.getenv('REDIS_HOST', 'redis')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            r = redis_client.Redis(host=redis_host, port=redis_port, db=0, socket_connect_timeout=1)
            r.ping()
            redis_status = "connected"
        except:
            redis_status = "not_configured"
        
        health_data["redis"] = {
            "status": redis_status
        }
        
        # Check AI models (optional, don't fail if not available)
        model_paths = {
            "pill_recognition": os.getenv('PILL_RECOGNITION_MODEL_PATH', 'models/pill_recognition.pt'),
            "drug_interaction": os.getenv('DDI_MODEL_PATH', 'models/biobert_ddi_model.pt'),
            "price_anomaly": os.getenv('PRICE_ANOMALY_MODEL_PATH', 'models/price_anomaly_model.joblib')
        }
        
        ai_models = {}
        for name, path in model_paths.items():
            ai_models[name] = Path(path).exists()
        
        models_loaded = sum(ai_models.values())
        ai_status = f"{models_loaded}/{len(ai_models)} models available"
        
        health_data["ai_models"] = {
            "status": ai_status,
            "details": ai_models
        }
        
        return health_data
    
    return app


# Create application instance
app = create_application()
