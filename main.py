"""
FastAPI Microservice Backend - Main Application
Entry point that loads and registers all service modules.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# Import service modules
from services import deskew_service
from services import algo_trading_service
from services import scheduler_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler_service.start()
    yield
    scheduler_service.stop()


# Create main FastAPI application
app = FastAPI(
    title="FastAPI Microservice Backend",
    version="1.0.0",
    description="Python microservice backend for AI, ML, and analytics workloads",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track application start time
START_TIME = time.time()

# Register service routers
app.include_router(deskew_service.router, prefix="/api/v1")
app.include_router(algo_trading_service.router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "fastapi-microservice-backend",
        "version": "1.0.0",
        "status": "running",
        "services": [
            "deskew"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "deskew": "/api/v1/deskew"
        }
    }


# Global health check
@app.get("/health")
async def health_check():
    """Global health check endpoint."""
    uptime = time.time() - START_TIME
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "services": {
            "deskew": "healthy"
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path),
            "message": "The requested resource was not found"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
