"""Shared Pydantic models."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str
    details: Optional[dict[str, Any]] = None
    service: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Standard health check response format."""

    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    checks: dict[str, bool] = Field(default_factory=dict)


class ServiceConfig(BaseModel):
    """Service configuration in registry."""

    name: str
    url: str
    protocols: list[str]
    health_check_path: str = "/health"
    timeout_seconds: int = 30
    retry_policy: Optional["RetryPolicy"] = None


class RetryPolicy(BaseModel):
    """Retry policy configuration."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
