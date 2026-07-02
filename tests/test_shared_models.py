"""Tests for shared Pydantic models."""
from shared.models import (
    ErrorResponse,
    HealthCheckResponse,
    RetryPolicy,
    ServiceConfig,
)


def test_error_response_creation() -> None:
    """Test ErrorResponse model creation."""
    error = ErrorResponse(
        error="Test error", details={"field": "value"}, service="test-service"
    )

    assert error.error == "Test error"
    assert error.details == {"field": "value"}
    assert error.service == "test-service"
    assert error.timestamp is not None
    assert error.request_id is None


def test_health_check_response_creation() -> None:
    """Test HealthCheckResponse model creation."""
    health = HealthCheckResponse(
        service="test-service", status="healthy", version="1.0.0", uptime_seconds=123.45
    )

    assert health.service == "test-service"
    assert health.status == "healthy"
    assert health.version == "1.0.0"
    assert health.uptime_seconds == 123.45
    assert health.checks == {}


def test_service_config_creation() -> None:
    """Test ServiceConfig model creation."""
    config = ServiceConfig(
        name="test-service", url="http://localhost:8001", protocols=["rest", "grpc"]
    )

    assert config.name == "test-service"
    assert config.url == "http://localhost:8001"
    assert config.protocols == ["rest", "grpc"]
    assert config.health_check_path == "/health"
    assert config.timeout_seconds == 30
    assert config.retry_policy is None


def test_retry_policy_defaults() -> None:
    """Test RetryPolicy model with defaults."""
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.backoff_seconds == 1.0
    assert policy.backoff_multiplier == 2.0


def test_service_config_with_retry_policy() -> None:
    """Test ServiceConfig with custom retry policy."""
    retry = RetryPolicy(max_attempts=5, backoff_seconds=2.0, backoff_multiplier=3.0)
    config = ServiceConfig(
        name="test-service",
        url="http://localhost:8001",
        protocols=["rest"],
        retry_policy=retry,
    )

    assert config.retry_policy is not None
    assert config.retry_policy.max_attempts == 5
    assert config.retry_policy.backoff_seconds == 2.0
    assert config.retry_policy.backoff_multiplier == 3.0
