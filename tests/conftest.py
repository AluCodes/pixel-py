"""Pytest configuration and shared fixtures."""
import pytest
from hypothesis import settings


# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


@pytest.fixture
def sample_service_config() -> dict:
    """Sample service configuration for testing."""
    return {
        "name": "test-service",
        "url": "http://localhost:8001",
        "protocols": ["rest"],
        "health_check_path": "/health",
        "timeout_seconds": 30,
    }
