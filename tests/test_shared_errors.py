"""Tests for custom exception classes."""
import pytest

from shared.errors import (
    ServiceNotFoundError,
    ServiceUnavailableError,
    UnsupportedVersionError,
)


def test_service_unavailable_error() -> None:
    """Test ServiceUnavailableError exception."""
    error = ServiceUnavailableError("test-service", "Connection timeout")

    assert error.service_name == "test-service"
    assert error.message == "Connection timeout"
    assert "test-service" in str(error)
    assert "Connection timeout" in str(error)


def test_service_not_found_error() -> None:
    """Test ServiceNotFoundError exception."""
    error = ServiceNotFoundError("unknown-service")

    assert error.service_name == "unknown-service"
    assert "unknown-service" in str(error)


def test_unsupported_version_error() -> None:
    """Test UnsupportedVersionError exception."""
    error = UnsupportedVersionError("2022-01", ["2024-01", "2023-12"])

    assert error.version == "2022-01"
    assert error.supported_versions == ["2024-01", "2023-12"]
    assert "2022-01" in str(error)
    assert "2024-01" in str(error)
    assert "2023-12" in str(error)
