"""Custom exception classes."""
from typing import Any, Optional


class ServiceUnavailableError(Exception):
    """Raised when a downstream service is unavailable."""

    def __init__(self, service_name: str, message: str) -> None:
        self.service_name = service_name
        self.message = message
        super().__init__(f"Service {service_name} unavailable: {message}")


class ServiceNotFoundError(Exception):
    """Raised when a service is not found in the registry."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        super().__init__(f"Service not found: {service_name}")


class UnsupportedVersionError(Exception):
    """Raised when an unsupported API version is requested."""

    def __init__(self, version: str, supported_versions: list[str]) -> None:
        self.version = version
        self.supported_versions = supported_versions
        super().__init__(
            f"Unsupported API version: {version}. "
            f"Supported versions: {', '.join(supported_versions)}"
        )
