"""Shared configuration utilities."""
import os
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceConfig(BaseSettings):
    """Base configuration for all microservices."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Algo Trading Service
    massive_api_key: str
    postgres_user_algo_trading: str
    postgres_password_algo_trading: str
    postgres_db_algo_trading: str
    postgres_host_algo_trading: str
    postgres_port_algo_trading: int

    ibkr_host: str
    ibkr_port: int = 7497
    ibkr_client_id: int = 108

    # Alpaca (paper trading / staging broker)
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_paper: bool = True

    # Pairs trading risk limits
    max_concurrent_pairs: int = 5
    max_pair_pct: float = 0.02

    # Secrets (loaded from Docker secrets or env vars)
    api_key: Optional[str] = Field(None, alias="API_KEY")
    database_url: Optional[str] = Field(None, alias="DATABASE_URL")

    @classmethod
    def load_secret(cls, secret_name: str) -> Optional[str]:
        """Load secret from Docker secrets or environment variable.

        Args:
            secret_name: Name of the secret to load

        Returns:
            Secret value or None if not found
        """
        # Try to load from Docker secrets first
        secret_path = f"/run/secrets/{secret_name}"
        if os.path.exists(secret_path):
            with open(secret_path) as f:
                return f.read().strip()

        # Fall back to environment variable
        return os.getenv(secret_name)

    def validate_secrets(self, required_secrets: list[str]) -> None:
        """Validate that required secrets are present.

        Args:
            required_secrets: List of required secret names

        Raises:
            ValueError: If any required secret is missing
        """
        missing = []
        for secret in required_secrets:
            value = getattr(self, secret, None) or self.load_secret(secret.upper())
            if not value:
                missing.append(secret)

        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")
