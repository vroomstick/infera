# config/settings.py
"""
Centralized configuration and logging utilities.
Loads environment variables from .env file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (monorepo structure)
# backend/config/settings.py -> go up 3 levels to find .env at root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Also try loading from backend/ folder if running from there
backend_env = Path(__file__).parent.parent / ".env"
if backend_env.exists():
    load_dotenv(dotenv_path=backend_env, override=True)


class Settings:
    """Application settings loaded from environment variables."""
    
    # PostgreSQL is required - no SQLite fallback
    DEFAULT_DATABASE_URL = "postgresql://infera:infera_dev_password@localhost:5432/infera"
    
    def __init__(self):
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", self.DEFAULT_DATABASE_URL)
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Security settings
        self.API_KEY: str = os.getenv("INFERA_API_KEY", "")  # Optional API auth
        self.CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
        self.RATE_LIMIT: str = os.getenv("RATE_LIMIT", "60/minute")
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        
        # Validate all settings at startup
        self._validate_all_settings()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def require_api_key(self) -> bool:
        """Check if API key authentication is enabled."""
        return bool(self.API_KEY)
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL (always True in v4+)."""
        return self.DATABASE_URL.startswith("postgresql")
    
    def _validate_all_settings(self) -> None:
        """
        Validate all settings at startup.
        
        Raises ValueError with clear messages if any required settings are invalid.
        This prevents runtime surprises from misconfiguration.
        """
        errors = []
        
        # Validate DATABASE_URL
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL is required. Set it in .env or environment variables.")
        elif not self.DATABASE_URL.startswith("postgresql"):
            errors.append(
                f"DATABASE_URL must start with 'postgresql://'. "
                f"Got: {self.DATABASE_URL[:30]}... (truncated for security)"
            )
        else:
            # Validate URL format (basic check)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.DATABASE_URL)
                if not parsed.hostname:
                    errors.append("DATABASE_URL must include a hostname")
                if not parsed.path or parsed.path == '/':
                    errors.append("DATABASE_URL must include a database name")
            except Exception as e:
                errors.append(f"DATABASE_URL format invalid: {e}")
        
        # Validate LOG_LEVEL
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}. Got: {self.LOG_LEVEL}"
            )
        
        # Validate ENVIRONMENT
        valid_environments = ['development', 'production', 'testing']
        if self.ENVIRONMENT.lower() not in valid_environments:
            errors.append(
                f"ENVIRONMENT must be one of {valid_environments}. Got: {self.ENVIRONMENT}"
            )
        
        # Validate RATE_LIMIT format (basic check)
        if self.RATE_LIMIT:
            # Should be like "60/minute" or "100/hour"
            if '/' not in self.RATE_LIMIT:
                errors.append(
                    f"RATE_LIMIT must be in format 'N/unit' (e.g., '60/minute'). Got: {self.RATE_LIMIT}"
                )
            else:
                try:
                    rate, unit = self.RATE_LIMIT.split('/')
                    int(rate)  # Should be a number
                    if unit not in ['second', 'minute', 'hour', 'day']:
                        errors.append(
                            f"RATE_LIMIT unit must be one of: second, minute, hour, day. Got: {unit}"
                        )
                except ValueError:
                    errors.append(
                        f"RATE_LIMIT rate must be a number. Got: {self.RATE_LIMIT}"
                    )
        
        # Warn about missing optional but recommended settings
        warnings = []
        if not self.OPENAI_API_KEY and self.ENVIRONMENT.lower() == 'production':
            warnings.append(
                "WARNING: OPENAI_API_KEY not set. GPT summarization will not work."
            )
        
        # Raise error if any validation failed
        if errors:
            error_msg = "Configuration validation failed:\n"
            error_msg += "\n".join(f"  ❌ {e}" for e in errors)
            if warnings:
                error_msg += "\n\nWarnings:\n"
                error_msg += "\n".join(f"  ⚠️  {w}" for w in warnings)
            raise ValueError(error_msg)
        
        # Log warnings but don't fail
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(warning)


# Global settings instance
settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure handler
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Configure format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    return logger


