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
        
        # Validate database URL
        if not self.DATABASE_URL.startswith("postgresql"):
            raise ValueError(
                "PostgreSQL is required. Set DATABASE_URL=postgresql://... "
                "or run: docker compose up -d db"
            )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def require_api_key(self) -> bool:
        """Check if API key authentication is enabled."""
        return bool(self.API_KEY)


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


