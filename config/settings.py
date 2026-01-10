# config/settings.py
"""
Centralized configuration and logging utilities.
Loads environment variables from .env file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./infera.db")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Security settings
        self.API_KEY: str = os.getenv("INFERA_API_KEY", "")  # Optional API auth
        self.CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
        self.RATE_LIMIT: str = os.getenv("RATE_LIMIT", "60/minute")
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        
        # Validate required settings
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def require_api_key(self) -> bool:
        """Check if API key authentication is enabled."""
        return bool(self.API_KEY)
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.DATABASE_URL.startswith("sqlite")
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.DATABASE_URL.startswith("postgresql")


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


