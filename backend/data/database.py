# data/database.py
"""
Database engine, session management, and initialization.
"""

import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, get_logger

logger = get_logger(__name__)


def mask_db_url(url: str) -> str:
    """
    Mask credentials in database URL for safe logging.
    
    Example:
        postgresql://user:secret123@host/db → postgresql://user:***@host/db
    """
    # Match password in URL pattern: ://user:password@
    return re.sub(r'(://[^:]+:)[^@]+(@)', r'\1***\2', url)

# Create engine (PostgreSQL only - no SQLite support)
engine = create_engine(
    settings.DATABASE_URL,
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get a database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables and pgvector extension.
    Call this once at application startup.
    """
    from .models import Base
    from sqlalchemy import text
    
    # Log masked URL to prevent credential leakage
    logger.info(f"Initializing database: {mask_db_url(settings.DATABASE_URL)}")
    
    # Create pgvector extension (required for vector search)
    # Use autocommit mode for DDL statements
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension created/enabled")
        except Exception as e:
            logger.warning(f"Could not create pgvector extension (may already exist): {e}")
            # Continue anyway - extension might already exist
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


