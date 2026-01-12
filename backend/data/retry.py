"""
Database retry logic for handling transient connection failures.

Uses tenacity to automatically retry database operations that fail due to
transient network or connection issues.
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from sqlalchemy.exc import (
    OperationalError,
    DisconnectionError,
    TimeoutError as SQLTimeoutError
)
from config.settings import get_logger

logger = get_logger(__name__)

# Retryable exceptions (transient failures)
RETRYABLE_DB_EXCEPTIONS = (
    OperationalError,
    DisconnectionError,
    SQLTimeoutError,
)


def db_retry(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """
    Decorator for database operations that should retry on transient failures.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        min_wait: Minimum wait time between retries in seconds (default: 1.0)
        max_wait: Maximum wait time between retries in seconds (default: 10.0)
    
    Usage:
        @db_retry()
        def create_filing(db: Session, ...):
            # DB operation
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_DB_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True  # Re-raise the exception after all retries fail
    )

