"""
Canonical error codes and exception classes for Infera.

Provides consistent error handling across the application.
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(str, Enum):
    """Canonical error codes for Infera."""
    
    # Ingest errors
    INGEST_001 = "INF-INGEST-001"  # SEC EDGAR download failure
    INGEST_002 = "INF-INGEST-002"  # CIK lookup failed
    
    # Parse errors
    PARSE_001 = "INF-PARSE-001"    # Item 1A section not found
    PARSE_002 = "INF-PARSE-002"    # HTML cleaning failed
    
    # Embedding errors
    EMBED_001 = "INF-EMBED-001"    # Embedding computation failed
    EMBED_002 = "INF-EMBED-002"    # Batch embedding partial failure
    
    # Database errors
    DB_001 = "INF-DB-001"          # Database write failure
    DB_002 = "INF-DB-002"          # Database connection failed
    
    # API errors
    API_001 = "INF-API-001"        # Bad request (validation)
    API_002 = "INF-API-002"        # Unauthorized
    API_003 = "INF-API-003"        # Rate limit exceeded
    
    # Validation errors
    VALID_001 = "INF-VALID-001"    # Data validation failed
    VALID_002 = "INF-VALID-002"    # Ticker format invalid
    VALID_003 = "INF-VALID-003"    # Embedding dimension mismatch


class InferaError(Exception):
    """Base exception for Infera with error codes."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details
            }
        }
    
    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"


# Convenience exception classes
class IngestError(InferaError):
    """Error during SEC EDGAR ingestion."""
    pass


class ParseError(InferaError):
    """Error during HTML parsing or section extraction."""
    pass


class EmbeddingError(InferaError):
    """Error during embedding computation."""
    pass


class DatabaseError(InferaError):
    """Error during database operations."""
    pass


class ValidationError(InferaError):
    """Error during data validation."""
    pass


class APIError(InferaError):
    """Error in API layer."""
    pass

