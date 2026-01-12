"""
Data validation and contracts for Infera.

Prevents silent corruption by validating data at API boundary and database write.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError
import re

# Allowed risk categories (from classification_service)
ALLOWED_CATEGORIES = {
    "Cybersecurity",
    "Regulatory",
    "Supply Chain",
    "Competition",
    "Economic",
    "Litigation",
    "Personnel",
    "Intellectual Property",
    "Other"
}


class ParagraphValidation(BaseModel):
    """Validation model for paragraph text."""
    text: str
    
    @validator('text')
    def text_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Paragraph text must be non-empty")
        if len(v.strip()) < 10:
            raise ValueError("Paragraph text must be at least 10 characters")
        return v.strip()


class ScoreValidation(BaseModel):
    """Validation model for risk scores."""
    score: Optional[float] = None
    
    @validator('score')
    def score_in_range(cls, v):
        if v is None:
            return v  # NULL is allowed
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be a number")
        if v < 0 or v > 1:
            raise ValueError(f"Score must be in [0, 1] range, got {v}")
        return float(v)


class CategoryValidation(BaseModel):
    """Validation model for risk categories."""
    category: str
    
    @validator('category')
    def category_in_allowed_set(cls, v):
        if v not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Category must be one of {sorted(ALLOWED_CATEGORIES)}, got '{v}'"
            )
        return v


class EmbeddingValidation(BaseModel):
    """Validation model for embeddings."""
    embedding: List[float]
    
    @validator('embedding')
    def embedding_correct_dimension(cls, v):
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list")
        if len(v) != 768:
            raise ValueError(f"Embedding must be 768-dimensional (FinBERT), got {len(v)}")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numbers")
        return v


class FilingDateValidation(BaseModel):
    """Validation model for filing dates."""
    filing_date: Optional[datetime] = None
    
    @validator('filing_date')
    def date_not_future(cls, v):
        if v is None:
            return v
        if v > datetime.now():
            raise ValueError(f"Filing date cannot be in the future: {v}")
        # Allow dates back to 1900 (reasonable for SEC filings)
        if v < datetime(1900, 1, 1):
            raise ValueError(f"Filing date seems invalid (before 1900): {v}")
        return v


class TickerValidation(BaseModel):
    """Validation model for ticker symbols."""
    ticker: str
    
    @validator('ticker')
    def ticker_format(cls, v):
        if not v:
            raise ValueError("Ticker cannot be empty")
        v_upper = v.upper().strip()
        if len(v_upper) < 1 or len(v_upper) > 5:
            raise ValueError(f"Ticker must be 1-5 characters, got '{v_upper}' (length {len(v_upper)})")
        # Allow alphanumeric and some special chars (e.g., BRK.A, BRK.B)
        if not re.match(r'^[A-Z0-9.]+$', v_upper):
            raise ValueError(f"Ticker must contain only uppercase letters, numbers, and dots, got '{v_upper}'")
        return v_upper


def validate_paragraph_text(text: str) -> str:
    """Validate paragraph text. Raises ValueError if invalid."""
    try:
        validated = ParagraphValidation(text=text)
        return validated.text
    except ValidationError as e:
        raise ValueError(f"Paragraph validation failed: {e}")


def validate_score(score: Optional[float]) -> Optional[float]:
    """Validate risk score. Raises ValueError if invalid."""
    try:
        validated = ScoreValidation(score=score)
        return validated.score
    except ValidationError as e:
        raise ValueError(f"Score validation failed: {e}")


def validate_category(category: str) -> str:
    """Validate risk category. Raises ValueError if invalid."""
    try:
        validated = CategoryValidation(category=category)
        return validated.category
    except ValidationError as e:
        raise ValueError(f"Category validation failed: {e}")


def validate_embedding(embedding: List[float]) -> List[float]:
    """Validate embedding vector. Raises ValueError if invalid."""
    try:
        validated = EmbeddingValidation(embedding=embedding)
        return validated.embedding
    except ValidationError as e:
        raise ValueError(f"Embedding validation failed: {e}")


def validate_filing_date(filing_date: Optional[datetime]) -> Optional[datetime]:
    """Validate filing date. Raises ValueError if invalid."""
    try:
        validated = FilingDateValidation(filing_date=filing_date)
        return validated.filing_date
    except ValidationError as e:
        raise ValueError(f"Filing date validation failed: {e}")


def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol. Raises ValueError if invalid."""
    try:
        validated = TickerValidation(ticker=ticker)
        return validated.ticker
    except ValidationError as e:
        raise ValueError(f"Ticker validation failed: {e}")

