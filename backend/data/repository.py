# data/repository.py
"""
Repository helpers for database CRUD operations.

Requires PostgreSQL with pgvector extension.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from .models import Company, Filing, Section, Paragraph, Score, Summary, ScoreVector
from config.settings import settings
from services.validation import (
    validate_paragraph_text,
    validate_score,
    validate_embedding,
    validate_filing_date,
    validate_ticker
)
from .retry import db_retry

# PostgreSQL with pgvector is always used now
USE_PGVECTOR = settings.is_postgres


# === Company ===

@db_retry()
def get_or_create_company(db: Session, ticker: str, name: Optional[str] = None) -> Company:
    """Get existing company by ticker or create new one."""
    # Validate ticker
    ticker = validate_ticker(ticker)
    
    company = db.query(Company).filter(Company.ticker == ticker).first()
    if not company:
        company = Company(ticker=ticker, name=name)
        db.add(company)
        db.commit()
        db.refresh(company)
    return company


def get_company_by_ticker(db: Session, ticker: str) -> Optional[Company]:
    """Get company by ticker symbol."""
    return db.query(Company).filter(Company.ticker == ticker.upper()).first()


# === Filing ===

@db_retry()
def create_filing(
    db: Session,
    company_id: int,
    filing_type: str = "10-K",
    filing_date: Optional[datetime] = None,
    accession_number: Optional[str] = None,
    source_file: Optional[str] = None
) -> Filing:
    """Create a new filing record."""
    # Validate filing date
    if filing_date:
        filing_date = validate_filing_date(filing_date)
    
    filing = Filing(
        company_id=company_id,
        filing_type=filing_type,
        filing_date=filing_date,
        accession_number=accession_number,
        source_file=source_file
    )
    db.add(filing)
    db.commit()
    db.refresh(filing)
    return filing


def get_filing_by_id(db: Session, filing_id: int) -> Optional[Filing]:
    """Get filing by ID."""
    return db.query(Filing).filter(Filing.id == filing_id).first()


def get_filings_by_ticker(db: Session, ticker: str) -> List[Filing]:
    """Get all filings for a company."""
    company = get_company_by_ticker(db, ticker)
    if not company:
        return []
    return db.query(Filing).filter(Filing.company_id == company.id).all()


# === Section ===

def create_section(
    db: Session,
    filing_id: int,
    section_type: str,
    raw_text: str
) -> Section:
    """Create a new section record."""
    word_count = len(raw_text.split()) if raw_text else 0
    section = Section(
        filing_id=filing_id,
        section_type=section_type,
        raw_text=raw_text,
        word_count=word_count
    )
    db.add(section)
    db.commit()
    db.refresh(section)
    return section


def get_sections_by_filing(db: Session, filing_id: int) -> List[Section]:
    """Get all sections for a filing."""
    return db.query(Section).filter(Section.filing_id == filing_id).all()


# === Paragraph ===

def create_paragraph(
    db: Session,
    section_id: int,
    text: str,
    position: int
) -> Paragraph:
    """Create a new paragraph record."""
    # Validate paragraph text
    text = validate_paragraph_text(text)
    
    word_count = len(text.split()) if text else 0
    paragraph = Paragraph(
        section_id=section_id,
        text=text,
        position=position,
        word_count=word_count
    )
    db.add(paragraph)
    db.commit()
    db.refresh(paragraph)
    return paragraph


def create_paragraphs_bulk(
    db: Session,
    section_id: int,
    texts: List[str]
) -> List[Paragraph]:
    """Create multiple paragraphs at once."""
    paragraphs = []
    for i, text in enumerate(texts):
        # Validate paragraph text (skip invalid ones with warning)
        try:
            validated_text = validate_paragraph_text(text)
        except ValueError as e:
            from config.settings import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Skipping invalid paragraph at position {i}: {e}")
            continue
        
        p = Paragraph(
            section_id=section_id,
            text=validated_text,
            position=i,
            word_count=len(validated_text.split()) if validated_text else 0
        )
        paragraphs.append(p)
    db.add_all(paragraphs)
    db.commit()
    for p in paragraphs:
        db.refresh(p)
    return paragraphs


def get_paragraphs_by_section(db: Session, section_id: int) -> List[Paragraph]:
    """Get all paragraphs for a section, ordered by position."""
    return db.query(Paragraph).filter(
        Paragraph.section_id == section_id
    ).order_by(Paragraph.position).all()


# === Score ===

def create_score(
    db: Session,
    paragraph_id: int,
    method: str,
    score: float,
    embedding: Optional[bytes] = None,
    top_terms: Optional[str] = None
) -> Score:
    """Create a new score record."""
    # Validate score
    validated_score = validate_score(score)
    if validated_score is None:
        raise ValueError("Score cannot be None for Score model (use NULL handling in application layer)")
    
    score_obj = Score(
        paragraph_id=paragraph_id,
        method=method,
        score=validated_score,
        embedding=embedding,
        top_terms=top_terms
    )
    db.add(score_obj)
    db.commit()
    db.refresh(score_obj)
    return score_obj


def get_top_scored_paragraphs(
    db: Session,
    section_id: int,
    method: str = "embedding",
    limit: int = 5
) -> List[tuple]:
    """Get top scored paragraphs for a section.
    
    Returns list of (Paragraph, Score) tuples.
    """
    results = db.query(Paragraph, Score).join(
        Score, Paragraph.id == Score.paragraph_id
    ).filter(
        Paragraph.section_id == section_id,
        Score.method == method
    ).order_by(Score.score.desc()).limit(limit).all()
    return results


# === Summary ===

def create_summary(
    db: Session,
    filing_id: int,
    section_type: str,
    summary_text: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None
) -> Summary:
    """Create a new summary record."""
    summary = Summary(
        filing_id=filing_id,
        section_type=section_type,
        summary_text=summary_text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


def get_summary_by_filing(
    db: Session,
    filing_id: int,
    section_type: str = "Item 1A"
) -> Optional[Summary]:
    """Get summary for a filing and section type."""
    return db.query(Summary).filter(
        Summary.filing_id == filing_id,
        Summary.section_type == section_type
    ).first()


# === Vector Search (PostgreSQL + pgvector) ===

def create_score_vector(
    db: Session,
    paragraph_id: int,
    embedding: List[float]
) -> "ScoreVector":
    """
    Create a vector embedding record for pgvector search.
    
    Only available when using PostgreSQL with pgvector.
    """
    if not USE_PGVECTOR:
        raise RuntimeError("Vector search requires PostgreSQL with pgvector")
    
    # Validate embedding
    validated_embedding = validate_embedding(embedding)
    
    score_vector = ScoreVector(
        paragraph_id=paragraph_id,
        embedding=validated_embedding
    )
    db.add(score_vector)
    db.commit()
    db.refresh(score_vector)
    return score_vector


def vector_search(
    db: Session,
    query_embedding: List[float],
    limit: int = 10,
    threshold: float = 0.30,
    ticker: Optional[str] = None
) -> List[Tuple[Paragraph, float]]:
    """
    Perform vector similarity search using pgvector.
    
    Args:
        db: Database session
        query_embedding: Query embedding vector
        limit: Maximum results to return
        threshold: Minimum similarity threshold
        ticker: Optional filter by company ticker
        
    Returns:
        List of (Paragraph, similarity_score) tuples
    """
    if not USE_PGVECTOR:
        raise RuntimeError("Vector search requires PostgreSQL with pgvector")
    
    # Build the query with pgvector cosine distance
    query_str = """
        SELECT 
            p.id,
            p.text,
            p.position,
            p.section_id,
            (1 - (sv.embedding <=> :query_embedding::vector)) as similarity
        FROM score_vectors sv
        JOIN paragraphs p ON sv.paragraph_id = p.id
        JOIN sections s ON p.section_id = s.id
        JOIN filings f ON s.filing_id = f.id
        JOIN companies c ON f.company_id = c.id
        WHERE (1 - (sv.embedding <=> :query_embedding::vector)) >= :threshold
    """
    
    if ticker:
        query_str += " AND c.ticker = :ticker"
    
    query_str += " ORDER BY similarity DESC LIMIT :limit"
    
    params = {
        "query_embedding": query_embedding,
        "threshold": threshold,
        "limit": limit
    }
    if ticker:
        params["ticker"] = ticker.upper()
    
    results = db.execute(text(query_str), params).fetchall()
    
    # Convert to Paragraph objects with similarity scores
    output = []
    for row in results:
        para = db.query(Paragraph).filter(Paragraph.id == row[0]).first()
        if para:
            output.append((para, row[4]))  # (paragraph, similarity)
    
    return output


def keyword_search(
    db: Session,
    query: str,
    limit: int = 10,
    ticker: Optional[str] = None
) -> List[Tuple[Paragraph, float]]:
    """
    Perform full-text keyword search using PostgreSQL ts_rank.
    
    Args:
        db: Database session
        query: Search query text
        limit: Maximum results to return
        ticker: Optional filter by company ticker
        
    Returns:
        List of (Paragraph, relevance_score) tuples
    """
    if not USE_PGVECTOR:
        raise RuntimeError("Keyword search requires PostgreSQL")
    
    query_str = """
        SELECT 
            p.id,
            ts_rank(to_tsvector('english', p.text), 
                    websearch_to_tsquery('english', :query)) as relevance
        FROM paragraphs p
        JOIN sections s ON p.section_id = s.id
        JOIN filings f ON s.filing_id = f.id
        JOIN companies c ON f.company_id = c.id
        WHERE to_tsvector('english', p.text) @@ websearch_to_tsquery('english', :query)
    """
    
    if ticker:
        query_str += " AND c.ticker = :ticker"
    
    query_str += " ORDER BY relevance DESC LIMIT :limit"
    
    params = {"query": query, "limit": limit}
    if ticker:
        params["ticker"] = ticker.upper()
    
    results = db.execute(text(query_str), params).fetchall()
    
    # Convert to Paragraph objects with relevance scores
    output = []
    for row in results:
        para = db.query(Paragraph).filter(Paragraph.id == row[0]).first()
        if para:
            output.append((para, row[1]))  # (paragraph, relevance)
    
    return output


def hybrid_search_rrf(
    db: Session,
    query: str,
    query_embedding: List[float],
    limit: int = 10,
    k: int = 60,
    ticker: Optional[str] = None
) -> List[Tuple[Paragraph, float]]:
    """
    Hybrid search combining vector and keyword search with RRF fusion.
    
    Reciprocal Rank Fusion formula: RRF_score = Σ(1 / (k + rank))
    
    Args:
        db: Database session
        query: Search query text (for keyword search)
        query_embedding: Query embedding vector (for vector search)
        limit: Maximum results to return
        k: RRF constant (default 60)
        ticker: Optional filter by company ticker
        
    Returns:
        List of (Paragraph, rrf_score) tuples
    """
    if not USE_PGVECTOR:
        raise RuntimeError("Hybrid search requires PostgreSQL with pgvector")
    
    # Get vector search results
    vector_results = vector_search(
        db, query_embedding, 
        limit=limit * 3,  # Get more candidates for fusion
        threshold=0.20,
        ticker=ticker
    )
    
    # Get keyword search results
    keyword_results = keyword_search(
        db, query,
        limit=limit * 3,
        ticker=ticker
    )
    
    # RRF Fusion
    rrf_scores = {}
    
    # Score vector results
    for rank, (para, _) in enumerate(vector_results):
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[para.id] = {
            "paragraph": para,
            "rrf_score": rrf_score,
            "vector_rank": rank + 1
        }
    
    # Add keyword results (additive)
    for rank, (para, _) in enumerate(keyword_results):
        rrf_score = 1.0 / (k + rank + 1)
        if para.id in rrf_scores:
            rrf_scores[para.id]["rrf_score"] += rrf_score
            rrf_scores[para.id]["keyword_rank"] = rank + 1
        else:
            rrf_scores[para.id] = {
                "paragraph": para,
                "rrf_score": rrf_score,
                "keyword_rank": rank + 1
            }
    
    # Sort by RRF score and return top results
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    return [(r["paragraph"], r["rrf_score"]) for r in sorted_results[:limit]]


