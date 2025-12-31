# data/repository.py
"""
Repository helpers for database CRUD operations.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime

from .models import Company, Filing, Section, Paragraph, Score, Summary


# === Company ===

def get_or_create_company(db: Session, ticker: str, name: Optional[str] = None) -> Company:
    """Get existing company by ticker or create new one."""
    company = db.query(Company).filter(Company.ticker == ticker.upper()).first()
    if not company:
        company = Company(ticker=ticker.upper(), name=name)
        db.add(company)
        db.commit()
        db.refresh(company)
    return company


def get_company_by_ticker(db: Session, ticker: str) -> Optional[Company]:
    """Get company by ticker symbol."""
    return db.query(Company).filter(Company.ticker == ticker.upper()).first()


# === Filing ===

def create_filing(
    db: Session,
    company_id: int,
    filing_type: str = "10-K",
    filing_date: Optional[datetime] = None,
    source_file: Optional[str] = None
) -> Filing:
    """Create a new filing record."""
    filing = Filing(
        company_id=company_id,
        filing_type=filing_type,
        filing_date=filing_date,
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
        p = Paragraph(
            section_id=section_id,
            text=text,
            position=i,
            word_count=len(text.split()) if text else 0
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
    score_obj = Score(
        paragraph_id=paragraph_id,
        method=method,
        score=score,
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


