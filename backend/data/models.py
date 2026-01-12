# data/models.py
"""
SQLAlchemy ORM models for Infera.

Supports both SQLite (development) and PostgreSQL + pgvector (production).
"""

import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Check if we're using PostgreSQL (enables pgvector)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./infera.db")
USE_PGVECTOR = DATABASE_URL.startswith("postgresql")

if USE_PGVECTOR:
    try:
        from pgvector.sqlalchemy import Vector
        VECTOR_DIM = 384  # all-MiniLM-L6-v2 dimension
    except ImportError:
        USE_PGVECTOR = False

Base = declarative_base()


class Company(Base):
    """Company/ticker metadata."""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    industry = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    filings = relationship("Filing", back_populates="company", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Company(ticker='{self.ticker}', name='{self.name}')>"


class Filing(Base):
    """SEC 10-K filing metadata."""
    __tablename__ = "filings"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    filing_type = Column(String(20), default="10-K")  # 10-K, 10-Q, etc.
    filing_date = Column(DateTime, nullable=True)
    accession_number = Column(String(50), nullable=True)
    source_file = Column(String(500), nullable=True)  # Original HTML file path
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="filings")
    sections = relationship("Section", back_populates="filing", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="filing", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Filing(company_id={self.company_id}, type='{self.filing_type}', date='{self.filing_date}')>"


class Section(Base):
    """Extracted section from a filing (e.g., Item 1A Risk Factors)."""
    __tablename__ = "sections"
    
    id = Column(Integer, primary_key=True, index=True)
    filing_id = Column(Integer, ForeignKey("filings.id"), nullable=False)
    section_type = Column(String(50), nullable=False)  # "Item 1A", "Item 7", etc.
    raw_text = Column(Text, nullable=True)  # Full section text
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    filing = relationship("Filing", back_populates="sections")
    paragraphs = relationship("Paragraph", back_populates="section", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Section(filing_id={self.filing_id}, type='{self.section_type}')>"


class Paragraph(Base):
    """Individual paragraph from a section with embedding and score."""
    __tablename__ = "paragraphs"
    
    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    text = Column(Text, nullable=False)
    position = Column(Integer, nullable=True)  # Order within section
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    section = relationship("Section", back_populates="paragraphs")
    scores = relationship("Score", back_populates="paragraph", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Paragraph(section_id={self.section_id}, position={self.position})>"


class Score(Base):
    """Risk score for a paragraph."""
    __tablename__ = "scores"
    
    id = Column(Integer, primary_key=True, index=True)
    paragraph_id = Column(Integer, ForeignKey("paragraphs.id"), nullable=False)
    method = Column(String(50), nullable=False)  # "tfidf", "embedding", etc.
    score = Column(Float, nullable=False)  # 0.0 to 1.0
    embedding = Column(LargeBinary, nullable=True)  # Serialized numpy array (SQLite)
    top_terms = Column(Text, nullable=True)  # JSON list of contributing terms
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    paragraph = relationship("Paragraph", back_populates="scores")
    
    def __repr__(self):
        return f"<Score(paragraph_id={self.paragraph_id}, method='{self.method}', score={self.score:.3f})>"


# For PostgreSQL with pgvector, we add a separate table for vector search
# This enables native vector similarity search while maintaining backward compatibility
if USE_PGVECTOR:
    class ScoreVector(Base):
        """
        Vector embeddings for PostgreSQL pgvector search.
        
        This table stores embeddings in native vector format for fast similarity search.
        Linked to Score by paragraph_id for compatibility.
        """
        __tablename__ = "score_vectors"
        
        id = Column(Integer, primary_key=True, index=True)
        paragraph_id = Column(Integer, ForeignKey("paragraphs.id"), nullable=False, unique=True)
        embedding = Column(Vector(VECTOR_DIM), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        def __repr__(self):
            return f"<ScoreVector(paragraph_id={self.paragraph_id})>"


class Summary(Base):
    """GPT-generated summary for a filing."""
    __tablename__ = "summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    filing_id = Column(Integer, ForeignKey("filings.id"), nullable=False)
    section_type = Column(String(50), nullable=False)  # "Item 1A", etc.
    summary_text = Column(Text, nullable=False)
    model = Column(String(50), nullable=True)  # "gpt-4o", etc.
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    filing = relationship("Filing", back_populates="summaries")
    
    def __repr__(self):
        return f"<Summary(filing_id={self.filing_id}, section='{self.section_type}')>"


