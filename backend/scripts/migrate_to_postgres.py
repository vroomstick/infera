#!/usr/bin/env python3
"""
Migration script: SQLite -> PostgreSQL with pgvector

This script:
1. Reads all data from SQLite database
2. Creates tables in PostgreSQL
3. Migrates all records
4. Re-embeds paragraphs and stores in score_vectors table

Usage:
    # Set environment variables
    export DATABASE_URL="postgresql://infera:password@localhost:5432/infera"
    
    # Run migration
    python scripts/migrate_to_postgres.py --sqlite-path ./infera.db
    
    # Or specify both databases
    python scripts/migrate_to_postgres.py \\
        --sqlite-path ./infera.db \\
        --postgres-url "postgresql://infera:password@localhost:5432/infera"
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from typing import List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from config.settings import get_logger

logger = get_logger(__name__)


def migrate_data(sqlite_path: str, postgres_url: str, re_embed: bool = True):
    """
    Migrate data from SQLite to PostgreSQL.
    
    Args:
        sqlite_path: Path to SQLite database
        postgres_url: PostgreSQL connection URL
        re_embed: Whether to re-embed paragraphs for pgvector
    """
    logger.info(f"Starting migration: {sqlite_path} -> PostgreSQL")
    
    # Connect to SQLite
    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_engine = create_engine(sqlite_url)
    SQLiteSession = sessionmaker(bind=sqlite_engine)
    sqlite_db = SQLiteSession()
    
    # Connect to PostgreSQL
    pg_engine = create_engine(postgres_url)
    PGSession = sessionmaker(bind=pg_engine)
    pg_db = PGSession()
    
    # Enable pgvector extension
    with pg_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector extension enabled")
    
    # Import models (this will create tables)
    os.environ["DATABASE_URL"] = postgres_url  # Switch to postgres mode
    from data.models import Base, Company, Filing, Section, Paragraph, Score, Summary
    
    try:
        from data.models import ScoreVector
        has_pgvector = True
    except ImportError:
        has_pgvector = False
        logger.warning("ScoreVector not available - pgvector migration will be skipped")
    
    # Create all tables in PostgreSQL
    Base.metadata.create_all(bind=pg_engine)
    logger.info("PostgreSQL tables created")
    
    # Migrate companies
    logger.info("Migrating companies...")
    companies = sqlite_db.execute(text("SELECT * FROM companies")).fetchall()
    company_map = {}  # old_id -> new_id
    
    for row in tqdm(companies, desc="Companies"):
        # Check if exists
        existing = pg_db.execute(
            text("SELECT id FROM companies WHERE ticker = :ticker"),
            {"ticker": row.ticker}
        ).fetchone()
        
        if existing:
            company_map[row.id] = existing[0]
        else:
            result = pg_db.execute(
                text("""
                    INSERT INTO companies (ticker, name, industry, created_at)
                    VALUES (:ticker, :name, :industry, :created_at)
                    RETURNING id
                """),
                {
                    "ticker": row.ticker,
                    "name": row.name,
                    "industry": row.industry,
                    "created_at": row.created_at
                }
            )
            company_map[row.id] = result.fetchone()[0]
    pg_db.commit()
    logger.info(f"Migrated {len(companies)} companies")
    
    # Migrate filings
    logger.info("Migrating filings...")
    filings = sqlite_db.execute(text("SELECT * FROM filings")).fetchall()
    filing_map = {}
    
    for row in tqdm(filings, desc="Filings"):
        result = pg_db.execute(
            text("""
                INSERT INTO filings (company_id, filing_type, filing_date, accession_number, source_file, created_at)
                VALUES (:company_id, :filing_type, :filing_date, :accession_number, :source_file, :created_at)
                RETURNING id
            """),
            {
                "company_id": company_map[row.company_id],
                "filing_type": row.filing_type,
                "filing_date": row.filing_date,
                "accession_number": row.accession_number,
                "source_file": row.source_file,
                "created_at": row.created_at
            }
        )
        filing_map[row.id] = result.fetchone()[0]
    pg_db.commit()
    logger.info(f"Migrated {len(filings)} filings")
    
    # Migrate sections
    logger.info("Migrating sections...")
    sections = sqlite_db.execute(text("SELECT * FROM sections")).fetchall()
    section_map = {}
    
    for row in tqdm(sections, desc="Sections"):
        result = pg_db.execute(
            text("""
                INSERT INTO sections (filing_id, section_type, raw_text, word_count, created_at)
                VALUES (:filing_id, :section_type, :raw_text, :word_count, :created_at)
                RETURNING id
            """),
            {
                "filing_id": filing_map[row.filing_id],
                "section_type": row.section_type,
                "raw_text": row.raw_text,
                "word_count": row.word_count,
                "created_at": row.created_at
            }
        )
        section_map[row.id] = result.fetchone()[0]
    pg_db.commit()
    logger.info(f"Migrated {len(sections)} sections")
    
    # Migrate paragraphs
    logger.info("Migrating paragraphs...")
    paragraphs = sqlite_db.execute(text("SELECT * FROM paragraphs")).fetchall()
    paragraph_map = {}
    paragraph_texts = {}  # For re-embedding
    
    for row in tqdm(paragraphs, desc="Paragraphs"):
        result = pg_db.execute(
            text("""
                INSERT INTO paragraphs (section_id, text, position, word_count, created_at)
                VALUES (:section_id, :text, :position, :word_count, :created_at)
                RETURNING id
            """),
            {
                "section_id": section_map[row.section_id],
                "text": row.text,
                "position": row.position,
                "word_count": row.word_count,
                "created_at": row.created_at
            }
        )
        new_id = result.fetchone()[0]
        paragraph_map[row.id] = new_id
        paragraph_texts[new_id] = row.text
    pg_db.commit()
    logger.info(f"Migrated {len(paragraphs)} paragraphs")
    
    # Migrate scores (without embedding - will re-embed)
    logger.info("Migrating scores...")
    scores = sqlite_db.execute(text("SELECT * FROM scores")).fetchall()
    
    for row in tqdm(scores, desc="Scores"):
        pg_db.execute(
            text("""
                INSERT INTO scores (paragraph_id, method, score, top_terms, created_at)
                VALUES (:paragraph_id, :method, :score, :top_terms, :created_at)
            """),
            {
                "paragraph_id": paragraph_map[row.paragraph_id],
                "method": row.method,
                "score": row.score,
                "top_terms": row.top_terms,
                "created_at": row.created_at
            }
        )
    pg_db.commit()
    logger.info(f"Migrated {len(scores)} scores")
    
    # Migrate summaries
    logger.info("Migrating summaries...")
    summaries = sqlite_db.execute(text("SELECT * FROM summaries")).fetchall()
    
    for row in tqdm(summaries, desc="Summaries"):
        pg_db.execute(
            text("""
                INSERT INTO summaries (filing_id, section_type, summary_text, model, prompt_tokens, completion_tokens, created_at)
                VALUES (:filing_id, :section_type, :summary_text, :model, :prompt_tokens, :completion_tokens, :created_at)
            """),
            {
                "filing_id": filing_map[row.filing_id],
                "section_type": row.section_type,
                "summary_text": row.summary_text,
                "model": row.model,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "created_at": row.created_at
            }
        )
    pg_db.commit()
    logger.info(f"Migrated {len(summaries)} summaries")
    
    # Re-embed paragraphs for pgvector
    if re_embed and has_pgvector:
        logger.info("Re-embedding paragraphs for pgvector...")
        
        # Import embedding function
        from services.scoring_service import embed_texts
        
        # Batch embed
        para_ids = list(paragraph_texts.keys())
        texts = [paragraph_texts[pid] for pid in para_ids]
        
        # Process in batches
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_ids = para_ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            embeddings = embed_texts(batch_texts)
            
            for para_id, embedding in zip(batch_ids, embeddings):
                pg_db.execute(
                    text("""
                        INSERT INTO score_vectors (paragraph_id, embedding, created_at)
                        VALUES (:paragraph_id, :embedding, :created_at)
                    """),
                    {
                        "paragraph_id": para_id,
                        "embedding": embedding.tolist(),
                        "created_at": datetime.utcnow()
                    }
                )
            pg_db.commit()
        
        logger.info(f"Created {len(para_ids)} vector embeddings")
        
        # Create index for fast search
        logger.info("Creating vector search index...")
        pg_db.execute(text("""
            CREATE INDEX IF NOT EXISTS score_vectors_embedding_idx 
            ON score_vectors 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        pg_db.commit()
        logger.info("Vector index created")
    
    # Close connections
    sqlite_db.close()
    pg_db.close()
    
    logger.info("="*60)
    logger.info("MIGRATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Companies: {len(companies)}")
    logger.info(f"Filings: {len(filings)}")
    logger.info(f"Sections: {len(sections)}")
    logger.info(f"Paragraphs: {len(paragraphs)}")
    logger.info(f"Scores: {len(scores)}")
    logger.info(f"Summaries: {len(summaries)}")
    if re_embed and has_pgvector:
        logger.info(f"Vector embeddings: {len(paragraph_texts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate Infera database from SQLite to PostgreSQL with pgvector"
    )
    parser.add_argument(
        "--sqlite-path",
        default="./infera.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("DATABASE_URL", "postgresql://infera:infera_dev_password@localhost:5432/infera"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip re-embedding paragraphs"
    )
    args = parser.parse_args()
    
    migrate_data(
        sqlite_path=args.sqlite_path,
        postgres_url=args.postgres_url,
        re_embed=not args.skip_embeddings
    )

