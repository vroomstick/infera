-- Infera Database Initialization Script
-- Runs automatically on first container startup
-- Creates schema if it doesn't exist

-- ============================================
-- 1. Enable Extensions
-- ============================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Log extension versions
DO $$
BEGIN
    RAISE NOTICE 'Extensions enabled:';
    RAISE NOTICE '  - pgvector: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
    RAISE NOTICE '  - timescaledb: %', (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb');
END $$;

-- ============================================
-- 2. Create Tables (if not exist)
-- ============================================

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Filings table
CREATE TABLE IF NOT EXISTS filings (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    filing_type VARCHAR(20) DEFAULT '10-K',
    filing_date TIMESTAMP,
    accession_number VARCHAR(50),
    source_file VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sections table
CREATE TABLE IF NOT EXISTS sections (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    section_type VARCHAR(50) NOT NULL,
    raw_text TEXT,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paragraphs table
CREATE TABLE IF NOT EXISTS paragraphs (
    id SERIAL PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    position INTEGER,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scores table (stores risk scores, embeddings as binary for SQLite compatibility)
CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    paragraph_id INTEGER NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
    method VARCHAR(50) NOT NULL,
    score FLOAT NOT NULL,
    embedding BYTEA,
    top_terms TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Score vectors table (native pgvector for fast similarity search)
-- 768 dimensions for FinBERT embeddings
CREATE TABLE IF NOT EXISTS score_vectors (
    id SERIAL PRIMARY KEY,
    paragraph_id INTEGER NOT NULL UNIQUE REFERENCES paragraphs(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Summaries table (GPT-generated summaries)
CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    section_type VARCHAR(50) NOT NULL,
    summary_text TEXT NOT NULL,
    model VARCHAR(50),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 3. Create Indexes (if not exist)
-- ============================================

-- B-tree indexes for foreign keys and lookups
CREATE INDEX IF NOT EXISTS ix_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS ix_filings_company_id ON filings(company_id);
CREATE INDEX IF NOT EXISTS ix_sections_filing_id ON sections(filing_id);
CREATE INDEX IF NOT EXISTS ix_paragraphs_section_id ON paragraphs(section_id);
CREATE INDEX IF NOT EXISTS ix_scores_paragraph_id ON scores(paragraph_id);
CREATE INDEX IF NOT EXISTS ix_score_vectors_paragraph_id ON score_vectors(paragraph_id);
CREATE INDEX IF NOT EXISTS ix_summaries_filing_id ON summaries(filing_id);

-- HNSW index for fast vector similarity search (cosine distance)
-- This enables sub-millisecond nearest neighbor queries
CREATE INDEX IF NOT EXISTS ix_score_vectors_embedding 
    ON score_vectors 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search on paragraph text
CREATE INDEX IF NOT EXISTS ix_paragraphs_text_search 
    ON paragraphs 
    USING gin (to_tsvector('english', text));

-- ============================================
-- 4. Verify Setup
-- ============================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    
    RAISE NOTICE '';
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Infera Database Initialized Successfully';
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Tables created: %', table_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Tables:';
    RAISE NOTICE '  - companies';
    RAISE NOTICE '  - filings';
    RAISE NOTICE '  - sections';
    RAISE NOTICE '  - paragraphs';
    RAISE NOTICE '  - scores';
    RAISE NOTICE '  - score_vectors (pgvector)';
    RAISE NOTICE '  - summaries';
    RAISE NOTICE '';
    RAISE NOTICE 'Vector search: HNSW index on score_vectors.embedding';
    RAISE NOTICE 'Full-text search: GIN index on paragraphs.text';
    RAISE NOTICE '==========================================';
END $$;
