-- Infera Database Initialization Script
-- Creates pgvector extension and necessary setup

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for vector search (will be applied after tables are created by SQLAlchemy)
-- These are created by the migration script after initial data load

