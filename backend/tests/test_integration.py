"""
Integration tests for the full Infera pipeline.

Tests the complete flow: load fixture → clean → segment → score → store → API
Uses fixtures only (no live SEC EDGAR) for CI-safe, deterministic testing.
"""

import pytest
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import SessionLocal, init_db
from data import repository as repo
from data.models import Company, Filing, Section, Paragraph, Score, ScoreVector, Summary
from services.pipeline_service import run_analysis_pipeline


# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def clean_db(db_session):
    """Clean database before each test."""
    # Delete all data in reverse dependency order
    db_session.query(ScoreVector).delete()
    db_session.query(Score).delete()
    db_session.query(Paragraph).delete()
    db_session.query(Section).delete()
    db_session.query(Summary).delete()
    db_session.query(Filing).delete()
    db_session.query(Company).delete()
    db_session.commit()
    yield db_session


class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""
    
    def test_pipeline_standard_2023(self, clean_db):
        """Test pipeline with standard 2023 format fixture."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="TEST",
            filing_date=datetime(2023, 12, 31),
            accession_number="0001234567-23-000001",
            skip_summary=True,
            skip_scoring=False
        )
        
        # Assertions
        assert result is not None
        assert "filing_id" in result
        assert "section_id" in result
        assert "paragraph_count" in result
        assert result["paragraph_count"] > 0
        assert result["ticker"] == "TEST"
        
        # Verify database state
        filing = repo.get_filing_by_id(clean_db, result["filing_id"])
        assert filing is not None
        assert filing.accession_number == "0001234567-23-000001"
        
        sections = repo.get_sections_by_filing(clean_db, filing.id)
        assert len(sections) > 0
        
        paragraphs = repo.get_paragraphs_by_section(clean_db, sections[0].id)
        assert len(paragraphs) == result["paragraph_count"]
        
        # Verify scores exist
        scores = clean_db.query(Score).filter(
            Score.paragraph_id.in_([p.id for p in paragraphs])
        ).all()
        assert len(scores) > 0
        
        # Verify scores are in valid range
        for score in scores:
            assert 0 <= score.score <= 1
        
        # Verify embeddings exist (ScoreVector)
        score_vectors = clean_db.query(ScoreVector).filter(
            ScoreVector.paragraph_id.in_([p.id for p in paragraphs])
        ).all()
        assert len(score_vectors) > 0
        
        # Verify embedding dimensions
        for sv in score_vectors:
            assert len(sv.embedding) == 768  # FinBERT dimension
    
    def test_pipeline_standard_2024(self, clean_db):
        """Test pipeline with standard 2024 format fixture."""
        fixture_path = FIXTURES_DIR / "sample_10k_2024.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="TEST2",
            filing_date=datetime(2024, 12, 31),
            accession_number="0001234567-24-000001",
            skip_summary=True,
            skip_scoring=False
        )
        
        assert result is not None
        assert result["paragraph_count"] > 0
        
        # Verify no duplicates created
        company = repo.get_company_by_ticker(clean_db, "TEST2")
        if company:
            filings = repo.get_filings_by_ticker(clean_db, "TEST2")
            assert len(filings) == 1
    
    def test_pipeline_nonstandard_format(self, clean_db):
        """Test pipeline with non-standard header format."""
        fixture_path = FIXTURES_DIR / "sample_10k_nonstandard.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="TEST3",
            filing_date=datetime(2023, 6, 30),
            accession_number="0001234567-23-000002",
            skip_summary=True,
            skip_scoring=False
        )
        
        assert result is not None
        assert result["paragraph_count"] > 0
    
    def test_pipeline_small_filing(self, clean_db):
        """Test pipeline with small filing."""
        fixture_path = FIXTURES_DIR / "sample_10k_small.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="SMALL",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        assert result is not None
        assert result["paragraph_count"] > 0
        assert result["paragraph_count"] < 10  # Small filing should have few paragraphs
    
    def test_pipeline_large_filing(self, clean_db):
        """Test pipeline with large filing."""
        fixture_path = FIXTURES_DIR / "sample_10k_large.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="LARGE",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        assert result is not None
        assert result["paragraph_count"] > 5  # Large filing should have many paragraphs
    
    def test_pipeline_idempotency(self, clean_db):
        """Test that running pipeline twice doesn't create duplicates."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        accession = "0001234567-23-000003"
        
        # First run
        result1 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="IDEMP",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False
        )
        
        filing_id_1 = result1["filing_id"]
        paragraph_count_1 = result1["paragraph_count"]
        
        # Second run (should skip)
        result2 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="IDEMP",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False
        )
        
        # Should be skipped
        assert result2.get("skipped") is True
        assert result2["filing_id"] == filing_id_1
        
        # Verify no duplicates in database
        filings = clean_db.query(Filing).filter(Filing.accession_number == accession).all()
        assert len(filings) == 1
        
        sections = repo.get_sections_by_filing(clean_db, filing_id_1)
        paragraphs = []
        for section in sections:
            paragraphs.extend(repo.get_paragraphs_by_section(clean_db, section.id))
        
        # Should have same number of paragraphs (no duplicates)
        assert len(paragraphs) == paragraph_count_1
    
    def test_pipeline_force_flag(self, clean_db):
        """Test --force flag wipes derived data and reprocesses."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        accession = "0001234567-23-000004"
        
        # First run
        result1 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="FORCE",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False
        )
        
        filing_id = result1["filing_id"]
        sections1 = repo.get_sections_by_filing(clean_db, filing_id)
        paragraphs1 = []
        for section in sections1:
            paragraphs1.extend(repo.get_paragraphs_by_section(clean_db, section.id))
        
        # Second run with --force
        result2 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="FORCE",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False,
            force=True
        )
        
        # Should reprocess
        assert result2.get("skipped") is None
        assert result2["filing_id"] == filing_id  # Same filing, but data wiped
        
        # Verify data was wiped and recreated
        sections2 = repo.get_sections_by_filing(clean_db, filing_id)
        paragraphs2 = []
        for section in sections2:
            paragraphs2.extend(repo.get_paragraphs_by_section(clean_db, section.id))
        
        # Should have same structure (may have different IDs)
        assert len(paragraphs2) == len(paragraphs1)
    
    def test_pipeline_update_flag(self, clean_db):
        """Test --update flag recomputes scores but preserves metadata."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        accession = "0001234567-23-000005"
        
        # First run
        result1 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="UPDATE",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False
        )
        
        filing_id = result1["filing_id"]
        sections1 = repo.get_sections_by_filing(clean_db, filing_id)
        paragraphs1 = []
        for section in sections1:
            paragraphs1.extend(repo.get_paragraphs_by_section(clean_db, section.id))
        
        # Get original scores
        original_scores = {}
        for para in paragraphs1:
            score = clean_db.query(Score).filter(Score.paragraph_id == para.id).first()
            if score:
                original_scores[para.id] = score.score
        
        # Second run with --update
        result2 = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="UPDATE",
            filing_date=datetime(2023, 12, 31),
            accession_number=accession,
            skip_summary=True,
            skip_scoring=False,
            update=True
        )
        
        # Should update scores
        assert result2.get("skipped") is None
        assert result2["filing_id"] == filing_id
        
        # Verify sections and paragraphs preserved
        sections2 = repo.get_sections_by_filing(clean_db, filing_id)
        paragraphs2 = []
        for section in sections2:
            paragraphs2.extend(repo.get_paragraphs_by_section(clean_db, section.id))
        
        assert len(paragraphs2) == len(paragraphs1)
        
        # Verify scores were recomputed
        for para in paragraphs2:
            score = clean_db.query(Score).filter(Score.paragraph_id == para.id).first()
            assert score is not None
            assert 0 <= score.score <= 1


class TestDatabaseState:
    """Tests for database state consistency."""
    
    def test_no_orphaned_records(self, clean_db):
        """Test that all records have proper foreign key relationships."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        
        run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="ORPHAN",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        # Verify all sections have valid filing_id
        sections = clean_db.query(Section).all()
        for section in sections:
            filing = repo.get_filing_by_id(clean_db, section.filing_id)
            assert filing is not None
        
        # Verify all paragraphs have valid section_id
        paragraphs = clean_db.query(Paragraph).all()
        for para in paragraphs:
            section = clean_db.query(Section).filter(Section.id == para.section_id).first()
            assert section is not None
        
        # Verify all scores have valid paragraph_id
        scores = clean_db.query(Score).all()
        for score in scores:
            para = clean_db.query(Paragraph).filter(Paragraph.id == score.paragraph_id).first()
            assert para is not None
        
        # Verify all score_vectors have valid paragraph_id
        score_vectors = clean_db.query(ScoreVector).all()
        for sv in score_vectors:
            para = clean_db.query(Paragraph).filter(Paragraph.id == sv.paragraph_id).first()
            assert para is not None
    
    def test_row_counts_match_expectations(self, clean_db):
        """Test that row counts match expected relationships."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="COUNTS",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        filing_id = result["filing_id"]
        
        # Count relationships
        sections = repo.get_sections_by_filing(clean_db, filing_id)
        total_paragraphs = 0
        total_scores = 0
        total_score_vectors = 0
        
        for section in sections:
            paragraphs = repo.get_paragraphs_by_section(clean_db, section.id)
            total_paragraphs += len(paragraphs)
            
            for para in paragraphs:
                scores = clean_db.query(Score).filter(Score.paragraph_id == para.id).all()
                total_scores += len(scores)
                
                sv = clean_db.query(ScoreVector).filter(ScoreVector.paragraph_id == para.id).first()
                if sv:
                    total_score_vectors += 1
        
        # Verify counts match
        assert total_paragraphs == result["paragraph_count"]
        assert total_scores == total_paragraphs  # One score per paragraph
        assert total_score_vectors == total_paragraphs  # One vector per paragraph


class TestAPISchema:
    """Tests for API response schema validation."""
    
    def test_pipeline_result_schema(self, clean_db):
        """Test that pipeline result has expected schema."""
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        
        result = run_analysis_pipeline(
            filepath=str(fixture_path),
            ticker="SCHEMA",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        # Required fields
        assert "ticker" in result
        assert "filing_id" in result
        assert "section_id" in result
        assert "paragraph_count" in result
        
        # Type checks
        assert isinstance(result["ticker"], str)
        assert isinstance(result["filing_id"], int)
        assert isinstance(result["section_id"], int)
        assert isinstance(result["paragraph_count"], int)
        
        # Value checks
        assert result["ticker"] == "SCHEMA"
        assert result["filing_id"] > 0
        assert result["section_id"] > 0
        assert result["paragraph_count"] > 0

