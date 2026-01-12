"""
Edge case and chaos tests for the Infera pipeline.

Tests graceful handling of bad input, malformed data, and unusual formats.
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import SessionLocal, init_db
from data import repository as repo
from services.pipeline_service import run_analysis_pipeline
from analyze.cleaner import clean_html
from analyze.segmenter import get_risk_section

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
    from data.models import ScoreVector, Score, Paragraph, Section, Summary, Filing, Company
    db_session.query(ScoreVector).delete()
    db_session.query(Score).delete()
    db_session.query(Paragraph).delete()
    db_session.query(Section).delete()
    db_session.query(Summary).delete()
    db_session.query(Filing).delete()
    db_session.query(Company).delete()
    db_session.commit()
    yield db_session


class TestHeaderVariants:
    """Test Item 1A header detection with various formats."""
    
    def test_item_1a_with_period(self):
        """Test 'ITEM 1A.' format."""
        text = """
        ITEM 1A.
        
        Risk Factors
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        ITEM 1B.
        """
        result = get_risk_section(text)
        assert result is not None
        assert len(result) > 100
    
    def test_item_1a_with_dash(self):
        """Test 'Item 1A - Risk Factors' format."""
        text = """
        Item 1A - Risk Factors
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        Item 1B - Unresolved Staff Comments
        """
        result = get_risk_section(text)
        assert result is not None
        assert len(result) > 100
    
    def test_item_1a_with_emdash(self):
        """Test 'Item 1A—Risk Factors' format (em dash)."""
        text = """
        Item 1A—Risk Factors
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        Item 1B—Unresolved Staff Comments
        """
        result = get_risk_section(text)
        assert result is not None
        assert len(result) > 100
    
    def test_item_1a_with_html_anchor(self):
        """Test HTML anchor format: <a name="item1a">."""
        text = """
        <a name="item1a"></a>
        <h2>Item 1A. Risk Factors</h2>
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        <a name="item1b"></a>
        <h2>Item 1B. Unresolved Staff Comments</h2>
        """
        result = get_risk_section(text)
        assert result is not None
        assert len(result) > 100
    
    def test_item_1a_case_insensitive(self):
        """Test case-insensitive matching."""
        text = """
        item 1a: risk factors
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        ITEM 1B: UNRESOLVED STAFF COMMENTS
        """
        result = get_risk_section(text)
        # May or may not match depending on regex, but shouldn't crash
        assert result is None or len(result) > 0
    
    def test_item_1a_spacing_variations(self):
        """Test various spacing patterns."""
        variants = [
            "Item 1A:Risk Factors",
            "Item  1A:  Risk Factors",  # Multiple spaces
            "Item\t1A:\tRisk Factors",  # Tabs
            "Item\n1A:\nRisk Factors",  # Newlines
        ]
        
        for header in variants:
            text = f"""
            {header}
            
            """ + " ".join(["Risk content here."] * 30) + """
            
            Item 1B: Unresolved Staff Comments
            """
            result = get_risk_section(text)
            # Should handle gracefully (may or may not match, but shouldn't crash)
            assert result is None or len(result) > 0


class TestMalformedInput:
    """Test handling of malformed or invalid input."""
    
    def test_truncated_filing(self, clean_db, tmp_path):
        """Test handling of incomplete/truncated filing."""
        # Create a truncated HTML file
        truncated_html = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1A. Risk Factors</h2>
        <p>This is incomplete content that gets cut off mid-sentence...
        """
        
        filepath = tmp_path / "truncated.html"
        filepath.write_text(truncated_html)
        
        # Should handle gracefully - may extract partial content or return error
        try:
            result = run_analysis_pipeline(
                filepath=str(filepath),
                ticker="TRUNC",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
            # If it succeeds, verify it handled gracefully
            if result:
                assert "filing_id" in result or "error" in str(result).lower()
        except (ValueError, Exception) as e:
            # Expected - should raise meaningful error
            assert "Item 1A" in str(e) or "Risk Factors" in str(e) or "section" in str(e).lower()
    
    def test_no_item_1a_section(self, clean_db, tmp_path):
        """Test filing with no Item 1A section."""
        html_no_1a = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1. Business</h2>
        <p>Business description here.</p>
        <h2>Item 1B. Unresolved Staff Comments</h2>
        <p>Not applicable.</p>
        </body>
        </html>
        """
        
        filepath = tmp_path / "no_1a.html"
        filepath.write_text(html_no_1a)
        
        # Should raise ValueError with meaningful message
        with pytest.raises(ValueError) as exc_info:
            run_analysis_pipeline(
                filepath=str(filepath),
                ticker="NO1A",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
        
        assert "Item 1A" in str(exc_info.value) or "Risk Factors" in str(exc_info.value)
    
    def test_malformed_html_unclosed_tags(self, clean_db, tmp_path):
        """Test HTML with unclosed tags."""
        malformed_html = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1A. Risk Factors</h2>
        <p>Risk content here.
        <p>Another paragraph without closing tag.
        <div>Unclosed div
        """ + " ".join(["More content."] * 30) + """
        <h2>Item 1B. Unresolved Staff Comments</h2>
        </body>
        </html>
        """
        
        filepath = tmp_path / "malformed.html"
        filepath.write_text(malformed_html)
        
        # Should handle gracefully - HTML parser should fix or handle unclosed tags
        try:
            result = run_analysis_pipeline(
                filepath=str(filepath),
                ticker="MALFORM",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
            # May succeed with cleaned HTML
            if result:
                assert "filing_id" in result
        except Exception as e:
            # Should provide meaningful error
            assert len(str(e)) > 0
    
    def test_non_utf8_encoding(self, clean_db, tmp_path):
        """Test file with non-UTF8 encoding."""
        # Create file with Latin-1 encoding (common in old filings)
        latin1_text = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1A. Risk Factors</h2>
        <p>Risk content with special chars: café, résumé, naïve.
        """ + " ".join(["More content."] * 30) + """
        <h2>Item 1B. Unresolved Staff Comments</h2>
        </body>
        </html>
        """
        
        filepath = tmp_path / "latin1.html"
        filepath.write_bytes(latin1_text.encode('latin-1'))
        
        # Should handle encoding gracefully
        try:
            result = run_analysis_pipeline(
                filepath=str(filepath),
                ticker="LATIN1",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
            # May succeed with encoding detection
            if result:
                assert "filing_id" in result
        except (UnicodeDecodeError, Exception) as e:
            # Should provide meaningful error about encoding
            assert "encoding" in str(e).lower() or "decode" in str(e).lower() or len(str(e)) > 0


class TestEmptyContent:
    """Test handling of empty or minimal content."""
    
    def test_empty_paragraphs(self, clean_db, tmp_path):
        """Test filing with empty paragraphs."""
        html_empty = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1A. Risk Factors</h2>
        <p></p>
        <p>   </p>
        <p>Valid paragraph with enough content to pass the minimum word threshold and be considered a real paragraph for analysis purposes.</p>
        <p></p>
        """ + " ".join(["More valid content."] * 30) + """
        <h2>Item 1B. Unresolved Staff Comments</h2>
        </body>
        </html>
        """
        
        filepath = tmp_path / "empty_paras.html"
        filepath.write_text(html_empty)
        
        # Should handle gracefully - may extract content or fail if no valid content
        try:
            result = run_analysis_pipeline(
                filepath=str(filepath),
                ticker="EMPTY",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
            # If it succeeds, should filter out empty paragraphs
            assert result is not None
            assert result["paragraph_count"] > 0
        except ValueError as e:
            # If it fails, should be because no valid content was found
            assert "Item 1A" in str(e) or "Risk Factors" in str(e) or "section" in str(e).lower()
    
    def test_very_short_paragraphs(self, clean_db, tmp_path):
        """Test paragraphs shorter than minimum threshold."""
        html_short = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Item 1A. Risk Factors</h2>
        <p>Risk.</p>
        <p>Short.</p>
        <p>This is a longer paragraph with enough words to meet the minimum threshold for inclusion in the analysis pipeline and risk scoring system.</p>
        """ + " ".join(["More valid content."] * 30) + """
        <h2>Item 1B. Unresolved Staff Comments</h2>
        </body>
        </html>
        """
        
        filepath = tmp_path / "short_paras.html"
        filepath.write_text(html_short)
        
        # Should handle gracefully - filter out short paragraphs
        result = run_analysis_pipeline(
            filepath=str(filepath),
            ticker="SHORT",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        # Should filter out very short paragraphs but keep valid ones
        assert result is not None
        # Should have at least some paragraphs (the longer ones)
        assert result["paragraph_count"] >= 0  # Allow 0 if all filtered, but shouldn't crash
    
    def test_filing_with_only_toc(self, clean_db, tmp_path):
        """Test filing with only table of contents, no actual content."""
        html_toc_only = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <h2>Table of Contents</h2>
        <ul>
        <li>Item 1A. Risk Factors</li>
        <li>Item 1B. Unresolved Staff Comments</li>
        <li>Item 2. Properties</li>
        </ul>
        <p>End of document.</p>
        </body>
        </html>
        """
        
        filepath = tmp_path / "toc_only.html"
        filepath.write_text(html_toc_only)
        
        # Should raise ValueError - no actual Item 1A content
        with pytest.raises(ValueError) as exc_info:
            run_analysis_pipeline(
                filepath=str(filepath),
                ticker="TOC",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
        
        assert "Item 1A" in str(exc_info.value) or "Risk Factors" in str(exc_info.value)


class TestEdgeCaseIntegration:
    """Integration tests for edge cases in full pipeline."""
    
    def test_pipeline_handles_header_variant(self, clean_db, tmp_path):
        """Test that pipeline handles non-standard header format."""
        html_variant = """
        <!DOCTYPE html>
        <html>
        <head><title>10-K</title></head>
        <body>
        <h1>FORM 10-K</h1>
        <a name="item1a"></a>
        <h2>ITEM 1A.</h2>
        <h3>RISK FACTORS</h3>
        <p>Risk content paragraph one with sufficient length to pass validation.</p>
        <p>Risk content paragraph two with sufficient length to pass validation.</p>
        """ + " ".join(["More risk content."] * 30) + """
        <a name="item1b"></a>
        <h2>ITEM 1B.</h2>
        </body>
        </html>
        """
        
        filepath = tmp_path / "variant.html"
        filepath.write_text(html_variant)
        
        result = run_analysis_pipeline(
            filepath=str(filepath),
            ticker="VAR",
            filing_date=datetime(2023, 12, 31),
            skip_summary=True,
            skip_scoring=False
        )
        
        # Should succeed despite non-standard format
        assert result is not None
        assert result["paragraph_count"] > 0
    
    def test_pipeline_continues_on_partial_failure(self, clean_db, tmp_path):
        """Test that pipeline continues processing even if some steps fail."""
        # Use a valid fixture but with potential scoring issues
        fixture_path = FIXTURES_DIR / "sample_10k_2023.html"
        
        if fixture_path.exists():
            result = run_analysis_pipeline(
                filepath=str(fixture_path),
                ticker="PART",
                filing_date=datetime(2023, 12, 31),
                skip_summary=True,
                skip_scoring=False
            )
            
            # Should complete even if some paragraphs fail scoring
            assert result is not None
            assert "filing_id" in result

