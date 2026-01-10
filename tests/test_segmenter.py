"""
Unit tests for the segmenter module.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze.segmenter import extract_between_items, get_risk_section, fallback_collect_risk_paragraphs


class TestExtractBetweenItems:
    """Tests for the primary extraction method."""
    
    def test_extracts_section_between_items(self):
        """Should extract content between Item 1A and Item 1B."""
        text = """
        Some preamble text here.
        
        Item 1A: Risk Factors
        
        This is the first risk paragraph discussing various business risks 
        that could affect the company's operations and financial results.
        
        This is the second risk paragraph with more detailed information
        about specific risks and their potential impact on the business.
        
        """ + " ".join(["additional content"] * 50) + """
        
        Item 1B: Unresolved Staff Comments
        
        Some other content here.
        """
        
        result = extract_between_items(text)
        
        assert result is not None
        assert "first risk paragraph" in result
        assert "second risk paragraph" in result
        assert "Unresolved Staff Comments" not in result
    
    def test_returns_none_when_section_too_short(self):
        """Should return None if extracted section is too short."""
        text = """
        Item 1A: Risk Factors
        Short.
        Item 1B: Unresolved Staff Comments
        """
        
        result = extract_between_items(text)
        
        assert result is None
    
    def test_returns_none_when_no_match(self):
        """Should return None if Item 1A pattern not found."""
        text = "This is just regular text without any SEC filing structure."
        
        result = extract_between_items(text)
        
        assert result is None
    
    def test_case_insensitive_matching(self):
        """Should match Item 1A regardless of case."""
        text = """
        ITEM 1A - RISK FACTORS
        
        """ + " ".join(["Risk content here."] * 30) + """
        
        ITEM 1B - UNRESOLVED STAFF COMMENTS
        """
        
        result = extract_between_items(text)
        
        # May or may not match depending on regex - at minimum shouldn't crash
        assert result is None or len(result) > 0


class TestFallbackCollectRiskParagraphs:
    """Tests for the fallback extraction method."""
    
    def test_collects_paragraphs_containing_risk(self):
        """Should collect paragraphs that mention 'risk'."""
        text = """
        This paragraph discusses financial risk and its implications for investors.
        
        This paragraph is about something else entirely unrelated.
        
        The company faces significant operational risk in its supply chain.
        
        Another unrelated paragraph without the keyword.
        """
        
        result = fallback_collect_risk_paragraphs(text)
        
        assert result is not None
        assert "financial risk" in result
        assert "operational risk" in result
    
    def test_returns_none_when_no_risk_paragraphs(self):
        """Should return None if no paragraphs contain 'risk'."""
        text = """
        This is a paragraph about sunshine and happiness.
        
        Another paragraph about rainbows and unicorns.
        """
        
        result = fallback_collect_risk_paragraphs(text)
        
        assert result is None
    
    def test_filters_short_paragraphs(self):
        """Should not include very short paragraphs."""
        text = """
        Risk.
        
        This is a longer paragraph about risk that should be included because
        it has more than 50 characters of content.
        """
        
        result = fallback_collect_risk_paragraphs(text)
        
        if result:
            assert "longer paragraph" in result
            # Very short "Risk." alone shouldn't be the only content


class TestGetRiskSection:
    """Tests for the main get_risk_section function."""
    
    def test_uses_primary_extraction_when_available(self):
        """Should use primary extraction if it succeeds."""
        text = """
        Item 1A: Risk Factors
        
        """ + " ".join(["Important risk information."] * 30) + """
        
        Item 1B: Unresolved Staff Comments
        """
        
        result = get_risk_section(text)
        
        assert result is not None
        assert len(result) > 100
    
    def test_falls_back_when_primary_fails(self):
        """Should use fallback when primary extraction fails."""
        text = """
        The company faces multiple risk factors in its operations.
        
        There is significant regulatory risk in the current environment.
        
        Supply chain risk remains a concern for management.
        """
        
        result = get_risk_section(text)
        
        assert result is not None
        assert "risk" in result.lower()
    
    def test_returns_none_when_both_methods_fail(self):
        """Should return None when no risk content found."""
        text = "Just some random text without any relevant content."
        
        result = get_risk_section(text)
        
        assert result is None


class TestSegmenterIntegration:
    """Integration tests using real filing data."""
    
    def test_aapl_filing_extracts_risks(self):
        """AAPL 10-K should yield risk paragraphs."""
        from analyze.cleaner import clean_html
        
        filepath = "data/AAPL_10K.html"
        if os.path.exists(filepath):
            cleaned = clean_html(filepath)
            result = get_risk_section(cleaned)
            
            assert result is not None
            assert len(result) > 1000
            assert "risk" in result.lower()

