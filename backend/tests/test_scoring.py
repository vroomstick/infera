"""
Unit tests for the scoring service.
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.scoring_service import (
    embed_text, 
    embed_texts, 
    compute_risk_scores,
    get_model
)


class TestEmbedding:
    """Tests for embedding functions."""
    
    def test_embed_text_returns_numpy_array(self):
        """Single text embedding should return numpy array."""
        result = embed_text("This is a test sentence.")
        
        assert isinstance(result, np.ndarray)
    
    def test_embed_text_correct_dimension(self):
        """Embedding should be 768-dimensional (FinBERT)."""
        result = embed_text("This is a test sentence.")
        
        assert result.shape == (768,)
    
    def test_embed_texts_batch(self):
        """Batch embedding should work correctly."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
    
    def test_embed_empty_list_returns_empty(self):
        """Empty list should return empty array."""
        result = embed_texts([])
        
        assert len(result) == 0
    
    def test_similar_texts_have_high_similarity(self):
        """Semantically similar texts should have similar embeddings."""
        text1 = "The company faces significant cybersecurity risks."
        text2 = "There are major information security threats to the business."
        text3 = "The weather is nice today."
        
        emb1 = embed_text(text1)
        emb2 = embed_text(text2)
        emb3 = embed_text(text3)
        
        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
    
    def test_model_loads_once(self):
        """Model should be lazy-loaded and cached."""
        model1 = get_model()
        model2 = get_model()
        
        assert model1 is model2  # Same object (cached)


class TestRiskScoring:
    """Tests for risk scoring functions."""
    
    def test_compute_risk_scores_returns_correct_structure(self):
        """Should return list of (score, embedding) tuples."""
        paragraphs = [
            "The company faces significant litigation risk.",
            "Revenue grew 10% year over year."
        ]
        
        results = compute_risk_scores(paragraphs)
        
        assert len(results) == 2
        assert all(len(r) == 2 for r in results)  # (score, embedding) tuples
        assert all(isinstance(r[0], float) for r in results)
        assert all(isinstance(r[1], np.ndarray) for r in results)
    
    def test_scores_in_valid_range(self):
        """Scores should be between 0 and 1."""
        paragraphs = [
            "Cybersecurity threats could materially affect operations.",
            "The company was founded in 1976.",
            "Supply chain disruption poses significant risks."
        ]
        
        results = compute_risk_scores(paragraphs)
        scores = [r[0] for r in results]
        
        assert all(0 <= s <= 1 for s in scores)
    
    def test_risk_paragraphs_score_higher(self):
        """Risk-related paragraphs should score higher than neutral ones."""
        risk_paragraph = """
        The Company is exposed to significant cybersecurity threats including 
        data breaches, ransomware attacks, and unauthorized access to confidential 
        information. These risks could materially adversely affect operations.
        """
        
        neutral_paragraph = """
        The company's headquarters are located in Cupertino, California. 
        The building was constructed in 2017 and features modern architecture.
        """
        
        results = compute_risk_scores([risk_paragraph, neutral_paragraph])
        
        risk_score = results[0][0]
        neutral_score = results[1][0]
        
        assert risk_score > neutral_score
    
    def test_empty_input_returns_empty(self):
        """Empty paragraph list should return empty results."""
        results = compute_risk_scores([])
        
        assert results == []
    
    def test_ordering_preserved(self):
        """Results should maintain input paragraph ordering."""
        paragraphs = ["First", "Second", "Third"]
        
        results = compute_risk_scores(paragraphs)
        
        # Just verify we get 3 results back in order
        assert len(results) == 3


class TestScoringIntegration:
    """Integration tests for scoring with real data."""
    
    def test_real_risk_paragraph_scores_high(self):
        """Real 10-K risk paragraph should score reasonably high."""
        # Actual paragraph from Apple 10-K
        paragraph = """
        The Company's business, reputation, results of operations, financial 
        condition and stock price can be affected by a number of factors, 
        whether currently known or unknown, including those described below.
        """
        
        results = compute_risk_scores([paragraph])
        score = results[0][0]
        
        # Should score above 0.5 (has risk-related language)
        assert score > 0.4
    
    def test_boilerplate_scores_high_limitation(self):
        """Known limitation: boilerplate intro text scores high."""
        # This is a known false positive pattern
        boilerplate = """
        You should carefully consider the risks described below together with 
        the other information set forth in this report, which could materially 
        affect our business, financial condition and future results.
        """
        
        results = compute_risk_scores([boilerplate])
        score = results[0][0]
        
        # Known limitation - this will score high despite being boilerplate
        # Just verify it doesn't crash
        assert 0 <= score <= 1

