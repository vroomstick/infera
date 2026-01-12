# services/scoring_service.py
"""
Embedding and scoring service using sentence-transformers.
"""

import os
import sys
import json
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger
from data.database import SessionLocal
from data import repository as repo
from data.models import Paragraph, Score

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)

# Risk-related prompt for computing similarity scores
RISK_PROMPT = """
Significant business risks including lawsuits, litigation, regulatory compliance,
cybersecurity threats, data breaches, supply chain disruption, economic downturn,
inflation, pandemic, natural disasters, product recalls, fraud, labor shortage,
competitive pressure, market volatility, and operational failure.
"""

# Lazy-loaded model
_model = None


def get_model() -> SentenceTransformer:
    """Get or load the sentence transformer model (FinBERT for financial domain)."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: ProsusAI/finbert")
        _model = SentenceTransformer('ProsusAI/finbert')
        logger.info("Model loaded successfully (768-dim embeddings)")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts at once (more efficient)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def compute_risk_scores(paragraphs: List[str]) -> List[Tuple[float, np.ndarray]]:
    """
    Compute risk severity scores for paragraphs.
    
    Args:
        paragraphs: List of paragraph texts
        
    Returns:
        List of (score, embedding) tuples, one per paragraph
    """
    if not paragraphs:
        return []
    
    logger.info(f"Computing embeddings for {len(paragraphs)} paragraphs...")
    
    # Embed paragraphs and risk prompt
    all_texts = [RISK_PROMPT] + paragraphs
    embeddings = embed_texts(all_texts)
    
    risk_embedding = embeddings[0:1]  # Shape: (1, dim)
    paragraph_embeddings = embeddings[1:]  # Shape: (n, dim)
    
    # Compute cosine similarity to risk prompt
    similarities = cosine_similarity(paragraph_embeddings, risk_embedding).flatten()
    
    # Normalize to 0-1 range (cosine similarity is already -1 to 1, but for text usually 0 to 1)
    scores = np.clip(similarities, 0, 1)
    
    logger.info(f"Scores computed. Range: {scores.min():.3f} - {scores.max():.3f}")
    
    return list(zip(scores.tolist(), paragraph_embeddings))


def score_section_paragraphs(section_id: int, store_embeddings: bool = True) -> List[Dict]:
    """
    Score all paragraphs in a section and store results.
    
    Args:
        section_id: Database ID of the section
        store_embeddings: Whether to store full embeddings (uses more space)
        
    Returns:
        List of dicts with paragraph_id, score, and position
    """
    db = SessionLocal()
    
    try:
        # Get paragraphs
        paragraphs = repo.get_paragraphs_by_section(db, section_id)
        
        if not paragraphs:
            logger.warning(f"No paragraphs found for section {section_id}")
            return []
        
        logger.info(f"Scoring {len(paragraphs)} paragraphs for section {section_id}")
        
        # Extract texts
        texts = [p.text for p in paragraphs]
        
        # Compute scores with per-paragraph error handling
        scored = []
        failed_count = 0
        
        # Process in batches for efficiency, but handle individual failures
        try:
            results = compute_risk_scores(texts)
            
            # Store scores
            for para, (score, embedding) in zip(paragraphs, results):
                try:
                    # Serialize embedding if storing
                    embedding_bytes = None
                    if store_embeddings:
                        embedding_bytes = pickle.dumps(embedding)
                    
                    # Create score record
                    score_obj = repo.create_score(
                        db,
                        paragraph_id=para.id,
                        method="embedding",
                        score=float(score),
                        embedding=embedding_bytes
                    )
                    
                    # Store in ScoreVector for pgvector
                    from data.repository import create_score_vector
                    try:
                        create_score_vector(db, para.id, embedding.tolist())
                    except Exception as e:
                        logger.warning(f"Failed to store ScoreVector for paragraph {para.id}: {e}")
                        # Continue - ScoreVector is optional for backward compatibility
                    
                    scored.append({
                        "paragraph_id": para.id,
                        "position": para.position,
                        "score": float(score),
                        "text_preview": para.text[:100] + "..." if len(para.text) > 100 else para.text
                    })
                except Exception as e:
                    failed_count += 1
                    text_preview = para.text[:100] + "..." if len(para.text) > 100 else para.text
                    logger.error(
                        f"Failed to score paragraph {para.id} (position {para.position}): {e}. "
                        f"Text preview: {text_preview}"
                    )
                    # Store NULL score to indicate failure (Score model doesn't support NULL, so we'll skip)
                    # In a production system, you'd want a status field, but for now we just log and continue
                    continue
        except Exception as e:
            # If batch computation fails, try per-paragraph
            logger.warning(f"Batch scoring failed: {e}. Falling back to per-paragraph processing...")
            
            for para in paragraphs:
                try:
                    # Compute score for single paragraph
                    para_results = compute_risk_scores([para.text])
                    if not para_results:
                        raise ValueError("No results returned")
                    
                    score, embedding = para_results[0]
                    
                    # Serialize embedding if storing
                    embedding_bytes = None
                    if store_embeddings:
                        embedding_bytes = pickle.dumps(embedding)
                    
                    # Create score record
                    score_obj = repo.create_score(
                        db,
                        paragraph_id=para.id,
                        method="embedding",
                        score=float(score),
                        embedding=embedding_bytes
                    )
                    
                    # Store in ScoreVector for pgvector
                    from data.repository import create_score_vector
                    try:
                        create_score_vector(db, para.id, embedding.tolist())
                    except Exception as e:
                        logger.warning(f"Failed to store ScoreVector for paragraph {para.id}: {e}")
                    
                    scored.append({
                        "paragraph_id": para.id,
                        "position": para.position,
                        "score": float(score),
                        "text_preview": para.text[:100] + "..." if len(para.text) > 100 else para.text
                    })
                except Exception as para_error:
                    failed_count += 1
                    text_preview = para.text[:100] + "..." if len(para.text) > 100 else para.text
                    logger.error(
                        f"Failed to score paragraph {para.id} (position {para.position}): {para_error}. "
                        f"Text preview: {text_preview}"
                    )
                    continue
        
        success_count = len(scored)
        total_count = len(paragraphs)
        
        if failed_count > 0:
            logger.warning(
                f"Scoring completed with {failed_count} failures. "
                f"Success: {success_count}/{total_count} paragraphs scored successfully."
            )
        else:
            logger.info(f"Stored {success_count} scores (all successful)")
        
        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Add summary to return value
        result = {
            "scored": scored,
            "success_count": success_count,
            "failed_count": failed_count,
            "total_count": total_count
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_top_scored_paragraphs(section_id: int, top_n: int = 5) -> List[Dict]:
    """
    Get the top N scored paragraphs for a section.
    
    Args:
        section_id: Database ID of the section
        top_n: Number of top paragraphs to return
        
    Returns:
        List of dicts with paragraph text, score, and position
    """
    db = SessionLocal()
    
    try:
        results = repo.get_top_scored_paragraphs(db, section_id, method="embedding", limit=top_n)
        
        top = []
        for para, score in results:
            top.append({
                "paragraph_id": para.id,
                "position": para.position,
                "score": float(score.score),
                "text": para.text
            })
        
        return top
        
    finally:
        db.close()


if __name__ == "__main__":
    # Test scoring on existing data
    import argparse
    
    parser = argparse.ArgumentParser(description="Score paragraphs in a section")
    parser.add_argument("--section-id", type=int, required=True, help="Section ID to score")
    parser.add_argument("--no-embeddings", action="store_true", help="Don't store embeddings")
    args = parser.parse_args()
    
    results = score_section_paragraphs(args.section_id, store_embeddings=not args.no_embeddings)
    
    print("\n" + "="*60)
    print("TOP SCORED PARAGRAPHS")
    print("="*60)
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {r['score']:.3f}")
        print(f"   {r['text_preview']}")


