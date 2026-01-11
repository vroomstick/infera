# services/trend_service.py
"""
Year-over-Year trend analysis for risk factor changes.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger
from data.database import SessionLocal
from data import repository as repo
from data.models import Company, Filing, Section, Paragraph, Score
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)

# Thresholds for trend detection
NEW_RISK_THRESHOLD = 0.65      # Below this = likely new risk
REMOVED_RISK_THRESHOLD = 0.65  # Similar paragraph in prior year must exceed this
DRIFT_THRESHOLD = 0.15         # Significant narrative change


def get_filing_embeddings(filing_id: int) -> Dict[int, np.ndarray]:
    """
    Get all paragraph embeddings for a filing.
    
    Returns:
        Dict mapping paragraph_id to embedding array
    """
    db = SessionLocal()
    try:
        sections = repo.get_sections_by_filing(db, filing_id)
        risk_section = next((s for s in sections if s.section_type == "Item 1A"), None)
        
        if not risk_section:
            return {}
        
        paragraphs = repo.get_paragraphs_by_section(db, risk_section.id)
        
        embeddings = {}
        for para in paragraphs:
            score = db.query(Score).filter(
                Score.paragraph_id == para.id,
                Score.method == "embedding",
                Score.embedding.isnot(None)
            ).first()
            
            if score and score.embedding:
                embeddings[para.id] = pickle.loads(score.embedding)
        
        return embeddings
    finally:
        db.close()


def get_paragraph_texts(filing_id: int) -> Dict[int, str]:
    """Get paragraph texts for a filing."""
    db = SessionLocal()
    try:
        sections = repo.get_sections_by_filing(db, filing_id)
        risk_section = next((s for s in sections if s.section_type == "Item 1A"), None)
        
        if not risk_section:
            return {}
        
        paragraphs = repo.get_paragraphs_by_section(db, risk_section.id)
        return {p.id: p.text for p in paragraphs}
    finally:
        db.close()


def find_best_match(embedding: np.ndarray, candidates: Dict[int, np.ndarray]) -> Tuple[int, float]:
    """
    Find the best matching paragraph from candidates.
    
    Returns:
        (paragraph_id, similarity_score) or (-1, 0.0) if no candidates
    """
    if not candidates:
        return (-1, 0.0)
    
    embedding = embedding.reshape(1, -1)
    best_id = -1
    best_sim = 0.0
    
    for para_id, candidate_emb in candidates.items():
        sim = cosine_similarity(embedding, candidate_emb.reshape(1, -1))[0][0]
        if sim > best_sim:
            best_sim = sim
            best_id = para_id
    
    return (best_id, float(best_sim))


def analyze_yoy_changes(
    current_filing_id: int,
    prior_filing_id: int
) -> Dict:
    """
    Analyze year-over-year changes in risk disclosures.
    
    Detects:
    - New risks (low similarity to any prior paragraph)
    - Removed risks (prior paragraphs with no current match)
    - Narrative drift (overall semantic shift)
    
    Args:
        current_filing_id: ID of current year's filing
        prior_filing_id: ID of prior year's filing
        
    Returns:
        Dict with new_risks, removed_risks, drift_score, and detailed analysis
    """
    logger.info(f"Analyzing YoY changes: {prior_filing_id} -> {current_filing_id}")
    
    # Get embeddings for both filings
    current_embeddings = get_filing_embeddings(current_filing_id)
    prior_embeddings = get_filing_embeddings(prior_filing_id)
    
    if not current_embeddings or not prior_embeddings:
        return {
            "error": "Missing embeddings for one or both filings",
            "current_count": len(current_embeddings),
            "prior_count": len(prior_embeddings)
        }
    
    # Get texts for context
    current_texts = get_paragraph_texts(current_filing_id)
    prior_texts = get_paragraph_texts(prior_filing_id)
    
    # Find new risks (current paragraphs with low similarity to all prior)
    new_risks = []
    for para_id, embedding in current_embeddings.items():
        best_match_id, best_sim = find_best_match(embedding, prior_embeddings)
        
        if best_sim < NEW_RISK_THRESHOLD:
            text = current_texts.get(para_id, "")
            new_risks.append({
                "paragraph_id": para_id,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "best_prior_match_similarity": round(best_sim, 4),
                "interpretation": "Likely new risk not present in prior year"
            })
    
    # Find removed risks (prior paragraphs with low similarity to all current)
    removed_risks = []
    for para_id, embedding in prior_embeddings.items():
        best_match_id, best_sim = find_best_match(embedding, current_embeddings)
        
        if best_sim < REMOVED_RISK_THRESHOLD:
            text = prior_texts.get(para_id, "")
            removed_risks.append({
                "paragraph_id": para_id,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "best_current_match_similarity": round(best_sim, 4),
                "interpretation": "Risk disclosure no longer present or significantly modified"
            })
    
    # Compute overall narrative drift
    # Average embedding for each filing, then compute distance
    current_centroid = np.mean(list(current_embeddings.values()), axis=0).reshape(1, -1)
    prior_centroid = np.mean(list(prior_embeddings.values()), axis=0).reshape(1, -1)
    
    drift_similarity = float(cosine_similarity(current_centroid, prior_centroid)[0][0])
    drift_score = 1.0 - drift_similarity  # Higher = more drift
    
    # Interpretation
    if drift_score > DRIFT_THRESHOLD:
        drift_interpretation = "Significant narrative shift detected"
    elif drift_score > 0.05:
        drift_interpretation = "Moderate narrative evolution"
    else:
        drift_interpretation = "Stable risk narrative"
    
    return {
        "current_filing_id": current_filing_id,
        "prior_filing_id": prior_filing_id,
        "current_paragraph_count": len(current_embeddings),
        "prior_paragraph_count": len(prior_embeddings),
        "new_risks": {
            "count": len(new_risks),
            "details": new_risks[:10]  # Limit output
        },
        "removed_risks": {
            "count": len(removed_risks),
            "details": removed_risks[:10]
        },
        "narrative_drift": {
            "score": round(drift_score, 4),
            "similarity": round(drift_similarity, 4),
            "interpretation": drift_interpretation
        },
        "summary": {
            "total_changes": len(new_risks) + len(removed_risks),
            "change_rate": round((len(new_risks) + len(removed_risks)) / 
                                 (len(current_embeddings) + len(prior_embeddings)) * 100, 1)
        }
    }


def get_ticker_filing_history(ticker: str) -> List[Dict]:
    """Get all filings for a ticker, sorted by date."""
    db = SessionLocal()
    try:
        filings = repo.get_filings_by_ticker(db, ticker)
        return sorted([
            {
                "id": f.id,
                "filing_date": f.filing_date.isoformat() if f.filing_date else None,
                "filing_type": f.filing_type
            }
            for f in filings
        ], key=lambda x: x["filing_date"] or "", reverse=True)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Analyze YoY risk changes")
    parser.add_argument("--current", type=int, required=True, help="Current year filing ID")
    parser.add_argument("--prior", type=int, required=True, help="Prior year filing ID")
    args = parser.parse_args()
    
    result = analyze_yoy_changes(args.current, args.prior)
    print(json.dumps(result, indent=2, default=str))

