# services/benchmark_service.py
"""
Peer benchmarking service for comparing risk profiles across companies.
"""

import os
import sys
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger
from data.database import SessionLocal
from data import repository as repo
from data.models import Company, Filing, Section, Paragraph, Score

logger = get_logger(__name__)

# Score thresholds (same as scoring_service.py)
HIGH_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.45


def classify_score(score: float) -> str:
    """Classify a score into severity category."""
    if score >= HIGH_THRESHOLD:
        return "high"
    elif score >= MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "low"


def get_filing_risk_profile(filing_id: int) -> Dict:
    """
    Compute risk distribution for a single filing.
    
    Returns:
        Dict with risk counts, percentages, and statistics
    """
    db = SessionLocal()
    try:
        # Get the Item 1A section for this filing
        sections = repo.get_sections_by_filing(db, filing_id)
        risk_section = next((s for s in sections if s.section_type == "Item 1A"), None)
        
        if not risk_section:
            return {"error": f"No Item 1A section found for filing {filing_id}"}
        
        # Get all scored paragraphs
        paragraphs = repo.get_paragraphs_by_section(db, risk_section.id)
        
        scores = []
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for para in paragraphs:
            score_obj = db.query(Score).filter(
                Score.paragraph_id == para.id,
                Score.method == "embedding"
            ).first()
            
            if score_obj:
                scores.append(score_obj.score)
                category = classify_score(score_obj.score)
                distribution[category] += 1
        
        total = len(scores)
        if total == 0:
            return {"error": "No scored paragraphs found"}
        
        return {
            "filing_id": filing_id,
            "total_paragraphs": total,
            "distribution": distribution,
            "percentages": {
                k: round(v / total * 100, 1) for k, v in distribution.items()
            },
            "statistics": {
                "mean": round(float(np.mean(scores)), 4),
                "median": round(float(np.median(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "min": round(float(min(scores)), 4),
                "max": round(float(max(scores)), 4)
            }
        }
    finally:
        db.close()


def get_company_risk_profile(ticker: str) -> Dict:
    """Get risk profile for a company (latest filing)."""
    db = SessionLocal()
    try:
        company = repo.get_company_by_ticker(db, ticker)
        if not company:
            return {"error": f"Company not found: {ticker}"}
        
        # Get most recent filing
        filings = repo.get_filings_by_ticker(db, ticker)
        if not filings:
            return {"error": f"No filings found for {ticker}"}
        
        latest_filing = sorted(filings, key=lambda f: f.filing_date or f.created_at, reverse=True)[0]
        
        profile = get_filing_risk_profile(latest_filing.id)
        if "error" not in profile:
            profile["ticker"] = ticker
            profile["company_name"] = company.name
        
        return profile
    finally:
        db.close()


def compare_peers(tickers: List[str]) -> Dict:
    """
    Compare risk profiles across multiple companies.
    
    Args:
        tickers: List of ticker symbols to compare
        
    Returns:
        Dict with individual profiles and comparative analysis
    """
    profiles = {}
    for ticker in tickers:
        profiles[ticker] = get_company_risk_profile(ticker)
    
    # Filter out errors
    valid_profiles = {k: v for k, v in profiles.items() if "error" not in v}
    
    if len(valid_profiles) < 2:
        return {
            "profiles": profiles,
            "comparison": None,
            "error": "Need at least 2 valid profiles for comparison"
        }
    
    # Comparative analysis
    comparison = {
        "rankings": {},
        "insights": []
    }
    
    # Rank by high-risk percentage
    by_high_risk = sorted(
        valid_profiles.items(),
        key=lambda x: x[1]["percentages"]["high"],
        reverse=True
    )
    comparison["rankings"]["by_high_risk_pct"] = [t for t, _ in by_high_risk]
    
    # Rank by mean score
    by_mean_score = sorted(
        valid_profiles.items(),
        key=lambda x: x[1]["statistics"]["mean"],
        reverse=True
    )
    comparison["rankings"]["by_mean_score"] = [t for t, _ in by_mean_score]
    
    # Generate insights
    highest_risk = by_high_risk[0]
    lowest_risk = by_high_risk[-1]
    
    if highest_risk[1]["percentages"]["high"] > 0:
        diff = highest_risk[1]["percentages"]["high"] - lowest_risk[1]["percentages"]["high"]
        comparison["insights"].append(
            f"{highest_risk[0]} has {diff:.1f}% more high-severity risks than {lowest_risk[0]}"
        )
    
    # Compare mean scores
    mean_diff = highest_risk[1]["statistics"]["mean"] - lowest_risk[1]["statistics"]["mean"]
    comparison["insights"].append(
        f"Mean risk score ranges from {lowest_risk[1]['statistics']['mean']:.3f} ({lowest_risk[0]}) "
        f"to {highest_risk[1]['statistics']['mean']:.3f} ({highest_risk[0]})"
    )
    
    # Score volatility comparison
    by_std = sorted(valid_profiles.items(), key=lambda x: x[1]["statistics"]["std"], reverse=True)
    comparison["insights"].append(
        f"{by_std[0][0]} shows highest risk score variability (σ={by_std[0][1]['statistics']['std']:.3f})"
    )
    
    return {
        "profiles": profiles,
        "comparison": comparison
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Compare risk profiles across companies")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "MSFT"], 
                        help="Ticker symbols to compare")
    args = parser.parse_args()
    
    result = compare_peers(args.tickers)
    print(json.dumps(result, indent=2, default=str))

