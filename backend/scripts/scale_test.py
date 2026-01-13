#!/usr/bin/env python3
"""
Scale Test: Run Infera pipeline on 50+ SEC filings

This script:
1. Fetches 10-K filings from SEC EDGAR for 50+ companies
2. Runs the full analysis pipeline on each
3. Tracks success/failure rates and performance metrics
4. Generates a scale test report

Usage:
    python scripts/scale_test.py
    python scripts/scale_test.py --limit 100
    python scripts/scale_test.py --skip-existing
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger, settings
from ingest.sec_fetcher import SECFetcher
from data.database import SessionLocal
from data import repository as repo

logger = get_logger(__name__)

# S&P 500 subset + other major companies for scale testing
# Diverse industries: Tech, Finance, Healthcare, Consumer, Energy, Industrial
SCALE_TEST_TICKERS = [
    # Tech (15)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "CSCO", "IBM", "QCOM",
    # Finance (10)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "BLK",
    # Healthcare (10)
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer (10)
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Energy (5)
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial (5)
    "CAT", "BA", "HON", "UPS", "GE",
    # Additional for 60 total
    "DIS", "NFLX", "PYPL", "UBER", "SQ"
]


def run_scale_test(
    tickers: List[str] = None,
    limit: int = 60,
    skip_existing: bool = False,
    skip_summary: bool = True,
    output_file: str = "scale_test_results.json"
) -> Dict:
    """
    Run scale test on multiple SEC filings.
    
    Args:
        tickers: List of ticker symbols (defaults to SCALE_TEST_TICKERS)
        limit: Maximum number of filings to process
        skip_existing: Skip tickers already in database
        skip_summary: Skip GPT summarization (faster)
        output_file: Output file for results
        
    Returns:
        Dict with test results and metrics
    """
    tickers = tickers or SCALE_TEST_TICKERS[:limit]
    
    logger.info("="*60)
    logger.info("INFERA SCALE TEST")
    logger.info("="*60)
    logger.info(f"Target: {len(tickers)} companies")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"Skip summary: {skip_summary}")
    logger.info("="*60)
    
    # Initialize results
    results = {
        "start_time": datetime.utcnow().isoformat(),
        "target_count": len(tickers),
        "skip_existing": skip_existing,
        "skip_summary": skip_summary,
        "database_url": "postgresql",  # Postgres only
        "successes": [],
        "failures": [],
        "skipped": [],
        "metrics": {
            "total_paragraphs": 0,
            "total_words": 0,
            "total_time_seconds": 0,
            "avg_time_per_filing": 0,
            "avg_paragraphs_per_filing": 0,
        }
    }
    
    # Check existing
    db = SessionLocal()
    existing_tickers = set()
    if skip_existing:
        companies = db.query(repo.Company).all()
        existing_tickers = {c.ticker for c in companies}
        logger.info(f"Found {len(existing_tickers)} existing companies in database")
    db.close()
    
    # Initialize fetcher
    fetcher = SECFetcher()
    
    # Process each ticker
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        ticker = ticker.upper()
        logger.info(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")
        
        # Skip if exists
        if skip_existing and ticker in existing_tickers:
            logger.info(f"  ⏭️ Skipping {ticker} (already exists)")
            results["skipped"].append(ticker)
            continue
        
        # Process filing
        filing_start = time.time()
        try:
            result = fetcher.fetch_and_analyze(
                ticker=ticker,
                skip_summary=skip_summary
            )
            filing_time = time.time() - filing_start
            
            if result:
                logger.info(f"  ✅ {ticker}: {result['paragraph_count']} paragraphs, {result['word_count']} words ({filing_time:.1f}s)")
                results["successes"].append({
                    "ticker": ticker,
                    "filing_id": result.get("filing_id"),
                    "paragraph_count": result.get("paragraph_count", 0),
                    "word_count": result.get("word_count", 0),
                    "time_seconds": round(filing_time, 2)
                })
                results["metrics"]["total_paragraphs"] += result.get("paragraph_count", 0)
                results["metrics"]["total_words"] += result.get("word_count", 0)
            else:
                logger.warning(f"  ❌ {ticker}: No result returned")
                results["failures"].append({
                    "ticker": ticker,
                    "error": "No result returned",
                    "time_seconds": round(filing_time, 2)
                })
                
        except Exception as e:
            filing_time = time.time() - filing_start
            error_msg = str(e)[:200]  # Truncate long errors
            logger.error(f"  ❌ {ticker}: {error_msg}")
            results["failures"].append({
                "ticker": ticker,
                "error": error_msg,
                "time_seconds": round(filing_time, 2)
            })
    
    # Calculate final metrics
    total_time = time.time() - start_time
    results["metrics"]["total_time_seconds"] = round(total_time, 2)
    
    success_count = len(results["successes"])
    if success_count > 0:
        results["metrics"]["avg_time_per_filing"] = round(
            sum(s["time_seconds"] for s in results["successes"]) / success_count, 2
        )
        results["metrics"]["avg_paragraphs_per_filing"] = round(
            results["metrics"]["total_paragraphs"] / success_count, 1
        )
    
    results["end_time"] = datetime.utcnow().isoformat()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SCALE TEST COMPLETE")
    logger.info("="*60)
    logger.info(f"Successes: {len(results['successes'])}/{len(tickers)}")
    logger.info(f"Failures: {len(results['failures'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Total paragraphs: {results['metrics']['total_paragraphs']}")
    logger.info(f"Total words: {results['metrics']['total_words']:,}")
    if success_count > 0:
        logger.info(f"Avg time/filing: {results['metrics']['avg_time_per_filing']:.1f}s")
        logger.info(f"Avg paragraphs/filing: {results['metrics']['avg_paragraphs_per_filing']:.1f}")
    
    # Success rate
    attempted = len(tickers) - len(results["skipped"])
    if attempted > 0:
        success_rate = len(results["successes"]) / attempted * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
        results["metrics"]["success_rate"] = round(success_rate, 1)
    
    # Save results
    output_path = Path(output_file)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print failure summary
    if results["failures"]:
        logger.info("\nFailure details:")
        for f in results["failures"]:
            logger.info(f"  {f['ticker']}: {f['error'][:50]}...")
    
    return results


def print_database_summary():
    """Print current database contents summary."""
    db = SessionLocal()
    try:
        companies = db.query(repo.Company).all()
        filings = db.query(repo.Filing).all()
        paragraphs = db.query(repo.Paragraph).all()
        
        logger.info("\nDatabase Summary:")
        logger.info(f"  Companies: {len(companies)}")
        logger.info(f"  Filings: {len(filings)}")
        logger.info(f"  Paragraphs: {len(paragraphs)}")
        
        if companies:
            tickers = [c.ticker for c in companies]
            logger.info(f"  Tickers: {', '.join(sorted(tickers))}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Infera scale test on 50+ SEC filings"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of filings to process (default: 60)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers already in database"
    )
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="Generate GPT summaries (slower, uses API credits)"
    )
    parser.add_argument(
        "--output",
        default="scale_test_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Custom list of tickers to test"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print database summary, don't run test"
    )
    args = parser.parse_args()
    
    if args.summary_only:
        print_database_summary()
    else:
        results = run_scale_test(
            tickers=args.tickers,
            limit=args.limit,
            skip_existing=args.skip_existing,
            skip_summary=not args.with_summary,
            output_file=args.output
        )
        
        # Exit with error if too many failures
        success_rate = results["metrics"].get("success_rate", 0)
        if success_rate < 80:
            logger.warning(f"Success rate {success_rate}% below 80% threshold")
            sys.exit(1)

