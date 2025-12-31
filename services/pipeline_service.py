# services/pipeline_service.py
"""
Main pipeline orchestrator that runs the full analysis and persists results.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, get_logger
from data.database import SessionLocal, init_db
from data import repository as repo
from data.models import Filing, Section, Paragraph, Summary

from analyze.cleaner import clean_html
from analyze.segmenter import get_risk_section

from openai import OpenAI

logger = get_logger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)


def extract_ticker_from_filename(filepath: str) -> str:
    """Extract ticker symbol from filename like 'AAPL_10K.html'."""
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if parts:
        return parts[0].upper()
    return "UNKNOWN"


def split_into_paragraphs(text: str, min_words: int = 30) -> list:
    """Split section text into paragraphs."""
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para and len(para.split()) >= min_words:
            paragraphs.append(para)
    return paragraphs


def summarize_text(text: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Generate GPT summary for the given text.
    Returns dict with summary_text, prompt_tokens, completion_tokens.
    """
    logger.info(f"Generating GPT summary using model: {model}")
    
    prompt = f"""Summarize the following risk factors from a company's 10-K filing into a clear, concise executive-level overview (3-5 bullet points max). Focus on the most serious risks to the business:

{text}

Summary:"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    summary_text = response.choices[0].message.content.strip()
    usage = response.usage
    
    logger.info(f"Summary generated. Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
    
    return {
        "summary_text": summary_text,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model": model
    }


def run_analysis_pipeline(
    filepath: str,
    ticker: Optional[str] = None,
    filing_date: Optional[datetime] = None,
    skip_summary: bool = False,
    skip_scoring: bool = False
) -> Dict[str, Any]:
    """
    Run the full analysis pipeline on a 10-K HTML file.
    
    Steps:
    1. Clean HTML -> extract text
    2. Segment -> extract Item 1A (Risk Factors)
    3. Split into paragraphs
    4. Store in database
    5. Score paragraphs with embeddings (optional)
    6. Generate GPT summary (optional)
    
    Args:
        filepath: Path to 10-K HTML file
        ticker: Company ticker (auto-extracted from filename if not provided)
        filing_date: Filing date (defaults to today if not provided)
        skip_summary: If True, skip GPT summarization step
        skip_scoring: If True, skip embedding-based scoring
        
    Returns:
        Dict with filing_id, section_id, paragraph_count, summary (if generated)
    """
    logger.info(f"Starting analysis pipeline for: {filepath}")
    
    # Initialize database
    init_db()
    
    # Extract ticker if not provided
    if not ticker:
        ticker = extract_ticker_from_filename(filepath)
    
    if not filing_date:
        filing_date = datetime.now()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Step 1: Get or create company
        logger.info(f"Processing ticker: {ticker}")
        company = repo.get_or_create_company(db, ticker)
        
        # Step 2: Create filing record
        filing = repo.create_filing(
            db,
            company_id=company.id,
            filing_type="10-K",
            filing_date=filing_date,
            source_file=filepath
        )
        logger.info(f"Created filing record: id={filing.id}")
        
        # Step 3: Clean HTML
        logger.info("Cleaning HTML...")
        cleaned_text = clean_html(filepath)
        logger.info(f"Cleaned text: {len(cleaned_text)} characters")
        
        # Step 4: Extract Risk Factors section
        logger.info("Extracting Item 1A (Risk Factors)...")
        risk_section = get_risk_section(cleaned_text)
        
        if not risk_section:
            raise ValueError("Could not extract Risk Factors section from filing")
        
        logger.info(f"Extracted section: {len(risk_section.split())} words")
        
        # Step 5: Store section
        section = repo.create_section(
            db,
            filing_id=filing.id,
            section_type="Item 1A",
            raw_text=risk_section
        )
        logger.info(f"Created section record: id={section.id}")
        
        # Step 6: Split into paragraphs and store
        paragraphs = split_into_paragraphs(risk_section)
        logger.info(f"Split into {len(paragraphs)} paragraphs")
        
        stored_paragraphs = repo.create_paragraphs_bulk(db, section.id, paragraphs)
        logger.info(f"Stored {len(stored_paragraphs)} paragraphs")
        
        result = {
            "ticker": ticker,
            "filing_id": filing.id,
            "section_id": section.id,
            "paragraph_count": len(stored_paragraphs),
            "word_count": section.word_count
        }
        
        # Step 7: Score paragraphs with embeddings (optional)
        if not skip_scoring:
            from services.scoring_service import score_section_paragraphs, get_top_scored_paragraphs
            
            logger.info("Scoring paragraphs with embeddings...")
            scores = score_section_paragraphs(section.id, store_embeddings=True)
            
            # Get top 5 for result
            top_scored = get_top_scored_paragraphs(section.id, top_n=5)
            result["top_scored"] = top_scored
            logger.info(f"Scored {len(scores)} paragraphs")
        
        # Step 8: Generate summary (optional)
        if not skip_summary:
            # Use top paragraphs for summary (first 10 or all if less)
            top_text = "\n\n".join(paragraphs[:10])
            summary_result = summarize_text(top_text)
            
            # Store summary
            summary = repo.create_summary(
                db,
                filing_id=filing.id,
                section_type="Item 1A",
                summary_text=summary_result["summary_text"],
                model=summary_result["model"],
                prompt_tokens=summary_result["prompt_tokens"],
                completion_tokens=summary_result["completion_tokens"]
            )
            logger.info(f"Created summary record: id={summary.id}")
            
            result["summary"] = summary_result["summary_text"]
            result["summary_id"] = summary.id
        
        # Step 9: Generate report
        from services.report_service import generate_report_from_pipeline_result
        
        report_path = generate_report_from_pipeline_result(result)
        result["report_path"] = report_path
        
        logger.info("Pipeline completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Infera analysis pipeline")
    parser.add_argument("--file", required=True, help="Path to 10-K HTML file")
    parser.add_argument("--ticker", help="Company ticker (optional)")
    parser.add_argument("--skip-summary", action="store_true", help="Skip GPT summarization")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip embedding scoring")
    args = parser.parse_args()
    
    result = run_analysis_pipeline(
        args.file,
        ticker=args.ticker,
        skip_summary=args.skip_summary,
        skip_scoring=args.skip_scoring
    )
    
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    for key, value in result.items():
        if key == "summary":
            print(f"\n{key}:\n{value}")
        else:
            print(f"{key}: {value}")

