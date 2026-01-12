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

logger = get_logger(__name__)

# Lazy OpenAI client initialization
_openai_client = None


def _get_openai_client():
    """Get OpenAI client, creating it lazily on first use."""
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required for summarization. "
                "Set it in your .env file or environment."
            )
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def extract_ticker_from_filename(filepath: str) -> str:
    """Extract ticker symbol from filename like 'AAPL_10K.html'."""
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if parts:
        return parts[0].upper()
    return "UNKNOWN"


def split_into_paragraphs(text: str, min_words: int = 30) -> list:
    """
    Split section text into paragraphs.
    
    Tries double newlines first, falls back to single newlines.
    Filters out short paragraphs (less than min_words).
    """
    paragraphs = []
    
    # Try splitting on double newlines first
    splits = text.split("\n\n")
    
    # If we only got 1 chunk, try single newlines instead
    if len(splits) <= 1:
        splits = text.split("\n")
    
    for para in splits:
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
    
    client = _get_openai_client()
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
    accession_number: Optional[str] = None,
    skip_summary: bool = False,
    skip_scoring: bool = False,
    force: bool = False,
    update: bool = False
) -> Dict[str, Any]:
    """
    Run the full analysis pipeline on a 10-K HTML file.
    
    Steps:
    1. Check for existing filing (idempotency)
    2. Clean HTML -> extract text
    3. Segment -> extract Item 1A (Risk Factors)
    4. Split into paragraphs
    5. Store in database
    6. Score paragraphs with embeddings (optional)
    7. Generate GPT summary (optional)
    
    Args:
        filepath: Path to 10-K HTML file
        ticker: Company ticker (auto-extracted from filename if not provided)
        filing_date: Filing date (defaults to today if not provided)
        accession_number: SEC accession number (for duplicate detection)
        skip_summary: If True, skip GPT summarization step
        skip_scoring: If True, skip embedding-based scoring
        force: If True, wipe derived data and reprocess even if filing exists
        update: If True, recompute scores/embeddings but preserve filing metadata
        
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
        
        # Step 2: Check for existing filing (idempotency)
        existing_filing = None
        if accession_number:
            existing_filing = repo.get_filing_by_accession_number(db, accession_number)
            if existing_filing:
                logger.info(f"Found existing filing by accession_number: {accession_number}")
        elif filing_date:
            existing_filing = repo.get_filing_by_ticker_and_date(db, ticker, filing_date)
            if existing_filing:
                logger.info(f"Found existing filing by ticker+date: {ticker} {filing_date}")
        
        if existing_filing:
            if force:
                logger.info("--force flag: wiping derived data and reprocessing...")
                # Delete derived data: sections, paragraphs, scores, summaries
                from data.models import Section, Paragraph, Score, Summary, ScoreVector
                sections = repo.get_sections_by_filing(db, existing_filing.id)
                for section in sections:
                    paragraphs = repo.get_paragraphs_by_section(db, section.id)
                    for para in paragraphs:
                        # Delete scores and score vectors
                        db.query(Score).filter(Score.paragraph_id == para.id).delete()
                        db.query(ScoreVector).filter(ScoreVector.paragraph_id == para.id).delete()
                    # Delete paragraphs
                    db.query(Paragraph).filter(Paragraph.section_id == section.id).delete()
                # Delete sections and summaries
                db.query(Section).filter(Section.filing_id == existing_filing.id).delete()
                db.query(Summary).filter(Summary.filing_id == existing_filing.id).delete()
                db.commit()
                logger.info("Deleted all derived data, will reprocess...")
                filing = existing_filing
            elif update:
                logger.info("--update flag: will recompute scores/embeddings...")
                filing = existing_filing
            else:
                logger.info(f"Filing already exists (id={existing_filing.id}), skipping. Use --force to reprocess or --update to recompute scores.")
                db.close()
                return {
                    "ticker": ticker,
                    "filing_id": existing_filing.id,
                    "message": "Filing already exists, skipped. Use --force to reprocess or --update to recompute scores.",
                    "skipped": True
                }
        else:
            # Step 3: Create new filing record
            filing = repo.create_filing(
                db,
                company_id=company.id,
                filing_type="10-K",
                filing_date=filing_date,
                accession_number=accession_number,
                source_file=filepath
            )
            logger.info(f"Created filing record: id={filing.id}")
        
        # Step 4: Clean HTML
        logger.info("Cleaning HTML...")
        cleaned_text = clean_html(filepath)
        logger.info(f"Cleaned text: {len(cleaned_text)} characters")
        
        # Step 5: Extract Risk Factors section
        logger.info("Extracting Item 1A (Risk Factors)...")
        risk_section = get_risk_section(cleaned_text)
        
        if not risk_section:
            raise ValueError("Could not extract Risk Factors section from filing")
        
        logger.info(f"Extracted section: {len(risk_section.split())} words")
        
        # Step 6: Store section (or reuse existing if --update)
        existing_sections = repo.get_sections_by_filing(db, filing.id)
        section = None
        for s in existing_sections:
            if s.section_type == "Item 1A":
                section = s
                logger.info(f"Reusing existing section: id={section.id}")
                break
        
        if not section:
            section = repo.create_section(
                db,
                filing_id=filing.id,
                section_type="Item 1A",
                raw_text=risk_section
            )
            logger.info(f"Created section record: id={section.id}")
        
        # Step 7: Split into paragraphs and store (only if new section or --force)
        if not existing_filing or force or not repo.get_paragraphs_by_section(db, section.id):
            paragraphs = split_into_paragraphs(risk_section)
            logger.info(f"Split into {len(paragraphs)} paragraphs")
            
            stored_paragraphs = repo.create_paragraphs_bulk(db, section.id, paragraphs)
            logger.info(f"Stored {len(stored_paragraphs)} paragraphs")
        else:
            stored_paragraphs = repo.get_paragraphs_by_section(db, section.id)
            logger.info(f"Reusing existing {len(stored_paragraphs)} paragraphs")
        
        result = {
            "ticker": ticker,
            "filing_id": filing.id,
            "section_id": section.id,
            "paragraph_count": len(stored_paragraphs),
            "word_count": section.word_count
        }
        
        # Step 8: Score paragraphs with embeddings (optional, always run if --update)
        if not skip_scoring or update:
            from services.scoring_service import score_section_paragraphs, get_top_scored_paragraphs
            
            logger.info("Scoring paragraphs with embeddings...")
            scoring_result = score_section_paragraphs(section.id, store_embeddings=True)
            
            # Handle new return format (dict with scored, success_count, etc.)
            if isinstance(scoring_result, dict):
                scores = scoring_result.get("scored", [])
                success_count = scoring_result.get("success_count", 0)
                failed_count = scoring_result.get("failed_count", 0)
                total_count = scoring_result.get("total_count", 0)
                
                if failed_count > 0:
                    logger.warning(
                        f"Scoring completed with {failed_count} failures. "
                        f"Success: {success_count}/{total_count} paragraphs scored successfully."
                    )
                else:
                    logger.info(f"Scored {success_count}/{total_count} paragraphs successfully")
            else:
                # Backward compatibility with old return format
                scores = scoring_result
                logger.info(f"Scored {len(scores)} paragraphs")
            
            # Get top 5 for result
            top_scored = get_top_scored_paragraphs(section.id, top_n=5)
            result["top_scored"] = top_scored
        
        # Step 9: Generate summary (optional)
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
        
        # Step 10: Generate report
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
    parser.add_argument("--accession-number", help="SEC accession number (for duplicate detection)")
    parser.add_argument("--skip-summary", action="store_true", help="Skip GPT summarization")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip embedding scoring")
    parser.add_argument("--force", action="store_true", help="Wipe derived data and reprocess even if filing exists")
    parser.add_argument("--update", action="store_true", help="Recompute scores/embeddings but preserve filing metadata")
    args = parser.parse_args()
    
    result = run_analysis_pipeline(
        args.file,
        ticker=args.ticker,
        accession_number=getattr(args, 'accession_number', None),
        skip_summary=args.skip_summary,
        skip_scoring=args.skip_scoring,
        force=args.force,
        update=args.update
    )
    
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    for key, value in result.items():
        if key == "summary":
            print(f"\n{key}:\n{value}")
        else:
            print(f"{key}: {value}")

