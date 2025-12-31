# api/main.py
"""
FastAPI application for Infera SEC Filing Analysis.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger
from data.database import get_db, init_db
from data import repository as repo
from data.models import Company, Filing, Section, Summary

logger = get_logger(__name__)


# === Pydantic Models ===

class AnalyzeRequest(BaseModel):
    """Request body for analyze endpoint."""
    file_path: str
    ticker: Optional[str] = None
    skip_summary: bool = False
    skip_scoring: bool = False


class AnalyzeResponse(BaseModel):
    """Response from analyze endpoint."""
    ticker: str
    filing_id: int
    section_id: int
    paragraph_count: int
    word_count: int
    report_path: Optional[str] = None
    summary: Optional[str] = None


class ScoredParagraph(BaseModel):
    """A paragraph with its risk score."""
    paragraph_id: int
    position: int
    score: float
    text: str


class FilingDetail(BaseModel):
    """Detailed filing response."""
    id: int
    ticker: str
    filing_type: str
    filing_date: Optional[str] = None
    paragraph_count: int
    word_count: int
    summary: Optional[str] = None
    top_risks: List[ScoredParagraph]


class FilingSummary(BaseModel):
    """Brief filing info for list endpoint."""
    id: int
    ticker: str
    filing_type: str
    filing_date: Optional[str] = None
    has_summary: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Starting Infera API...")
    init_db()
    yield
    logger.info("Shutting down Infera API...")


# === App ===

app = FastAPI(
    title="Infera API",
    description="AI-powered SEC 10-K Filing Risk Analysis",
    version="1.0.0",
    lifespan=lifespan
)


# === Routes ===

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_filing(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Run the analysis pipeline on a 10-K filing.
    
    Provide a file path to an HTML 10-K filing and the pipeline will:
    1. Clean and extract text
    2. Segment the Risk Factors section
    3. Score paragraphs with embeddings
    4. Generate a GPT summary (optional)
    5. Create a Markdown report
    """
    from services.pipeline_service import run_analysis_pipeline
    
    # Validate file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    try:
        result = run_analysis_pipeline(
            filepath=request.file_path,
            ticker=request.ticker,
            skip_summary=request.skip_summary,
            skip_scoring=request.skip_scoring
        )
        
        return AnalyzeResponse(
            ticker=result["ticker"],
            filing_id=result["filing_id"],
            section_id=result["section_id"],
            paragraph_count=result["paragraph_count"],
            word_count=result["word_count"],
            report_path=result.get("report_path"),
            summary=result.get("summary")
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filings", response_model=List[FilingSummary], tags=["Filings"])
async def list_filings(db: Session = Depends(get_db)):
    """List all analyzed filings."""
    filings = db.query(Filing).all()
    
    result = []
    for f in filings:
        company = db.query(Company).filter(Company.id == f.company_id).first()
        summary = repo.get_summary_by_filing(db, f.id, "Item 1A")
        
        result.append(FilingSummary(
            id=f.id,
            ticker=company.ticker if company else "UNKNOWN",
            filing_type=f.filing_type,
            filing_date=f.filing_date.strftime("%Y-%m-%d") if f.filing_date else None,
            has_summary=summary is not None
        ))
    
    return result


@app.get("/filings/{filing_id}", response_model=FilingDetail, tags=["Filings"])
async def get_filing(filing_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific filing."""
    filing = repo.get_filing_by_id(db, filing_id)
    
    if not filing:
        raise HTTPException(status_code=404, detail=f"Filing not found: {filing_id}")
    
    company = db.query(Company).filter(Company.id == filing.company_id).first()
    sections = repo.get_sections_by_filing(db, filing_id)
    
    # Get Item 1A section
    risk_section = next((s for s in sections if s.section_type == "Item 1A"), None)
    
    # Get summary
    summary = repo.get_summary_by_filing(db, filing_id, "Item 1A")
    
    # Get top scored paragraphs
    top_risks = []
    if risk_section:
        scored = repo.get_top_scored_paragraphs(db, risk_section.id, method="embedding", limit=5)
        for para, score in scored:
            top_risks.append(ScoredParagraph(
                paragraph_id=para.id,
                position=para.position,
                score=score.score,
                text=para.text
            ))
    
    return FilingDetail(
        id=filing.id,
        ticker=company.ticker if company else "UNKNOWN",
        filing_type=filing.filing_type,
        filing_date=filing.filing_date.strftime("%Y-%m-%d") if filing.filing_date else None,
        paragraph_count=len(repo.get_paragraphs_by_section(db, risk_section.id)) if risk_section else 0,
        word_count=risk_section.word_count if risk_section else 0,
        summary=summary.summary_text if summary else None,
        top_risks=top_risks
    )


@app.get("/filings/{filing_id}/report", response_class=PlainTextResponse, tags=["Filings"])
async def get_filing_report(filing_id: int, db: Session = Depends(get_db)):
    """Get the Markdown report for a filing."""
    filing = repo.get_filing_by_id(db, filing_id)
    
    if not filing:
        raise HTTPException(status_code=404, detail=f"Filing not found: {filing_id}")
    
    company = db.query(Company).filter(Company.id == filing.company_id).first()
    ticker = company.ticker if company else "UNKNOWN"
    
    # Find the most recent report file for this ticker
    report_dir = "reports"
    if not os.path.exists(report_dir):
        raise HTTPException(status_code=404, detail="No reports directory found")
    
    # Look for reports matching this ticker
    reports = [f for f in os.listdir(report_dir) if f.startswith(ticker) and f.endswith("_risk_report.md")]
    
    if not reports:
        raise HTTPException(status_code=404, detail=f"No report found for ticker: {ticker}")
    
    # Get most recent
    reports.sort(reverse=True)
    report_path = os.path.join(report_dir, reports[0])
    
    with open(report_path, "r") as f:
        return f.read()


# === Main ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


