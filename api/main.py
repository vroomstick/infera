# api/main.py
"""
FastAPI application for Infera SEC Filing Analysis.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Security
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, get_logger
from data.database import get_db, init_db
from data import repository as repo
from data.models import Company, Filing, Section, Paragraph, Score, Summary

logger = get_logger(__name__)

# === Security Configuration ===

# API Key authentication (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key if authentication is enabled.
    Set INFERA_API_KEY environment variable to enable.
    """
    if not settings.require_api_key:
        return None  # Auth disabled
    
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API key required"}
        )
    return api_key


# Rate limiter setup
if RATE_LIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


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


# --- Semantic Search Models ---

class SearchResult(BaseModel):
    """A paragraph matching a search query."""
    paragraph_id: int
    filing_id: int
    ticker: str
    score: float
    text: str
    position: int


class SearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: List[SearchResult]
    count: int


# --- Benchmark Models ---

class RiskStatistics(BaseModel):
    """Statistical summary of risk scores."""
    mean: float
    median: float
    std: float
    min: float
    max: float


class RiskProfile(BaseModel):
    """Risk profile for a company."""
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    filing_id: Optional[int] = None
    total_paragraphs: Optional[int] = None
    distribution: Optional[Dict[str, int]] = None
    percentages: Optional[Dict[str, float]] = None
    statistics: Optional[RiskStatistics] = None
    error: Optional[str] = None


class ComparisonData(BaseModel):
    """Comparison rankings and insights."""
    rankings: Dict[str, List[str]]
    insights: List[str]


class PeerComparisonResponse(BaseModel):
    """Peer comparison response."""
    profiles: Dict[str, RiskProfile]
    comparison: Optional[ComparisonData] = None
    error: Optional[str] = None


# --- YoY Trends Models ---

class RiskChange(BaseModel):
    """A new or removed risk."""
    paragraph_id: int
    text_preview: str
    best_prior_match_similarity: Optional[float] = None
    best_current_match_similarity: Optional[float] = None
    interpretation: str


class RiskChanges(BaseModel):
    """Collection of risk changes."""
    count: int
    details: List[RiskChange]


class NarrativeDrift(BaseModel):
    """Narrative drift analysis."""
    score: float
    similarity: float
    interpretation: str


class YoYSummary(BaseModel):
    """Summary of year-over-year changes."""
    total_changes: int
    change_rate: float


class YoYAnalysisResponse(BaseModel):
    """Year-over-year risk analysis response."""
    current_filing_id: int
    prior_filing_id: int
    current_paragraph_count: int
    prior_paragraph_count: int
    new_risks: RiskChanges
    removed_risks: RiskChanges
    narrative_drift: NarrativeDrift
    summary: YoYSummary


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

# === Middleware ===

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting middleware
if RATE_LIMIT_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# === Error Handlers ===

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that sanitizes errors in production.
    Prevents leaking stack traces and internal details.
    """
    if settings.is_production:
        # Production: generic error message
        logger.error(f"Unhandled error: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."}
        )
    else:
        # Development: show full error for debugging
        logger.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": type(exc).__name__}
        )


# === Security Helpers ===

# Allowed directories for file operations
ALLOWED_DATA_DIRS = [
    os.path.abspath("data"),
    os.path.abspath(".sec_cache"),
]


def validate_file_path(file_path: str) -> str:
    """
    Validate that a file path is within allowed directories.
    
    Prevents path traversal attacks (e.g., reading /etc/passwd).
    
    Args:
        file_path: User-provided file path
        
    Returns:
        Absolute path if valid
        
    Raises:
        HTTPException: If path is outside allowed directories or doesn't exist
    """
    abs_path = os.path.abspath(file_path)
    
    # Check if path is within any allowed directory
    is_allowed = any(
        abs_path.startswith(allowed_dir + os.sep) or abs_path == allowed_dir
        for allowed_dir in ALLOWED_DATA_DIRS
    )
    
    if not is_allowed:
        logger.warning(f"Path traversal attempt blocked: {file_path}")
        raise HTTPException(
            status_code=403, 
            detail="File path not allowed. Files must be in the data/ directory."
        )
    
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    if os.path.isdir(abs_path):
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")
    
    return abs_path


# === Routes ===

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_filing(
    request: AnalyzeRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Run the analysis pipeline on a 10-K filing.
    
    Provide a file path to an HTML 10-K filing and the pipeline will:
    1. Clean and extract text
    2. Segment the Risk Factors section
    3. Score paragraphs with embeddings
    4. Generate a GPT summary (optional)
    5. Create a Markdown report
    
    Note: File must be in the data/ directory (path traversal protection).
    """
    from services.pipeline_service import run_analysis_pipeline
    
    # Validate file path (prevents path traversal attacks)
    validated_path = validate_file_path(request.file_path)
    
    try:
        result = run_analysis_pipeline(
            filepath=validated_path,
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


# === Analytics Endpoints ===

@app.get("/search", response_model=SearchResponse, tags=["Analytics"])
async def semantic_search(
    q: str,
    limit: int = 10,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Semantic search across all risk paragraphs.
    
    Find paragraphs similar to a natural language query like:
    - "cybersecurity threats and data breaches"
    - "supply chain disruption"
    - "regulatory compliance issues"
    
    Args:
        q: Natural language search query
        limit: Maximum results to return (default 10)
        ticker: Optional filter by company ticker (AAPL, TSLA, MSFT)
    """
    from services.scoring_service import embed_text
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(q.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    
    # Embed the query
    query_embedding = embed_text(q).reshape(1, -1)
    
    # Build base query for paragraphs with embeddings
    query = db.query(Paragraph, Score, Filing, Company).join(
        Score, Paragraph.id == Score.paragraph_id
    ).join(
        Section, Paragraph.section_id == Section.id
    ).join(
        Filing, Section.filing_id == Filing.id
    ).join(
        Company, Filing.company_id == Company.id
    ).filter(
        Score.method == "embedding",
        Score.embedding.isnot(None)
    )
    
    # Filter by ticker if specified
    if ticker:
        query = query.filter(Company.ticker == ticker.upper())
    
    # Fetch all paragraphs with embeddings
    results = query.all()
    
    if not results:
        return SearchResponse(query=q, results=[], count=0)
    
    # Compute similarities
    similarities = []
    for para, score, filing, company in results:
        if score.embedding:
            para_embedding = pickle.loads(score.embedding).reshape(1, -1)
            sim = cosine_similarity(query_embedding, para_embedding)[0][0]
            similarities.append({
                "paragraph_id": para.id,
                "filing_id": filing.id,
                "ticker": company.ticker,
                "score": float(sim),
                "text": para.text,
                "position": para.position
            })
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results
    top_results = similarities[:limit]
    
    return SearchResponse(
        query=q,
        results=[SearchResult(**r) for r in top_results],
        count=len(top_results)
    )


@app.get("/benchmark", response_model=PeerComparisonResponse, tags=["Analytics"])
async def peer_benchmark(
    tickers: str = "AAPL,TSLA,MSFT",
    db: Session = Depends(get_db)
):
    """
    Compare risk profiles across companies.
    
    Provides:
    - Risk distribution (high/medium/low) per company
    - Statistical summary (mean, median, std)
    - Rankings and comparative insights
    
    Args:
        tickers: Comma-separated list of ticker symbols
    """
    from services.benchmark_service import compare_peers
    
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    if len(ticker_list) < 1:
        raise HTTPException(status_code=400, detail="At least one ticker required")
    
    result = compare_peers(ticker_list)
    
    # Convert to response model format
    profiles = {}
    for ticker, profile in result["profiles"].items():
        if "error" in profile:
            profiles[ticker] = RiskProfile(error=profile["error"])
        else:
            profiles[ticker] = RiskProfile(
                ticker=profile.get("ticker"),
                company_name=profile.get("company_name"),
                filing_id=profile.get("filing_id"),
                total_paragraphs=profile.get("total_paragraphs"),
                distribution=profile.get("distribution"),
                percentages=profile.get("percentages"),
                statistics=RiskStatistics(**profile["statistics"]) if profile.get("statistics") else None
            )
    
    comparison = None
    if result.get("comparison"):
        comparison = ComparisonData(
            rankings=result["comparison"]["rankings"],
            insights=result["comparison"]["insights"]
        )
    
    return PeerComparisonResponse(
        profiles=profiles,
        comparison=comparison,
        error=result.get("error")
    )


@app.get("/filings/{filing_id}/trends", tags=["Analytics"])
async def yoy_trends(
    filing_id: int,
    prior_filing_id: int,
    db: Session = Depends(get_db)
):
    """
    Analyze year-over-year changes in risk disclosures.
    
    Compares two filings to detect:
    - New risks added in current year
    - Risks removed from prior year
    - Overall narrative drift
    
    Args:
        filing_id: Current year filing ID
        prior_filing_id: Prior year filing ID to compare against
    """
    from services.trend_service import analyze_yoy_changes
    
    # Validate both filings exist
    current = repo.get_filing_by_id(db, filing_id)
    prior = repo.get_filing_by_id(db, prior_filing_id)
    
    if not current:
        raise HTTPException(status_code=404, detail=f"Current filing not found: {filing_id}")
    if not prior:
        raise HTTPException(status_code=404, detail=f"Prior filing not found: {prior_filing_id}")
    
    result = analyze_yoy_changes(filing_id, prior_filing_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


# === Fetch Endpoint ===

class FetchRequest(BaseModel):
    """Request body for fetch endpoint."""
    ticker: str
    year: Optional[int] = None
    analyze: bool = True
    skip_summary: bool = True


class FetchResponse(BaseModel):
    """Response from fetch endpoint."""
    ticker: str
    year: Optional[int] = None
    filepath: str
    filing_id: Optional[int] = None
    paragraph_count: Optional[int] = None
    word_count: Optional[int] = None
    message: str


@app.post("/fetch", response_model=FetchResponse, tags=["Ingest"])
async def fetch_filing(request: FetchRequest, api_key: str = Depends(verify_api_key)):
    """
    Fetch a 10-K filing from SEC EDGAR.
    
    Downloads the filing by ticker and optionally runs the analysis pipeline.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
        year: Fiscal year (e.g., 2023). If omitted, gets most recent.
        analyze: Run analysis pipeline after download (default: true)
        skip_summary: Skip GPT summarization (default: true)
    
    Example:
        POST /fetch {"ticker": "NVDA", "year": 2023}
    """
    from ingest.sec_fetcher import SECFetcher
    
    ticker = request.ticker.upper()
    
    try:
        fetcher = SECFetcher()
        
        if request.analyze:
            result = fetcher.fetch_and_analyze(
                ticker=ticker,
                year=request.year,
                skip_summary=request.skip_summary
            )
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Could not fetch 10-K for {ticker}")
            
            return FetchResponse(
                ticker=ticker,
                year=request.year,
                filepath=result.get("report_path", ""),
                filing_id=result.get("filing_id"),
                paragraph_count=result.get("paragraph_count"),
                word_count=result.get("word_count"),
                message=f"Successfully fetched and analyzed {ticker} 10-K"
            )
        else:
            filepath = fetcher.download_filing(ticker=ticker, year=request.year)
            
            if not filepath:
                raise HTTPException(status_code=404, detail=f"Could not fetch 10-K for {ticker}")
            
            return FetchResponse(
                ticker=ticker,
                year=request.year,
                filepath=filepath,
                message=f"Successfully downloaded {ticker} 10-K to {filepath}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Main ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


