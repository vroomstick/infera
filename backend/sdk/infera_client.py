"""
Infera Python SDK

A clean Python client for the Infera SEC Risk Analysis API.

Usage:
    from sdk.infera_client import InferaClient
    
    client = InferaClient(base_url="http://localhost:8000")
    
    # Analyze a filing
    result = client.analyze(file_path="data/AAPL_10K.html", ticker="AAPL")
    
    # Get explanation for a paragraph
    explanation = client.explain(paragraph_id=42)
    
    # Search for similar risks
    results = client.search("cybersecurity data breach", limit=10)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx


@dataclass
class TokenContribution:
    """A token and its contribution to the score."""
    token: str
    contribution: float
    position: int


@dataclass
class RiskExplanation:
    """Explanation of a paragraph's risk score."""
    paragraph_id: int
    text: str
    score: float
    confidence: float
    top_tokens: List[TokenContribution]
    risk_category: Optional[str]
    category_confidence: Optional[float]


@dataclass
class ScoredParagraph:
    """A paragraph with its risk score."""
    paragraph_id: int
    position: int
    score: float
    confidence: float
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """A search result."""
    paragraph_id: int
    filing_id: int
    ticker: str
    score: float
    text: str
    position: int


@dataclass
class AnalysisResult:
    """Result from analyzing a filing."""
    ticker: str
    filing_id: int
    section_id: int
    paragraph_count: int
    word_count: int
    report_path: Optional[str]
    summary: Optional[str]


@dataclass
class FilingSummary:
    """Brief filing info."""
    id: int
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    has_summary: bool


class InferaError(Exception):
    """Base exception for Infera client errors."""
    pass


class InferaClient:
    """
    Python client for the Infera SEC Risk Analysis API.
    
    Provides a clean interface for:
    - Analyzing 10-K filings
    - Explaining risk scores
    - Searching similar risks
    - Getting paragraph details
    
    Example:
        client = InferaClient()
        
        # Analyze a filing
        result = client.analyze("data/AAPL_10K.html", ticker="AAPL")
        print(f"Found {result.paragraph_count} paragraphs")
        
        # Explain a score
        explanation = client.explain(paragraph_id=42)
        for token in explanation.top_tokens[:5]:
            print(f"  {token.token}: +{token.contribution:.3f}")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the Infera client.
        
        Args:
            base_url: Base URL of the Infera API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client
    
    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an API request."""
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            raise InferaError(f"API error: {error_detail}")
        except httpx.RequestError as e:
            raise InferaError(f"Request failed: {e}")
    
    # === Health ===
    
    def health(self) -> Dict[str, str]:
        """Check API health."""
        return self._request("GET", "/health")
    
    # === Analysis ===
    
    def analyze(
        self,
        file_path: str,
        ticker: Optional[str] = None,
        skip_summary: bool = False,
        skip_scoring: bool = False,
    ) -> AnalysisResult:
        """
        Analyze a 10-K filing.
        
        Args:
            file_path: Path to the HTML filing (relative to data/ directory)
            ticker: Optional ticker symbol (extracted from filename if not provided)
            skip_summary: Skip GPT summarization
            skip_scoring: Skip risk scoring
            
        Returns:
            AnalysisResult with filing details and summary
        """
        data = self._request("POST", "/analyze", json={
            "file_path": file_path,
            "ticker": ticker,
            "skip_summary": skip_summary,
            "skip_scoring": skip_scoring,
        })
        return AnalysisResult(**data)
    
    def fetch(
        self,
        ticker: str,
        year: Optional[int] = None,
        analyze: bool = True,
        skip_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch a 10-K filing from SEC EDGAR.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL, TSLA)
            year: Fiscal year (e.g., 2023). If omitted, gets most recent.
            analyze: Run analysis pipeline after download
            skip_summary: Skip GPT summarization
            
        Returns:
            Fetch result with file path and optional analysis
        """
        return self._request("POST", "/fetch", json={
            "ticker": ticker,
            "year": year,
            "analyze": analyze,
            "skip_summary": skip_summary,
        })
    
    # === Filings ===
    
    def list_filings(self) -> List[FilingSummary]:
        """List all analyzed filings."""
        data = self._request("GET", "/filings")
        return [FilingSummary(**f) for f in data]
    
    def get_filing(self, filing_id: int) -> Dict[str, Any]:
        """Get detailed information about a filing."""
        return self._request("GET", f"/filings/{filing_id}")
    
    def get_report(self, filing_id: int) -> str:
        """Get the markdown report for a filing."""
        response = self.client.get(f"/filings/{filing_id}/report")
        response.raise_for_status()
        return response.text
    
    # === Explanation (Phase 5.1) ===
    
    def explain(
        self,
        paragraph_id: int,
        top_n: int = 10,
    ) -> RiskExplanation:
        """
        Explain why a paragraph received its risk score.
        
        Returns token-level attributions showing which words drive the score.
        
        Args:
            paragraph_id: Database ID of the paragraph
            top_n: Number of top contributing tokens to return
            
        Returns:
            RiskExplanation with top tokens and their contributions
        """
        data = self._request("GET", f"/explain/{paragraph_id}", params={"top_n": top_n})
        
        top_tokens = [TokenContribution(**t) for t in data.get("top_tokens", [])]
        
        return RiskExplanation(
            paragraph_id=data["paragraph_id"],
            text=data["text"],
            score=data["score"],
            confidence=data["confidence"],
            top_tokens=top_tokens,
            risk_category=data.get("risk_category"),
            category_confidence=data.get("category_confidence"),
        )
    
    def explain_text(
        self,
        text: str,
        top_n: int = 10,
    ) -> RiskExplanation:
        """
        Explain risk score for arbitrary text (not in database).
        
        Args:
            text: Risk paragraph text to analyze
            top_n: Number of top contributing tokens
            
        Returns:
            RiskExplanation for the text
        """
        data = self._request("POST", "/explain", params={"text": text, "top_n": top_n})
        
        top_tokens = [TokenContribution(**t) for t in data.get("top_tokens", [])]
        
        return RiskExplanation(
            paragraph_id=0,
            text=data["text"],
            score=data["score"],
            confidence=data["confidence"],
            top_tokens=top_tokens,
            risk_category=data.get("risk_category"),
            category_confidence=data.get("category_confidence"),
        )
    
    # === Paragraphs (Phase 5.2, 5.3) ===
    
    def get_paragraph(
        self,
        paragraph_id: int,
        include_embedding: bool = False,
    ) -> ScoredParagraph:
        """
        Get a paragraph with score and confidence.
        
        Args:
            paragraph_id: Database ID of the paragraph
            include_embedding: If true, include raw embedding vector
            
        Returns:
            ScoredParagraph with optional embedding
        """
        data = self._request(
            "GET",
            f"/paragraphs/{paragraph_id}",
            params={"include_embedding": include_embedding},
        )
        return ScoredParagraph(**data)
    
    # === Search ===
    
    def search(
        self,
        query: str,
        limit: int = 10,
        ticker: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantic search across all risk paragraphs.
        
        Args:
            query: Natural language search query
            limit: Maximum results to return
            ticker: Optional filter by company ticker
            
        Returns:
            List of matching paragraphs, sorted by similarity
        """
        params = {"q": query, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        
        data = self._request("GET", "/search", params=params)
        return [SearchResult(**r) for r in data.get("results", [])]
    
    # === Benchmark ===
    
    def benchmark(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Compare risk profiles across companies.
        
        Args:
            tickers: List of ticker symbols to compare
            
        Returns:
            Comparison with profiles, rankings, and insights
        """
        return self._request("GET", "/benchmark", params={"tickers": ",".join(tickers)})
    
    # === Cleanup ===
    
    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# === Convenience functions ===

def create_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
) -> InferaClient:
    """Create an Infera client."""
    return InferaClient(base_url=base_url, api_key=api_key)


if __name__ == "__main__":
    # Quick test
    print("Infera SDK")
    print("=" * 40)
    
    client = InferaClient()
    
    try:
        health = client.health()
        print(f"✅ API is {health['status']}")
        
        filings = client.list_filings()
        print(f"📁 {len(filings)} filings available")
        
        if filings:
            print(f"\nFilings: {', '.join(f.ticker for f in filings[:5])}")
    except InferaError as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

