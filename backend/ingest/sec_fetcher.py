# ingest/sec_fetcher.py
"""
Robust SEC EDGAR filing fetcher.

Downloads 10-K filings from SEC EDGAR with:
- Rate limiting (respects SEC's ~10 req/sec guidance)
- Exponential backoff + jitter for 429/503/403
- Response caching (24h TTL for submissions)
- Deterministic file storage by CIK/accession
- Content-Type validation

SEC EDGAR Requirements:
- User-Agent header with contact info is REQUIRED
- Rate limit: ~10 requests/second max
- All lookups are CIK-based (10-digit zero-padded)

Usage:
    python ingest/sec_fetcher.py --ticker AAPL --year 2023
    python ingest/sec_fetcher.py --ticker NVDA  # Gets most recent
"""

import os
import sys
import re
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_logger

logger = get_logger(__name__)

# SEC EDGAR URLs
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Default User-Agent (users should customize this)
DEFAULT_USER_AGENT = "Infera Research Project (contact: infera-research@example.com)"

# Common ticker -> CIK mappings (fallback cache)
KNOWN_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "V": "0001403161",
    "JNJ": "0000200406",
    "WMT": "0000104169",
    "PG": "0000080424",
    "UNH": "0000731766",
    "HD": "0000354950",
    "MA": "0001141391",
    "BAC": "0000070858",
    "XOM": "0000034088",
    "PFE": "0000078003",
    "KO": "0000021344",
}


def normalize_cik(cik_or_ticker: str) -> str:
    """
    Normalize a CIK or ticker to 10-digit zero-padded CIK format.
    
    Args:
        cik_or_ticker: Either a CIK (any format) or ticker symbol
        
    Returns:
        10-digit zero-padded CIK string (e.g., "0000320193")
    """
    value = str(cik_or_ticker).strip().upper()
    
    # If it's all digits, treat as CIK
    if value.isdigit():
        return value.zfill(10)
    
    # Otherwise, look up ticker in known mappings
    if value in KNOWN_CIKS:
        return KNOWN_CIKS[value]
    
    # Return None to signal lookup needed
    return None


class SECClient:
    """
    Robust SEC EDGAR client with rate limiting, retries, and caching.
    
    Features:
    - Rate limiting to stay under SEC's 10 req/sec limit
    - Exponential backoff with jitter for 429/503/403
    - Caching of submissions data (24h TTL)
    - Deterministic file paths for idempotent storage
    - Content-Type validation
    
    Usage:
        client = SECClient(user_agent="MyApp (me@example.com)")
        cik = client.ticker_to_cik("AAPL")
        submissions = client.get_submissions(cik)
        filings = client.list_filings(submissions, form="10-K", year=2023)
        html = client.fetch_primary_html(cik, filings[0])
    """
    
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: str = ".sec_cache",
        max_rps: float = 8.0,  # Conservative: under 10/sec
        max_retries: int = 5,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize SEC client.
        
        Args:
            user_agent: Required User-Agent string (include email)
            cache_dir: Directory for caching responses
            max_rps: Maximum requests per second
            max_retries: Maximum retry attempts
            cache_ttl_hours: Cache TTL in hours for submissions data
        """
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.min_interval = 1.0 / max_rps
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl_hours * 3600
        self.last_request_time = 0
        
        # Create session with retry adapter
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/html, */*",
        })
        
        # Ticker -> CIK cache (populated on first use)
        self._ticker_map: Dict[str, str] = dict(KNOWN_CIKS)
        self._ticker_map_loaded = False
        
        logger.info(f"SECClient initialized (max_rps={max_rps}, cache_dir={cache_dir})")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _request_with_retry(
        self, 
        url: str, 
        identifier: str = "",
        expect_json: bool = False
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff + jitter.
        
        Args:
            url: URL to fetch
            identifier: Identifier for logging (ticker/cik/accession)
            expect_json: If True, validate JSON response
            
        Returns:
            Response object
            
        Raises:
            Exception: After max retries exceeded
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            self._rate_limit()
            
            try:
                logger.debug(f"Request [{identifier}] attempt {attempt+1}: {url}")
                resp = self.session.get(url, timeout=30)
                
                if resp.status_code == 200:
                    logger.debug(f"Request [{identifier}] success: {resp.status_code}")
                    return resp
                
                # Rate limiting or temporary errors - retry with backoff
                if resp.status_code in (429, 503, 502, 504):
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Request [{identifier}] rate limited ({resp.status_code}), "
                        f"retry {attempt+1}/{self.max_retries} in {delay:.1f}s: {url}"
                    )
                    time.sleep(delay)
                    continue
                
                # 403 can be rate control in disguise
                if resp.status_code == 403:
                    delay = (2 ** attempt) + random.uniform(0.5, 2)
                    logger.warning(
                        f"Request [{identifier}] 403 Forbidden, "
                        f"retry {attempt+1}/{self.max_retries} in {delay:.1f}s: {url}"
                    )
                    time.sleep(delay)
                    continue
                
                # 404 is definitive - don't retry
                if resp.status_code == 404:
                    logger.error(f"Request [{identifier}] 404 Not Found: {url}")
                    resp.raise_for_status()
                
                # Other errors
                logger.error(f"Request [{identifier}] failed ({resp.status_code}): {url}")
                resp.raise_for_status()
                
            except requests.exceptions.Timeout as e:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Request [{identifier}] timeout, "
                    f"retry {attempt+1}/{self.max_retries} in {delay:.1f}s"
                )
                last_error = e
                time.sleep(delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request [{identifier}] error: {e}")
                last_error = e
                if attempt == self.max_retries - 1:
                    raise
        
        raise Exception(f"Max retries ({self.max_retries}) exceeded for [{identifier}]: {url}")
    
    def _get_cache_path(self, *parts: str, ext: str = "json") -> Path:
        """Get deterministic cache path for given identifiers."""
        return self.cache_dir / "/".join(parts[:-1]) / f"{parts[-1]}.{ext}" if len(parts) > 1 else self.cache_dir / f"{parts[0]}.{ext}"
    
    def _is_cache_valid(self, path: Path) -> bool:
        """Check if cache file exists and is within TTL."""
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.cache_ttl
    
    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        """
        Convert ticker symbol to 10-digit CIK.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            10-digit CIK or None if not found
        """
        ticker = ticker.upper().strip()
        
        # Check in-memory cache
        if ticker in self._ticker_map:
            return self._ticker_map[ticker]
        
        # Load full ticker map from SEC if not already loaded
        if not self._ticker_map_loaded:
            self._load_ticker_map()
            if ticker in self._ticker_map:
                return self._ticker_map[ticker]
        
        logger.warning(f"Ticker not found: {ticker}")
        return None
    
    def _load_ticker_map(self):
        """Load ticker -> CIK mapping from SEC."""
        cache_path = self.cache_dir / "company_tickers.json"
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            try:
                data = json.loads(cache_path.read_text())
                for entry in data.values():
                    ticker = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    if ticker and cik:
                        self._ticker_map[ticker] = cik
                self._ticker_map_loaded = True
                logger.info(f"Loaded {len(self._ticker_map)} tickers from cache")
                return
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fetch from SEC
        try:
            resp = self._request_with_retry(SEC_COMPANY_TICKERS, identifier="ticker_map")
            data = resp.json()
            
            # Cache the response
            cache_path.write_text(json.dumps(data))
            
            # Build mapping
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if ticker and cik:
                    self._ticker_map[ticker] = cik
            
            self._ticker_map_loaded = True
            logger.info(f"Loaded {len(self._ticker_map)} tickers from SEC")
            
        except Exception as e:
            logger.error(f"Failed to load ticker map: {e}")
            self._ticker_map_loaded = True  # Mark loaded to avoid retrying
    
    def get_submissions(self, cik: str, use_cache: bool = True) -> Dict:
        """
        Get company submissions (filing history) from SEC.
        
        Args:
            cik: 10-digit CIK
            use_cache: Whether to use cached data
            
        Returns:
            Submissions JSON data
        """
        cik = cik.zfill(10)
        cache_path = self.cache_dir / f"submissions_{cik}.json"
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_path):
            try:
                logger.debug(f"Using cached submissions for CIK {cik}")
                return json.loads(cache_path.read_text())
            except Exception as e:
                logger.warning(f"Cache read failed for {cik}: {e}")
        
        # Fetch from SEC
        url = SEC_SUBMISSIONS.format(cik=cik)
        resp = self._request_with_retry(url, identifier=f"CIK:{cik}")
        data = resp.json()
        
        # Cache the response
        cache_path.write_text(json.dumps(data))
        logger.info(f"Fetched submissions for CIK {cik}: {data.get('name', 'Unknown')}")
        
        return data
    
    def list_filings(
        self,
        submissions: Dict,
        form: str = "10-K",
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_amendments: bool = True,
    ) -> List[Dict]:
        """
        Filter filings from submissions data.
        
        Args:
            submissions: Submissions JSON from get_submissions()
            form: Form type to filter (e.g., "10-K", "10-Q")
            year: Filter by fiscal year (approximate - uses filing date)
            start_date: Filter filings after this date (YYYY-MM-DD)
            end_date: Filter filings before this date (YYYY-MM-DD)
            include_amendments: Include amended filings (e.g., "10-K/A")
            
        Returns:
            List of filing metadata dicts
        """
        recent = submissions.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        documents = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        
        results = []
        form_upper = form.upper()
        
        for i, f in enumerate(forms):
            f_upper = f.upper()
            
            # Match form type
            if include_amendments:
                if not (f_upper == form_upper or f_upper == f"{form_upper}/A"):
                    continue
            else:
                if f_upper != form_upper:
                    continue
            
            filing_date = dates[i] if i < len(dates) else ""
            
            # Year filter (fiscal year ends in prior year for many filings)
            if year and filing_date:
                filing_year = int(filing_date[:4])
                # 10-K filed in Q1 of year N is for FY N-1
                # Allow filings from both year and year+1
                if filing_year != year and filing_year != year + 1:
                    continue
            
            # Date range filters
            if start_date and filing_date < start_date:
                continue
            if end_date and filing_date > end_date:
                continue
            
            # Format accession (remove dashes for URL)
            accession = accessions[i] if i < len(accessions) else ""
            accession_nodash = accession.replace("-", "")
            
            results.append({
                "form": f,
                "accession": accession_nodash,
                "accession_formatted": accession,
                "filing_date": filing_date,
                "primary_document": documents[i] if i < len(documents) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
            })
        
        return results
    
    def fetch_primary_html(
        self,
        cik: str,
        filing: Dict,
        output_dir: Optional[str] = None,
    ) -> Tuple[bytes, Path]:
        """
        Fetch primary document HTML for a filing.
        
        Args:
            cik: 10-digit CIK
            filing: Filing dict from list_filings()
            output_dir: Optional output directory (defaults to cache)
            
        Returns:
            Tuple of (raw bytes, file path)
        """
        cik = cik.zfill(10)
        accession = filing["accession"]
        primary_doc = filing["primary_document"]
        
        # Deterministic storage path
        if output_dir:
            out_path = Path(output_dir) / f"{cik}_{accession}_{primary_doc}"
        else:
            out_path = self.cache_dir / cik / accession / primary_doc
        
        # Check cache
        if out_path.exists():
            logger.debug(f"Using cached document: {out_path}")
            return out_path.read_bytes(), out_path
        
        # Fetch from SEC
        url = SEC_ARCHIVES.format(cik=cik, accession=accession, document=primary_doc)
        resp = self._request_with_retry(
            url, 
            identifier=f"CIK:{cik}/ACC:{accession}"
        )
        
        # Validate Content-Type
        content_type = resp.headers.get("Content-Type", "")
        if not any(t in content_type.lower() for t in ["text/html", "text/plain", "application/xhtml"]):
            logger.warning(f"Unexpected Content-Type: {content_type} for {url}")
        
        # Save to deterministic path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        
        logger.info(f"Fetched document: {out_path} ({len(resp.content):,} bytes)")
        
        return resp.content, out_path


class SECFetcher:
    """
    High-level SEC filing fetcher for the Infera pipeline.
    
    Wraps SECClient with convenience methods for fetching and analyzing 10-Ks.
    """
    
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        output_dir: str = "data",
        cache_dir: str = ".sec_cache",
    ):
        self.client = SECClient(
            user_agent=user_agent,
            cache_dir=cache_dir,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_filing(
        self,
        ticker: str,
        year: Optional[int] = None,
        output_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download a 10-K filing by ticker.
        
        Args:
            ticker: Stock ticker symbol
            year: Fiscal year (optional, defaults to most recent)
            output_name: Custom output filename (optional)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        ticker = ticker.upper()
        logger.info(f"Fetching 10-K for {ticker}" + (f" (FY{year})" if year else " (most recent)"))
        
        # Get CIK
        cik = self.client.ticker_to_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return None
        
        logger.info(f"Found CIK: {cik}")
        
        # Get submissions
        try:
            submissions = self.client.get_submissions(cik)
        except Exception as e:
            logger.error(f"Failed to get submissions for {ticker}: {e}")
            return None
        
        # Find 10-K filings
        filings = self.client.list_filings(submissions, form="10-K", year=year)
        
        if not filings:
            logger.error(f"No 10-K filings found for {ticker}" + (f" in {year}" if year else ""))
            return None
        
        # Get the first (most recent) matching filing
        filing = filings[0]
        logger.info(f"Found filing: {filing['form']} dated {filing['filing_date']}")
        
        # Fetch the document
        try:
            content, cache_path = self.client.fetch_primary_html(cik, filing)
        except Exception as e:
            logger.error(f"Failed to fetch document: {e}")
            return None
        
        # Generate output filename
        if not output_name:
            year_str = filing["filing_date"][:4] if filing["filing_date"] else "unknown"
            output_name = f"{ticker}_10K_{year_str}.html"
        
        output_path = self.output_dir / output_name
        
        # Copy to output directory
        output_path.write_bytes(content)
        
        logger.info(f"✅ Downloaded: {output_path} ({len(content):,} bytes)")
        return str(output_path)
    
    def fetch_and_analyze(
        self,
        ticker: str,
        year: Optional[int] = None,
        skip_summary: bool = True,
    ) -> Optional[Dict]:
        """
        Fetch a filing and run the analysis pipeline.
        
        Args:
            ticker: Stock ticker symbol
            year: Fiscal year (optional)
            skip_summary: Skip GPT summarization (default True)
            
        Returns:
            Pipeline result dict, or None if failed
        """
        # Download the filing
        filepath = self.download_filing(ticker, year)
        
        if not filepath:
            return None
        
        # Run the pipeline
        from services.pipeline_service import run_analysis_pipeline
        
        logger.info(f"Running analysis pipeline on {filepath}...")
        result = run_analysis_pipeline(
            filepath=filepath,
            ticker=ticker,
            skip_summary=skip_summary,
        )
        
        return result


# Convenience functions for simple usage

def fetch_filing(
    ticker: str,
    year: Optional[int] = None,
    output_dir: str = "data",
) -> Optional[str]:
    """
    Fetch a 10-K filing by ticker.
    
    Args:
        ticker: Stock ticker symbol
        year: Fiscal year (optional)
        output_dir: Output directory
        
    Returns:
        Path to downloaded file
    """
    fetcher = SECFetcher(output_dir=output_dir)
    return fetcher.download_filing(ticker, year)


def fetch_and_analyze(
    ticker: str,
    year: Optional[int] = None,
    skip_summary: bool = True,
) -> Optional[Dict]:
    """
    Fetch and analyze a 10-K filing.
    
    Args:
        ticker: Stock ticker symbol
        year: Fiscal year (optional)
        skip_summary: Skip GPT summarization
        
    Returns:
        Pipeline result dict
    """
    fetcher = SECFetcher()
    return fetcher.fetch_and_analyze(ticker, year, skip_summary)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch SEC 10-K filings from EDGAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest/sec_fetcher.py --ticker AAPL
  python ingest/sec_fetcher.py --ticker NVDA --year 2023
  python ingest/sec_fetcher.py --ticker MSFT --analyze
        """
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--year", type=int, help="Fiscal year (e.g., 2023)")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Run analysis pipeline after download")
    parser.add_argument("--skip-summary", action="store_true", default=True, help="Skip GPT summary (default: true)")
    args = parser.parse_args()
    
    fetcher = SECFetcher(output_dir=args.output_dir)
    
    if args.analyze:
        result = fetcher.fetch_and_analyze(
            ticker=args.ticker,
            year=args.year,
            skip_summary=args.skip_summary,
        )
        
        if result:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Ticker: {result['ticker']}")
            print(f"Filing ID: {result['filing_id']}")
            print(f"Paragraphs: {result['paragraph_count']}")
            print(f"Words: {result['word_count']}")
        else:
            print("❌ Fetch and analyze failed")
            sys.exit(1)
    else:
        filepath = fetcher.download_filing(
            ticker=args.ticker,
            year=args.year,
        )
        
        if filepath:
            print(f"\n✅ Downloaded: {filepath}")
            print(f"Run analysis with: python services/pipeline_service.py --file {filepath}")
        else:
            print("❌ Download failed")
            sys.exit(1)
