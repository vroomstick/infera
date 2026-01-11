# Infera API Documentation

Base URL: `http://localhost:8000`

## Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Run analysis pipeline |
| POST | `/fetch` | **Fetch 10-K from SEC EDGAR** |
| GET | `/filings` | List all filings |
| GET | `/filings/{id}` | Get filing details |
| GET | `/filings/{id}/report` | Get Markdown report |
| GET | `/filings/{id}/trends` | YoY trend analysis |
| GET | `/search` | Semantic search |
| GET | `/benchmark` | Peer comparison |

---

## System Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-10T07:00:16.321965"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

## Analysis Endpoints

### POST /analyze

Run the full analysis pipeline on a 10-K filing.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes | Path to 10-K HTML file |
| `ticker` | string | No | Company ticker (auto-detected from filename) |
| `skip_summary` | boolean | No | Skip GPT summarization (default: false) |
| `skip_scoring` | boolean | No | Skip embedding scoring (default: false) |

**Response:**
```json
{
  "ticker": "AAPL",
  "filing_id": 1,
  "section_id": 1,
  "paragraph_count": 13,
  "word_count": 756,
  "report_path": "reports/AAPL_2026-01-10_risk_report.md",
  "summary": "• Economic risks from adverse conditions..."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/AAPL_10K.html",
    "skip_summary": true
  }'
```

---

## Filing Endpoints

### GET /filings

List all analyzed filings.

**Response:**
```json
[
  {
    "id": 1,
    "ticker": "AAPL",
    "filing_type": "10-K",
    "filing_date": "2025-12-31",
    "has_summary": true
  },
  {
    "id": 2,
    "ticker": "TSLA",
    "filing_type": "10-K",
    "filing_date": "2026-01-10",
    "has_summary": false
  }
]
```

**Example:**
```bash
curl http://localhost:8000/filings
```

---

### GET /filings/{filing_id}

Get detailed information about a specific filing including top risk factors.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `filing_id` | integer | Filing ID from database |

**Response:**
```json
{
  "id": 1,
  "ticker": "AAPL",
  "filing_type": "10-K",
  "filing_date": "2025-12-31",
  "paragraph_count": 13,
  "word_count": 756,
  "summary": "Executive summary text...",
  "top_risks": [
    {
      "paragraph_id": 1,
      "position": 0,
      "score": 0.742,
      "text": "The Company's business, reputation..."
    },
    {
      "paragraph_id": 6,
      "position": 5,
      "score": 0.657,
      "text": "Investment in new business strategies..."
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/filings/1
```

---

### GET /filings/{filing_id}/report

Get the Markdown report for a filing.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `filing_id` | integer | Filing ID from database |

**Response:** Plain text Markdown

```markdown
# Infera Risk Analysis Report: AAPL

**Generated:** 2026-01-10 01:58:25

## Executive Summary
...
```

**Example:**
```bash
curl http://localhost:8000/filings/1/report
```

---

## Ingest Endpoints

### POST /fetch

Fetch a 10-K filing directly from SEC EDGAR and optionally analyze it.

**Request Body:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `ticker` | string | Yes | — | Stock ticker symbol (e.g., "NVDA") |
| `year` | integer | No | — | Fiscal year (e.g., 2023). If omitted, gets most recent. |
| `analyze` | boolean | No | true | Run analysis pipeline after download |
| `skip_summary` | boolean | No | true | Skip GPT summarization |

**Response:**
```json
{
  "ticker": "NVDA",
  "year": 2023,
  "filepath": "data/NVDA_10K_2024.html",
  "filing_id": 7,
  "paragraph_count": 41,
  "word_count": 3629,
  "message": "Successfully fetched and analyzed NVDA 10-K"
}
```

**Examples:**
```bash
# Fetch and analyze NVIDIA's latest 10-K
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'

# Fetch specific fiscal year
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "year": 2022}'

# Download only (no analysis)
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"ticker": "GOOGL", "analyze": false}'
```

**Supported Tickers:**
Any ticker listed on SEC EDGAR. Common examples: AAPL, MSFT, TSLA, NVDA, GOOGL, AMZN, META, JPM, V, JNJ.

---

## Analytics Endpoints

### GET /search

Semantic search across all risk paragraphs. Find paragraphs similar to a natural language query.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | — | Search query (min 3 characters) |
| `limit` | integer | No | 10 | Max results to return |
| `ticker` | string | No | — | Filter by company ticker |

**Response:**
```json
{
  "query": "cybersecurity threats",
  "results": [
    {
      "paragraph_id": 66,
      "filing_id": 6,
      "ticker": "MSFT",
      "score": 0.718,
      "text": "We describe the risks from cybersecurity threats...",
      "position": 3
    },
    {
      "paragraph_id": 32,
      "filing_id": 3,
      "ticker": "TSLA",
      "score": 0.699,
      "text": "We recognize the importance of assessing...",
      "position": 5
    }
  ],
  "count": 2
}
```

**Examples:**
```bash
# Search for cybersecurity risks
curl "http://localhost:8000/search?q=cybersecurity%20threats&limit=5"

# Search within a specific company
curl "http://localhost:8000/search?q=supply%20chain&ticker=AAPL"

# Search for regulatory risks
curl "http://localhost:8000/search?q=regulatory%20compliance%20violations"
```

---

### GET /benchmark

Compare risk profiles across multiple companies.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tickers` | string | No | `AAPL,TSLA,MSFT` | Comma-separated ticker symbols |

**Response:**
```json
{
  "profiles": {
    "AAPL": {
      "ticker": "AAPL",
      "filing_id": 1,
      "total_paragraphs": 13,
      "distribution": {
        "high": 5,
        "medium": 2,
        "low": 6
      },
      "percentages": {
        "high": 38.5,
        "medium": 15.4,
        "low": 46.2
      },
      "statistics": {
        "mean": 0.4771,
        "median": 0.4869,
        "std": 0.1626,
        "min": 0.1351,
        "max": 0.7418
      }
    },
    "TSLA": {
      "ticker": "TSLA",
      "filing_id": 5,
      "total_paragraphs": 12,
      "distribution": {
        "high": 2,
        "medium": 1,
        "low": 9
      },
      "percentages": {
        "high": 16.7,
        "medium": 8.3,
        "low": 75.0
      },
      "statistics": {
        "mean": 0.4009,
        "median": 0.3695,
        "std": 0.1355,
        "min": 0.2179,
        "max": 0.7258
      }
    }
  },
  "comparison": {
    "rankings": {
      "by_high_risk_pct": ["AAPL", "MSFT", "TSLA"],
      "by_mean_score": ["AAPL", "TSLA", "MSFT"]
    },
    "insights": [
      "AAPL has 21.8% more high-severity risks than TSLA",
      "Mean risk score ranges from 0.401 (TSLA) to 0.477 (AAPL)",
      "AAPL shows highest risk score variability (σ=0.163)"
    ]
  }
}
```

**Examples:**
```bash
# Compare all three companies
curl "http://localhost:8000/benchmark?tickers=AAPL,TSLA,MSFT"

# Compare just two companies
curl "http://localhost:8000/benchmark?tickers=AAPL,TSLA"
```

---

### GET /filings/{filing_id}/trends

Analyze year-over-year changes in risk disclosures between two filings.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `filing_id` | integer | Current year filing ID |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prior_filing_id` | integer | Yes | Prior year filing ID to compare against |

**Response:**
```json
{
  "current_filing_id": 1,
  "prior_filing_id": 2,
  "current_paragraph_count": 13,
  "prior_paragraph_count": 13,
  "new_risks": {
    "count": 3,
    "details": [
      {
        "paragraph_id": 5,
        "text_preview": "New regulatory requirements regarding...",
        "best_prior_match_similarity": 0.42,
        "interpretation": "Likely new risk not present in prior year"
      }
    ]
  },
  "removed_risks": {
    "count": 2,
    "details": [
      {
        "paragraph_id": 18,
        "text_preview": "Previous pandemic-related disruptions...",
        "best_current_match_similarity": 0.38,
        "interpretation": "Risk disclosure no longer present or significantly modified"
      }
    ]
  },
  "narrative_drift": {
    "score": 0.12,
    "similarity": 0.88,
    "interpretation": "Moderate narrative evolution"
  },
  "summary": {
    "total_changes": 5,
    "change_rate": 19.2
  }
}
```

**Interpretation Guide:**
- `narrative_drift.score` > 0.15: Significant narrative shift
- `narrative_drift.score` 0.05-0.15: Moderate evolution
- `narrative_drift.score` < 0.05: Stable narrative

**Examples:**
```bash
# Compare AAPL filings (current=1, prior=2)
curl "http://localhost:8000/filings/1/trends?prior_filing_id=2"

# Compare TSLA filings
curl "http://localhost:8000/filings/5/trends?prior_filing_id=3"
```

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (filing/resource doesn't exist) |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "detail": "Filing not found: 999"
}
```

---

## Interactive Documentation

When the API is running, interactive documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Security

### Authentication

API key authentication is **optional** and disabled by default. To enable:

```bash
# Set in .env or environment
export INFERA_API_KEY=your-secret-key
```

When enabled, include the key in requests:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/AAPL_10K.html"}'
```

**Protected endpoints** (when auth is enabled):
- `POST /analyze`
- `POST /fetch`

**Public endpoints** (always accessible):
- `GET /health`
- `GET /filings`
- `GET /filings/{id}`
- `GET /filings/{id}/report`
- `GET /filings/{id}/trends`
- `GET /search`
- `GET /benchmark`

---

### Rate Limiting

Rate limiting is enforced via `slowapi`. Default: **60 requests/minute** per IP.

Configure via environment variable:
```bash
export RATE_LIMIT=100/minute  # or 10/second, 1000/hour
```

**Rate limit exceeded response:**
```json
{
  "error": "Rate limit exceeded",
  "detail": "60 per 1 minute"
}
```

---

### CORS

Cross-Origin Resource Sharing is configurable:

```bash
# Allow specific origins (comma-separated)
export CORS_ORIGINS=https://example.com,https://app.example.com

# Allow all origins (default, for development)
export CORS_ORIGINS=*
```

---

### Path Traversal Protection

The `/analyze` endpoint validates file paths to prevent directory traversal attacks. Only files in these directories are accessible:
- `data/`
- `.sec_cache/`

Attempts to access files outside these directories return:
```json
{
  "detail": "File path not allowed. Files must be in the data/ directory."
}
```

---

### Production Mode

Set `ENVIRONMENT=production` to enable:
- Sanitized error messages (no stack traces)
- Generic 500 errors for unhandled exceptions

```bash
export ENVIRONMENT=production
```

