# Infera Error Codes

Canonical error codes for consistent error handling and debugging.

## Error Code Format

`INF-{CATEGORY}-{NUMBER}`

- `INF`: Infera prefix
- `{CATEGORY}`: Component (INGEST, PARSE, EMBED, DB, API, VALID)
- `{NUMBER}`: Sequential error number (001, 002, ...)

## Error Codes

| Code | Category | Description | HTTP Status | Example |
|------|----------|-------------|-------------|---------|
| `INF-INGEST-001` | Ingest | SEC EDGAR download failure | 502 | Network timeout, rate limit |
| `INF-INGEST-002` | Ingest | CIK lookup failed | 404 | Ticker not found in SEC database |
| `INF-PARSE-001` | Parse | Item 1A section not found | 422 | Filing missing Risk Factors section |
| `INF-PARSE-002` | Parse | HTML cleaning failed | 422 | Malformed HTML, encoding issues |
| `INF-EMBED-001` | Embed | Embedding computation failed | 500 | Model load error, OOM |
| `INF-EMBED-002` | Embed | Batch embedding partial failure | 207 | Some paragraphs failed (see details) |
| `INF-DB-001` | Database | Database write failure | 500 | Constraint violation, connection lost |
| `INF-DB-002` | Database | Database connection failed | 503 | PostgreSQL not available |
| `INF-API-001` | API | Bad request (validation) | 400 | Invalid ticker format, missing required field |
| `INF-API-002` | API | Unauthorized | 401 | Missing or invalid API key |
| `INF-API-003` | API | Rate limit exceeded | 429 | Too many requests |
| `INF-VALID-001` | Validation | Data validation failed | 422 | Score out of range, invalid category |
| `INF-VALID-002` | Validation | Ticker format invalid | 422 | Ticker must be 1-5 uppercase chars |
| `INF-VALID-003` | Validation | Embedding dimension mismatch | 422 | Expected 768-dim, got different |

## Usage

Errors are raised as `InferaError` exceptions with error codes:

```python
from services.errors import InferaError, ErrorCode

raise InferaError(
    code=ErrorCode.INGEST_001,
    message="Failed to download filing from SEC EDGAR",
    details={"ticker": "AAPL", "url": "https://..."}
)
```

API responses include error code in JSON:

```json
{
  "error": {
    "code": "INF-INGEST-001",
    "message": "Failed to download filing from SEC EDGAR",
    "details": {"ticker": "AAPL"}
  }
}
```

## Error Handling Philosophy

- **Fail fast**: Validate inputs at API boundary
- **Graceful degradation**: Partial failures (e.g., some paragraphs fail embedding) return partial results
- **Consistent format**: All errors include code, message, and optional details
- **Debuggable**: Error messages include context (ticker, paragraph_id, etc.)

