# Infera

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Infera** is an AI-powered ETL pipeline that analyzes SEC 10-K filings to identify, score, and summarize corporate risk factors. It combines traditional NLP with modern embeddings and LLMs to deliver actionable insights.

## Features

- **HTML Cleaning**: Extracts clean text from raw SEC HTML filings using BeautifulSoup
- **Section Segmentation**: Isolates Item 1A (Risk Factors) using regex pattern matching
- **Embedding-Based Scoring**: Uses `sentence-transformers` to compute semantic similarity scores for risk severity ranking
- **GPT Summarization**: Generates executive-level risk summaries using OpenAI GPT-4o
- **Semantic Search**: Find risk paragraphs by natural language query across all filings
- **Peer Benchmarking**: Compare risk profiles across companies with statistical insights
- **YoY Trend Analysis**: Detect new/removed risks and narrative drift between filing years
- **SEC EDGAR Integration**: Auto-fetch 10-K filings by ticker with rate limiting and caching
- **Database Persistence**: Stores all results in SQLite/PostgreSQL via SQLAlchemy
- **REST API**: FastAPI endpoints for programmatic access with optional authentication
- **Production Security**: API key auth, rate limiting, CORS, path traversal protection
- **Markdown Reports**: Auto-generated analysis reports

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGEST LAYER                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│  10-K HTML  │   Cleaner   │  Segmenter  │   Scorer    │     Database        │
│   (Input)   │ BeautifulSoup│   (Regex)   │ (Embeddings)│     (SQLite)        │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │             │                 │
       ▼             ▼             ▼             ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ANALYTICS LAYER                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│   Semantic Search   │  Peer Benchmarking  │      YoY Trend Analysis         │
│  (Query → Similar   │  (Compare risk      │  (Detect new/removed risks,     │
│   paragraphs)       │   profiles)         │   narrative drift)              │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
       │                         │                         │
       ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                     │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│    REST API         │   Markdown Reports  │      GPT Summaries              │
│   (FastAPI)         │   (Auto-generated)  │      (Executive briefs)         │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION LAYER                                   │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Labeled Dataset    │  Metrics & Plots    │      Error Analysis             │
│  (45 samples)       │  (Accuracy, ρ, P@k) │   (Failure modes)               │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| HTML Parsing | BeautifulSoup, lxml |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT-4o |
| Database | SQLAlchemy + SQLite (Postgres-ready) |
| API | FastAPI + Uvicorn |
| Testing | pytest (35 tests) |
| Containerization | Docker + Docker Compose |

## Project Structure

```
infera/
├── api/                    # FastAPI application
│   └── main.py            # API routes (9 endpoints)
├── analyze/               # Core analysis modules
│   ├── cleaner.py        # HTML → clean text
│   ├── segmenter.py      # Extract Risk Factors section
│   ├── scorer.py         # Legacy TF-IDF scorer
│   └── summarizer.py     # Legacy summarizer
├── config/               # Configuration
│   └── settings.py       # Environment variables, logging
├── data/                 # Database layer + 10-K filings
│   ├── *_10K.html        # SEC 10-K filings (AAPL, TSLA, MSFT)
│   ├── database.py       # SQLAlchemy engine/session
│   ├── models.py         # ORM models
│   └── repository.py     # CRUD operations
├── docs/                 # Documentation
│   ├── api.md            # API endpoint documentation
│   ├── results.md        # Evaluation results
│   ├── error_analysis.md # Failure mode analysis
│   └── methodology.md    # Technical approach
├── evaluation/           # Evaluation harness
│   ├── labeled_risks.json    # Ground truth labels (45 samples)
│   ├── eval_scorer.py        # Evaluation script
│   ├── compare_methods.py    # TF-IDF vs embeddings comparison
│   └── plots/                # Visualization outputs
├── samples/              # Example outputs (checked in)
│   ├── AAPL_sample_report.md     # Sample analysis report
│   ├── evaluation_summary.json   # Evaluation metrics summary
│   └── api_response_example.json # Example API response
├── services/             # Business logic services
│   ├── pipeline_service.py   # Main orchestrator
│   ├── scoring_service.py    # Embedding-based scoring
│   ├── benchmark_service.py  # Peer comparison analytics
│   ├── trend_service.py      # YoY trend analysis
│   └── report_service.py     # Report generation
├── tests/                # Unit tests (35 tests)
│   ├── test_cleaner.py       # HTML cleaning tests
│   ├── test_segmenter.py     # Section extraction tests
│   └── test_scoring.py       # Embedding scoring tests
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # API + test + eval services
├── Makefile              # 15+ commands
├── requirements.txt      # Dependencies
└── requirements-lock.txt # Pinned dependencies (125 packages)
```

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Setup
make install

# Run pipeline on sample filing
make run

# Start API server
make api

# Run tests
make test

# Run evaluation
make eval
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "OPENAI_API_KEY=your-key" > .env

# Run pipeline
python services/pipeline_service.py --file data/AAPL_10K.html --skip-summary

# Start API
python -m api.main
```

API available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option 3: Docker

```bash
# Build and run
make docker

# Or manually
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Run analysis pipeline on a filing |
| GET | `/filings` | List all analyzed filings |
| GET | `/filings/{id}` | Get filing details + top risks |
| GET | `/filings/{id}/report` | Get Markdown report |
| GET | `/filings/{id}/trends` | **YoY trend analysis** |
| GET | `/search` | **Semantic search** across paragraphs |
| GET | `/benchmark` | **Peer comparison** across companies |
| POST | `/fetch` | **Fetch 10-K from SEC EDGAR** |

See [`docs/api.md`](docs/api.md) for full API documentation with examples.

### Quick Examples

```bash
# Fetch a 10-K from SEC EDGAR
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "year": 2023}'

# Semantic search for cybersecurity risks
curl "http://localhost:8000/search?q=cybersecurity%20threats&limit=5"

# Compare risk profiles across companies
curl "http://localhost:8000/benchmark?tickers=AAPL,TSLA,MSFT"

# YoY trend analysis (compare filing 1 vs filing 2)
curl "http://localhost:8000/filings/1/trends?prior_filing_id=2"
```

## Evaluation Results

We evaluated the scoring pipeline on 45 hand-labeled risk paragraphs across 3 filings (AAPL, TSLA, MSFT).

| Metric | TF-IDF | Embeddings | Winner |
|--------|--------|------------|--------|
| **Accuracy** | 51.1% | **55.6%** | Embeddings |
| **Spearman ρ** | 0.300 | **0.399** | Embeddings |
| **High-Risk Recall** | 0% | **75%** | Embeddings |

![Method Comparison](evaluation/plots/method_comparison.png)

### Key Findings

- **Embeddings outperform TF-IDF** by 4.5 percentage points on accuracy
- **75% of high-risk paragraphs correctly identified** by embeddings (vs 0% for TF-IDF)
- **Spearman correlation of 0.40** shows scores meaningfully track human severity judgments

See [`docs/results.md`](docs/results.md) for full evaluation details.

## Sample Output

### Peer Benchmark Results

```json
{
  "comparison": {
    "insights": [
      "AAPL has 21.8% more high-severity risks than TSLA",
      "Mean risk score ranges from 0.386 (MSFT) to 0.477 (AAPL)",
      "AAPL shows highest risk score variability (σ=0.163)"
    ]
  }
}
```

### Risk Report

```markdown
# Infera Risk Analysis Report: AAPL

## Executive Summary
- **Economic and Financial Risks**: Exposure to adverse economic conditions...
- **Cybersecurity Threats**: Heightened risk of cyberattacks...

## Top Risk Factors (by severity score)
### 1. Risk Score: 74.2%
The Company's business, reputation, results of operations...
```

## Limitations

1. **Boilerplate sensitivity**: Standard SEC intro language scores high due to risk vocabulary
2. **No negation handling**: "No material impact" statements incorrectly flagged
3. **General-purpose embeddings**: Not fine-tuned on financial text
4. **Static risk prompt**: May miss industry-specific risks

See [`docs/error_analysis.md`](docs/error_analysis.md) for detailed failure analysis.

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/api.md`](docs/api.md) | API endpoint reference with examples |
| [`docs/results.md`](docs/results.md) | Evaluation methodology and metrics |
| [`docs/error_analysis.md`](docs/error_analysis.md) | Failure modes and examples |
| [`docs/methodology.md`](docs/methodology.md) | Technical approach and theory |

## Security

Production-ready security features:

| Feature | Description |
|---------|-------------|
| **API Key Auth** | Optional authentication via `X-API-Key` header |
| **Rate Limiting** | Configurable per-IP limits (default: 60/min) |
| **CORS** | Configurable cross-origin policy |
| **Path Traversal Protection** | Blocks `../` attacks on file endpoints |
| **Non-Root Docker** | Container runs as unprivileged user |
| **Error Sanitization** | Production mode hides stack traces |
| **Credential Masking** | Database URLs masked in logs |
| **Dependency Audit** | All vulnerabilities patched |

Configure via environment variables:
```bash
INFERA_API_KEY=secret      # Enable API authentication
CORS_ORIGINS=https://...   # Restrict CORS origins
RATE_LIMIT=100/minute      # Adjust rate limits
ENVIRONMENT=production     # Enable error sanitization
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_scoring.py -v
```

**35 tests** covering:
- HTML cleaning edge cases
- Section extraction patterns
- Embedding score computation
- Score ranking consistency

## Use Cases

- **Financial Analysts**: Quick risk assessment of potential investments
- **Compliance Teams**: Monitor regulatory risk disclosures
- **Researchers**: Analyze risk trends across companies/sectors
- **Due Diligence**: M&A target risk evaluation

## Future Enhancements

- [x] ~~Year-over-year risk comparison~~ (Implemented)
- [x] ~~Semantic search~~ (Implemented)
- [x] ~~Peer benchmarking~~ (Implemented)
- [x] ~~SEC EDGAR API integration~~ (Implemented via `POST /fetch`)
- [ ] PostgreSQL + pgvector for production deployment
- [ ] Streamlit dashboard for interactive exploration
- [ ] Fine-tuned embeddings on SEC filings (FinBERT)

## License

MIT

---

*Built with Python, FastAPI, and AI*
