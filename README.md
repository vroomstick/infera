# 🧠 Infera

**Infera** is an AI-powered ETL pipeline that analyzes SEC 10-K filings to identify, score, and summarize corporate risk factors. It combines traditional NLP with modern embeddings and LLMs to deliver actionable insights.

## ✨ Features

- **HTML Cleaning**: Extracts clean text from raw SEC HTML filings using BeautifulSoup
- **Section Segmentation**: Isolates Item 1A (Risk Factors) using regex pattern matching
- **Embedding-Based Scoring**: Uses `sentence-transformers` to compute semantic similarity scores for risk severity ranking
- **GPT Summarization**: Generates executive-level risk summaries using OpenAI GPT-4o
- **Database Persistence**: Stores all results in SQLite/PostgreSQL via SQLAlchemy
- **REST API**: FastAPI endpoints for programmatic access
- **Markdown Reports**: Auto-generated analysis reports

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  10-K HTML  │───▶│   Cleaner   │───▶│  Segmenter  │───▶│   Scorer    │
│   (Input)   │    │ BeautifulSoup│    │   (Regex)   │    │ (Embeddings)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Report    │◀───│  Summarizer │◀───│  Database   │◀───│  Paragraphs │
│ (Markdown)  │    │   (GPT-4o)  │    │  (SQLite)   │    │  + Scores   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| HTML Parsing | BeautifulSoup, lxml |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT-4o |
| Database | SQLAlchemy + SQLite (Postgres-ready) |
| API | FastAPI + Uvicorn |
| Reports | Markdown (pdfkit for PDF) |

## 📁 Project Structure

```
infera/
├── api/                    # FastAPI application
│   └── main.py            # API routes and models
├── analyze/               # Core analysis modules
│   ├── cleaner.py        # HTML → clean text
│   ├── segmenter.py      # Extract Risk Factors section
│   ├── scorer.py         # Legacy TF-IDF scorer
│   └── summarizer.py     # Legacy summarizer
├── config/               # Configuration
│   └── settings.py       # Environment variables, logging
├── data/                 # Database layer
│   ├── database.py       # SQLAlchemy engine/session
│   ├── models.py         # ORM models
│   └── repository.py     # CRUD operations
├── services/             # Business logic services
│   ├── pipeline_service.py   # Main orchestrator
│   ├── scoring_service.py    # Embedding-based scoring
│   └── report_service.py     # Report generation
├── reports/              # Generated reports
├── .env                  # Environment variables (not in git)
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd infera

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=sqlite:///./infera.db
LOG_LEVEL=INFO
```

### 3. Run the Pipeline (local CLI)

```bash
# Analyze a 10-K filing
python services/pipeline_service.py --file data/AAPL_10K.html

# Skip GPT summary (faster, no API cost)
python services/pipeline_service.py --file data/AAPL_10K.html --skip-summary
```

### 4. Start the API

```bash
python -m api.main
```

API available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### 5. Quick API checks

```bash
# Health
curl -s http://localhost:8000/health

# List filings (after running the pipeline once)
curl -s http://localhost:8000/filings | python -m json.tool

# Filing details (replace 1 with your filing_id)
curl -s http://localhost:8000/filings/1 | python -m json.tool

# Report (Markdown text)
curl -s http://localhost:8000/filings/1/report
```

### 6. Outputs
- Database: `infera.db` (ignored by git) stores companies, filings, sections, paragraphs, scores, summaries.
- Reports: Markdown files under `reports/` (ignored by git).

### 7. Notes / Options
- To avoid OpenAI costs, add `--skip-summary` when running the pipeline.
- Default DB is SQLite; set `DATABASE_URL` to Postgres if desired.
- First run downloads the sentence-transformer model (requires internet).

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Run analysis pipeline |
| GET | `/filings` | List all analyzed filings |
| GET | `/filings/{id}` | Get filing details + top risks |
| GET | `/filings/{id}/report` | Get Markdown report |

### Example: Analyze a Filing

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/AAPL_10K.html"}'
```

### Example: Get Filing Details

```bash
curl http://localhost:8000/filings/1
```

## 📊 Sample Output

The pipeline generates Markdown reports like:

```markdown
# Infera Risk Analysis Report: AAPL

## Executive Summary
- **Economic and Financial Risks**: Exposure to adverse economic conditions...
- **Cybersecurity Threats**: Heightened risk of cyberattacks...
- **Strategic and Operational Risks**: New business strategies may disrupt...

## Top Risk Factors (by severity score)

### 1. Risk Score: 74.2%
The Company's business, reputation, results of operations...

### 2. Risk Score: 65.7%
Investment in new business strategies and acquisitions...
```

## 🧪 How It Works

1. **Cleaning**: BeautifulSoup removes HTML tags, scripts, and noise
2. **Segmentation**: Regex extracts Item 1A (Risk Factors) section
3. **Embedding**: Each paragraph is embedded using `all-MiniLM-L6-v2`
4. **Scoring**: Cosine similarity to a risk-focused prompt ranks severity (0-100%)
5. **Storage**: All data persisted to SQLite via SQLAlchemy
6. **Summarization**: Top paragraphs sent to GPT-4o for executive summary
7. **Reporting**: Markdown report generated with scores and summary

## 🎯 Use Cases

- **Financial Analysts**: Quick risk assessment of potential investments
- **Compliance Teams**: Monitor regulatory risk disclosures
- **Researchers**: Analyze risk trends across companies/sectors
- **Due Diligence**: M&A target risk evaluation

## 📈 Future Enhancements

- [ ] SEC EDGAR API integration for automatic filing fetch
- [ ] PostgreSQL + pgvector for semantic search
- [ ] Year-over-year risk comparison
- [ ] Risk taxonomy classification
- [ ] Streamlit dashboard
- [ ] Docker Compose deployment

## 📄 License

MIT

---

*Built with ❤️ using Python, FastAPI, and AI*
