# 🧠 INFERA — Build Checklist (2-Day MVP)

## ✅ Phase 0 — Setup
- [x] Create repo, venv, `.gitignore`, `requirements.txt`
- [x] Add `.env` (OPENAI_API_KEY, DATABASE_URL, LOG_LEVEL)
- [x] Install deps (BeautifulSoup, sentence-transformers, FastAPI, SQLAlchemy, OpenAI, etc.)

## ✅ Phase 1 — Core Pipeline
- [x] `analyze/cleaner.py`: clean HTML with BeautifulSoup (strip scripts/styles/nav)
- [x] `analyze/segmenter.py`: regex extract Item 1A Risk Factors (primary + fallback)
- [x] `services/pipeline_service.py`: orchestrate clean → segment → paragraphs → (optional) summarize → report

## ✅ Phase 2 — Persistence Layer
- [x] `config/settings.py`: env-driven config + logging
- [x] `data/database.py`: SQLAlchemy engine/session (SQLite by default; Postgres-ready)
- [x] `data/models.py`: Company, Filing, Section, Paragraph, Score, Summary
- [x] `data/repository.py`: CRUD helpers

## ✅ Phase 3 — Scoring (ML)
- [x] `services/scoring_service.py`: sentence-transformers (all-MiniLM-L6-v2), cosine similarity to risk prompt, store scores (+ embeddings optional)
- [x] Store top scored paragraphs for reporting/API

## ✅ Phase 4 — Summarization (LLM)
- [x] GPT-4o summary over top paragraphs (optional via `--skip-summary`)
- [x] Store summary text + token counts in DB

## ✅ Phase 5 — Reporting
- [x] `services/report_service.py`: generate Markdown report (summary + top risks + metadata)
- [x] Reports saved under `reports/` (PDF optional via wkhtmltopdf/pdfkit)

## ✅ Phase 6 — API
- [x] `api/main.py` (FastAPI)
  - `GET /health`
  - `POST /analyze` (run pipeline on provided file)
  - `GET /filings` (list)
  - `GET /filings/{id}` (details + top risks)
  - `GET /filings/{id}/report` (Markdown)

## ✅ Phase 7 — Docs & Demo
- [x] README with Quick Start, API usage, architecture, sample output
- [x] Example run on `data/AAPL_10K.html` (stored in DB, report generated)

## 🧠 Next Steps (Post-MVP)
- [ ] Postgres + Docker Compose (optionally pgvector for semantic search)
- [ ] SEC fetcher (EDGAR) for auto-ingest
- [ ] Tests (unit/integration; mock OpenAI)
- [ ] Second filing + basic compare (YOY/peer-lite)
- [ ] Optional Streamlit/dashboard
