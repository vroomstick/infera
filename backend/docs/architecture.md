# Infera Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        SEC[("SEC EDGAR<br/>10-K Filings")]
        HTML["Raw HTML<br/>Filing"]
    end

    subgraph Ingest["🔄 Ingest Pipeline"]
        FETCH["sec_fetcher.py<br/>Rate-limited fetch"]
        CLEAN["cleaner.py<br/>BeautifulSoup"]
        SEGMENT["segmenter.py<br/>Item 1A extraction"]
        SPLIT["Split into<br/>paragraphs"]
    end

    subgraph ML["🧠 ML Layer"]
        EMBED["FinBERT<br/>Embeddings"]
        SCORE["scoring_service.py<br/>Cosine similarity"]
        ATTR["attribution_service.py<br/>Token attribution"]
        CLASS["classification_service.py<br/>Risk taxonomy"]
    end

    subgraph LLM["🤖 LLM Layer"]
        GPT["GPT-4o<br/>Summarization"]
        STRUCT["structured_summarizer.py<br/>Pydantic validation"]
        PROD["production_summarizer.py<br/>Retry + fallback"]
    end

    subgraph Storage["💾 Storage"]
        DB[("SQLite / PostgreSQL<br/>+ Embeddings")]
    end

    subgraph API["🌐 API Layer"]
        FAST["FastAPI<br/>main.py"]
        AUTH["Auth + Rate Limit"]
    end

    subgraph Output["📤 Output"]
        JSON["JSON Responses"]
        REPORT["Markdown Reports"]
        SDK["Python SDK"]
        AGENT["LangGraph / OpenAI<br/>Agent Tools"]
    end

    SEC --> FETCH
    HTML --> CLEAN
    FETCH --> CLEAN
    CLEAN --> SEGMENT
    SEGMENT --> SPLIT
    SPLIT --> DB
    SPLIT --> EMBED
    EMBED --> SCORE
    SCORE --> DB
    EMBED --> ATTR
    EMBED --> CLASS
    SPLIT --> GPT
    GPT --> STRUCT
    STRUCT --> PROD
    PROD --> DB
    DB --> FAST
    FAST --> AUTH
    AUTH --> JSON
    AUTH --> REPORT
    JSON --> SDK
    SDK --> AGENT
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Pipeline as Pipeline Service
    participant ML as ML Services
    participant LLM as GPT-4o
    participant DB as Database

    User->>API: POST /fetch {ticker: "AAPL"}
    API->>Pipeline: run_analysis_pipeline()
    
    Note over Pipeline: 1. Fetch from SEC EDGAR
    Pipeline->>Pipeline: clean_html()
    Pipeline->>Pipeline: get_risk_section()
    Pipeline->>Pipeline: split_into_paragraphs()
    Pipeline->>DB: Store paragraphs
    
    Note over ML: 2. Score & Classify
    Pipeline->>ML: compute_risk_scores()
    ML->>ML: FinBERT embeddings
    ML->>ML: Cosine similarity
    ML->>DB: Store scores
    
    Note over LLM: 3. Summarize (optional)
    Pipeline->>LLM: Generate summary
    LLM->>DB: Store summary
    
    Pipeline->>API: Return result
    API->>User: {filing_id, paragraph_count, ...}
```

---

## Component Details

### Ingest Layer

| Component | File | Purpose |
|-----------|------|---------|
| SEC Fetcher | `ingest/sec_fetcher.py` | Rate-limited 10-K download from EDGAR |
| Cleaner | `analyze/cleaner.py` | HTML → clean text (BeautifulSoup) |
| Segmenter | `analyze/segmenter.py` | Extract Item 1A (Risk Factors) |

### ML Layer

| Component | File | Model | Purpose |
|-----------|------|-------|---------|
| Scoring | `services/scoring_service.py` | FinBERT | Cosine similarity scoring |
| Attribution | `services/attribution_service.py` | FinBERT | Token-level explainability |
| Classification | `services/classification_service.py` | FinBERT | 8-category risk taxonomy |

### LLM Layer

| Component | File | Model | Purpose |
|-----------|------|-------|---------|
| Summarizer | `analyze/summarizer.py` | GPT-4o | Executive summaries |
| Structured | `analyze/structured_summarizer.py` | GPT-4o | Pydantic-validated JSON |
| Production | `analyze/production_summarizer.py` | GPT-4o | Retry + fallback |

### API Layer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/fetch` | POST | Fetch 10-K from SEC EDGAR |
| `/analyze` | POST | Run analysis on local file |
| `/filings` | GET | List analyzed filings |
| `/filings/{id}` | GET | Get filing details |
| `/search` | GET | Semantic search |
| `/benchmark` | GET | Peer comparison |
| `/explain/{id}` | GET | Token attribution |
| `/paragraphs/{id}` | GET | Paragraph with embedding |

### Agent Integration

| Component | File | Framework |
|-----------|------|-----------|
| Python SDK | `sdk/infera_client.py` | httpx |
| LangGraph | `examples/langgraph_tool.py` | langgraph |
| OpenAI Functions | `examples/openai_functions.py` | JSON Schema |

---

## Database Schema

```mermaid
erDiagram
    COMPANY ||--o{ FILING : has
    FILING ||--o{ SECTION : contains
    SECTION ||--o{ PARAGRAPH : contains
    PARAGRAPH ||--o{ SCORE : has
    FILING ||--o| SUMMARY : has

    COMPANY {
        int id PK
        string ticker
        string name
        datetime created_at
    }

    FILING {
        int id PK
        int company_id FK
        string filing_type
        date filing_date
        string source_file
    }

    SECTION {
        int id PK
        int filing_id FK
        string section_type
        text raw_text
        int word_count
    }

    PARAGRAPH {
        int id PK
        int section_id FK
        int position
        text text
    }

    SCORE {
        int id PK
        int paragraph_id FK
        string method
        float score
        blob embedding
    }

    SUMMARY {
        int id PK
        int filing_id FK
        string section_name
        text summary_text
        string model
    }
```

---

## Embedding Flow

```mermaid
flowchart LR
    subgraph Input
        TEXT["Paragraph Text"]
    end

    subgraph FinBERT["FinBERT Model"]
        TOK["Tokenize"]
        ENC["Encode"]
        POOL["Mean Pooling"]
    end

    subgraph Scoring
        PROMPT["Risk Prompt<br/>'material adverse impact...'"]
        COS["Cosine Similarity"]
        SCORE["Risk Score<br/>0.0 - 1.0"]
    end

    TEXT --> TOK
    TOK --> ENC
    ENC --> POOL
    POOL --> COS
    PROMPT --> COS
    COS --> SCORE
```

---

## Deployment Options

```mermaid
flowchart TB
    subgraph Dev["Development"]
        LOCAL["Local Python<br/>uvicorn"]
        SQLITE[("SQLite")]
    end

    subgraph Staging["Staging"]
        DOCKER["Docker Compose"]
        PG_STG[("PostgreSQL")]
    end

    subgraph Prod["Production"]
        K8S["Kubernetes /<br/>Cloud Run"]
        PG_PROD[("PostgreSQL<br/>+ pgvector")]
        REDIS[("Redis Cache")]
    end

    LOCAL --> DOCKER
    DOCKER --> K8S
    SQLITE --> PG_STG
    PG_STG --> PG_PROD
```

---

*Last updated: January 2026*

