# Infera

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](tests/)
[![Accuracy](https://img.shields.io/badge/accuracy-56.6%25-blue.svg)](backend/docs/v4_developer_handbook.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Infera** is an AI-powered ETL pipeline that analyzes SEC 10-K filings to identify, score, and summarize corporate risk factors. It combines domain-specific embeddings (FinBERT) with LLMs (GPT-4o) to deliver explainable, actionable insights.

## ✨ Key Features

| Category | Features |
|----------|----------|
| **Analysis** | HTML cleaning, Item 1A extraction, paragraph scoring, GPT summarization |
| **Explainability** | Token attribution, risk taxonomy (8 categories), confidence scores |
| **Search** | Semantic search across all filings, peer benchmarking, YoY trends |
| **Agent-Ready** | Python SDK, LangGraph tools, OpenAI function calling |
| **Production** | API auth, rate limiting, retry logic, graceful fallbacks |

## 📊 Evaluation Results

Evaluated on **286 hand-labeled paragraphs** across 6 companies (AAPL, TSLA, MSFT, NVDA, AMZN, GOOGL).

| Metric | FinBERT | TF-IDF | Improvement |
|--------|---------|--------|-------------|
| **Accuracy** | 56.6% | 39.8% | **+16.8 pts** |
| **Human Correlation** | ρ = 0.59 | ρ = 0.23 | **+0.36** |
| **Statistical Significance** | — | — | **p < 0.0001** |

> "FinBERT beats TF-IDF by 16.8 points with 95% CI [49.0%, 60.1%]"

See [v4 Developer Handbook](backend/docs/v4_developer_handbook.md) for full methodology.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGEST LAYER                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│  SEC EDGAR  │   Cleaner   │  Segmenter  │  FinBERT    │     Database        │
│  (10-K HTML)│ BeautifulSoup│   (Regex)   │ (Scoring)   │  (SQLite/Postgres)  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ML LAYER                                       │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Token Attribution  │   Risk Taxonomy     │      Confidence Scores          │
│  (Explainability)   │   (8 categories)    │      (Percentile-based)         │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                   │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│    REST API         │   Python SDK        │      Agent Tools                │
│   (FastAPI)         │   (infera_client)   │   (LangGraph, OpenAI)           │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

See [Architecture Diagram](backend/docs/architecture.md) for detailed Mermaid diagrams.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Embeddings | **FinBERT** (ProsusAI/finbert) |
| LLM | OpenAI GPT-4o |
| API | FastAPI + Uvicorn |
| Database | SQLAlchemy + SQLite (Postgres-ready) |
| Agent Framework | LangGraph, langchain-core |
| Testing | pytest (35 tests) |

## 🚀 Quick Start

### Option 1: Make (Recommended)

```bash
# Clone and setup
git clone https://github.com/yourusername/infera.git
cd infera/backend

# Install dependencies
make install

# Start API
make api

# Open http://localhost:8000/docs
```

### Option 2: Manual

```bash
cd backend
python -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt

# Set OpenAI key
echo "OPENAI_API_KEY=sk-..." > ../.env

# Start API
uvicorn api.main:app --reload
```

### Option 3: Docker

```bash
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/fetch` | Fetch 10-K from SEC EDGAR |
| POST | `/analyze` | Analyze local filing |
| GET | `/filings` | List analyzed filings |
| GET | `/filings/{id}` | Get filing details |
| GET | `/search` | Semantic search |
| GET | `/benchmark` | Peer comparison |
| GET | `/explain/{id}` | **Token attribution** |
| GET | `/paragraphs/{id}` | **Get paragraph + embedding** |

### Examples

```bash
# Fetch and analyze NVIDIA's 10-K
curl -X POST http://localhost:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "year": 2023}'

# Semantic search for cybersecurity risks
curl "http://localhost:8000/search?q=cybersecurity%20data%20breach&limit=5"

# Explain why paragraph 42 scored high
curl "http://localhost:8000/explain/42"

# Compare AAPL vs TSLA risk profiles
curl "http://localhost:8000/benchmark?tickers=AAPL,TSLA"
```

## 🤖 Agent Integration

### Python SDK

```python
from sdk.infera_client import InferaClient

client = InferaClient()

# Search for risks
results = client.search("supply chain disruption", limit=5)

# Explain a score
explanation = client.explain(paragraph_id=42)
for token in explanation.top_tokens[:5]:
    print(f"  {token.token}: +{token.contribution:.4f}")
```

### LangGraph Tools

```python
from examples.langgraph_tool import create_infera_agent
from langchain_core.messages import HumanMessage

agent = create_infera_agent()
response = agent.invoke({
    "messages": [HumanMessage("What are Tesla's main risks?")]
})
```

### OpenAI Function Calling

```python
from examples.openai_functions import INFERA_FUNCTIONS, execute_function

# Use with OpenAI API
response = client.chat.completions.create(
    model="gpt-4o",
    tools=INFERA_FUNCTIONS,
    messages=[{"role": "user", "content": "Analyze AAPL risks"}]
)
```

## 🔬 Explainability

Every prediction is explainable:

```json
{
  "paragraph_id": 42,
  "score": 0.82,
  "confidence": 0.73,
  "risk_category": "Cybersecurity",
  "top_tokens": [
    {"token": "cybersecurity", "contribution": 0.0234},
    {"token": "breach", "contribution": 0.0189},
    {"token": "unauthorized", "contribution": 0.0156}
  ]
}
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [v4 Developer Handbook](backend/docs/v4_developer_handbook.md) | **Complete technical reference** |
| [Architecture](backend/docs/architecture.md) | System diagrams (Mermaid) |
| [Decisions & Tradeoffs](backend/docs/decisions.md) | Why we made each choice |
| [Theoretical Background](backend/docs/theoretical_background.md) | Academic grounding |
| [Ablation Study](backend/docs/ablation_study.md) | Component contribution analysis |
| [API Reference](backend/docs/api.md) | Endpoint documentation |
| [Failure Modes](backend/docs/similarity_failure_modes.md) | When the model fails |

## 🔒 Security

| Feature | Description |
|---------|-------------|
| API Key Auth | Optional `X-API-Key` header |
| Rate Limiting | 60 requests/min per IP |
| CORS | Configurable origins |
| Path Traversal | Blocked |
| Error Sanitization | Production mode hides traces |

## 📈 Key Findings

| Finding | Result |
|---------|--------|
| Best Model | FinBERT (+16.8 pts vs TF-IDF) |
| Accuracy | 56.6% (95% CI: 49.0% - 60.1%) |
| Human Correlation | ρ = 0.59 |
| Top Risk Keywords | "harm", "failure", "significant" |
| Risk Categories | Regulatory (43%), Competitive (21%) |
| GPT Faithfulness | 93.9% keyword overlap |
| Optimal Temperature | 0.0 (99.5% consistency) |

## ⚠️ Known Limitations

1. **Poor calibration** (ECE = 0.36): Score ≠ probability
2. **Prompt sensitivity** (ρ = 0.41): Rankings depend on prompt wording
3. **56.6% accuracy**: Room for improvement with fine-tuning
4. **SQLite only**: Postgres + pgvector for production (coming soon)

## 🗺️ Roadmap

- [x] Token attribution / explainability
- [x] FinBERT model selection (empirical comparison)
- [x] 286-sample labeled dataset
- [x] Bootstrap confidence intervals
- [x] Agent tooling (SDK, LangGraph, OpenAI)
- [ ] PostgreSQL + pgvector
- [ ] CI/CD pipeline
- [ ] Integration tests
- [ ] Streamlit dashboard

## 🧪 Testing

```bash
# Run all tests
make test

# With coverage
pytest --cov=. --cov-report=html
```

## 📄 License

MIT

---

*Built with Python, FinBERT, FastAPI, and GPT-4o*
