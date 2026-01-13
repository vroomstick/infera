# Infera

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Infera** is a production-grade ML system that extracts, scores, and explains corporate risk factors from SEC 10-K filings. Built for both human analysts and AI agents, it combines domain-specific embeddings (FinBERT) with explainable AI to deliver transparent, actionable risk intelligence.

## Philosophy: For Agent or For Human

Infera serves two distinct use cases:

- **For Humans:** GPT-4o summarization transforms dense legal text into executive summaries. Reports highlight top risks with confidence scores and category classifications.

- **For Agents:** Structured JSON outputs, token-level attributions, and raw embeddings enable programmatic analysis. LangGraph tools and OpenAI function calling integrate directly into agent workflows.

Both modes share the same core pipeline: HTML cleaning → Item 1A extraction → FinBERT scoring → explainable outputs.

## Key Features

| Category | Features |
|----------|----------|
| **Ingestion** | SEC EDGAR fetching, HTML cleaning, Item 1A extraction, paragraph splitting |
| **Scoring** | FinBERT embeddings, cosine similarity, percentile-based confidence, risk taxonomy (8 categories) |
| **Explainability** | Token attribution, risk category classification, confidence scores |
| **Search** | RRF fusion (vector + keyword), semantic search, peer benchmarking, YoY trend analysis |
| **Agent Integration** | Python SDK, LangGraph tools, OpenAI function calling, raw embedding access |
| **Production** | API auth, rate limiting, retry logic, graceful fallbacks, Docker support |

## Evaluation Results

Evaluated on **286 hand-labeled paragraphs** across 6 companies (AAPL, TSLA, MSFT, NVDA, AMZN, GOOGL).

### Zero-Shot Classification

| Method | Accuracy | F1 | ROC-AUC | Human Correlation (ρ) |
|--------|----------|----|---------|----------------------|
| **FinBERT** ⭐ | **56.6%** | 0.61 | 0.68 | **0.590** |
| TF-IDF | 39.8% | 0.45 | 0.52 | 0.23 |
| **Improvement** | **+16.8 pts** | +0.16 | +0.16 | **+0.36** |

**95% Confidence Interval:** 56.6% (49.0% - 60.1%)

### Supervised Learning (Evaluated)

| Method | Accuracy | F1 | ROC-AUC | Notes |
|--------|----------|----|---------|-------|
| Zero-shot (baseline) | 62.1% | 0.65 | 0.68 | Current default |
| **Supervised LR** | **86.2%** | **0.84** | **0.89** | Logistic regression on FinBERT embeddings |
| **Improvement** | **+24.1 pts** | **+0.19** | **+0.21** | Model available, not integrated by default |

**Note:** Supervised model (`supervised_scorer.pkl`) exists but requires manual integration. Zero-shot remains default for determinism.

### Retrieval Evaluation

Evaluated on **40 manually labeled queries** with ground truth relevance judgments.

| Method | MRR@10 | nDCG@10 | Recall@10 | Status |
|--------|--------|---------|-----------|--------|
| **RRF Fusion (k=60)** ⭐ | **0.744 ± 0.336** | **0.768 ± 0.215** | **0.861 ± 0.186** | **Default** |
| Vector-only | 0.705 ± 0.350 | 0.743 ± 0.249 | 0.826 ± 0.237 | FinBERT embeddings |
| Keyword-only | 0.103 ± 0.292 | 0.107 ± 0.298 | 0.069 ± 0.236 | PostgreSQL full-text |

**Decision:** RRF fusion combines vector and keyword search for best performance across all metrics.

### Calibration Analysis (Evaluated)

| Metric | Before | After (Isotonic) | Improvement |
|--------|--------|-----------------|-------------|
| **ECE** | 0.363 | **0.200** | **-44.8%** |
| **Brier Score** | 0.42 | 0.35 | -16.7% |

**Note:** Calibration model (`calibrator.pkl`) exists but requires manual integration. Current outputs use uncalibrated similarity scores.

**Scale tested on 55 S&P 500 companies with 100% success rate.**

See [Developer Handbook](backend/docs/v4_developer_handbook.md#consolidated-results-phase-7) for complete results.

## Architecture

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

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Embeddings | **FinBERT** (ProsusAI/finbert, 768-dim) |
| LLM | OpenAI GPT-4o (optional, for summarization) |
| API | FastAPI + Uvicorn |
| Database | SQLAlchemy + SQLite / PostgreSQL + pgvector |
| Agent Framework | LangGraph, langchain-core |
| Testing | pytest (35 tests) |

## Quick Start

### Option 1: Make (Recommended)

```bash
# Clone and setup
git clone https://github.com/vroomstick/infera.git
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

# Set OpenAI key (optional, for summarization)
echo "OPENAI_API_KEY=sk-..." > ../.env

# Start API
uvicorn api.main:app --reload
```

### Option 3: Docker (Production)

```bash
# Set environment variables (optional)
export POSTGRES_PASSWORD=your_password
export OPENAI_API_KEY=sk-...  # Optional, for summarization

# Start services
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/fetch` | Fetch 10-K from SEC EDGAR |
| POST | `/analyze` | Analyze local filing |
| GET | `/filings` | List analyzed filings |
| GET | `/filings/{id}` | Get filing details |
| GET | `/filings/{id}/report` | Get Markdown report |
| GET | `/filings/{id}/trends` | YoY trend analysis (requires `prior_filing_id` query param) |
| GET | `/search` | Semantic search (RRF fusion) |
| GET | `/benchmark` | Peer comparison |
| GET | `/explain/{paragraph_id}` | Token attribution |
| POST | `/explain` | Explain arbitrary text |
| GET | `/paragraphs/{paragraph_id}` | Get paragraph + embedding |

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

# Get YoY trends
curl "http://localhost:8000/filings/1/trends?prior_filing_id=2"
```

## Agent Integration

### Python SDK

```python
from sdk.infera_client import InferaClient

client = InferaClient(base_url="http://localhost:8000")

# Search for risks
results = client.search("supply chain disruption", limit=5)
for result in results:
    print(f"{result.ticker}: {result.score:.3f} - {result.text[:100]}")

# Explain a score
explanation = client.explain(paragraph_id=42)
print(f"Score: {explanation.score:.3f}")
print(f"Category: {explanation.risk_category}")
for token in explanation.top_tokens[:5]:
    print(f"  {token.token}: +{token.contribution:.4f}")

# Get paragraph with embedding
para = client.get_paragraph(paragraph_id=42, include_embedding=True)
print(f"Embedding dimension: {len(para.embedding)}")
```

### LangGraph Tools

```python
from examples.langgraph_tool import create_infera_agent
from langchain_core.messages import HumanMessage

agent = create_infera_agent()
response = agent.invoke({
    "messages": [HumanMessage("What are Tesla's main risks?")]
})
print(response["messages"][-1].content)
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

# Execute function calls
for tool_call in response.choices[0].message.tool_calls:
    result = execute_function(tool_call.function.name, tool_call.function.arguments)
    print(result)
```

## Explainability

Every prediction is explainable with token-level attributions:

```json
{
  "paragraph_id": 42,
  "text": "We face significant cybersecurity threats...",
  "score": 0.82,
  "confidence": 0.73,
  "risk_category": "Cybersecurity",
  "category_confidence": 0.89,
  "top_tokens": [
    {"token": "cybersecurity", "contribution": 0.0234, "position": 5},
    {"token": "breach", "contribution": 0.0189, "position": 12},
    {"token": "unauthorized", "contribution": 0.0156, "position": 18}
  ]
}
```

## Semantic Contract

**✅ SEMANTIC FREEZE DECLARED** — All output contracts are locked.

| Output | Type | Meaning | Calibrated? | Use Case |
|--------|------|---------|-------------|----------|
| `similarity_score` | float [0,1] | Ranking signal only | ❌ No | Order paragraphs by relevance |
| `confidence` | float [0,1] | Percentile rank | ❌ No | "How confident is this a top risk?" |
| `search_default` | method | RRF fusion (k=60) | N/A | Semantic search |

**Note:** `prob_high` (calibrated probability) and supervised scoring are evaluated but not integrated by default. See [Developer Handbook](backend/docs/v4_developer_handbook.md#semantic-contract-phase-7) for full contract details.

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Handbook](backend/docs/v4_developer_handbook.md) | **Complete technical reference** (286 pages) |
| [Architecture](backend/docs/architecture.md) | System diagrams (Mermaid) |
| [Decisions & Tradeoffs](backend/docs/decisions.md) | Why we made each choice |
| [Theoretical Background](backend/docs/theoretical_background.md) | Academic grounding |
| [Ablation Study](backend/docs/ablation_study.md) | Component contribution analysis |
| [API Reference](backend/docs/api.md) | Endpoint documentation |
| [Failure Modes](backend/docs/similarity_failure_modes.md) | When the model fails |

## Security

| Feature | Description |
|---------|-------------|
| API Key Auth | Optional `X-API-Key` header (set `INFERA_API_KEY`) |
| Rate Limiting | 60 requests/min per IP |
| CORS | Configurable origins |
| Path Traversal | Blocked (file operations restricted to `data/` directory) |
| Error Sanitization | Production mode hides stack traces |

## Key Findings

| Finding | Result |
|---------|--------|
| Best Model | FinBERT (+16.8 pts vs TF-IDF, p < 0.0001) |
| Zero-Shot Accuracy | 56.6% (95% CI: 49.0% - 60.1%) |
| Human Correlation | ρ = 0.59 (statistically significant) |
| Top Risk Keywords | "harm" (+21.5%), "failure" (+15.7%), "significant" (+14.5%) |
| Risk Categories | Regulatory (43%), Competitive (21%), Financial (18%) |
| GPT Faithfulness | 93.9% keyword overlap, 100% claims verifiable |
| Optimal Temperature | 0.0 (99.5% consistency) |
| Scale Test | 55 filings, 100% success rate |
| Retrieval (RRF) | MRR=0.74, nDCG@10=0.77, Recall@10=0.86 |

## Known Limitations

1. **Calibration:** ECE = 0.36 (scores ≠ probabilities). Calibration model evaluated but not integrated by default.
2. **Prompt sensitivity** (ρ = 0.41): Rankings depend on prompt wording.
3. **Zero-shot accuracy:** 62.1% (supervised option available at 86.2% but not integrated).

## Completed (v5)

- [x] Token attribution / explainability
- [x] FinBERT model selection (empirical comparison, p < 0.0001)
- [x] 286-sample labeled dataset
- [x] Bootstrap confidence intervals
- [x] Agent tooling (SDK, LangGraph, OpenAI)
- [x] PostgreSQL + pgvector (production database)
- [x] Scale test (55 companies, 100% success)
- [x] Retrieval evaluation (40 labeled queries, RRF fusion)
- [x] Supervised learning exploration (86.2% accuracy)
- [x] Calibration analysis (ECE: 0.363 → 0.200)
- [x] Semantic freeze (output contracts locked)

## Future Work

- [ ] Integrate calibration model into default pipeline
- [ ] Integrate supervised scorer as optional enhancement
- [ ] CI/CD pipeline
- [ ] Streamlit dashboard

## Testing

```bash
# Run all tests
make test

# With coverage
pytest --cov=. --cov-report=html
```

## License

MIT

---

*Built with Python, FinBERT (ProsusAI), FastAPI, PostgreSQL, and pgvector | v5 | [Developer Handbook](backend/docs/v4_developer_handbook.md)*
