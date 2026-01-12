# Technical Decisions & Tradeoffs

This document explains every major design decision in Infera, why alternatives were rejected, and what tradeoffs were accepted.

---

## 1. Embedding Model Selection

### Decision: FinBERT over MiniLM

**Options considered:**
| Model | Size | Domain | Accuracy |
|-------|------|--------|----------|
| all-MiniLM-L6-v2 | 22M | General | 44.8% |
| all-mpnet-base-v2 | 110M | General | 46.9% |
| **ProsusAI/finbert** | 110M | **Financial** | **56.6%** |

**Why FinBERT:**
- +9.8 points over MiniLM (statistically significant, p < 0.0001)
- Trained on financial text (Reuters, analyst reports)
- Better at understanding financial terminology ("material adverse", "regulatory risk")

**Tradeoff accepted:**
- Slower inference (110M vs 22M params)
- Larger memory footprint
- Worth it for +9.8 accuracy points

**Evidence:** `evaluation/model_comparison_results.json`

---

## 2. Scoring Method

### Decision: Cosine Similarity vs Classification

**Options considered:**
| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| Classification | Train classifier on labeled data | Highest accuracy | Requires large labeled dataset |
| **Cosine similarity** | Compare to risk prompt | Zero-shot, no training | Less accurate |
| Keyword matching | Count risk words | Simple | Too crude |

**Why cosine similarity:**
- No training required — works immediately
- Transparent: score = similarity to "material adverse impact"
- Explainable: can show what drives similarity

**Tradeoff accepted:**
- 56.6% accuracy vs potential 70%+ with trained classifier
- Poor calibration (ECE = 0.36) — score ≠ probability
- Sensitive to prompt wording (ρ = 0.41 prompt sensitivity)

**Mitigation:** Token attribution explains each score.

---

## 3. Risk Prompt

### Decision: "Material adverse impact" baseline

**Prompts tested:**
| Prompt | Spearman ρ | Stability |
|--------|------------|-----------|
| critical business threat | 0.32 | Low |
| **material adverse condition** | 0.41 | Medium |
| significant operational concern | 0.38 | Medium |
| negative_impact (custom) | 0.61 | High |

**Why "material adverse":**
- Matches SEC regulatory language
- Interpretable to financial professionals
- Reasonable correlation with human labels

**Why not "negative_impact":**
- Higher stability (ρ = 0.61) but less intuitive
- "Material adverse" is industry standard terminology

**Tradeoff accepted:**
- Moderate prompt sensitivity (ρ = 0.41)
- Rankings can change with prompt wording

---

## 4. GPT Temperature

### Decision: Temperature 0.0

**Tested:**
| Temperature | Consistency | Creativity |
|-------------|-------------|------------|
| **0.0** | **99.5%** | None |
| 0.2 | 93.3% | Minimal |
| 0.5 | 85.0% | Some |
| 0.7 | 85.9% | More |

**Why 0.0:**
- Maximum reproducibility for factual summarization
- Same input → same output
- No hallucination from "creative" sampling

**Tradeoff accepted:**
- Less varied language (robotic at times)
- Worth it for consistency in financial reporting

**Evidence:** `evaluation/temperature_results.json`

---

## 5. GPT Summarization Prompt

### Decision: Baseline "3-5 bullet points"

**Prompts tested:**
| Prompt | LLM-Judge Score | Tokens |
|--------|-----------------|--------|
| **baseline** | **4.80/5.0** | 605 |
| executive | 4.65 | 598 |
| concise | 4.40 | 525 |
| structured | 4.15 | 583 |
| analyst | 4.10 | 827 |

**Why baseline:**
- Highest quality score
- Balanced coverage and conciseness
- Natural language (not forced JSON)

**Tradeoff accepted:**
- Structured output (JSON) scored lower but is machine-parseable
- Use `structured_summarizer.py` when JSON is required

**Evidence:** `evaluation/prompt_comparison_results.json`

---

## 6. Database

### Decision: SQLite (Postgres-ready)

**Options:**
| Database | Pros | Cons |
|----------|------|------|
| **SQLite** | Zero setup, portable | No concurrent writes |
| PostgreSQL | Production-grade, pgvector | Requires server |
| MongoDB | Flexible schema | Overkill for structured data |

**Why SQLite:**
- Single-file database for development
- No Docker/server required
- SQLAlchemy makes Postgres migration trivial

**Tradeoff accepted:**
- No concurrent writes (fine for single-user)
- Embeddings stored as pickled blobs (not searchable)

**Migration path:** Phase 6.8 adds PostgreSQL + pgvector.

---

## 7. Embedding Storage

### Decision: Pickle blobs in SQLite

**Options:**
| Storage | Pros | Cons |
|---------|------|------|
| **Pickle in DB** | Simple, portable | No vector search |
| pgvector | Native vector similarity | Requires Postgres |
| Pinecone/Weaviate | Managed, scalable | External dependency |

**Why pickle:**
- Works with SQLite
- Embeddings loaded into memory for search
- Good enough for <10k paragraphs

**Tradeoff accepted:**
- All embeddings loaded into RAM for search
- Won't scale past ~100k paragraphs
- Migrate to pgvector for production

---

## 8. Confidence Scores

### Decision: Percentile-based, not calibrated

**Options:**
| Method | Meaning | Accuracy |
|--------|---------|----------|
| Raw score | "Similarity to risk prompt" | Meaningless probability |
| **Percentile** | "Higher than X% of scores" | Interpretable |
| Platt scaling | "Calibrated probability" | Accurate but complex |

**Why percentile:**
- Simple to compute and explain
- "This paragraph is in the top 10% of risk scores"
- No training required

**Tradeoff accepted:**
- Not a true probability (ECE = 0.36)
- A 0.80 score ≠ 80% chance of high risk

---

## 9. Section Extraction

### Decision: Regex with fallback

**Options:**
| Method | Pros | Cons |
|--------|------|------|
| **Regex patterns** | Fast, deterministic | Brittle on unusual formats |
| LLM extraction | Handles any format | Slow, expensive |
| Trained NER | Accurate | Requires training data |

**Why regex:**
- SEC filings have standard format (Item 1A, Item 1B)
- Handles 95%+ of filings
- Fallback to keyword collection for edge cases

**Tradeoff accepted:**
- May fail on unusual formats
- Fallback may collect non-risk paragraphs

---

## 10. API Framework

### Decision: FastAPI over Flask

**Options:**
| Framework | Pros | Cons |
|-----------|------|------|
| **FastAPI** | Async, auto-docs, Pydantic | Newer, less tutorials |
| Flask | Simple, mature | No async, manual docs |
| Django | Full-featured | Overkill |

**Why FastAPI:**
- Auto-generated OpenAPI docs
- Pydantic validation built-in
- Async support for I/O-bound operations
- Modern Python type hints

---

## 11. Agent Integration

### Decision: LangGraph over LangChain

**Options:**
| Framework | Pros | Cons |
|-----------|------|------|
| **LangGraph** | Graph-based, modern | Newer |
| LangChain | Mature, more examples | Older patterns |
| Custom | No dependencies | More work |

**Why LangGraph:**
- Current standard for agentic workflows
- Better state management
- Compatible with LangChain tools

---

## Summary Table

| Decision | Choice | Alternative | Why |
|----------|--------|-------------|-----|
| Embedding model | FinBERT | MiniLM | +9.8 accuracy points |
| Scoring | Cosine similarity | Classifier | Zero-shot, explainable |
| Risk prompt | "Material adverse" | "Negative impact" | Industry standard |
| GPT temperature | 0.0 | 0.2-0.7 | Max consistency |
| GPT prompt | Baseline | Structured | Highest quality score |
| Database | SQLite | PostgreSQL | Zero setup |
| Embedding storage | Pickle | pgvector | Simplicity |
| Confidence | Percentile | Calibrated | Interpretable |
| Section extraction | Regex | LLM | Speed, cost |
| API | FastAPI | Flask | Modern, auto-docs |
| Agent framework | LangGraph | LangChain | Current standard |

---

*Last updated: January 2026*

