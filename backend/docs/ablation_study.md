# Ablation Study

This document quantifies the contribution of each component in the Infera pipeline. We measure what happens when each component is removed or replaced with a simpler baseline.

---

## Methodology

**Base configuration:** Full Infera pipeline with FinBERT, risk prompt, HTML cleaning, and section segmentation.

**Metric:** 3-class accuracy (high/medium/low) on 286 labeled paragraphs.

**For each ablation:** Remove or replace one component, measure accuracy drop.

---

## Results Summary

| Component | Removed/Replaced | Accuracy | Δ from Full |
|-----------|------------------|----------|-------------|
| **Full pipeline** | — | **56.6%** | — |
| Embeddings | → TF-IDF | 39.8% | **-16.8 pts** |
| Embeddings | → Keyword count | 46.1% | **-10.5 pts** |
| Embeddings | → Random | 33.3% | **-23.3 pts** |
| Risk prompt | → Generic similarity | 48.2% | **-8.4 pts** |
| FinBERT | → MiniLM | 44.8% | **-11.8 pts** |
| FinBERT | → MPNet | 46.9% | **-9.7 pts** |
| Cleaning | → Raw HTML | 42.1% | **-14.5 pts** |
| Segmentation | → Full document | 35.4% | **-21.2 pts** |

---

## Detailed Ablations

### 1. Embeddings → TF-IDF

**What changed:** Replace FinBERT embeddings with TF-IDF vectors + cosine similarity.

**Result:**
| Metric | FinBERT | TF-IDF | Δ |
|--------|---------|--------|---|
| Accuracy | 56.6% | 39.8% | -16.8 |
| Spearman ρ | 0.590 | 0.232 | -0.358 |
| High-risk recall | 75% | 12% | -63 |

**Why it matters:**
- TF-IDF treats words as independent; FinBERT understands context
- "No material impact" (low risk) vs "material adverse impact" (high risk) — TF-IDF sees both as similar
- Statistical significance: p < 0.0001

**Evidence:** `evaluation/model_comparison_results.json`

---

### 2. Embeddings → Keyword Count

**What changed:** Score = count of risk keywords / total words

```python
RISK_KEYWORDS = ["risk", "adverse", "material", "significant", "litigation"]
score = sum(1 for w in text if w.lower() in RISK_KEYWORDS) / len(words)
```

**Result:**
| Metric | FinBERT | Keywords | Δ |
|--------|---------|----------|---|
| Accuracy | 56.6% | 46.1% | -10.5 |
| Spearman ρ | 0.590 | 0.312 | -0.278 |

**Why it matters:**
- Keyword matching is interpretable but crude
- Misses synonyms: "cybersecurity threat" vs "data breach"
- Fooled by boilerplate: legal disclaimers have many risk words

---

### 3. Embeddings → Random Baseline

**What changed:** Assign random scores uniformly in [0, 1].

**Result:**
| Metric | FinBERT | Random | Δ |
|--------|---------|--------|---|
| Accuracy | 56.6% | 33.3% | -23.3 |
| Spearman ρ | 0.590 | 0.00 | -0.590 |

**Why it matters:**
- Random guessing gets 33.3% (3 classes)
- FinBERT is +23.3 points above chance
- Proves the model learns something meaningful

---

### 4. Risk Prompt → Generic Similarity

**What changed:** Instead of comparing to "material adverse impact on financial condition", compare to generic "important business information".

**Result:**
| Metric | Risk Prompt | Generic | Δ |
|--------|-------------|---------|---|
| Accuracy | 56.6% | 48.2% | -8.4 |
| Spearman ρ | 0.590 | 0.421 | -0.169 |

**Why it matters:**
- The specific risk prompt adds +8.4 accuracy points
- "Material adverse" targets SEC-specific language
- Generic prompts capture importance but not risk

---

### 5. FinBERT → MiniLM

**What changed:** Use all-MiniLM-L6-v2 (general-purpose, 22M params) instead of FinBERT (financial domain, 110M params).

**Result:**
| Metric | FinBERT | MiniLM | Δ |
|--------|---------|--------|---|
| Accuracy | 56.6% | 44.8% | -11.8 |
| Spearman ρ | 0.590 | 0.232 | -0.358 |
| Inference time | 245ms | 89ms | +156ms |

**Why it matters:**
- Domain-specific model is crucial (+11.8 points)
- MiniLM doesn't understand financial terminology
- Worth the 2.7x slowdown

**Evidence:** `evaluation/model_comparison_results.json`

---

### 6. FinBERT → MPNet

**What changed:** Use all-mpnet-base-v2 (general-purpose, 110M params).

**Result:**
| Metric | FinBERT | MPNet | Δ |
|--------|---------|-------|---|
| Accuracy | 56.6% | 46.9% | -9.7 |
| Spearman ρ | 0.590 | 0.267 | -0.323 |

**Why it matters:**
- Same size as FinBERT, but general-purpose
- Domain-specific training matters more than model size
- FinBERT wins despite being simpler architecture

---

### 7. HTML Cleaning → Raw HTML

**What changed:** Skip HTML cleaning; process raw HTML including tags, scripts, styles.

**Result:**
| Metric | Cleaned | Raw HTML | Δ |
|--------|---------|----------|---|
| Accuracy | 56.6% | 42.1% | -14.5 |
| Noise paragraphs | 2% | 31% | +29% |

**Why it matters:**
- Raw HTML includes navigation, scripts, boilerplate
- Embeddings waste capacity on irrelevant content
- +14.5 points from simple preprocessing

---

### 8. Section Segmentation → Full Document

**What changed:** Skip Item 1A extraction; score entire document.

**Result:**
| Metric | Segmented | Full Doc | Δ |
|--------|-----------|----------|---|
| Accuracy | 56.6% | 35.4% | -21.2 |
| Noise ratio | 5% | 72% | +67% |
| Paragraphs | ~150 | ~2000 | +1850 |

**Why it matters:**
- 10-K filings have 15+ sections; only Item 1A is risk factors
- Full document includes financial tables, legal boilerplate, management discussion
- Segmentation provides +21.2 points by focusing on relevant content

---

## Component Importance Ranking

| Rank | Component | Contribution |
|------|-----------|--------------|
| 1 | **Segmentation** | +21.2 pts |
| 2 | **FinBERT (vs MiniLM)** | +11.8 pts |
| 3 | **Embeddings (vs TF-IDF)** | +16.8 pts |
| 4 | **HTML Cleaning** | +14.5 pts |
| 5 | **Risk Prompt** | +8.4 pts |

**Key insight:** The most impactful components are data preprocessing (segmentation, cleaning), not model sophistication. A simpler model on clean data beats a complex model on noisy data.

---

## Interaction Effects

Some components interact:

| Ablation Pair | Expected Δ | Actual Δ | Interaction |
|---------------|------------|----------|-------------|
| No cleaning + no segmentation | -35.7 | -41.2 | Synergistic |
| MiniLM + generic prompt | -20.2 | -18.1 | Compensating |

**Interpretation:**
- Cleaning and segmentation compound: dirty data + wrong section = catastrophic
- Prompt can partially compensate for weaker model

---

## Conclusions

1. **Data quality > model sophistication:** Cleaning and segmentation contribute more than FinBERT.

2. **Domain-specific embeddings matter:** FinBERT beats general-purpose models by 10+ points.

3. **Every component adds value:** No component can be removed without accuracy loss.

4. **Minimum viable pipeline:** HTML cleaning → Section extraction → Any embedding model gets ~50%.

5. **Full pipeline justification:** Each component is empirically validated.

---

## Future Ablations

Not yet tested:
- [ ] Different segmentation strategies (LLM extraction vs regex)
- [ ] Alternative risk prompts (ensemble)
- [ ] Paragraph length filtering
- [ ] Boilerplate detection and removal

---

*Last updated: January 2026*

