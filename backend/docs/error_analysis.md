# Error Analysis

## Overview

This document analyzes the failure modes of Infera's risk scoring models. Understanding where and why the models fail is critical for improving accuracy and setting appropriate expectations.

## Error Summary

| Method | False Positives (low→high) | False Negatives (high→low) | Total Errors |
|--------|----------------------------|----------------------------|--------------|
| **Embeddings** | 4 | 1 | 20 (44.4%) |
| **TF-IDF** | 0 | 5 | 22 (48.9%) |

**Key insight**: The methods fail in opposite ways:
- **Embeddings**: Over-triggers on risk language (too aggressive)
- **TF-IDF**: Under-triggers on everything (too conservative)

---

## Embedding Model Errors

### False Positives: Low-Risk Paragraphs Scored as High

The embedding model incorrectly flagged 4 low-risk paragraphs as high-risk:

#### Error 1: Boilerplate Introduction (AAPL)
| Attribute | Value |
|-----------|-------|
| **ID** | 1 |
| **True Label** | low |
| **Predicted** | high |
| **Score** | 0.742 |

> "The Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, whether currently known or unknown, including those described below..."

**Why it failed**: This is standard introductory language that appears at the start of every risk section. The model is fooled by "materially and adversely affected" language.

**Pattern**: SEC-mandated boilerplate with risk terminology.

---

#### Error 2: Boilerplate Introduction (TSLA)
| Attribute | Value |
|-----------|-------|
| **ID** | 18 |
| **True Label** | low |
| **Predicted** | high |
| **Score** | 0.726 |

> "You should carefully consider the risks described below together with the other information set forth in this report, which could materially affect our business, financial condition and future results..."

**Why it failed**: Same pattern—introductory language with "materially affect" phrasing.

---

#### Error 3: Boilerplate Introduction (MSFT)
| Attribute | Value |
|-----------|-------|
| **ID** | 32 |
| **True Label** | low |
| **Predicted** | high |
| **Score** | 0.691 |

> "Our operations and financial results are subject to various risks and uncertainties, including those described below, that could adversely affect our business, operations, financial condition..."

**Why it failed**: Generic risk framing language without specific risk content.

---

#### Error 4: Negative Risk Statement (MSFT)
| Attribute | Value |
|-----------|-------|
| **ID** | 37 |
| **True Label** | low |
| **Predicted** | high |
| **Score** | 0.560 |

> "We describe the risks from cybersecurity threats... As of the date of this Form 10-K, we do not believe any risks from cybersecurity threats have materially affected or are reasonably likely to materially affect us..."

**Why it failed**: The paragraph discusses cybersecurity but **explicitly states NO material impact**. The model sees "cybersecurity threats" and triggers, missing the negation.

**Pattern**: Semantic negation not captured by similarity scoring.

---

### False Negatives: High-Risk Paragraphs Scored as Low

The embedding model missed 1 high-risk paragraph:

#### Error 5: Customer Credit Risk (TSLA)
| Attribute | Value |
|-----------|-------|
| **ID** | 21 |
| **True Label** | high |
| **Predicted** | low |
| **Score** | 0.411 |

> "Finally, our vehicle and solar energy system financing programs and our energy storage sales programs also expose us to customer credit risk. In the event of a widespread economic downturn or other catastrophic event, our customers may be unable or unwilling to satisfy their payment obligations..."

**Why it failed**: The language is more conversational ("In the event of...") and doesn't use the intense risk keywords ("litigation," "breach," "disruption") that the prompt emphasizes.

**Pattern**: Indirect risk language with less dramatic vocabulary.

---

## TF-IDF Model Errors

### Systematic Failure

TF-IDF exhibits a **systematic bias toward predicting "low"** because its scores are uniformly low:

| Score Range | Count |
|-------------|-------|
| 0.00 - 0.05 | 35 |
| 0.05 - 0.10 | 8 |
| 0.10 - 0.15 | 2 |
| > 0.15 | 0 |

With thresholds set at 0.15 (high) and 0.08 (medium), almost everything falls into "low."

### False Negatives: All High-Risk Paragraphs Missed

TF-IDF failed to identify **any** high-risk paragraphs:

| ID | Source | Score | True | Why Missed |
|----|--------|-------|------|------------|
| 3 | AAPL | 0.015 | high | "credit risk," "counterparty" not in prompt |
| 4 | AAPL | 0.047 | high | "supply shortages" doesn't match exactly |
| 6 | AAPL | 0.000 | high | "malicious attacks" not in prompt vocabulary |
| 8 | AAPL | 0.000 | high | "acquisitions," "disrupt" not captured |
| 21 | TSLA | 0.042 | high | "customer credit risk" not matched |

**Root cause**: TF-IDF requires exact keyword matches. Paraphrased risks ("supply shortages" vs "supply chain") receive zero signal.

---

## Failure Patterns Summary

### Embeddings

| Pattern | Count | Example |
|---------|-------|---------|
| **Boilerplate introduction** | 3 | "Our business may be materially adversely affected..." |
| **Negated risk statement** | 1 | "We do not believe risks have materially affected..." |
| **Indirect risk language** | 1 (FN) | "In the event of a downturn..." |

### TF-IDF

| Pattern | Count | Example |
|---------|-------|---------|
| **Vocabulary mismatch** | 5 | "malicious attacks" vs "cybersecurity" |
| **Paraphrased concepts** | 5 | "supply shortages" vs "supply chain" |

---

## Recommendations

### For Embeddings

1. **Add boilerplate detection**: Filter or downweight standard SEC introduction paragraphs
2. **Handle negation**: Detect "do not believe," "unlikely to," etc. and adjust scores
3. **Expand risk prompt**: Include more varied risk vocabulary

### For TF-IDF

1. **Expand keyword list**: Add synonyms and related terms
2. **Use n-grams**: Capture phrases like "supply chain" not just "supply"
3. **Consider abandoning**: Embeddings are clearly superior for this task

### General

1. **Threshold optimization**: Use a validation set to tune score cutoffs
2. **Ensemble approach**: Combine both methods—use TF-IDF as a filter
3. **Domain fine-tuning**: Train embeddings on SEC filings specifically

---

## Error Examples for Training

These misclassified examples could be used to improve the model:

### Boilerplate Phrases to Ignore (FP Reduction)
```
"The Company's business, reputation, results of operations..."
"You should carefully consider the risks described below..."
"Our operations and financial results are subject to various risks..."
```

### Negation Patterns to Detect
```
"we do not believe... have materially affected"
"unlikely to materially affect"
"no material impact"
```

### Indirect Risk Language to Capture (FN Reduction)
```
"In the event of a widespread economic downturn..."
"may be unable or unwilling to satisfy..."
"expose us to customer credit risk"
```

---

*Analysis conducted January 2026 on 45-sample evaluation set*

