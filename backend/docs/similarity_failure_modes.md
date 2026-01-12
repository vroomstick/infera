# Similarity Failure Mode Analysis

This document identifies and categorizes cases where cosine similarity fails for risk scoring.

## Summary

- **Total samples evaluated:** 286
- **Accuracy:** 54.5%
- **Total failures:** 130
- **Thresholds:** high ≥ 0.7, medium ≥ 0.65

---

## Failure Type Breakdown

| Failure Type | Count | % of Failures | Description |
|--------------|-------|---------------|-------------|
| unknown_false_positive | 52 | 40.0% |  |
| boundary_error | 43 | 33.1% |  |
| missed_severity | 16 | 12.3% | Real risk that embeddings underweighted |
| short_text_bias | 15 | 11.5% |  |
| boilerplate | 2 | 1.5% | Generic risk language that appears in every filing without substance |
| cross_reference | 1 | 0.8% | References other sections without substantive content |
| hedging_language | 1 | 0.8% | Overly cautious legal language without material concern |

---

## Failure Type Details

### Boilerplate

*Generic risk language that appears in every filing without substance*

**2 examples (showing top 5):**

#### Example 1: AMZN (ID: 101)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.7958
- **Indicators found:** from time to time

> We are also subject to labor union efforts to organize groups of our employees from time to time. These organizational efforts, if successful, decrease our operational flexibility, which could adverse......

#### Example 2: AAPL (ID: 1)

- **True label:** low
- **Predicted:** high
- **Score:** 0.7283
- **Indicators found:** from time to time

> The Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, whether currently known or unknown, including those described bel......

### Unknown False Positive

*Unknown type*

**5 examples (showing top 5):**

#### Example 1: GOOGL (ID: 170)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.8520

> Acquisitions, joint ventures, investments, and divestitures could result in operating difficulties, dilution, and other consequences that could harm our business, financial condition, and operating re......

#### Example 2: NVDA (ID: 76)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.8512

> Stakeholder groups may find us insufficiently responsive to the implications of climate change, and therefore we may face legal action or reputational harm. We may not achieve our stated sustainabilit......

#### Example 3: NVDA (ID: 63)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.8402

> Increased scrutiny from shareholders, regulators, and others regarding our corporate sustainability practices could result in financial, reputational, or operational harm and liability....

#### Example 4: GOOGL (ID: 122)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.8206

> Our ongoing investment in new businesses, products, services, and technologies is inherently risky, and could divert management attention and harm our business, financial condition, and operating resu......

#### Example 5: GOOGL (ID: 131)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.8173

> We face a number of manufacturing and supply chain risks that could harm our business, financial condition, and operating results....

### Boundary Error

*Unknown type*

**5 examples (showing top 5):**

#### Example 1: GOOGL (ID: 136)

- **True label:** medium
- **Predicted:** low
- **Score:** 0.4194
- **Indicators found:** predicted: low, true: medium

> Our international operations are significant to our revenues and net income, and we plan to continue to grow internationally. International revenues accounted for approximately 53% of our consolidated......

#### Example 2: NVDA (ID: 71)

- **True label:** medium
- **Predicted:** low
- **Score:** 0.4939
- **Indicators found:** predicted: low, true: medium

> Our accelerated computing platforms experience rapid changes in technology, customer requirements, competitive products, and industry standards....

#### Example 3: GOOGL (ID: 156)

- **True label:** medium
- **Predicted:** low
- **Score:** 0.4986
- **Indicators found:** predicted: low, true: medium

> The General Data Protection Regulation and the United Kingdom General Data Protection Regulations, which apply to all of our activities conducted from an establishment in the EU or the United Kingdom,......

#### Example 4: NVDA (ID: 57)

- **True label:** medium
- **Predicted:** low
- **Score:** 0.5343
- **Indicators found:** predicted: low, true: medium

> We may not be able to realize the potential benefits of business investments or acquisitions, nor successfully integrate acquisition targets....

#### Example 5: MSFT (ID: 44)

- **True label:** medium
- **Predicted:** low
- **Score:** 0.5521
- **Indicators found:** predicted: low, true: medium

> Certain forecasted transactions, assets, and liabilities are exposed to foreign currency risk. We monitor our foreign currency exposures daily to maximize the economic effectiveness of our foreign cur......

### Short Text Bias

*Unknown type*

**5 examples (showing top 5):**

#### Example 1: AMZN (ID: 115)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.7141
- **Indicators found:** text length: 61 chars

> We Face Additional Tax Liabilities and Collection Obligations...

#### Example 2: AMZN (ID: 93)

- **True label:** low
- **Predicted:** high
- **Score:** 0.7154
- **Indicators found:** text length: 63 chars

> We Are Impacted by Fraudulent or Unlawful Activities of Sellers...

#### Example 3: AMZN (ID: 274)

- **True label:** low
- **Predicted:** high
- **Score:** 0.7181
- **Indicators found:** text length: 34 chars

> impairment of other relationships;...

#### Example 4: AMZN (ID: 236)

- **True label:** low
- **Predicted:** high
- **Score:** 0.7472
- **Indicators found:** text length: 92 chars

> shorter payable and longer receivable cycles and the resultant negative impact on cash flow;...

#### Example 5: NVDA (ID: 55)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.7515
- **Indicators found:** text length: 69 chars

> Business disruptions could harm our operations and financial results....

### Missed Severity

*Real risk that embeddings underweighted*

**5 examples (showing top 5):**

#### Example 1: GOOGL (ID: 179)

- **True label:** high
- **Predicted:** low
- **Score:** 0.4239

> Our performance and future success depends in large part upon the continued service of key technical leads as well as members of our senior management team. For instance, Sundar Pichai is critical to ......

#### Example 2: GOOGL (ID: 130)

- **True label:** high
- **Predicted:** low
- **Score:** 0.4908

> Furthermore, failure to maintain and enhance our brands could harm our business, reputation, financial condition, and operating results. Our success will depend largely on our ability to remain a tech......

#### Example 3: GOOGL (ID: 159)

- **True label:** high
- **Predicted:** low
- **Score:** 0.4955

> The EU's Digital Markets Act, which will require in-scope companies to obtain user consent for combining data across certain products and require search engines to share anonymized data with rival com......

#### Example 4: AMZN (ID: 211)

- **True label:** high
- **Predicted:** low
- **Score:** 0.5532

> the extent to which we fail to maintain our unique culture of innovation, customer obsession, and long-term thinking, which has been critical to our growth and success;...

#### Example 5: AMZN (ID: 100)

- **True label:** high
- **Predicted:** low
- **Score:** 0.5733

> We depend on our senior management and other key personnel, including our President and CEO. We do not have "key person" life insurance policies. We also rely on other highly skilled personnel. Compet......

### Cross Reference

*References other sections without substantive content*

**1 examples (showing top 5):**

#### Example 1: AMZN (ID: 110)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.7317
- **Indicators found:** described elsewhere

> We have a rapidly evolving business model. The trading price of our common stock fluctuates significantly in response to, among other risks, the risks described elsewhere in this Item 1A, as well as: ......

### Hedging Language

*Overly cautious legal language without material concern*

**1 examples (showing top 5):**

#### Example 1: GOOGL (ID: 169)

- **True label:** medium
- **Predicted:** high
- **Score:** 0.7560
- **Indicators found:** difficult to predict

> Our operating results may fluctuate, which makes our results difficult to predict and could cause our results to fall short of expectations....

---

## Key Insights

### When Cosine Similarity Lies

1. **Boilerplate text**: Standard legal language that appears in every 10-K scores high because it contains risk-related words, but conveys no specific threat.

2. **Risk mitigation descriptions**: Text about insurance, hedging, or controls gets scored as high-risk because it mentions risks (even though it's describing protection).

3. **Hedging language**: Legal disclaimers using words like 'may', 'could', 'potential' trigger high similarity to the risk prompt.

4. **Boundary cases**: Medium vs Low and Medium vs High distinctions are inherently fuzzy—cosine similarity provides a continuous score that must be discretized.

### Recommendations

1. **Add boilerplate filter**: Pre-filter common boilerplate phrases before scoring.

2. **Negation detection**: Implement simple negation/mitigation detection to downweight risk mitigation text.

3. **Calibrate thresholds**: Use probability calibration to convert raw scores to risk probabilities.

4. **Ensemble approach**: Combine embedding similarity with keyword rules for edge cases.

---

*Generated by `evaluation/failure_mode_analysis.py`*